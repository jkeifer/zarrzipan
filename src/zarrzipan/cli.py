import asyncio
import json
import re
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlsplit

import click
import numpy as np
import obstore as obs
import xarray as xr
import yaml

from obstore.store import HTTPStore
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.text import Text

from zarrzipan.types import BenchmarkInput, CodecPipeline
from zarrzipan.zarrzipan import run_comparison

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')


class ArrayDict(TypedDict):
    name: str
    href: str
    slice: dict[str, list[int]] | None


class CodecPipelineDict(TypedDict):
    name: str
    steps: list[dict[str, Any]]


JobDict = TypedDict(
    'JobDict',
    {
        'array': str,
        'pipelines': list[str],
        'chunk-shapes': list[list[int] | None],
        'iterations': int,
    },
)


class ConfigDict(TypedDict):
    arrays: list[ArrayDict]
    pipelines: list[CodecPipelineDict]
    jobs: list[JobDict]


@dataclass(frozen=True)
class Job:
    array: ArrayDict
    pipelines: list[CodecPipelineDict]
    chunk_shapes: list[list[int] | None]
    iterations: int

    @classmethod
    def from_config(
        cls,
        arrays: dict[str, ArrayDict],
        pipelines: dict[str, CodecPipelineDict],
        job: JobDict,
    ) -> 'Job':
        return cls(
            array=arrays[job['array']],
            pipelines=[pipelines[p] for p in job['pipelines']],
            chunk_shapes=job.get('chunk-shapes', [None]),
            iterations=job.get('iterations', 1),
        )


@dataclass(frozen=True)
class Config:
    """
    A container for all the config information from the config file.
    """

    jobs: list[Job]

    @classmethod
    def from_file(cls, config_file: Path) -> 'Config':
        with config_file.open('r') as f:
            config = yaml.safe_load(f.read())

        arrays = {array['name']: array for array in config['arrays']}
        pipelines = {pipeline['name']: pipeline for pipeline in config['pipelines']}

        jobs = [Job.from_config(arrays, pipelines, job) for job in config['jobs']]

        return cls(jobs=jobs)


def _get_output_filepath(output_dir: Path) -> Path:
    """
    Given a directory of files with the same naming style
    (output1.json, output2.json, ...) figure out the next filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'output(\d+)\.json')

    output_files_n = 0
    for f in output_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            try:
                # Extract the number and convert it to an integer
                n = int(match.group(1))
                if n > output_files_n:
                    output_files_n = n
            except ValueError:
                # For when the extracted group might not be a valid integer
                continue

    return output_dir / f'output{output_files_n + 1}.json'


def _fetch_array(data_dir: Path, name: str, href: str) -> Path:
    parts = urlsplit(href)
    filename = parts.path.split('/')[-1]

    filepath = data_dir / name / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        click.echo(f'Fetching {href}')

        store = HTTPStore(f'{parts.scheme}://{parts.netloc}')
        resp = obs.get(store, parts.path)

        with filepath.open('wb') as f:
            for chunk in resp:
                f.write(chunk)

    return filepath


def _create_table(results: list[dict[str, Any]]) -> Table:
    """Create a rich table matching the markdown table"""
    # Define the desired column order and their corresponding keys in the JSON data
    desired_columns = [
        ('Dataset Name', 'dataset_name'),
        ('Pipeline Name', 'pipeline_name'),
        ('Chunk Shape', 'chunk_shape'),
        ('Iterations', 'iterations'),
        ('Compression Ratio', 'size_ratio'),
        ('Space Saving', 'size_space_saving'),
        ('Avg Compress Time (ms)', 'time_compress_avg_ms'),
        ('Avg Decompress Time (ms)', 'time_decompress_avg_ms'),
        ('Lossiness (MAE)', 'lossiness_mae'),
    ]

    mean_compression_ratio = np.mean([item['size_ratio'] for item in results])

    # Helper function to format values
    def format_value(key, value):
        if key == 'size_ratio':
            color = 'cyan'
            if value > mean_compression_ratio + 1:
                color = 'green'
            elif value < max(mean_compression_ratio - 1, 1):
                color = 'red'
            return Text(f'{value:.2f}x', style=color)
        if key == 'lossiness_mae':
            return f'{value:.4f}'
        if key == 'chunk_shape':
            return str(value).replace('[', '(').replace(']', ')')
        if isinstance(value, float):
            return f'{value:.2f}'
        return str(value)

    # Create a rich table
    table = Table(title='Compression Benchmark Results')

    # Add columns to the table
    for header_name, _ in desired_columns:
        table.add_column(header_name, justify='right', style='cyan', no_wrap=False)

    # Add rows to the table
    for data_item in results:
        row_values = [format_value(key, data_item[key]) for _, key in desired_columns]
        table.add_row(*row_values)

    return table


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Print more output.')
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['v'] = verbose


@cli.command()
@click.pass_context
@click.option(
    '--config-file',
    '-f',
    default='config.yaml',
    help='Filepath to config file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    '--data-dir',
    default=DATA_DIR,
    help=(f"Directory to cached data in. If not specified '{DATA_DIR}' will be used"),
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    '--output-file',
    '-o',
    required=False,
    help=(
        'Filepath to output into. If specified will append to an '
        'existing file. If not specified a new file will be created '
        f"within '{OUTPUT_DIR}'"
    ),
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    '--output-dir',
    default=OUTPUT_DIR,
    help=(
        'Directory to put output into. If not specified '
        f"'{OUTPUT_DIR}' will be used unless output-file is specified"
    ),
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    '--hide',
    '-h',
    is_flag=True,
    help='Do not show the table.',
)
def run(
    ctx,
    config_file: Path,
    data_dir: Path = DATA_DIR,
    output_file: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    hide: bool = False,
) -> None:
    """Run the jobs in the config file fetching data if needed"""

    config = Config.from_file(config_file)

    if not output_file:
        output_file = _get_output_filepath(output_dir)

    results = []
    for job in track(config.jobs, description='Running jobs...'):
        local_path = _fetch_array(data_dir, job.array['name'], job.array['href'])

        # Open the dataset from local with xarray
        ds = xr.open_dataset(local_path, chunks=None)

        # TODO: Add handling for when there is more than one variable
        if len(ds.keys()) == 1:
            data_array = next(v.squeeze() for v in ds.values())
        else:
            raise ValueError('More than one variable in dataset')

        if job.array['slice'] is not None:
            data_array = data_array.isel(
                {k: slice(*v) for k, v in job.array['slice'].items()},
            )

        if ctx.obj['v']:
            click.echo(f'Array has shape: {data_array.shape}')

        pipelines = [
            CodecPipeline(name=pipeline['name'], codec_configs=pipeline['steps'])
            for pipeline in job.pipelines
        ]

        for chunk_shape in job.chunk_shapes:
            dataset = BenchmarkInput(
                name=job.array['name'],
                array=data_array.values,
                chunk_shape=tuple(chunk_shape) if chunk_shape else data_array.shape,
            )

            if ctx.obj['v']:
                click.echo(
                    f'Running benchmark comparison with chunk shape: {chunk_shape} ...',
                )

            run_results = asyncio.run(
                run_comparison(
                    datasets=[dataset],
                    pipelines=pipelines,
                    iterations=job.iterations,
                ),
            )

            with output_file.open('a') as f:
                for result in run_results.to_ndjson():
                    f.write(result)
                    f.write('\n')

            results.extend(run_results.to_dicts())

    if not hide:
        table = _create_table(results)

        # Print the table
        console = Console()
        console.print(table)


@cli.command()
@click.argument(
    'output_file',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def show(output_file: Path) -> None:
    """Show a table with all the outputs from a given file"""
    with output_file.open('r') as f:
        results = [json.loads(line) for line in f.readlines()]

    table = _create_table(results)

    # Print the table
    console = Console()
    console.print(table)
