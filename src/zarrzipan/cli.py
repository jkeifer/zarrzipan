import asyncio
import json
import re
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click
import numpy as np
import obstore as obs
import xarray as xr
import yaml

from obstore.store import HTTPStore
from rich.console import Console
from rich.table import Table
from rich.text import Text

from zarrzipan.types import BenchmarkInput, CodecPipeline
from zarrzipan.zarrzipan import run_comparison

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')


@dataclass(frozen=True)
class Config:
    """
    A container for all the config information from the config file.
    """

    arrays: dict[str, dict[str, str | tuple[int, int]]]
    codec_pipelines: dict[str, list[dict[str, any]]]
    jobs: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, config) -> 'Config':
        arrays = {}
        for array in config['arrays']:
            arrays[array['name']] = {'href': array['href'], 'slice': array.get('slice')}
        codec_pipelines = {}
        for pipeline in config['codec-pipelines']:
            codec_pipelines[pipeline['name']] = pipeline['steps']

        return cls(arrays=arrays, codec_pipelines=codec_pipelines, jobs=config['jobs'])


def _parse_config(config_file):
    with open(config_file) as f:
        data = yaml.safe_load(f.read())

    config = Config.from_dict(data)
    return config


def _get_output_filepath():
    output_files_n = 0
    pattern = re.compile(r'output(\d+)\.json')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in OUTPUT_DIR.iterdir():
        match = pattern.match(f.name)
        if match:
            try:
                # Extract the number and convert it to an integer
                n = int(match.group(1))
                if n > output_files_n:
                    output_files_n = n
            except ValueError:
                # This handles cases where the extracted group might not be a valid integer
                continue
    return OUTPUT_DIR / f'output{output_files_n + 1}.json'


def _fetch_array(name: str, href: str):
    parts = urlsplit(href)
    filename = parts.path.split('/')[-1]

    filepath = DATA_DIR / name / filename

    if not filepath.exists():
        click.echo(f'Fetching {href}')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        store = HTTPStore(f'{parts.scheme}://{parts.netloc}')
        resp = obs.get(store, parts.path)

        with open(filepath, 'wb') as f:
            for chunk in resp:
                f.write(chunk)

    return filepath


def _load_output(output_file):
    # 0. Load the data
    with open(output_file) as f:
        ndjson_data = f.read()

    # 1. Clean and prepare the ndjson_data string.
    #    Remove empty lines and leading/trailing whitespace from each line, then join.
    cleaned_data_lines = [
        line.strip() for line in ndjson_data.splitlines() if line.strip()
    ]
    continuous_json_string = ''.join(cleaned_data_lines)

    # 2. Replace 'NaN' with 'null' in the entire string to make it JSON compliant.
    json_compliant_string = continuous_json_string.replace('NaN', 'null')

    # 3. Replace the '}{' pattern with '},{' to create a comma-separated list of JSON objects.
    #    This handles cases where JSON objects are directly concatenated without a comma.
    array_elements_string = json_compliant_string.replace('}{', '},{')

    # 4. Wrap the modified string with '[' and ']' to form a valid JSON array string.
    final_json_array_string = f'[{array_elements_string}]'

    # 5. Parse the resulting string using json.loads().
    return json.loads(final_json_array_string)


def _create_table(results):
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

    # Create a rich table
    table = Table(title='Compression Benchmark Results')

    # Add columns to the table based on desired_columns
    for header_name, _ in desired_columns:
        table.add_column(header_name, justify='left', style='cyan', no_wrap=False)

    # Add rows to the table
    for data_item in results:
        row_values = []
        for header_name, key in desired_columns:
            value = data_item.get(key, 'N/A')
            if value in ("NaN", "null", 'N/A'):
                return "-"

            # Format specific values
            if key == 'size_ratio':
                if value > mean_compression_ratio + 1:
                    color = 'green'
                elif value < max(mean_compression_ratio - 1, 1):
                    color = 'red'
                else:
                    color = 'cyan'
                formatted_value = f'{value:.2f}x'
                value = Text(formatted_value, style=color)
            elif key == 'lossiness_mae':
                value = f'{value:.4f}'
            elif key == 'chunk_shape':
                # Remove brackets and quotes if present from chunk_shape string representation
                value = str(value).replace('[', '').replace(']', '').replace("'", '')
            elif isinstance(value, (int, float)):
                value = f'{value:.2f}'

            row_values.append(str(value))
        table.add_row(*row_values)

    return table


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Print more output.')
@click.pass_context
def cli(ctx, verbose):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called directly)
    ctx.ensure_object(dict)
    ctx.obj['v'] = verbose


@cli.command()
@click.pass_context
@click.option(
    '--config',
    default='config.yaml',
    help='Filepath to config file',
    type=click.Path(exists=True),
)
@click.option(
    '--output-file',
    required=False,
    help=(
        'Filepath to output into. If specified will append to an '
        'existing file. If not specified a new file will be created '
        "within '{OUTPUT_DIR}'"
    ),
    type=click.Path(dir_okay=False),
)
def run(ctx, config: str, output_file: str | None = None) -> None:
    """Run the jobs in the config file fetching data if needed"""
    config = _parse_config(config)
    results = []

    if not output_file:
        output_file = _get_output_filepath()

    for i, job in enumerate(config.jobs):
        if ctx.obj['v']:
            click.echo(f'Running job [{i + 1}/{len(config.jobs)}]')

        array = config.arrays[job['array-name']]
        local_path = _fetch_array(job['array-name'], array['href'])
        ds = xr.open_dataset(local_path, chunks=None)

        if len(ds.keys()) == 1:
            data_array = next(v.squeeze() for v in ds.values())
        else:
            raise ValueError('More than one variable in dataset')

        if array['slice'] is not None:
            data_array = data_array.isel(
                **{k: slice(*v) for k, v in array['slice'].items()},
            )
        if ctx.obj['v']:
            click.echo(f'Array has shape: {data_array.shape}')

        pipelines = [
            CodecPipeline(name=name, codec_configs=config.codec_pipelines[name])
            for name in job['codec-pipelines']
        ]

        for chunk_shape in job.get('chunk-shapes', [None]):
            dataset = BenchmarkInput(
                name=job['array-name'],
                array=data_array.values,
                chunk_shape=chunk_shape,
            )

            if ctx.obj['v']:
                click.echo(
                    f'Running benchmark comparison with chunk shape: {chunk_shape} ...',
                )

            run_results = asyncio.run(
                run_comparison(
                    datasets=[dataset],
                    pipelines=pipelines,
                    iterations=job['iterations'] or 1,
                ),
            )

            with open(output_file, 'a') as f:
                for result in run_results.to_ndjson(indent=2):
                    f.write(result)
                    f.write('\n')

            results.extend(run_results.to_dicts())

    table = _create_table(results)

    # Print the table
    console = Console()
    console.print(table)


@cli.command()
@click.argument('output_file', type=click.Path(exists=True, dir_okay=False))
def show(output_file) -> None:
    """Show a table with all the outputs from a given file"""
    results = _load_output(output_file)
    table = _create_table(results)

    # Print the table
    console = Console()
    console.print(table)
