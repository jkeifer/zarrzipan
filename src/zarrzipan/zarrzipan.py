import json
import time
import tracemalloc

import numcodecs
import numpy as np
import zarr

from numcodecs.abc import Codec


def get_codec_pipeline(codec_configs: list[dict[str, Any]]) -> list[Codec]:
    """Initializes a list of Numcodecs codecs from a list of configurations."""
    codecs = []
    for config in codec_configs:
        config = config.copy()
        codec_id = config.pop('id')
        codec_cls = numcodecs.get_codec({'id': codec_id})
        codecs.append(codec_cls.from_config(config))
    return codecs


def _encode_by_chunk(data: Array, pipeline: list[Codec]) -> list[bytes]:
    """Encodes a Zarr array chunk by chunk using the given codec pipeline."""
    return [pipeline.encode(chunk) for chunk in data.blocks]


def _decode_by_chunk(
    chunks: list[bytes],
    pipeline: list[Codec],
    chunk_coords: list[tuple[int, ...]],
    out: Array,
) -> Array:
    """Decodes chunks and writes them to the output Zarr array."""
    for i, chunk_bytes in enumerate(chunks):
        decoded_chunk = pipeline.decode(chunk_bytes)
        out.blocks[chunk_coords[i]] = decoded_chunk
    return out


from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

# -- 1. Structured Input Classes --


@dataclass(frozen=True)
class BenchmarkInput:
    """
    A container for a named dataset and its chunking strategy.

    Validates that the input data is a NumPy array.
    """

    name: str
    array: np.ndarray
    chunk_shape: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.array, np.ndarray):
            raise TypeError("Input 'array' must be a numpy.ndarray.")


@dataclass(frozen=True)
class PipelineConfig:
    """A container for a named codec pipeline configuration."""

    name: str
    codec_configs: list[dict[str, Any]]


# -- 2. Structured Output & Result Classes --


@dataclass(frozen=True)
class BenchmarkTimings:
    avg_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float


@dataclass(frozen=True)
class BenchmarkRatio:
    uncompressed_size_bytes: int
    compressed_size_bytes: int
    ratio: float = field(init=False)

    def __post_init__(self):
        ratio = (
            self.uncompressed_size_bytes / self.compressed_size_bytes
            if self.compressed_size_bytes > 0
            else float('inf')
        )
        object.__setattr__(self, 'ratio', ratio)


@dataclass(frozen=True)
class BenchmarkMemory:
    avg_peak_bytes: float


@dataclass(frozen=True)
class BenchmarkLossiness:
    mae: float
    mse: float
    max_abs_error: float


@dataclass(frozen=True)
class BenchmarkResult:
    """
    A frozen dataclass holding all results for a single benchmark run.
    Includes serialization methods to simplify downstream analysis.
    """

    # Metadata
    dataset_name: str
    pipeline_name: str
    chunk_shape: tuple[int, ...]
    iterations: int
    # Core Results
    size_stats: BenchmarkRatio
    compress_memory_stats: BenchmarkMemory
    decompress_memory_stats: BenchmarkMemory
    compress_timings: BenchmarkTimings
    decompress_timings: BenchmarkTimings
    lossiness_stats: BenchmarkLossiness | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flattens the nested result into a single-level dictionary."""
        flat_dict = {
            'dataset_name': self.dataset_name,
            'pipeline_name': self.pipeline_name,
            'chunk_shape': str(
                self.chunk_shape
            ),  # Use string representation for compatibility
            'iterations': self.iterations,
        }
        for parent_key, dataclass_instance in [
            ('size', self.size_stats),
            ('mem_compress', self.compress_memory_stats),
            ('mem_decompress', self.decompress_memory_stats),
            ('time_compress', self.compress_timings),
            ('time_decompress', self.decompress_timings),
        ]:
            for key, value in asdict(dataclass_instance).items():
                flat_dict[f'{parent_key}_{key}'] = value

        if self.lossiness_stats:
            for key, value in asdict(self.lossiness_stats).items():
                flat_dict[f'lossiness_{key}'] = value
        else:  # Ensure keys exist even for lossless runs
            for key in asdict(BenchmarkLossiness(0, 0, 0)):
                flat_dict[f'lossiness_{key}'] = None

        return flat_dict

    def to_json(self, **json_kwargs) -> str:
        """Serializes the flattened result dictionary to a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)


@dataclass
class ComparisonResults:
    """A collection of benchmark results with methods for exporting."""

    results: list[BenchmarkResult]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Returns all results as a list of flattened dictionaries."""
        return [result.to_dict() for result in self.results]

    def to_ndjson(self, filepath: str, **json_kwargs):
        """
        Writes all results to a file in NDJSON (newline-delimited JSON) format.

        Args:
            filepath: The path to the output file.
            **json_kwargs: Keyword arguments passed to json.dumps (e.g., sort_keys=True).
        """
        with open(filepath, 'w') as f:
            for result in self.results:
                f.write(result.to_json(**json_kwargs) + '\n')

    def __iter__(self) -> Iterator[BenchmarkResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


# -- 3. Core Benchmarking Logic --


def _run_operation_benchmark(
    op: Callable, it: int
) -> tuple[list[float], list[int], Any]:
    # ... (This helper is unchanged but copied for completeness)
    timings_s, peak_mem_bytes, op_result = [], [], None
    tracemalloc.start()
    for i in range(it):
        tracemalloc.clear_traces()
        start_time = time.perf_counter()
        result = op()
        _, peak_mem = tracemalloc.get_traced_memory()
        end_time = time.perf_counter()
        timings_s.append(end_time - start_time)
        peak_mem_bytes.append(peak_mem)
        if i == 0:
            op_result = result
    tracemalloc.stop()
    return timings_s, peak_mem_bytes, op_result


def _encode_by_chunk(data: Array, pipeline: list[Codec]) -> list[bytes]:
    """Encodes a Zarr array chunk by chunk using the given codec pipeline."""
    encoded_chunks = []
    for chunk_coords in np.ndindex(data.chunks):
        chunk = data.blocks[chunk_coords]
        encoded = chunk
        for codec in pipeline:
            encoded = codec.encode(encoded)
        encoded_chunks.append(encoded)
    return encoded_chunks


def _decode_by_chunk(
    chunks: list[bytes],
    pipeline: list[Codec],
    out: Array,
) -> Array:
    """Decodes chunks and writes them to the output Zarr array."""
    for i, chunk_coords in enumerate(np.ndindex(out.chunks)):
        decoded_chunk = chunks[i]
        for codec in reversed(pipeline):
            decoded_chunk = codec.decode(decoded_chunk)
        out.blocks[chunk_coords] = decoded_chunk
    return out


def benchmark_pipeline(
    pipeline_config: PipelineConfig,
    data_input: BenchmarkInput,
    iterations: int,
) -> BenchmarkResult:
    """The core function to benchmark one pipeline against one dataset."""
    pipeline = get_codec_pipeline(pipeline_config.codec_configs)

    # Convert numpy array to zarr array
    z_array = zarr.array(data_input.array, chunks=data_input.chunk_shape)

    # Benchmark Encoding
    ct_s, c_mem, compressed = _run_operation_benchmark(
        lambda: _encode_by_chunk(z_array, pipeline),
        iterations,
    )

    # Prepare an empty array for decoding
    z_out = zarr.empty_like(z_array)

    # Benchmark Decoding
    dt_s, d_mem, decompressed_array = _run_operation_benchmark(
        lambda: _decode_by_chunk(compressed, pipeline, z_out),
        iterations,
    )

    # Collate results
    lossiness = None
    if np.issubdtype(z_array.dtype, np.number):
        diff = np.abs(z_array[:].astype('f8') - decompressed_array[:].astype('f8'))
        lossiness = BenchmarkLossiness(
            mae=np.mean(diff),
            mse=np.mean(diff**2),
            max_abs_error=np.max(diff),
        )

    return BenchmarkResult(
        dataset_name=data_input.name,
        pipeline_name=pipeline_config.name,
        chunk_shape=z_array.chunks,
        iterations=iterations,
        size_stats=BenchmarkRatio(z_array.nbytes, sum(len(c) for c in compressed)),
        compress_memory_stats=BenchmarkMemory(np.mean(c_mem)),
        decompress_memory_stats=BenchmarkMemory(np.mean(d_mem)),
        compress_timings=BenchmarkTimings(*(np.array(ct_s) * 1000).tolist()),
        decompress_timings=BenchmarkTimings(*(np.array(dt_s) * 1000).tolist()),
        lossiness_stats=lossiness,
    )


# -- 4. Main Comparison Runner --


def run_comparison(
    datasets: list[BenchmarkInput],
    pipelines: list[PipelineConfig],
    iterations: int = 3,
) -> ComparisonResults:
    """
    Runs a benchmark for every combination of dataset and pipeline.

    Args:
        datasets: A list of datasets to test.
        pipelines: A list of codec pipelines to test.
        iterations: The number of times to repeat each benchmark for averaging.

    Returns:
        A ComparisonResults object containing all benchmark results.
    """
    all_results = []
    total_runs = len(datasets) * len(pipelines)
    print(
        f'Starting comparison: {len(datasets)} datasets x {len(pipelines)} pipelines = {total_runs} total benchmarks.'
    )

    for i, data_input in enumerate(datasets):
        for j, pipeline_config in enumerate(pipelines):
            print(
                f"  Running ({i * len(pipelines) + j + 1}/{total_runs}): Dataset='{data_input.name}', Pipeline='{pipeline_config.name}'..."
            )
            result = benchmark_pipeline(pipeline_config, data_input, iterations)
            all_results.append(result)

    print('Comparison finished.')
    return ComparisonResults(all_results)
