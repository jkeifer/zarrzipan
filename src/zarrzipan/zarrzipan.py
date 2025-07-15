import time
import numpy as np
import zarr
import tracemalloc
import json
from zarr.util import iter_chunks
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Sequence, Callable, Tuple, Optional, Iterator

# -- 1. Structured Input Classes --

@dataclass(frozen=True)
class BenchmarkInput:
    """
    A container for a named dataset and its chunking strategy.

    Validates that the input data is a NumPy array.
    """
    name: str
    array: np.ndarray
    chunk_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if not isinstance(self.array, np.ndarray):
            raise TypeError("Input 'array' must be a numpy.ndarray.")

@dataclass(frozen=True)
class PipelineConfig:
    """A container for a named codec pipeline configuration."""
    name: str
    codec_configs: List[Dict[str, Any]]

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
        ratio = (self.uncompressed_size_bytes / self.compressed_size_bytes
                 if self.compressed_size_bytes > 0 else float('inf'))
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
    chunk_shape: Tuple[int, ...]
    iterations: int
    # Core Results
    size_stats: BenchmarkRatio
    compress_memory_stats: BenchmarkMemory
    decompress_memory_stats: BenchmarkMemory
    compress_timings: BenchmarkTimings
    decompress_timings: BenchmarkTimings
    lossiness_stats: Optional[BenchmarkLossiness] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flattens the nested result into a single-level dictionary."""
        flat_dict = {
            "dataset_name": self.dataset_name,
            "pipeline_name": self.pipeline_name,
            "chunk_shape": str(self.chunk_shape), # Use string representation for compatibility
            "iterations": self.iterations,
        }
        for parent_key, dataclass_instance in [
            ("size", self.size_stats),
            ("mem_compress", self.compress_memory_stats),
            ("mem_decompress", self.decompress_memory_stats),
            ("time_compress", self.compress_timings),
            ("time_decompress", self.decompress_timings),
        ]:
            for key, value in asdict(dataclass_instance).items():
                flat_dict[f"{parent_key}_{key}"] = value

        if self.lossiness_stats:
            for key, value in asdict(self.lossiness_stats).items():
                flat_dict[f"lossiness_{key}"] = value
        else: # Ensure keys exist even for lossless runs
            for key in asdict(BenchmarkLossiness(0,0,0)):
                 flat_dict[f"lossiness_{key}"] = None

        return flat_dict

    def to_json(self, **json_kwargs) -> str:
        """Serializes the flattened result dictionary to a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)

@dataclass
class ComparisonResults:
    """A collection of benchmark results with methods for exporting."""
    results: List[BenchmarkResult]

    def to_dicts(self) -> List[Dict[str, Any]]:
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

def _run_operation_benchmark(op: Callable, it: int) -> Tuple[List[float], List[int], Any]:
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
        if i == 0: op_result = result
    tracemalloc.stop()
    return timings_s, peak_mem_bytes, op_result

def _encode_by_chunk(data, pipeline, chunk_shape):
    # ... (This helper is unchanged)
    chunks = []
    for sel in iter_chunks(data.shape, chunk_shape):
        encoded = data[sel]
        for codec in pipeline: encoded = codec.encode(encoded)
        chunks.append(encoded)
    return chunks

def _decode_by_chunk(chunks, pipeline, chunk_shape, out_shape, out_dtype):
    # ... (This helper is unchanged)
    decoded = np.empty(out_shape, dtype=out_dtype)
    for i, sel in enumerate(iter_chunks(out_shape, chunk_shape)):
        d_chunk = chunks[i]
        for codec in reversed(pipeline): d_chunk = codec.decode(d_chunk)
        chunk_arr = np.frombuffer(d_chunk, dtype=out_dtype).reshape(decoded[sel].shape)
        decoded[sel] = chunk_arr
    return decoded

def benchmark_pipeline(
    pipeline_config: PipelineConfig,
    data_input: BenchmarkInput,
    iterations: int,
) -> BenchmarkResult:
    """The core function to benchmark one pipeline against one dataset."""
    pipeline = get_codec_pipeline(pipeline_config.codec_configs)
    data = data_input.array
    chunk_shape = data_input.chunk_shape if data_input.chunk_shape else data.shape

    # Benchmark Encoding
    ct_s, c_mem, compressed = _run_operation_benchmark(lambda: _encode_by_chunk(data, pipeline, chunk_shape), iterations)
    # Benchmark Decoding
    dt_s, d_mem, decompressed = _run_operation_benchmark(lambda: _decode_by_chunk(compressed, pipeline, chunk_shape, data.shape, data.dtype), iterations)

    # Collate results
    lossiness = None
    if np.issubdtype(data.dtype, np.number) and isinstance(decompressed, np.ndarray):
        diff = np.abs(data.astype('f8') - decompressed.astype('f8'))
        lossiness = BenchmarkLossiness(mae=np.mean(diff), mse=np.mean(diff**2), max_abs_error=np.max(diff))

    return BenchmarkResult(
        dataset_name=data_input.name, pipeline_name=pipeline_config.name,
        chunk_shape=chunk_shape, iterations=iterations,
        size_stats=BenchmarkRatio(data.nbytes, sum(len(c) for c in compressed)),
        compress_memory_stats=BenchmarkMemory(np.mean(c_mem)),
        decompress_memory_stats=BenchmarkMemory(np.mean(d_mem)),
        compress_timings=BenchmarkTimings(*(np.array(ct_s) * 1000).tolist()),
        decompress_timings=BenchmarkTimings(*(np.array(dt_s) * 1000).tolist()),
        lossiness_stats=lossiness,
    )

# -- 4. Main Comparison Runner --

def run_comparison(
    datasets: List[BenchmarkInput],
    pipelines: List[PipelineConfig],
    iterations: int = 3
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
    print(f"Starting comparison: {len(datasets)} datasets x {len(pipelines)} pipelines = {total_runs} total benchmarks.")

    for i, data_input in enumerate(datasets):
        for j, pipeline_config in enumerate(pipelines):
            print(f"  Running ({i*len(pipelines)+j+1}/{total_runs}): Dataset='{data_input.name}', Pipeline='{pipeline_config.name}'...")
            result = benchmark_pipeline(pipeline_config, data_input, iterations)
            all_results.append(result)

    print("Comparison finished.")
    return ComparisonResults(all_results)

# -- 5. Example Usage --

if __name__ == '__main__':
    # a. Define datasets to test
    datasets_to_test = [
        BenchmarkInput(
            name="sequential_float32",
            array=np.arange(1_000_000, dtype='f4'),
            # Test this dataset both chunked and as a single block
            chunk_shape=(65536,)
        ),
        BenchmarkInput(
            name="sequential_float32_single_block",
            array=np.arange(1_000_000, dtype='f4'),
            chunk_shape=None # Will default to the full array shape
        ),
        BenchmarkInput(
            name="random_int16_2d",
            array=np.random.randint(0, 5000, size=(2000, 2000), dtype='i2'),
            chunk_shape=(256, 256)
        ),
    ]

    # b. Define pipelines to compare
    pipelines_to_test = [
        PipelineConfig(
            name="blosc_lz4_bitshuffle",
            codec_configs=[{'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': zarr.codecs.Blosc.BIT}]
        ),
        PipelineConfig(
            name="quantize_f4_d2_blosc_zstd",
            codec_configs=[
                {'id': 'quantize', 'digits': 2, 'dtype': 'f4'},
                {'id': 'blosc', 'cname': 'zstd', 'clevel': 3}
            ]
        ),
        PipelineConfig(name="just_lz4", codec_configs=[{'id': 'lz4'}]),
    ]

    # c. Run the comparison
    comparison_results = run_comparison(datasets=datasets_to_test, pipelines=pipelines_to_test, iterations=3)

    # d. Use the results
    print("\n--- Example of accessing and exporting results ---")

    # Get the first result and print its flattened dictionary
    first_result = comparison_results.results[0]
    print("\nFlattened dictionary of the first result:")
    print(first_result.to_dict())

    # Save all results to an NDJSON file
    output_filepath = "benchmark_results.ndjson"
    comparison_results.to_ndjson(output_filepath, sort_keys=True)
    print(f"\nAll results saved to '{output_filepath}'. Each line is a JSON record.")
    print("This file can be easily loaded into pandas, Spark, or other analysis tools.")

    # Example of loading into pandas (if installed)
    try:
        import pandas as pd
        df = pd.DataFrame(comparison_results.to_dicts())
        print("\n--- Pandas DataFrame Preview ---")
        print(df[['dataset_name', 'pipeline_name', 'size_ratio', 'time_compress_avg_ms', 'time_decompress_avg_ms', 'lossiness_mae']].head())
    except ImportError:
        print("\n(Pandas not installed, skipping DataFrame preview.)")
