from __future__ import annotations

import logging
import time
import tracemalloc

from collections.abc import Callable
from functools import partial
from typing import Any

import numcodecs
import numpy as np

from zarr.abc.codec import Codec
from zarr.abc.codec import CodecPipeline as ZCodecPipeline
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.registry import get_ndbuffer_class

from zarrzipan.types import (
    BenchmarkInput,
    BenchmarkLossiness,
    BenchmarkMemory,
    BenchmarkRatio,
    BenchmarkResult,
    BenchmarkTimings,
    CodecPipeline,
    ComparisonResults,
)

logger = logging.getLogger(__name__)


def get_codecs(codec_configs: list[dict[str, Any]]) -> list[Codec]:
    """Initializes a list of Numcodecs codecs from a list of configurations."""
    codecs = []
    for config in codec_configs:
        config = config.copy()
        codec_id = config.pop('id')
        codec_cls = numcodecs.get_codec({'id': codec_id})
        codecs.append(codec_cls.from_config(config))
    return codecs


async def _encode_by_chunk(
    input: BenchmarkInput,
    pipeline: ZCodecPipeline,
) -> dict[tuple[slice, ...], tuple[Buffer, ArraySpec]]:
    """Encodes a numpy array chunk by chunk using the given codec pipeline."""
    encoded_chunks: dict[tuple[slice, ...], tuple[Buffer, ArraySpec]] = {}
    for chunk_slice in input.yield_chunk_slices():
        chunk, chunk_array_spec = input.get_chunk(chunk_slice)
        buffer = get_ndbuffer_class().from_numpy_array(chunk)
        encoded_chunk = next(iter(await pipeline.encode([(buffer, chunk_array_spec)])))

        if encoded_chunk is None:
            raise RuntimeError('Codec pipeline returned None for encoded chunk.')

        encoded_chunks[chunk_slice] = (encoded_chunk, chunk_array_spec)
    return encoded_chunks


async def _decode_by_chunk(
    input: BenchmarkInput,
    pipeline: ZCodecPipeline,
    chunks: dict[tuple[slice, ...], tuple[Buffer, ArraySpec]],
) -> np.ndarray:
    """Decodes chunks and writes them to a new numpy array."""
    out = np.empty_like(input.array)
    for chunk_slice, chunk_data in chunks.items():
        buffer, chunk_array_spec = chunk_data
        decoded_chunk = next(iter(await pipeline.decode([(buffer, chunk_array_spec)])))

        if decoded_chunk is None:
            raise RuntimeError('Codec pipeline returned None for decoded chunk.')

        out[chunk_slice] = decoded_chunk.as_numpy_array()
    return out


async def _run_operation_benchmark(
    op: Callable,
    it: int,
) -> tuple[list[float], list[int], Any]:
    timings_s, peak_mem_bytes, op_result = [], [], None
    tracemalloc.start()
    for i in range(it):
        tracemalloc.clear_traces()
        start_time = time.perf_counter()
        result = await op()
        _, peak_mem = tracemalloc.get_traced_memory()
        end_time = time.perf_counter()
        timings_s.append(end_time - start_time)
        peak_mem_bytes.append(peak_mem)
        if i == 0:
            op_result = result
    tracemalloc.stop()
    return timings_s, peak_mem_bytes, op_result


async def benchmark_pipeline(
    pipeline: CodecPipeline,
    data_input: BenchmarkInput,
    iterations: int,
) -> BenchmarkResult:
    """The core function to benchmark one pipeline against one dataset."""
    chunk_shape = data_input.chunk_shape or data_input.array.shape
    original_array = data_input.array
    _pipeline = pipeline.build_for_input(data_input)

    ct_s, c_mem, compressed = await _run_operation_benchmark(
        partial(
            _encode_by_chunk,
            data_input,
            _pipeline,
        ),
        iterations,
    )

    dt_s, d_mem, decompressed_array = await _run_operation_benchmark(
        partial(
            _decode_by_chunk,
            data_input,
            _pipeline,
            compressed,
        ),
        iterations,
    )

    lossiness = None
    # Lossiness is only meaningful for numerical data types.
    if np.issubdtype(original_array.dtype, np.number):
        diff = np.abs(original_array.astype('f8') - decompressed_array.astype('f8'))
        lossiness = BenchmarkLossiness(
            mae=np.mean(diff),
            mse=np.mean(diff**2),
            max_abs_error=np.max(diff),
        )

    ct_ms = np.array(ct_s) * 1000
    dt_ms = np.array(dt_s) * 1000

    return BenchmarkResult(
        dataset_name=data_input.name,
        pipeline_name=pipeline.name,
        chunk_shape=chunk_shape,
        iterations=iterations,
        size_stats=BenchmarkRatio(
            original_array.nbytes,
            sum(len(c[0]) for c in compressed.values()),
        ),
        compress_memory_stats=BenchmarkMemory(float(np.mean(c_mem))),
        decompress_memory_stats=BenchmarkMemory(float(np.mean(d_mem))),
        compress_timings=BenchmarkTimings(
            np.mean(ct_ms),
            np.min(ct_ms),
            np.max(ct_ms),
            np.std(ct_ms),
        ),
        decompress_timings=BenchmarkTimings(
            np.mean(dt_ms),
            np.min(dt_ms),
            np.max(dt_ms),
            np.std(dt_ms),
        ),
        lossiness_stats=lossiness,
    )


async def run_comparison(
    datasets: list[BenchmarkInput],
    pipelines: list[CodecPipeline],
    iterations: int = 3,
) -> ComparisonResults:
    """
    Runs a benchmark for every combination of dataset and pipeline.
    """
    all_results = []
    total_runs = len(datasets) * len(pipelines)
    logger.info(
        'Starting comparison: %d datasets x %d pipelines = %d total benchmarks.',
        len(datasets),
        len(pipelines),
        total_runs,
    )

    for i, data_input in enumerate(datasets):
        for j, pipeline_config in enumerate(pipelines):
            logger.info(
                "  Running (%d/%d): Dataset='%s', Pipeline='%s'...",
                i * len(pipelines) + j + 1,
                total_runs,
                data_input.name,
                pipeline_config.name,
            )
            result = await benchmark_pipeline(pipeline_config, data_input, iterations)
            all_results.append(result)

    logger.info('Comparison finished.')
    return ComparisonResults(all_results)
