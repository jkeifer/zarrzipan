import numpy as np
import pytest

from zarrzipan.types import BenchmarkInput, CodecPipeline
from zarrzipan.zarrzipan import run_comparison


@pytest.mark.asyncio
async def test_run_comparison_floats():
    # a. Define datasets to test
    arr_f4 = np.arange(1_000_000, dtype='f4').reshape((1000, 1000))
    datasets_to_test = [
        BenchmarkInput(
            name='sequential_float32',
            array=arr_f4,
            # Test this dataset both chunked and as a single block
            chunk_shape=(100, 100),
        ),
        BenchmarkInput(
            name='sequential_float32_single_block',
            array=arr_f4,
            chunk_shape=arr_f4.shape,  # Will default to the full array shape
        ),
    ]

    # b. Define pipelines to compare
    pipelines_to_test = [
        CodecPipeline(
            name='blosc_lz4_bitshuffle',
            codec_configs=[
                {
                    'name': 'bytes',
                    'configuration': {
                        'endian': 'little',
                    },
                },
                {
                    'name': 'blosc',
                    'configuration': {
                        'cname': 'lz4',
                        'clevel': 5,
                        'shuffle': 'shuffle',
                    },
                },
            ],
        ),
        CodecPipeline(
            name='quantize_f4_d2_blosc_zstd',
            codec_configs=[
                {
                    'name': 'numcodecs.quantize',
                    'configuration': {'digits': 2, 'dtype': 'f4'},
                },
                {
                    'name': 'bytes',
                    'configuration': {
                        'endian': 'little',
                    },
                },
                {'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 3}},
            ],
        ),
        CodecPipeline(
            name='just_lz4',
            codec_configs=[
                {
                    'name': 'bytes',
                    'configuration': {
                        'endian': 'little',
                    },
                },
                {'name': 'numcodecs.lz4', 'configuration': {}},
            ],
        ),
    ]

    # c. Run the comparison
    comparison_results = await run_comparison(
        datasets=datasets_to_test,
        pipelines=pipelines_to_test,
        iterations=1,
    )

    # d. Assertions
    assert len(comparison_results.results) == len(datasets_to_test) * len(
        pipelines_to_test,
    )
    assert comparison_results.results[0].dataset_name == 'sequential_float32'
    assert comparison_results.results[0].pipeline_name == 'blosc_lz4_bitshuffle'
    assert len(list(comparison_results.to_ndjson())) == 6


@pytest.mark.asyncio
async def test_run_comparison_ints():
    # a. Define datasets to test
    datasets_to_test = [
        BenchmarkInput(
            name='random_int16_2d',
            array=np.random.default_rng().integers(
                low=0,
                high=5000,
                size=(2000, 2000),
                dtype='i2',
            ),
            chunk_shape=(256, 256),
        ),
    ]

    # b. Define pipelines to compare
    pipelines_to_test = [
        CodecPipeline(
            name='blosc_lz4_bitshuffle',
            codec_configs=[
                {
                    'name': 'bytes',
                    'configuration': {
                        'endian': 'little',
                    },
                },
                {
                    'name': 'blosc',
                    'configuration': {
                        'cname': 'lz4',
                        'clevel': 5,
                        'shuffle': 'shuffle',
                    },
                },
            ],
        ),
        CodecPipeline(
            name='just_lz4',
            codec_configs=[
                {
                    'name': 'bytes',
                    'configuration': {
                        'endian': 'little',
                    },
                },
                {'name': 'numcodecs.lz4', 'configuration': {}},
            ],
        ),
    ]

    # c. Run the comparison
    comparison_results = await run_comparison(
        datasets=datasets_to_test,
        pipelines=pipelines_to_test,
        iterations=1,
    )

    # d. Assertions
    assert len(comparison_results.results) == len(datasets_to_test) * len(
        pipelines_to_test,
    )
    assert comparison_results.results[0].dataset_name == 'random_int16_2d'
    assert comparison_results.results[0].pipeline_name == 'blosc_lz4_bitshuffle'
    assert len(list(comparison_results.to_ndjson())) == 2
