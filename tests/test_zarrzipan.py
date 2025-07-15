import numpy as np
import zarr
from numcodecs import Blosc
from zarrzipan.zarrzipan import BenchmarkInput, PipelineConfig, run_comparison


def test_run_comparison():
    # a. Define datasets to test
    datasets_to_test = [
        BenchmarkInput(
            name='sequential_float32',
            array=np.arange(1_000_000, dtype='f4'),
            # Test this dataset both chunked and as a single block
            chunk_shape=(65536,),
        ),
        BenchmarkInput(
            name='sequential_float32_single_block',
            array=np.arange(1_000_000, dtype='f4'),
            chunk_shape=None,  # Will default to the full array shape
        ),
        BenchmarkInput(
            name='random_int16_2d',
            array=np.random.randint(0, 5000, size=(2000, 2000), dtype='i2'),
            chunk_shape=(256, 256),
        ),
    ]

    # b. Define pipelines to compare
    pipelines_to_test = [
        PipelineConfig(
            name='blosc_lz4_bitshuffle',
            codec_configs=[
                {'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': Blosc.SHUFFLE}
            ],
        ),
        PipelineConfig(
            name='quantize_f4_d2_blosc_zstd',
            codec_configs=[
                {'id': 'quantize', 'digits': 2, 'dtype': 'f4'},
                {'id': 'blosc', 'cname': 'zstd', 'clevel': 3},
            ],
        ),
        PipelineConfig(name='just_lz4', codec_configs=[{'id': 'lz4'}]),
    ]

    # c. Run the comparison
    comparison_results = run_comparison(
        datasets=datasets_to_test, pipelines=pipelines_to_test, iterations=1
    )

    # d. Assertions
    assert len(comparison_results.results) == len(datasets_to_test) * len(
        pipelines_to_test
    )
    assert comparison_results.results[0].dataset_name == 'sequential_float32'
    assert comparison_results.results[0].pipeline_name == 'blosc_lz4_bitshuffle'
