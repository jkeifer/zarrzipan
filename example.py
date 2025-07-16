import asyncio
import logging

import numpy as np

from zarrzipan.types import BenchmarkInput, CodecPipeline
from zarrzipan.zarrzipan import run_comparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 1. Define a sample dataset (BenchmarkInput)
    #    For demonstration, we'll use a simple NumPy array.
    #    In a real scenario, this would be your actual raster data.
    data_array = np.arange(10000, dtype=np.float32).reshape(100, 100)
    dataset = BenchmarkInput(
        name='SampleFloat32Array',
        array=data_array,
        chunk_shape=(50, 50),
    )

    # 2. Define sample codec pipelines
    #    Here we use 'zlib' and 'blosc' for demonstration.
    #    You can define any valid Zarr codec configuration.
    pipeline_zlib = CodecPipeline(
        name='ZlibCompression',
        codec_configs=[{'id': 'zlib', 'level': 5}],
    )

    pipeline_blosc = CodecPipeline(
        name='BloscCompression',
        codec_configs=[{'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1}],
    )

    pipelines = [pipeline_zlib, pipeline_blosc]

    # 3. Run the comparison
    logger.info('Running benchmark comparison...')
    results = await run_comparison(
        datasets=[dataset],
        pipelines=pipelines,
        iterations=3,
    )

    # 4. Print results
    logger.info('\n--- Benchmark Results ---')
    for result in results.results:
        logger.info(
            'Dataset: %s, Pipeline: %s',
            result.dataset_name,
            result.pipeline_name,
        )
        logger.info('  Chunk Shape: %s', result.chunk_shape)
        logger.info('  Iterations: %s', result.iterations)
        logger.info('  Compression Ratio: %.2fx', result.size_stats.ratio)
        logger.info('  Space Saving: %.2f', result.size_stats.space_saving)
        logger.info('  Compress Time (avg ms): %.2f', result.compress_timings.avg_ms)
        logger.info(
            '  Decompress Time (avg ms): %.2f',
            result.decompress_timings.avg_ms,
        )
        if result.lossiness_stats:
            logger.info('  Lossiness (MAE): %.4f', result.lossiness_stats.mae)
        logger.info('-' * 30)


if __name__ == '__main__':
    asyncio.run(main())
