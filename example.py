import asyncio
import json
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
        codec_configs=[
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            {'name': 'numcodecs.zlib', 'configuration': {'level': 5}},
        ],
    )

    pipeline_blosc = CodecPipeline(
        name='BloscCompression',
        codec_configs=[
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            {
                'name': 'blosc',
                'configuration': {'cname': 'lz4', 'clevel': 5, 'shuffle': 'shuffle'},
            },
        ],
    )

    pipelines = [pipeline_zlib, pipeline_blosc]

    # 3. Run the comparison
    logger.info('Running benchmark comparison...')
    results = await run_comparison(
        datasets=[dataset],
        pipelines=pipelines,
        iterations=3,
    )

    # 4. Print results as JSON
    print(json.dumps(results.to_dicts(), indent=2))  # noqa: T201


if __name__ == '__main__':
    asyncio.run(main())
