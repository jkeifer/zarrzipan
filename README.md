# Zarrzipan: A raster compression benchmarking tool

Zarrzipan is a compression benchmarking tool based on Zarr.

## Usage

To get started with `zarrzipan`, you can run the provided example script:

```bash
uv run python example.py
```

This script demonstrates how to define a `BenchmarkInput` (your dataset) and `CodecPipeline` (your compression configuration) and then run a comparison using `zarrzipan.run_comparison`.

For more detailed usage, refer to the `example.py` file and the source code in `src/zarrzipan/`.
