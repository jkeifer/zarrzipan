# Zarrzipan: A raster compression benchmarking tool

Zarrzipan is a compression benchmarking tool based on Zarr. It features a
python API for declaratively describing numpy arrays and zarr compression
pipelines to be benchmarked.

Note that this is experimental software written with significant LLM
assistance. Some metrics such as memory usage and timing may not be accurately
measured.

## Usage

Zarrzipan includes a little CLI tool. Configure your arrays and jobs in a
yaml file as demonstrated in [zarrzipan.yaml](zarrzipan.yaml) and then run:

```bash
uv run zarrzipan compare | uv run zarrzipan render
```

The output of `zarrzipan compare` is ndjson, so if you are doing many runs you
can build out your results over time:

```bash
uv run zarrzipan compare >> output.json
```

And render the output independently:

```bash
uv run zarrzipan render -f output.json
```

### Benchmark Results

The following table summarizes the benchmark results described in [zarrzipan.yaml](zarrzipan.yaml):

<!--  [[[cog
import json
import subprocess
from cog import out

# Run the jobs defined in zarrzipan.yaml and capture the JSON output
result = subprocess.run(
    ["uv", "run", "zarrzipan", "compare"],
    stdout=subprocess.PIPE,
    text=True,
    check=True
)

# Parse the JSON output
data = [json.loads(line) for line in result.stdout.split("\n")[:-1]]

# Generate the Markdown table
out('| Dataset Name | Pipeline Name | Chunk Shape | Iterations | Compression Ratio | Space Saving | Avg Compress Time (ms) | Avg Decompress Time (ms) | Lossiness (MAE) |\n')
out('|---|---|---:|---:|---:|---:|---:|---:|---:|\n')
for row in data:
    lossiness_mae = row['lossiness_mae'] and f"{row['lossiness_mae']:.4f}"
    out(f"| {row['dataset_name']} | {row['pipeline_name']} | {row['chunk_shape']} | {row['iterations']} | {row['size_ratio']:.2f}x | {row['size_space_saving']:.2f} | {row['time_compress_avg_ms']:.2f} | {row['time_decompress_avg_ms']:.2f} | {lossiness_mae} |\n")
 ]]]  -->
| Dataset Name | Pipeline Name | Chunk Shape | Iterations | Compression Ratio | Space Saving | Avg Compress Time (ms) | Avg Decompress Time (ms) | Lossiness (MAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Single band COG | Zlib level 5 | (1024, 1024) | 1 | 2.08x | 0.48 | 3651.11 | 383.99 | None |
| Single band COG | Blosc (defaults) | (1024, 1024) | 1 | 1.15x | 0.87 | 88.99 | 43.78 | None |
| Single band COG | Zlib level 5 | (2048, 2048) | 1 | 2.06x | 0.49 | 3609.53 | 402.76 | None |
| Single band COG | Blosc (defaults) | (2048, 2048) | 1 | 1.15x | 0.87 | 79.99 | 41.48 | None |
| Single band COG | Zlib level 5 | (4096, 4096) | 1 | 2.05x | 0.49 | 3593.96 | 398.66 | None |
| Single band COG | Blosc (defaults) | (4096, 4096) | 1 | 1.15x | 0.87 | 65.32 | 42.13 | None |
<!-- [[[end]]] (sum: FYssP979Y0) -->

## Other tooling like this

### geotiff-benchmark

The [geotiff-benchmark](https://github.com/kokoalberti/geotiff-benchmark) tool
is a gdal-based compression benchmarking solution targeting the GeoTIFF format
specifically. While not know to us prior to the initial Zarrzipan development,
geotiff-benchmark is similar in its goals, and allows testing a set of
declaratively-configured GeoTIFF compression pipelines. [Read the corresponding
blog
post](https://kokoalberti.com/articles/geotiff-compression-optimization-guide/)
to learn more about the tool.

## What does "zarrzipan" mean?

Zarrzipan, pronounced like marzipan, is like "Zarr zip analysis", where zip is
a general allusion to compression and not a reference to the specific zip
archive format.
