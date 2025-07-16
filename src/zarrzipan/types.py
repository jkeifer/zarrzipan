from __future__ import annotations

import itertools
import json

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any

import numpy as np

from zarr.abc.codec import Codec
from zarr.abc.codec import CodecPipeline as ZCodecPipeline
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import ChunkCoords
from zarr.core.dtype import get_data_type_from_json
from zarr.core.dtype.common import check_dtype_spec_v3
from zarr.core.metadata.v3 import parse_codecs, validate_codecs
from zarr.registry import get_pipeline_class


def yield_chunk_slices(
    data_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> Iterator[tuple[slice, ...]]:
    chunk_counts = tuple((x // y for x, y in zip(data_shape, chunk_shape, strict=True)))
    for chunk_indices in itertools.product(*[range(x) for x in chunk_counts]):
        yield tuple(
            slice(
                j * chunk_shape[i],
                min((j + 1) * chunk_shape[i], data_shape[i]),
            )
            for i, j in enumerate(chunk_indices)
        )


@dataclass(frozen=True)
class BenchmarkInput:
    """
    A container for a named dataset and its chunking scheme.
    Validates that the input data is a NumPy array.
    """

    name: str
    array: np.ndarray
    chunk_shape: ChunkCoords
    _array_spec: ArraySpec = field(init=False)

    def __post_init__(self):
        if not isinstance(self.array, np.ndarray):
            raise TypeError("Input 'array' must be a numpy.ndarray.")

        if len(self.array.shape) != len(self.chunk_shape):
            raise ValueError(
                '`chunk_shape` and `shape` need to have the same number of dimensions.',
            )

        data_type_str = str(self.array.dtype)
        if not check_dtype_spec_v3(data_type_str):
            raise ValueError(f'Invalid data_type: {data_type_str!r}')
        data_type = get_data_type_from_json(data_type_str, zarr_format=3)

        # For benchmarking purposes, a fill value of 0 is generally acceptable
        # as we are primarily concerned with the compression of existing data.
        # This could be made configurable in the future if needed.
        fill_value_parsed = data_type.cast_scalar(0)

        array_spec = ArraySpec(
            shape=self.array.shape,
            dtype=data_type,
            fill_value=fill_value_parsed,
            config=ArrayConfig.from_dict({}),
            prototype=default_buffer_prototype(),
        )
        object.__setattr__(self, '_array_spec', array_spec)

    def yield_chunk_slices(self) -> Iterator[tuple[slice, ...]]:
        yield from yield_chunk_slices(self.array.shape, self.chunk_shape)

    def get_chunk(self, chunk_slice: tuple[slice, ...]) -> tuple[np.ndarray, ArraySpec]:
        chunk = self.array[chunk_slice]
        array_spec = replace(self._array_spec, shape=chunk.shape)
        return chunk, array_spec


@dataclass(frozen=True)
class CodecPipeline:
    """A container for a named codec pipeline configuration."""

    name: str
    codec_configs: list[dict[str, Any]]
    _codecs: tuple[Codec] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            '_codecs',
            parse_codecs(self.codec_configs),
        )

    def build_for_input(self, input: BenchmarkInput) -> ZCodecPipeline:
        validate_codecs(self._codecs, input._array_spec.dtype)
        codecs_parsed = tuple(
            deepcopy(c).evolve_from_array_spec(input._array_spec) for c in self._codecs
        )
        return get_pipeline_class().from_codecs(codecs_parsed)


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
    space_saving: float = field(init=False)

    def __post_init__(self):
        ratio = (
            self.uncompressed_size_bytes / self.compressed_size_bytes
            if self.compressed_size_bytes > 0
            else float('inf')
        )
        space_saving = (
            self.compressed_size_bytes / self.uncompressed_size_bytes
            if self.uncompressed_size_bytes > 0
            else float('inf')
        )
        object.__setattr__(self, 'ratio', ratio)
        object.__setattr__(self, 'space_saving', space_saving)


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
            'chunk_shape': str(self.chunk_shape),
            'iterations': self.iterations,
        }
        for parent_key, dataclass_instance in [
            ('size', self.size_stats),
            ('mem_compress', self.compress_memory_stats),
            ('mem_decompress', self.decompress_memory_stats),
            ('time_compress', self.compress_timings),
            ('time_decompress', self.decompress_timings),
        ]:
            if is_dataclass(dataclass_instance) and not isinstance(
                dataclass_instance,
                type,
            ):
                for key, value in asdict(dataclass_instance).items():
                    flat_dict[f'{parent_key}_{key}'] = value

        if self.lossiness_stats:
            for key, value in asdict(self.lossiness_stats).items():
                flat_dict[f'lossiness_{key}'] = value
        else:
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
        return [result.to_dict() for result in self.results]

    def to_ndjson(self, **json_kwargs) -> Iterator[str]:
        yield from (result.to_json(**json_kwargs) for result in self.results)

    def __iter__(self) -> Iterator[BenchmarkResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)
