# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass, replace

from vllm_torchtpu.logger import init_logger

# At Bq=128, a 32K-key FP32 output window occupies 16 MiB. Mosaic double
# buffers grid-window outputs, so long contexts need 32 MiB plus scorer
# scratch. A single <=64K window retains the original single-buffer layout.
SCORE_WINDOW = 32768
QUERY_TILE = 128
KEY_TILE = 2048

# Measured candidate-union and gather policies. Keep these separate from
# hardware lane widths and the packed-key format.
CANDIDATE_DENSE_UNION_MAX_BLOCKS = 16384
CANDIDATE_GATHER_ROWS_PER_SUBCORE = 512
CANDIDATE_GATHER_MAX_READS = 8
# (TPU generation, score width, candidate count, records per candidate block).
PUBLISH_CANDIDATE_SHAPES = frozenset({(7, 16384, 2048, 8)})


@dataclass(frozen=True)
class PagedMQAConfig:
    """Paged-decode policy; explicit overrides support kernel tuning.

    Four 16384-key buffers occupy 16 MiB. Four pending QK fragments use only
    half of v7's MRB at eight queries / 32 heads.
    """

    query_tile: int = 8
    key_tile: int = 16384
    issue_depth: int = 4
    query_group: int = 8
    key_buffers: int = 4

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (
                self.query_tile,
                self.key_tile,
                self.issue_depth,
                self.query_group,
                self.key_buffers,
            )
        ):
            raise ValueError(
                "Paged MQA tiles and issue_depth must be positive integers"
            )
        if self.key_buffers < 2:
            raise ValueError("Paged MQA needs at least two K buffers to overlap DMA")


# TPU v7, H32/D128, transposed packed FP8 pages; measured at P-state 7.
PACKED_FP8_MQA_CONFIG = PagedMQAConfig(6, 32768, 8, 6, 4)

# Dense tiles follow configured decode width, capped at one 256-row H32 MXU tile.
PACKED_FP8_MAX_QUERY_TILE = 8

# (queries, context, candidate blocks, topk, block size, cache banks).
# 96 = lcm(6 query rows, 32 SparseCore subcores). Three independent chunks
# expose TC/SC overlap without changing each query's six-row head grouping.
PACKED_FP8_INDEXER_QUERY_ROWS = {(288, 98304, 2048, 512, 8, 2): 96}

# Stage eight 16.5 KiB pages so their reads and writes can be issued together.
PACKED_FP8_WRITE_BATCH = 8


@dataclass(frozen=True)
class PersistentScoreConfig:
    key_tile: int
    compute_tile: int
    head_group: int

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.key_tile, self.compute_tile, self.head_group)
        ):
            raise ValueError("Score tiles and head_group must be positive integers")
        if self.key_tile % self.compute_tile:
            raise ValueError("key_tile must be a multiple of compute_tile")


PERSISTENT_SCORE_CONFIGS: dict[tuple[int, ...], PersistentScoreConfig] = {
    (6, 32, 128, 65536, 2): PersistentScoreConfig(99328, 1024, 32),
    (6, 32, 128, 131072, 2): PersistentScoreConfig(99328, 1024, 32),
}


# Decode tables are calibrated with decode_indexer's default score/top-k options.
# Request tiles bound per-group buffers; total batch size is not a lookup key.
logger = init_logger(__name__)
DEFAULT_DECODE_REQUEST_TILE = 16


@dataclass(frozen=True)
class DecodeTuningKey:
    device_kind: str
    queries: int
    heads: int
    score_capacity: int
    page_records: int
    k: int
    banks: int


# Calibrated at R48 on TPU7x, JAX 0.10.2 / libtpu 0.0.47; DSpark K5.
# Seeds 517/519, effective lengths 49408/98816; all other options default.
DECODE_REQUEST_TILES: dict[DecodeTuningKey, int] = {
    DecodeTuningKey("TPU7x", 6, 32, 65536, 128, 512, 2): 12,
    DecodeTuningKey("TPU7x", 6, 32, 131072, 128, 512, 2): 16,
}


def get_persistent_score_config(key: tuple[int, ...]) -> PersistentScoreConfig:
    """Choose persistent tiles; table misses keep the same kernel."""
    config = PERSISTENT_SCORE_CONFIGS.get(key)
    if config is None:
        # The native ragged-query suite exercises 4K DMA / 1K compute fragments.
        config = PersistentScoreConfig(4096, 1024, 32)
        logger.warning_once(
            "[decode_scores] No tuned config for %s; using %s", key, config
        )
    return config


def get_decode_request_tile(key: DecodeTuningKey, requests: int) -> int:
    """Bound each score group independently of the total batch size."""
    if requests < 1:
        raise ValueError("decode requests must be positive")
    tile = DECODE_REQUEST_TILES.get(key)
    if tile is None:
        tile = min(DEFAULT_DECODE_REQUEST_TILE, requests)
        logger.warning_once(
            "[decode_indexer] No tuned request_tile for %s; using %d", key, tile
        )
    if type(tile) is not int or tile < 1:
        raise ValueError(f"Invalid decode request_tile={tile} for {key}")
    return min(tile, requests)


@dataclass(frozen=True)
class SparseMQATuningKey:
    """Workload key for canonical halfplane int32 candidate caches."""

    device_kind: str
    heads: int
    capacity: int
    key_tile: int | None
    k: int


@dataclass(frozen=True)
class SparseMQABlockSizes:
    key_tile: int
    row_tile: int
    topk_block_size: int = 8192
    reconstruct_threshold: int = 8192
    compute_tile: int | None = None
    dma_tile: int | None = None


_TPU7X_SPARSE_MQA = SparseMQABlockSizes(
    key_tile=49152,
    row_tile=96,
    topk_block_size=16384,
    reconstruct_threshold=1024,
    compute_tile=8192,
    dma_tile=24576,
)


# Retain the score-buffer budgets without tying row tiles to request lengths.
SPARSE_MQA_BLOCK_SIZES: dict[SparseMQATuningKey, SparseMQABlockSizes] = {
    SparseMQATuningKey("TPU7x", 32, 40960, None, 512): SparseMQABlockSizes(4096, 288),
    SparseMQATuningKey("TPU7x", 32, 40960, 4096, 512): SparseMQABlockSizes(4096, 288),
    SparseMQATuningKey("TPU7x", 32, 49152, 4096, 512): SparseMQABlockSizes(4096, 288),
    SparseMQATuningKey("TPU7x", 32, 49152, 8192, 512): SparseMQABlockSizes(8192, 288),
    SparseMQATuningKey("TPU7x", 32, 49152, 16384, 512): SparseMQABlockSizes(
        16384, 144, 16384, 1024
    ),
    SparseMQATuningKey("TPU7x", 32, 49152, 24576, 512): SparseMQABlockSizes(
        24576, 144, 16384, 1024, compute_tile=4096
    ),
    SparseMQATuningKey("TPU7x", 32, 49152, 49152, 512): _TPU7X_SPARSE_MQA,
    SparseMQATuningKey("TPU7x", 32, 49152, None, 512): _TPU7X_SPARSE_MQA,
}
# 40K overflowed in 12/48 profiled rank-steps. The 48K bucket covers the
# captured maximum of 5138 blocks with headroom, at 576 MiB per 48 requests.
SPARSE_MQA_KEY_CAPACITY = 49152
# Basic config for table misses: bounded groups and a smaller MXU key tile.
# Keep the score-buffer budget below the measured 48 x 6 x 48K configuration.
DEFAULT_SPARSE_MQA_KEY_TILE = 4096
DEFAULT_SPARSE_MQA_ROW_TILE = 96
SPARSE_MQA_SCORE_ELEMENTS = 288 * SPARSE_MQA_KEY_CAPACITY


def get_sparse_mqa_block_sizes(
    key: SparseMQATuningKey, rows: int
) -> SparseMQABlockSizes:
    """Resolve scorer and selector tiles together for the complete pipeline."""
    if rows < 1 or key.capacity < 1:
        raise ValueError("sparse MQA rows and capacity must be positive")
    config = SPARSE_MQA_BLOCK_SIZES.get(key)
    if config is None:
        key_tile = key.key_tile
        if key_tile is None:
            key_tile = math.gcd(DEFAULT_SPARSE_MQA_KEY_TILE, key.capacity)
        limit = min(
            DEFAULT_SPARSE_MQA_ROW_TILE,
            max(1, SPARSE_MQA_SCORE_ELEMENTS // key.capacity),
        )
        config = SparseMQABlockSizes(key_tile, limit)
        logger.warning_once(
            "[sparse_mqa_indexer] No tuned block sizes for %s; using %s", key, config
        )
    if config.row_tile < 1:
        raise ValueError(f"Invalid sparse MQA row_tile={config.row_tile} for {key}")
    return replace(config, row_tile=min(config.row_tile, rows))
