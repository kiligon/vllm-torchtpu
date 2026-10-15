# SPDX-License-Identifier: Apache-2.0
"""Nightly multi-rank golden validation tests for Wan2.1 T2V on TPU v7x."""

import pytest
from torch_tpu._internal.utils import hardware

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Wan tests")

from vllm_omni.diffusion.data import DiffusionParallelConfig

from tests_omni.conftest import OmniRunner
from tests_omni.models.wan.utils import (
    WAN2_1_T2V_1_3B,
    WAN2_1_T2V_14B,
    WanTestConfig,
    run_wan_golden_test,
)

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

RING_UNSUPPORTED = (
    "Ring attention hangs on TPU: the ring K/V exchange never completes in "
    "torch_tpu's P2P send (ProcessGroupTpu.experimental_send)"
)

# Parallel degrees are passed to OmniRunner as extra kwargs; Omni silently
# ignores a parallel_config object.
CASES = [
    (WAN2_1_T2V_1_3B, {"cfg_parallel_size": 2}, "cfg2", ()),
    (WAN2_1_T2V_1_3B, {"ulysses_degree": 2}, "u2", ()),
    (WAN2_1_T2V_1_3B, {"tensor_parallel_size": 2}, "tp2", ()),
    (
        WAN2_1_T2V_14B,
        {"cfg_parallel_size": 2, "ulysses_degree": 4},
        "cfg2xu4",
        pytest.mark.multichip,
    ),
    (
        WAN2_1_T2V_14B,
        {"ulysses_degree": 8},
        "u8",
        pytest.mark.multichip,
    ),
    (
        WAN2_1_T2V_1_3B,
        {"ring_degree": 2},
        "ring2",
        pytest.mark.skip(reason=RING_UNSUPPORTED),
    ),
]


@pytest.fixture(autouse=True)
def _check_tpu_devices(degrees: dict[str, int]) -> None:
    world_size = DiffusionParallelConfig(**degrees).world_size
    # PCI scan: counts TPU devices without starting the TPU runtime here.
    device_count = hardware.get_tpu_device_count()
    if device_count < world_size:
        pytest.skip(f"needs {world_size} TPU devices, found {device_count}")


@pytest.mark.parametrize(
    "omni_runner,config,degrees",
    [
        pytest.param((c.path, None, degrees), c, degrees, id=case_id, marks=marks)
        for c, degrees, case_id, marks in CASES
    ],
    indirect=["omni_runner"],
)
def test_wan_multirank_golden_answers_tpu(
    omni_runner: OmniRunner,
    config: WanTestConfig,
    degrees: dict[str, int],
):
    """Verify multi-rank Wan2.1 T2V output against the CPU fp32 golden video."""
    world_size = DiffusionParallelConfig(**degrees).world_size
    parallel = omni_runner.engine.stage_clients[0].od_config.parallel_config
    assert parallel.world_size == world_size, (
        f"Engine runs {parallel.world_size} ranks, expected {world_size}"
    )
    run_wan_golden_test(omni_runner, config)
