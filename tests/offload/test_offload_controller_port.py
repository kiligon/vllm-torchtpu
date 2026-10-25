# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for offload controller port defaults and TCP reachability probe."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import socket
import sys
import threading
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

LINUX_EPHEMERAL_LOW = 32768
LINUX_EPHEMERAL_HIGH = 60999


class TestOffloadControllerPortAndProbe(unittest.TestCase):
    rs: Any

    @classmethod
    def setUpClass(cls) -> None:
        stubs: dict[str, object] = {}
        for mod in (
            "torch",
            "vllm",
            "vllm.config",
            "vllm.envs",
            "vllm.logger",
            "vllm.v1",
            "vllm.v1.kv_cache_interface",
            "vllm.v1.kv_offload",
            "vllm.v1.kv_offload.base",
            "vllm.v1.kv_offload.config",
        ):
            root = mod.split(".")[0]
            if mod not in sys.modules and importlib.util.find_spec(root) is None:
                stubs[mod] = MagicMock()
        if "vllm.v1.kv_offload.base" in stubs:
            base_stub = stubs["vllm.v1.kv_offload.base"]
            assert isinstance(base_stub, MagicMock)
            base_stub.LoadStoreSpec = object
            base_stub.OffloadingManager = object
            base_stub.OffloadingSpec = object
            base_stub.OffloadingWorker = object
        cls._had_rs = "vllm_torchtpu.offload.raiden_store" in sys.modules
        cls._has_stubs = bool(stubs)
        cls._patcher = patch.dict(sys.modules, stubs)
        cls._patcher.start()
        import vllm_torchtpu.offload.raiden_store as rs

        cls.rs = rs

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()
        if cls._has_stubs and not cls._had_rs:
            sys.modules.pop("vllm_torchtpu.offload.raiden_store", None)

    def test_default_controller_port_outside_linux_ephemeral_range(self) -> None:
        port = self.rs._DEFAULT_CONTROLLER_PORT
        self.assertGreaterEqual(port, 1024)
        self.assertLess(port, LINUX_EPHEMERAL_LOW)

    def test_controller_reachable_rejects_tcp_loopback_self_connect(self) -> None:
        worker_cls = self.rs.RaidenStoreOffloadingWorker
        worker = object.__new__(worker_cls)
        worker._controller_address = "127.0.0.1:48800"

        self_connected_sock = MagicMock()
        self_connected_sock.__enter__.return_value = self_connected_sock
        self_connected_sock.getsockname.return_value = ("127.0.0.1", 48800)
        self_connected_sock.getpeername.return_value = ("127.0.0.1", 48800)

        with patch.object(
            self.rs.socket,
            "create_connection",
            return_value=self_connected_sock,
        ):
            self.assertFalse(worker._controller_reachable())

    def test_controller_reachable_accepts_listening_server_and_rejects_closed(
        self,
    ) -> None:
        worker_cls = self.rs.RaidenStoreOffloadingWorker
        worker = object.__new__(worker_cls)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            worker._controller_address = f"[127.0.0.1]:{port}"
            self.assertTrue(worker._controller_reachable())

        self.assertFalse(worker._controller_reachable())

    def test_register_loop_retries_past_self_connect_until_server_listens(
        self,
    ) -> None:
        worker_cls = self.rs.RaidenStoreOffloadingWorker
        worker = object.__new__(worker_cls)
        worker._device_tensors = []
        worker._host_blocks_to_allocate = 4
        worker._controller_address = "127.0.0.1:28800"
        worker._rank = 0
        worker._mgr = None
        worker._backend_configs = []
        worker._stop_registration = threading.Event()

        self_connected_sock = MagicMock()
        self_connected_sock.__enter__.return_value = self_connected_sock
        self_connected_sock.getsockname.return_value = ("127.0.0.1", 28800)
        self_connected_sock.getpeername.return_value = ("127.0.0.1", 28800)

        server_connected_sock = MagicMock()
        server_connected_sock.__enter__.return_value = server_connected_sock
        server_connected_sock.getsockname.return_value = ("127.0.0.1", 50123)
        server_connected_sock.getpeername.return_value = ("127.0.0.1", 28800)

        fake_kcm_mod: Any = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
        kcm_calls: list[dict[str, object]] = []

        def fake_kv_cache_manager(**kwargs: object) -> str:
            kcm_calls.append(kwargs)
            return "registered-manager"

        fake_kcm_mod.KVCacheManager = fake_kv_cache_manager
        fake_torch_pkg: Any = types.ModuleType("tpu_sync.api.torch")
        fake_torch_pkg.kv_cache_manager = fake_kcm_mod
        fake_api_pkg: Any = types.ModuleType("tpu_sync.api")
        fake_api_pkg.torch = fake_torch_pkg
        fake_root_pkg: Any = types.ModuleType("tpu_sync")
        fake_root_pkg.api = fake_api_pkg

        with (
            patch.object(
                self.rs.socket,
                "create_connection",
                side_effect=[self_connected_sock, server_connected_sock],
            ) as create_conn,
            patch.object(worker._stop_registration, "wait", return_value=False),
            patch.dict(
                sys.modules,
                {
                    "tpu_sync": fake_root_pkg,
                    "tpu_sync.api": fake_api_pkg,
                    "tpu_sync.api.torch": fake_torch_pkg,
                    "tpu_sync.api.torch.kv_cache_manager": fake_kcm_mod,
                },
            ),
        ):
            worker._register_loop()

        self.assertEqual(create_conn.call_count, 2)
        self.assertEqual(len(kcm_calls), 1)
        self.assertEqual(worker._mgr, "registered-manager")

    def test_p4d2_launcher_managed_ports_disjoint_and_outside_ephemeral_range(
        self,
    ) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        launcher = (
            repo_root / "examples" / "disagg" / "launch_qwen35_p4d2_v2_baseline.sh"
        ).read_text(encoding="utf-8")

        defaults = {
            match.group(1): int(match.group(2))
            for match in re.finditer(
                r'^([A-Z0-9_]+)="\$\{\1:-([0-9]+)\}"$',
                launcher,
                flags=re.MULTILINE,
            )
        }
        required_port_vars = (
            "PREFILL_PORT",
            "DECODE_PORT",
            "PROXY_PORT",
            "KV_PORT",
            "PREFILL_TPU_KV_TRANSFER_PORT",
            "DECODE_TPU_KV_TRANSFER_PORT",
            "TPU_SIDE_CHANNEL_PORT",
            "PREFILL_CTRL_PORT",
            "DECODE_CTRL_PORT",
            "PREFILL_OFFLOAD_CONTROLLER_PORT",
            "DECODE_OFFLOAD_CONTROLLER_PORT",
            "PREFILL_SB_PORT_BASE",
            "DECODE_SB_PORT_BASE",
            "DECODE_DP_MASTER_PORT",
        )
        for name in required_port_vars:
            self.assertIn(name, defaults, f"Missing default for {name}")

        allocated: dict[int, str] = {}

        def claim(port: int, owner: str) -> None:
            self.assertGreaterEqual(
                port, 1024, f"{owner} port {port} is in privileged range"
            )
            self.assertFalse(
                LINUX_EPHEMERAL_LOW <= port <= LINUX_EPHEMERAL_HIGH,
                f"{owner} port {port} falls inside Linux ephemeral range "
                f"{LINUX_EPHEMERAL_LOW}..{LINUX_EPHEMERAL_HIGH}",
            )
            self.assertNotIn(
                port,
                allocated,
                f"{owner} port {port} collides with {allocated.get(port)}",
            )
            allocated[port] = owner

        for single_var in (
            "PROXY_PORT",
            "PREFILL_PORT",
            "DECODE_PORT",
            "TPU_SIDE_CHANNEL_PORT",
            "KV_PORT",
            "DECODE_DP_MASTER_PORT",
        ):
            claim(defaults[single_var], single_var)

        for offset in range(4):
            claim(
                defaults["PREFILL_TPU_KV_TRANSFER_PORT"] + offset,
                f"PREFILL_TPU_KV_TRANSFER_PORT+{offset}",
            )
            claim(
                defaults["DECODE_TPU_KV_TRANSFER_PORT"] + offset,
                f"DECODE_TPU_KV_TRANSFER_PORT+{offset}",
            )
            claim(
                defaults["PREFILL_CTRL_PORT"] + offset,
                f"PREFILL_CTRL_PORT+{offset}",
            )
            claim(
                defaults["PREFILL_CTRL_PORT"] + 100 + offset,
                f"PREFILL_CTRL_PORT+100+{offset}",
            )
            claim(
                defaults["DECODE_CTRL_PORT"] + offset,
                f"DECODE_CTRL_PORT+{offset}",
            )
            claim(
                defaults["DECODE_CTRL_PORT"] + 100 + offset,
                f"DECODE_CTRL_PORT+100+{offset}",
            )
            claim(
                defaults["PREFILL_OFFLOAD_CONTROLLER_PORT"] + offset,
                f"PREFILL_OFFLOAD_CONTROLLER_PORT+{offset}",
            )
            claim(
                defaults["DECODE_OFFLOAD_CONTROLLER_PORT"] + offset,
                f"DECODE_OFFLOAD_CONTROLLER_PORT+{offset}",
            )
            claim(
                defaults["PREFILL_SB_PORT_BASE"] + offset,
                f"PREFILL_SB_PORT_BASE+{offset}",
            )
            claim(
                defaults["DECODE_SB_PORT_BASE"] + offset,
                f"DECODE_SB_PORT_BASE+{offset}",
            )

    def test_kimi_k3_1p1d_config_controller_port_outside_ephemeral_range(
        self,
    ) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        config_text = (
            repo_root
            / "scripts"
            / "vllm"
            / "benchmarking"
            / "configs"
            / "kimi-k3-tp8-dp4-ep-1p1d.sh"
        ).read_text(encoding="utf-8")
        ports = [
            int(match.group(1))
            for match in re.finditer(
                r'\\"raiden_controller_port\\":([0-9]+)', config_text
            )
        ]
        self.assertTrue(ports, "Expected raiden_controller_port in kimi-k3 config")
        for port in ports:
            self.assertGreaterEqual(port, 1024)
            self.assertFalse(LINUX_EPHEMERAL_LOW <= port <= LINUX_EPHEMERAL_HIGH)


if __name__ == "__main__":
    unittest.main()
