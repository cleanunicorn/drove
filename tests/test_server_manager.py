"""Tests for ServerManager config change detection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from drove.backend import BACKEND_ASR, BACKEND_LLAMA, BACKEND_TTS
from drove.config import Config
from drove.server_manager import ServerManager, _estimate_model_memory, _ModelInstance


def make_config(tmp_path: Path) -> Config:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return Config(models_dir=models_dir, listen_port=8080)


def make_model(config: Config, name: str = "testmodel") -> Path:
    model_dir = config.models_dir / name
    model_dir.mkdir(parents=True)
    model_path = model_dir / f"{name}.gguf"
    model_path.write_bytes(b"")
    return model_path


def make_fake_instance(
    model_name: str,
    model_path: Path,
    config_mtimes: tuple[float, float],
    active_requests: int = 0,
) -> _ModelInstance:
    process = MagicMock()
    process.returncode = None  # process is running
    process.pid = 12345
    process.wait = AsyncMock(return_value=0)
    inst = _ModelInstance(model_name, process, 9999, model_path, config_mtimes)
    inst.active_requests = active_requests
    return inst


def patch_stop_instance(manager: ServerManager) -> Any:
    """Patch _stop_instance with a mock that, like the real one, removes the instance.

    Eviction re-evaluates the running set after each stop, so a mock that
    leaves the instance in place would loop forever.
    """

    async def remove(name: str) -> None:
        manager._instances.pop(name, None)

    return patch.object(manager, "_stop_instance", new_callable=AsyncMock, side_effect=remove)


def test_get_config_mtimes_no_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    mtimes = manager._get_config_mtimes(model_path)
    assert mtimes == (0.0, 0.0)


def test_get_config_mtimes_with_per_model_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    cfg_path = model_path.with_suffix(".toml")
    cfg_path.write_text("ctx_size = 4096\n")

    model_mtime, global_mtime = manager._get_config_mtimes(model_path)
    assert model_mtime == cfg_path.stat().st_mtime
    assert global_mtime == 0.0


def test_get_config_mtimes_with_global_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    global_cfg_path = config.models_dir / "_global.toml"
    global_cfg_path.write_text("n_gpu_layers = -1\n")

    model_mtime, global_mtime = manager._get_config_mtimes(model_path)
    assert model_mtime == 0.0
    assert global_mtime == global_cfg_path.stat().st_mtime


async def test_ensure_running_no_restart_when_config_unchanged(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    # Create config and record current mtimes
    cfg_path = model_path.with_suffix(".toml")
    cfg_path.write_text("ctx_size = 4096\n")
    mtimes = manager._get_config_mtimes(model_path)

    inst = make_fake_instance("testmodel", model_path, mtimes)
    manager._instances["testmodel"] = inst

    await manager.ensure_running("testmodel")

    # Instance is unchanged — no restart happened
    assert manager._instances.get("testmodel") is inst
    assert not inst.needs_restart


async def test_ensure_running_restarts_when_idle_and_config_changed(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    # Instance was started with no config file present
    old_mtimes = (0.0, 0.0)
    inst = make_fake_instance("testmodel", model_path, old_mtimes)
    manager._instances["testmodel"] = inst

    # Config file created after startup — mtimes differ
    cfg_path = model_path.with_suffix(".toml")
    cfg_path.write_text("ctx_size = 8192\n")

    async def fake_start(model_name: str, **_: Any) -> None:
        new_mtimes = manager._get_config_mtimes(model_path)
        manager._instances[model_name] = make_fake_instance(model_name, model_path, new_mtimes)

    with (
        patch_stop_instance(manager),
        patch.object(manager, "_evict_if_needed", new_callable=AsyncMock),
        patch.object(manager, "_start", new_callable=AsyncMock, side_effect=fake_start),
    ):
        await manager.ensure_running("testmodel")

    # A new instance replaced the old one
    assert manager._instances.get("testmodel") is not inst


async def test_ensure_running_sets_needs_restart_when_busy(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    old_mtimes = (0.0, 0.0)
    inst = make_fake_instance("testmodel", model_path, old_mtimes, active_requests=1)
    manager._instances["testmodel"] = inst

    # Change config while requests are in-flight
    cfg_path = model_path.with_suffix(".toml")
    cfg_path.write_text("ctx_size = 8192\n")

    await manager.ensure_running("testmodel")

    # No restart yet — server still serving the in-flight request
    assert manager._instances.get("testmodel") is inst
    assert inst.needs_restart


async def test_ensure_running_claim_increments_active_requests(tmp_path: Path) -> None:
    """ensure_running(claim=True) must atomically reserve a request slot.

    Without this, a concurrent ensure_running() for a different model could
    evict an idle-looking instance between the first ensure_running() return
    and the caller's request_started() call — killing the server mid-request
    and producing empty 502 'Upstream error:' responses.
    """
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    mtimes = (0.0, 0.0)
    inst = make_fake_instance("testmodel", model_path, mtimes)
    manager._instances["testmodel"] = inst

    assert inst.active_requests == 0
    await manager.ensure_running("testmodel", claim=True)
    # Slot was claimed atomically under the lock
    assert inst.active_requests == 1


async def test_ensure_running_claim_prevents_eviction_race(tmp_path: Path) -> None:
    """After claim=True, the model cannot be evicted by another ensure_running call."""
    config = Config(models_dir=tmp_path / "models", listen_port=8080, max_loaded_models=1)
    config.models_dir.mkdir()
    manager = ServerManager(config)

    model_a_path = make_model(config, "model-a")
    make_model(config, "model-b")

    mtimes = (0.0, 0.0)
    inst_a = make_fake_instance("model-a", model_a_path, mtimes)
    manager._instances["model-a"] = inst_a

    # Request 1 arrives for model-a and claims a slot.
    await manager.ensure_running("model-a", claim=True)
    assert inst_a.active_requests == 1

    # Request 2 arrives for model-b. _evict_if_needed should see that model-a
    # has an in-flight request and wait (we patch the wait loop to observe this).
    wait_count = 0

    original_sleep = __import__("asyncio").sleep

    async def fake_sleep(seconds: float) -> None:
        nonlocal wait_count
        wait_count += 1
        if wait_count >= 2:
            # Simulate the request finishing during the wait
            inst_a.active_requests = 0
        await original_sleep(0)

    async def fake_start_b(model_name: str, **_: Any) -> None:
        manager._instances[model_name] = make_fake_instance(
            model_name, config.models_dir / model_name / f"{model_name}.gguf", mtimes
        )

    with (
        patch("drove.server_manager.asyncio.sleep", side_effect=fake_sleep),
        patch.object(manager, "_start", new_callable=AsyncMock, side_effect=fake_start_b),
        patch_stop_instance(manager) as mock_stop,
    ):
        await manager.ensure_running("model-b")

    # model-a was only evicted after its in-flight request drained (wait_count >= 2)
    assert wait_count >= 2
    mock_stop.assert_awaited_with("model-a")


async def test_evict_prefers_idle_over_busy_lru(tmp_path: Path) -> None:
    """At capacity, an idle model is evicted even if a busier model is older (LRU).

    model-a is the least-recently-used but still has an active connection;
    model-b is idle. The idle model-b must be evicted, leaving the busy
    model-a running.
    """
    config = Config(models_dir=tmp_path / "models", listen_port=8080, max_loaded_models=2)
    config.models_dir.mkdir()
    manager = ServerManager(config)

    mtimes = (0.0, 0.0)
    inst_a = make_fake_instance("model-a", make_model(config, "model-a"), mtimes, active_requests=1)
    inst_b = make_fake_instance("model-b", make_model(config, "model-b"), mtimes)
    # model-a is the LRU (oldest last_request_time) but is busy; model-b is idle.
    inst_a.last_request_time = 100.0
    inst_b.last_request_time = 200.0
    manager._instances["model-a"] = inst_a
    manager._instances["model-b"] = inst_b

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed()

    # Idle model-b evicted; busy model-a left running despite being LRU.
    mock_stop.assert_awaited_once_with("model-b")


async def test_evict_drains_lru_when_all_busy(tmp_path: Path) -> None:
    """When every loaded model is busy, fall back to draining the LRU model."""
    config = Config(models_dir=tmp_path / "models", listen_port=8080, max_loaded_models=2)
    config.models_dir.mkdir()
    manager = ServerManager(config)

    mtimes = (0.0, 0.0)
    inst_a = make_fake_instance("model-a", make_model(config, "model-a"), mtimes, active_requests=1)
    inst_b = make_fake_instance("model-b", make_model(config, "model-b"), mtimes, active_requests=1)
    inst_a.last_request_time = 100.0  # LRU
    inst_b.last_request_time = 200.0
    manager._instances["model-a"] = inst_a
    manager._instances["model-b"] = inst_b

    original_sleep = __import__("asyncio").sleep

    async def fake_sleep(seconds: float) -> None:
        inst_a.active_requests = 0  # request drains during the wait
        await original_sleep(0)

    with (
        patch("drove.server_manager.asyncio.sleep", side_effect=fake_sleep),
        patch_stop_instance(manager) as mock_stop,
    ):
        # _evict_if_needed releases/re-acquires the lock while draining.
        async with manager._lock:
            await manager._evict_if_needed()

    mock_stop.assert_awaited_once_with("model-a")


async def test_evict_reselects_when_victim_reclaimed_during_drain(tmp_path: Path) -> None:
    """A victim re-claimed while draining is spared; an idle peer is evicted instead.

    All models start busy, so eviction drains the LRU (model-a) with the lock
    released. While the lock is released a fresh request re-claims model-a and
    model-b goes idle. On re-acquiring the lock, eviction must re-evaluate and
    stop the now-idle model-b rather than the re-claimed model-a.
    """
    config = Config(models_dir=tmp_path / "models", listen_port=8080, max_loaded_models=2)
    config.models_dir.mkdir()
    manager = ServerManager(config)

    mtimes = (0.0, 0.0)
    inst_a = make_fake_instance("model-a", make_model(config, "model-a"), mtimes, active_requests=1)
    inst_b = make_fake_instance("model-b", make_model(config, "model-b"), mtimes, active_requests=1)
    inst_a.last_request_time = 100.0  # LRU, selected as the initial drain victim
    inst_b.last_request_time = 200.0
    manager._instances["model-a"] = inst_a
    manager._instances["model-b"] = inst_b

    original_sleep = __import__("asyncio").sleep

    async def fake_sleep(seconds: float) -> None:
        inst_a.active_requests = 0  # model-a's in-flight request completes
        inst_b.active_requests = 0  # model-b goes idle
        await original_sleep(0)

    real_acquire = manager._lock.acquire
    acquire_calls = {"n": 0}

    async def acquire_reclaiming() -> bool:
        result = await real_acquire()
        acquire_calls["n"] += 1
        if acquire_calls["n"] == 2:  # the re-acquire after draining
            inst_a.active_requests = 1  # a new request re-claims model-a in the gap
        return result

    with (
        patch("drove.server_manager.asyncio.sleep", side_effect=fake_sleep),
        patch.object(manager._lock, "acquire", side_effect=acquire_reclaiming),
        patch_stop_instance(manager) as mock_stop,
    ):
        async with manager._lock:  # acquire call #1
            await manager._evict_if_needed()

    mock_stop.assert_awaited_once_with("model-b")


async def test_evict_skips_stop_when_capacity_frees_during_drain(tmp_path: Path) -> None:
    """If a slot frees up while draining the busy LRU, no model is stopped."""
    config = Config(models_dir=tmp_path / "models", listen_port=8080, max_loaded_models=2)
    config.models_dir.mkdir()
    manager = ServerManager(config)

    mtimes = (0.0, 0.0)
    inst_a = make_fake_instance("model-a", make_model(config, "model-a"), mtimes, active_requests=1)
    inst_b = make_fake_instance("model-b", make_model(config, "model-b"), mtimes, active_requests=1)
    inst_a.last_request_time = 100.0  # LRU
    inst_b.last_request_time = 200.0
    manager._instances["model-a"] = inst_a
    manager._instances["model-b"] = inst_b

    original_sleep = __import__("asyncio").sleep

    async def fake_sleep(seconds: float) -> None:
        inst_a.active_requests = 0  # model-a drains...
        inst_b.process.returncode = 0  # ...and model-b exits, freeing a slot
        await original_sleep(0)

    with (
        patch("drove.server_manager.asyncio.sleep", side_effect=fake_sleep),
        patch.object(manager, "_stop_instance", new_callable=AsyncMock) as mock_stop,
    ):
        async with manager._lock:
            await manager._evict_if_needed()

    mock_stop.assert_not_awaited()


# ── Memory budget (max_memory) ───────────────────────────────────────────────


def make_memory_manager(tmp_path: Path, max_memory: str) -> ServerManager:
    config = Config(
        models_dir=tmp_path / "models",
        listen_port=8080,
        max_loaded_models=0,  # isolate the memory budget from the count limit
        max_memory=max_memory,
    )
    config.models_dir.mkdir()
    return ServerManager(config)


def add_instance(
    manager: ServerManager, name: str, est_memory_bytes: int, last_request_time: float
) -> _ModelInstance:
    inst = make_fake_instance(name, make_model(manager._config, name), (0.0, 0.0))
    inst.est_memory_bytes = est_memory_bytes
    inst.last_request_time = last_request_time
    manager._instances[name] = inst
    return inst


async def test_evict_for_memory_budget(tmp_path: Path) -> None:
    """The LRU idle model is evicted when the incoming model would exceed max_memory."""
    manager = make_memory_manager(tmp_path, max_memory="100")
    add_instance(manager, "model-a", est_memory_bytes=40, last_request_time=100.0)  # LRU
    add_instance(manager, "model-b", est_memory_bytes=40, last_request_time=200.0)

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed(incoming_bytes=50)

    # Evicting model-a is enough: 40 (model-b) + 50 (incoming) fits in 100.
    mock_stop.assert_awaited_once_with("model-a")


async def test_evict_multiple_models_for_memory_budget(tmp_path: Path) -> None:
    """Memory pressure can require evicting more than one model."""
    manager = make_memory_manager(tmp_path, max_memory="100")
    add_instance(manager, "model-a", est_memory_bytes=60, last_request_time=100.0)  # LRU
    add_instance(manager, "model-b", est_memory_bytes=30, last_request_time=200.0)

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed(incoming_bytes=90)

    # 90 incoming leaves no room for either loaded model: both are evicted, LRU first.
    assert [c.args[0] for c in mock_stop.await_args_list] == ["model-a", "model-b"]


async def test_no_evict_when_memory_budget_fits(tmp_path: Path) -> None:
    manager = make_memory_manager(tmp_path, max_memory="200")
    add_instance(manager, "model-a", est_memory_bytes=40, last_request_time=100.0)
    add_instance(manager, "model-b", est_memory_bytes=40, last_request_time=200.0)

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed(incoming_bytes=50)

    mock_stop.assert_not_awaited()


async def test_oversized_model_evicts_all_and_still_starts(tmp_path: Path) -> None:
    """A model bigger than the whole budget evicts everything else but is not refused."""
    manager = make_memory_manager(tmp_path, max_memory="100")
    add_instance(manager, "model-a", est_memory_bytes=40, last_request_time=100.0)

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed(incoming_bytes=150)

    # model-a was evicted and _evict_if_needed returned so the load can proceed.
    mock_stop.assert_awaited_once_with("model-a")


async def test_memory_budget_prefers_idle_over_busy_lru(tmp_path: Path) -> None:
    """Under memory pressure, a busy LRU model is spared in favour of an idle one."""
    manager = make_memory_manager(tmp_path, max_memory="100")
    inst_a = add_instance(manager, "model-a", est_memory_bytes=40, last_request_time=100.0)
    inst_a.active_requests = 1
    add_instance(manager, "model-b", est_memory_bytes=40, last_request_time=200.0)

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed(incoming_bytes=50)

    mock_stop.assert_awaited_once_with("model-b")


def test_estimate_model_memory_single_file(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 123)
    assert _estimate_model_memory(model) == 123


def test_estimate_model_memory_sharded_gguf(tmp_path: Path) -> None:
    """All shards of a sharded GGUF count, but unrelated files do not."""
    (tmp_path / "big-00001-of-00002.gguf").write_bytes(b"x" * 100)
    (tmp_path / "big-00002-of-00002.gguf").write_bytes(b"x" * 50)
    (tmp_path / "other-00001-of-00001.gguf").write_bytes(b"x" * 999)
    (tmp_path / "readme.txt").write_bytes(b"x" * 999)

    assert _estimate_model_memory(tmp_path / "big-00001-of-00002.gguf") == 150


def test_estimate_model_memory_onnx_dir(tmp_path: Path) -> None:
    """ASR models load every .onnx file in the directory."""
    (tmp_path / "encoder-model.onnx").write_bytes(b"x" * 70)
    (tmp_path / "decoder_joint-model.onnx").write_bytes(b"x" * 30)
    (tmp_path / "vocab.txt").write_bytes(b"x" * 999)

    assert _estimate_model_memory(tmp_path / "encoder-model.onnx") == 100


def test_estimate_model_memory_missing_file(tmp_path: Path) -> None:
    assert _estimate_model_memory(tmp_path / "gone.gguf") == 0


async def test_idle_watcher_detects_config_change(tmp_path: Path) -> None:
    """The idle watcher should detect config changes and stop the instance,
    even when no new request arrives to trigger ensure_running()."""
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)

    # Instance started with no config file
    old_mtimes = (0.0, 0.0)
    inst = make_fake_instance("testmodel", model_path, old_mtimes)
    manager._instances["testmodel"] = inst

    # Config file created after startup — mtimes now differ
    cfg_path = model_path.with_suffix(".toml")
    cfg_path.write_text("ctx_size = 8192\n")

    with patch.object(manager, "_stop_instance", new_callable=AsyncMock) as mock_stop:
        # Run one iteration of the idle watcher by calling it with a short sleep
        with patch("drove.server_manager.asyncio.sleep", new_callable=AsyncMock):
            # _idle_watcher loops; after stopping it returns, so this will finish
            await manager._idle_watcher("testmodel")

        mock_stop.assert_awaited_once_with("testmodel")
    assert inst.needs_restart


async def test_wait_for_health_success(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)
    inst = make_fake_instance("testmodel", model_path, (0.0, 0.0))

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        await manager._wait_for_health(inst)


async def test_wait_for_health_unexpected_exit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)
    inst = make_fake_instance("testmodel", model_path, (0.0, 0.0))

    # Simulate process exit
    inst.process.returncode = 1
    inst._stderr_buf.extend(b"error: failed to load model")

    with pytest.raises(RuntimeError) as excinfo:
        await manager._wait_for_health(inst)

    assert "Model server exited unexpectedly during startup" in str(excinfo.value)
    # stderr is kept for the operator, never in the message that can reach a client
    assert "error: failed to load model" not in str(excinfo.value)
    assert any("error: failed to load model" in n for n in excinfo.value.__notes__)


async def test_wait_for_health_timeout(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.startup_timeout_seconds = 1.0
    manager = ServerManager(config)
    model_path = make_model(config)
    inst = make_fake_instance("testmodel", model_path, (0.0, 0.0))

    # Simulate always failing health check
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 503

    inst._stderr_buf.extend(b"some warning on stderr")

    # Mock time to traverse the deadline
    # _wait_for_health uses time.monotonic()
    # Initial call + loop call + loop exit
    with (
        patch("drove.server_manager.time.monotonic", side_effect=[0.0, 0.0, 2.0]),
        patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp),
        patch("drove.server_manager.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(TimeoutError) as excinfo:
            await manager._wait_for_health(inst)

    assert "did not become healthy within 1.0s" in str(excinfo.value)
    assert "some warning on stderr" not in str(excinfo.value)
    assert any("some warning on stderr" in n for n in excinfo.value.__notes__)


# ── ASR backend ──────────────────────────────────────────────────────────────


def make_asr_model(config: Config, name: str = "parakeet") -> Path:
    model_dir = config.models_dir / name
    model_dir.mkdir(parents=True)
    (model_dir / "encoder-model.onnx").write_bytes(b"")
    (model_dir / "decoder_joint-model.onnx").write_bytes(b"")
    primary = sorted(model_dir.glob("*.onnx"))[0]
    return primary


def test_build_asr_command(tmp_path: Path) -> None:
    from drove.model_config import ModelConfig

    config = make_config(tmp_path)
    manager = ServerManager(config)
    primary = make_asr_model(config)
    cfg = ModelConfig(asr_model="nemo-parakeet-tdt-0.6b-v3", asr_quantization="int8")

    with patch("drove.server_manager._onnx_asr_available", return_value=True):
        cmd = manager._build_asr_command("parakeet", primary, cfg, 9000)

    assert cmd[1:3] == ["-m", "drove.workers.asr"]
    assert str(primary.parent) in cmd
    assert "nemo-parakeet-tdt-0.6b-v3" in cmd
    assert "9000" in cmd
    assert "--quantization" in cmd
    assert "int8" in cmd


def test_build_asr_command_requires_onnx_asr(tmp_path: Path) -> None:
    from drove.model_config import ModelConfig

    config = make_config(tmp_path)
    manager = ServerManager(config)
    primary = make_asr_model(config)

    with patch("drove.server_manager._onnx_asr_available", return_value=False):
        with pytest.raises(RuntimeError, match="onnx-asr"):
            manager._build_asr_command("parakeet", primary, ModelConfig(), 9000)


def test_infer_asr_model_type_from_download_info(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    primary = make_asr_model(config)
    primary.with_suffix(".toml").write_text(
        '[download]\nrepo_id = "istupakov/parakeet-tdt-0.6b-v3-onnx"\nfiles = []\n'
    )

    assert manager._infer_asr_model_type("parakeet", primary) == "nemo-parakeet-tdt-0.6b-v3"


def test_infer_asr_model_type_falls_back_to_model_name(tmp_path: Path) -> None:
    """Without download metadata, a recognised model name alone resolves the type."""
    config = make_config(tmp_path)
    manager = ServerManager(config)
    primary = make_asr_model(config, name="parakeet-tdt-0.6b-v3")

    inferred = manager._infer_asr_model_type("parakeet-tdt-0.6b-v3", primary)

    assert inferred == "nemo-parakeet-tdt-0.6b-v3"


def test_infer_asr_model_type_unknown_raises(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    primary = make_asr_model(config, name="mystery-model")

    with pytest.raises(RuntimeError, match="asr_model"):
        manager._infer_asr_model_type("mystery-model", primary)


# ── Concurrent cold starts during an eviction drain (#113) ───────────────────


def make_fake_process() -> MagicMock:
    """A fake backend process that exits when it is sent a signal."""
    process = MagicMock()
    process.returncode = None
    process.pid = 4242
    process.wait = AsyncMock(return_value=0)
    process.stderr.read = AsyncMock(return_value=b"")

    def exit_on_signal(_sig: int) -> None:
        process.returncode = 0

    process.send_signal.side_effect = exit_on_signal
    return process


class ColdStartRace:
    """Two cold-start requests for model-c that both wait on one drain of model-a.

    max_loaded_models=2 and models a and b are busy, so each request releases
    the manager lock to drain the LRU model-a. Each drain wait is gated: R1 is
    released first (with model-a drained) and runs to completion, then R2 is
    released. Any further drain is unexpected and fails the request. Only the
    process boundary is faked — ensure_running, eviction, _start and
    _stop_instance run for real.
    """

    def __init__(self, tmp_path: Path) -> None:
        config = Config(
            models_dir=tmp_path / "models",
            listen_port=8080,
            max_loaded_models=2,
            prompt_cache=False,
        )
        config.models_dir.mkdir()
        self.manager = ServerManager(config)
        now = time.monotonic()
        mtimes = (0.0, 0.0)
        self.inst_a = make_fake_instance(
            "model-a", make_model(config, "model-a"), mtimes, active_requests=1
        )
        self.inst_b = make_fake_instance(
            "model-b", make_model(config, "model-b"), mtimes, active_requests=1
        )
        self.inst_a.last_request_time = now - 20  # LRU: the drain victim
        self.inst_b.last_request_time = now - 10
        self.manager._instances["model-a"] = self.inst_a
        self.manager._instances["model-b"] = self.inst_b
        make_model(config, "model-c")
        self.spawned: list[MagicMock] = []
        self.requests: list[asyncio.Task[object | None]] = []
        self.draining = [asyncio.Event(), asyncio.Event()]
        self.drain_gates = [asyncio.Event(), asyncio.Event()]
        self.drain_calls = 0

    async def _spawn(self, *_args: Any, **_kwargs: Any) -> MagicMock:
        process = make_fake_process()
        self.spawned.append(process)
        return process

    async def _sleep(self, _seconds: float) -> None:
        if asyncio.current_task() not in self.requests:
            await asyncio.Event().wait()  # the idle watcher: park until cancelled
        call = self.drain_calls
        self.drain_calls += 1
        if call >= len(self.drain_gates):
            busy = {n: i.active_requests for n, i in self.manager._instances.items()}
            raise AssertionError(f"unexpected drain #{call + 1} with loaded models {busy}")
        self.draining[call].set()
        await self.drain_gates[call].wait()

    @contextlib.contextmanager
    def patched(self) -> Iterator[None]:
        with (
            patch("drove.server_manager.asyncio.sleep", new=self._sleep),
            patch("drove.server_manager.asyncio.create_subprocess_exec", new=self._spawn),
            patch("drove.server_manager.shutil.which", return_value="/usr/bin/llama-server"),
            patch.object(self.manager, "_wait_for_health", new=AsyncMock()),
        ):
            yield

    async def run(self, *, r2_claim: bool = True, before_r2: Any = None) -> None:
        """Drive the interleaving from #113: R1 and R2 for model-c both drain model-a."""
        manager = self.manager
        self.requests.append(asyncio.create_task(manager.ensure_running("model-c", claim=True)))
        await asyncio.wait_for(self.draining[0].wait(), 5)
        self.requests.append(asyncio.create_task(manager.ensure_running("model-c", claim=r2_claim)))
        await asyncio.wait_for(self.draining[1].wait(), 5)

        self.inst_a.active_requests = 0  # model-a's request finishes
        self.drain_gates[0].set()
        await asyncio.wait_for(self.requests[0], 5)  # R1 evicts A, starts and claims C
        if before_r2 is not None:
            before_r2()
        self.drain_gates[1].set()
        await asyncio.wait_for(self.requests[1], 5)

    def assert_no_orphans(self) -> None:
        tracked = {id(i.process) for i in self.manager._instances.values()}
        for process in self.spawned:
            assert id(process) in tracked or process.returncode is not None

    async def close(self) -> None:
        for task in self.requests:
            task.cancel()
        watchers = list(self.manager._idle_tasks.values())
        await self.manager.stop()
        await asyncio.gather(*self.requests, *watchers, return_exceptions=True)


async def test_concurrent_cold_start_reuses_peer_after_drain(tmp_path: Path) -> None:
    """A request whose model a peer started during the drain reuses that instance.

    Regression for #113: R2 must neither start a second model-c (orphaning
    R1's process) nor evict the busy model-b to make room for it.
    """
    race = ColdStartRace(tmp_path)
    manager = race.manager
    with race.patched():
        try:
            await race.run()

            assert len(race.spawned) == 1
            inst_c = manager._instances["model-c"]
            assert inst_c.process is race.spawned[0]
            assert inst_c.active_requests == 2  # both requests claimed the same C
            slot_1, slot_2 = (task.result() for task in race.requests)
            assert slot_1 is inst_c and slot_2 is inst_c
            assert "model-a" not in manager._instances
            race.inst_a.process.send_signal.assert_called_once()
            assert manager._instances["model-b"] is race.inst_b
            race.inst_b.process.send_signal.assert_not_called()
            race.assert_no_orphans()

            manager.request_finished("model-c", slot_1)
            assert inst_c.active_requests == 1
            manager.request_finished("model-c", slot_2)
            assert inst_c.active_requests == 0
            assert manager._instances["model-c"] is inst_c
            assert race.inst_b.active_requests == 1
        finally:
            await race.close()

    # Shutdown reaches the one model-c process: nothing is left untracked.
    race.spawned[0].send_signal.assert_called_once()


async def test_concurrent_cold_start_spares_busy_lru_peer(tmp_path: Path) -> None:
    """R2 does not drain R1's model-c even when it is the LRU busy model."""
    race = ColdStartRace(tmp_path)
    manager = race.manager

    def age_model_c() -> None:
        # model-c becomes older than model-b, so it is the LRU of the busy set.
        manager._instances["model-c"].last_request_time = race.inst_b.last_request_time - 5

    with race.patched():
        try:
            await race.run(before_r2=age_model_c)

            assert len(race.spawned) == 1
            inst_c = manager._instances["model-c"]
            assert inst_c.process is race.spawned[0]
            inst_c.process.send_signal.assert_not_called()
            assert inst_c.active_requests == 2
            assert manager._instances["model-b"] is race.inst_b
            race.assert_no_orphans()

            for task in race.requests:
                manager.request_finished("model-c", task.result())
            assert manager._instances["model-c"] is inst_c
            assert inst_c.active_requests == 0
            assert race.inst_b.active_requests == 1
        finally:
            await race.close()


async def test_concurrent_cold_start_reuse_without_claim(tmp_path: Path) -> None:
    """A peer-reusing request with claim=False leaves the request count alone."""
    race = ColdStartRace(tmp_path)
    with race.patched():
        try:
            await race.run(r2_claim=False)

            assert len(race.spawned) == 1
            assert race.manager._instances["model-c"].active_requests == 1
            assert race.requests[1].result() is None
        finally:
            await race.close()


async def test_concurrent_cold_start_replaces_peer_that_died(tmp_path: Path) -> None:
    """If the peer's model-c exited during the drain, it is cleaned up and restarted."""
    race = ColdStartRace(tmp_path)
    manager = race.manager

    def crash_model_c() -> None:
        manager._instances["model-c"].process.returncode = 1

    with race.patched():
        try:
            await race.run(before_r2=crash_model_c)

            assert len(race.spawned) == 2
            inst_c = manager._instances["model-c"]
            assert inst_c.process is race.spawned[1]
            assert inst_c.active_requests == 1  # only R2's claim is on the new process
            assert manager._instances["model-b"] is race.inst_b
            race.assert_no_orphans()

            # R1's request ran on the dead process: finishing it must not
            # release R2's slot on the replacement.
            slot_1, slot_2 = (task.result() for task in race.requests)
            assert slot_1 is not inst_c and slot_2 is inst_c
            manager.request_finished("model-c", slot_1)
            assert inst_c.active_requests == 1
            manager.request_finished("model-c", slot_2)
            assert inst_c.active_requests == 0
        finally:
            await race.close()


@pytest.mark.parametrize("state", ["live", "dead", "watcher"])
async def test_start_refuses_already_registered_model(tmp_path: Path, state: str) -> None:
    """_start never overwrites a tracked instance or idle watcher, live or not."""
    config = make_config(tmp_path)
    manager = ServerManager(config)
    inst = make_fake_instance("model-c", make_model(config, "model-c"), (0.0, 0.0))
    watcher: asyncio.Task[Any] | None = None
    if state == "watcher":
        watcher = asyncio.create_task(asyncio.Event().wait())
        manager._idle_tasks["model-c"] = watcher
    else:
        if state == "dead":
            inst.process.returncode = 1
        manager._instances["model-c"] = inst
    spawn = AsyncMock(return_value=make_fake_process())

    try:
        with (
            patch("drove.server_manager.asyncio.create_subprocess_exec", spawn),
            patch("drove.server_manager.shutil.which", return_value="/usr/bin/llama-server"),
            patch.object(manager, "_wait_for_health", new=AsyncMock()),
        ):
            async with manager._lock:
                with pytest.raises(RuntimeError, match="already registered"):
                    await manager._start("model-c")

        spawn.assert_not_called()
        if watcher is not None:
            assert manager._idle_tasks["model-c"] is watcher
            assert not watcher.cancelled()
            assert "model-c" not in manager._instances
        else:
            assert manager._instances["model-c"] is inst
            assert "model-c" not in manager._idle_tasks
            inst.process.send_signal.assert_not_called()
    finally:
        for task in [watcher, *manager._idle_tasks.values()]:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_evict_returns_when_target_already_running(tmp_path: Path) -> None:
    """At capacity, nothing is drained or evicted once the target model is running."""
    config = Config(models_dir=tmp_path / "models", listen_port=8080, max_loaded_models=2)
    config.models_dir.mkdir()
    manager = ServerManager(config)
    mtimes = (0.0, 0.0)
    manager._instances["model-a"] = make_fake_instance(
        "model-a", make_model(config, "model-a"), mtimes, active_requests=1
    )
    manager._instances["model-c"] = make_fake_instance(
        "model-c", make_model(config, "model-c"), mtimes, active_requests=1
    )

    with (
        patch("drove.server_manager.asyncio.sleep", side_effect=AssertionError("drained")),
        patch_stop_instance(manager) as mock_stop,
    ):
        async with manager._lock:
            await manager._evict_if_needed(model_name="model-c")

    mock_stop.assert_not_awaited()


async def test_memory_evict_returns_when_target_already_running(tmp_path: Path) -> None:
    """A running target's memory is not counted twice against max_memory."""
    manager = make_memory_manager(tmp_path, max_memory="100")
    add_instance(manager, "model-a", est_memory_bytes=60, last_request_time=100.0)
    add_instance(manager, "model-c", est_memory_bytes=30, last_request_time=200.0)

    with patch_stop_instance(manager) as mock_stop:
        await manager._evict_if_needed(incoming_bytes=30, model_name="model-c")

    mock_stop.assert_not_awaited()


def test_request_finished_ignores_slot_of_replaced_instance(tmp_path: Path) -> None:
    """A slot from a stopped instance never releases its replacement's requests."""
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config)
    old = make_fake_instance("testmodel", model_path, (0.0, 0.0), active_requests=1)
    new = make_fake_instance("testmodel", model_path, (0.0, 0.0), active_requests=1)
    manager._instances["testmodel"] = new

    manager.request_finished("testmodel", old)
    assert new.active_requests == 1

    manager.request_finished("testmodel", new)
    assert new.active_requests == 0


def test_request_started_returns_the_claimed_slot(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    inst = make_fake_instance("testmodel", make_model(config), (0.0, 0.0))
    manager._instances["testmodel"] = inst

    assert manager.request_started("testmodel") is inst
    assert inst.active_requests == 1
    assert manager.request_started("missing") is None


# A stand-in backend: floods stdout before binding (like a chatty model load)
# and on every request (like an access log), and writes a stderr burst ending
# in a sentinel. With an unread stdout pipe it blocks once ~192 KiB are queued.
_FLOODING_BACKEND = """
import http.server, sys
sys.stdout.buffer.write(b"x" * (1024 * 1024))
sys.stdout.flush()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        sys.stdout.buffer.write(b"GET 200 OK\\n" * 50000)
        sys.stdout.flush()
        sys.stderr.write("e" * 300000 + "\\nSENTINEL\\n")
        sys.stderr.flush()
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


@pytest.mark.parametrize("backend", [BACKEND_LLAMA, BACKEND_ASR, BACKEND_TTS])
async def test_backend_output_flood_does_not_block(tmp_path: Path, backend: str) -> None:
    config = make_config(tmp_path)
    config.llama_server_bin = sys.executable
    config.prompt_cache = False
    config.startup_timeout_seconds = 10
    make_model(config)
    manager = ServerManager(config)

    def command(*args: Any) -> list[str]:
        port = next(a for a in args if isinstance(a, int))
        return [sys.executable, "-c", _FLOODING_BACKEND, str(port)]

    def llama_args(*args: Any) -> list[str]:
        return command(*args)[1:]

    with (
        patch("drove.server_manager.detect_backend", return_value=backend),
        patch.object(manager, "_build_args", side_effect=llama_args),
        patch.object(manager, "_build_asr_command", side_effect=command),
        patch.object(manager, "_build_tts_command", side_effect=command),
    ):
        try:
            await asyncio.wait_for(manager.ensure_running("testmodel"), 20)
            inst = manager._instances["testmodel"]
            async with httpx.AsyncClient() as client:
                for _ in range(2):
                    url = f"http://127.0.0.1:{inst.port}/v1/models"
                    resp = await asyncio.wait_for(client.get(url), 10)
                    assert resp.status_code == 200
            for _ in range(200):
                if inst.stderr_text.endswith("SENTINEL"):
                    break
                await asyncio.sleep(0.02)
            assert inst.stderr_text.endswith("SENTINEL")
            assert len(inst._stderr_buf) <= 256 * 1024
        finally:
            await asyncio.wait_for(manager.stop(), 20)


def _hung_process() -> MagicMock:
    """A process that ignores every signal, never reaps, and never closes stderr."""

    async def never_reaped() -> int:
        await asyncio.Event().wait()
        return 0

    process = MagicMock(pid=12345, returncode=None)
    process.wait = AsyncMock(side_effect=never_reaped)
    process.stderr = asyncio.StreamReader()
    return process


@pytest.mark.parametrize("backend", [BACKEND_LLAMA, BACKEND_ASR, BACKEND_TTS])
async def test_stop_finishes_when_wait_never_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    backend: str,
) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    inst = _ModelInstance("stuck", _hung_process(), 9999, make_model(config), (0.0, 0.0), backend)
    inst.start_stderr_reader()
    manager._instances["stuck"] = inst
    monkeypatch.setattr("drove.server_manager.STOP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(manager, "_save_prompt_cache", AsyncMock())

    with patch("drove.server_manager.os.killpg"), caplog.at_level(logging.WARNING):
        await asyncio.wait_for(manager.stop_model("stuck"), 2)

    assert "stuck" not in manager._instances
    assert "did not exit" in caplog.text
    await asyncio.wait_for(manager._lock.acquire(), 1)
    manager._lock.release()
    # Whatever still holds stderr keeps being drained until it closes.
    stderr = inst.process.stderr
    assert stderr is not None and inst._stderr_task is not None
    stderr.feed_data(b"late output")
    await asyncio.sleep(0.05)
    assert inst.stderr_text == "late output"
    stderr.feed_eof()
    await asyncio.wait_for(asyncio.wait({inst._stderr_task}), 1)
    assert not inst._stderr_task.cancelled()


async def test_stop_finishes_for_child_holding_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The backend ignores SIGTERM and its own child keeps stderr open, so even
    # after SIGKILL asyncio never sees the pipes close.
    script = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.fork() == 0:
    flood = b"e" * (1024 * 1024) + b"\\nSENTINEL\\n"
    signal.signal(signal.SIGUSR1, lambda *_: os.write(2, flood))
    os.write(2, b"%d\\n" % os.getpid())
time.sleep(60)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stderr is not None
    holder_pid = int(await asyncio.wait_for(process.stderr.readline(), 5))
    config = make_config(tmp_path)
    manager = ServerManager(config)
    inst = _ModelInstance("stubborn", process, 9999, make_model(config), (0.0, 0.0))
    inst.start_stderr_reader()
    manager._instances["stubborn"] = inst
    monkeypatch.setattr("drove.server_manager.STOP_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(manager, "_save_prompt_cache", AsyncMock())
    try:
        await asyncio.wait_for(manager.stop(), 5)
        assert not manager._instances
        assert process.returncode == -signal.SIGKILL
        # The surviving holder's output is still drained, so it cannot block...
        os.kill(holder_pid, signal.SIGUSR1)
        for _ in range(150):
            if inst.stderr_text.endswith("SENTINEL"):
                break
            await asyncio.sleep(0.02)
        assert inst.stderr_text.endswith("SENTINEL")
        # ...and once it exits the reader reaches EOF and the pipe is released.
        os.kill(holder_pid, signal.SIGKILL)
        await asyncio.wait_for(process.wait(), 5)
        assert inst._stderr_task is not None
        await asyncio.wait_for(asyncio.wait({inst._stderr_task}), 1)
        assert not inst._stderr_task.cancelled()
    finally:
        for pid in (holder_pid, process.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.wait_for(process.wait(), 5)


async def test_cancelled_stop_kills_backend_and_keeps_draining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
sys.stderr.write("READY\\n")
sys.stderr.flush()
time.sleep(60)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stderr is not None
    assert await asyncio.wait_for(process.stderr.readline(), 5) == b"READY\n"
    config = make_config(tmp_path)
    manager = ServerManager(config)
    inst = _ModelInstance("stubborn", process, 9999, make_model(config), (0.0, 0.0))
    inst.start_stderr_reader()
    manager._instances["stubborn"] = inst
    monkeypatch.setattr(manager, "_save_prompt_cache", AsyncMock())
    try:
        stop = asyncio.create_task(manager.stop_model("stubborn"))
        await asyncio.sleep(0.2)  # inside the SIGTERM grace wait
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop
        assert not manager._instances
        # The untracked backend must not survive with an unread pipe.
        assert await asyncio.wait_for(process.wait(), 5) == -signal.SIGKILL
        assert inst._stderr_task is not None
        await asyncio.wait_for(asyncio.wait({inst._stderr_task}), 1)
        assert not inst._stderr_task.cancelled()
    finally:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.wait(), 5)


@pytest.mark.parametrize("name", ["./testmodel", "testmodel/", "org//testmodel", "../x", "", "/"])
async def test_ensure_running_rejects_noncanonical_names_before_lookup(
    tmp_path: Path, name: str
) -> None:
    """Alternate spellings must not become separate instance keys (#110)."""
    config = make_config(tmp_path)
    make_model(config)
    manager = ServerManager(config)
    with patch.object(manager, "_start", new_callable=AsyncMock) as start:
        with pytest.raises(ValueError, match="model name"):
            await manager.ensure_running(name)
    start.assert_not_awaited()
    assert manager._instances == {}


async def test_invalid_backend_is_detected_before_eviction(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.max_loaded_models = 1
    manager = ServerManager(config)
    other = make_model(config, "other")
    manager._instances["other"] = make_fake_instance("other", other, (0.0, 0.0))
    model_path = make_model(config, "bad")
    model_path.with_suffix(".toml").write_text('backend = "nope"\n')

    with (
        patch_stop_instance(manager) as stop,
        pytest.raises(ValueError, match="Unknown backend"),
    ):
        await manager.ensure_running("bad")

    stop.assert_not_awaited()
    assert list(manager._instances) == ["other"]


async def test_broken_sidecar_does_not_stop_idle_instance_with_changed_config(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    manager = ServerManager(config)
    model_path = make_model(config, "m")
    inst = make_fake_instance("m", model_path, (-1.0, -1.0))  # mtimes differ: config "changed"
    manager._instances["m"] = inst
    model_path.with_suffix(".toml").write_text("ctx_size = = 1\n")

    with patch_stop_instance(manager) as stop, pytest.raises(ValueError):
        await manager.ensure_running("m")

    stop.assert_not_awaited()
    assert manager._instances["m"] is inst


async def test_missing_binary_raises_backend_unavailable(tmp_path: Path) -> None:
    from drove.server_manager import BackendUnavailableError

    config = make_config(tmp_path)
    manager = ServerManager(config)
    make_model(config, "m")
    with patch("drove.server_manager.shutil.which", return_value=None):
        with pytest.raises(BackendUnavailableError, match="llama-server"):
            await manager.ensure_running("m")


async def test_exec_oserror_raises_backend_unavailable_with_cause(tmp_path: Path) -> None:
    from drove.server_manager import BackendUnavailableError

    config = make_config(tmp_path)
    manager = ServerManager(config)
    make_model(config, "m")
    boom = OSError(8, "Exec format error")
    with (
        patch("drove.server_manager.shutil.which", return_value="/usr/bin/llama-server"),
        patch(
            "drove.server_manager.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=boom),
        ),
        pytest.raises(BackendUnavailableError) as excinfo,
    ):
        await manager.ensure_running("m")
    assert excinfo.value.__cause__ is boom
