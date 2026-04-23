"""Tests for ServerManager config change detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from vllama.config import Config
from vllama.server_manager import ServerManager, _ModelInstance


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

    async def fake_start(model_name: str) -> None:
        new_mtimes = manager._get_config_mtimes(model_path)
        manager._instances[model_name] = make_fake_instance(model_name, model_path, new_mtimes)

    with (
        patch.object(manager, "_stop_instance", new_callable=AsyncMock),
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

    async def fake_start_b(model_name: str) -> None:
        manager._instances[model_name] = make_fake_instance(
            model_name, config.models_dir / model_name / f"{model_name}.gguf", mtimes
        )

    with (
        patch("vllama.server_manager.asyncio.sleep", side_effect=fake_sleep),
        patch.object(manager, "_start", new_callable=AsyncMock, side_effect=fake_start_b),
        patch.object(manager, "_stop_instance", new_callable=AsyncMock) as mock_stop,
    ):
        await manager.ensure_running("model-b")

    # model-a was only evicted after its in-flight request drained (wait_count >= 2)
    assert wait_count >= 2
    mock_stop.assert_awaited_with("model-a")


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
        with patch("vllama.server_manager.asyncio.sleep", new_callable=AsyncMock):
            # _idle_watcher loops; after stopping it returns, so this will finish
            await manager._idle_watcher("testmodel")

        mock_stop.assert_awaited_once_with("testmodel")
    assert inst.needs_restart
