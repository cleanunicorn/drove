"""Tests for ServerManager config change detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
