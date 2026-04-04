"""Manages the llama-server subprocess lifecycle."""

from __future__ import annotations

import asyncio
import logging
import shutil
import signal
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import psutil

if TYPE_CHECKING:
    from vllama.model_config import ModelConfig

from vllama.config import Config
from vllama.model_config import load_model_config

logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 0.5  # seconds between health poll attempts
HEALTH_CHECK_TIMEOUT = 60.0  # max seconds to wait for llama-server to be ready


class _ModelInstance:
    """State for a single running llama-server process."""

    def __init__(self, model_name: str, process: asyncio.subprocess.Process, port: int) -> None:
        self.model_name = model_name
        self.process = process
        self.port = port
        self.loaded_at: float = time.time()
        self.last_request_time: float = time.monotonic()
        self.active_requests: int = 0

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_request_time

    def get_process_stats(self) -> dict[str, object] | None:
        if not self.is_running:
            return None
        try:
            proc = psutil.Process(self.process.pid)
            mem = proc.memory_info()
            cpu_times = proc.cpu_times()
            elapsed = time.time() - proc.create_time()
            cpu_pct = (cpu_times.user + cpu_times.system) / elapsed * 100 if elapsed > 0 else 0
            return {
                "memory_rss_bytes": mem.rss,
                "cpu_percent": round(cpu_pct, 1),
            }
        except psutil.NoSuchProcess, psutil.AccessDenied:
            return None


class ServerManager:
    """Manages multiple llama-server subprocesses, one per model.

    Each model gets its own llama-server on a dynamically assigned port.
    Models are started lazily on first request and stopped after idle timeout.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._instances: dict[str, _ModelInstance] = {}
        self._idle_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return any(inst.is_running for inst in self._instances.values())

    @property
    def loaded_models(self) -> list[str]:
        return [name for name, inst in self._instances.items() if inst.is_running]

    @property
    def current_model(self) -> str | None:
        """For backwards compatibility — returns the first loaded model, or None."""
        models = self.loaded_models
        return models[0] if models else None

    @property
    def model_loaded_at(self) -> float | None:
        """For backwards compatibility — returns loaded_at of the first model."""
        models = self.loaded_models
        if models:
            return self._instances[models[0]].loaded_at
        return None

    @property
    def idle_seconds(self) -> float:
        """For backwards compatibility — returns minimum idle across all models."""
        running = [inst for inst in self._instances.values() if inst.is_running]
        if not running:
            return 0.0
        return min(inst.idle_seconds for inst in running)

    def base_url_for(self, model_name: str) -> str:
        inst = self._instances.get(model_name)
        port = inst.port if inst else 0
        return f"http://{self._config.llama_server_host}:{port}"

    @property
    def base_url(self) -> str:
        """For backwards compatibility — returns base_url of the first loaded model."""
        model = self.current_model
        if model:
            return self.base_url_for(model)
        return f"http://{self._config.llama_server_host}:0"

    def get_process_stats(self) -> dict[str, object] | None:
        """Return aggregated stats, or per-model stats if multiple models loaded."""
        running = {name: inst for name, inst in self._instances.items() if inst.is_running}
        if not running:
            return None
        if len(running) == 1:
            return next(iter(running.values())).get_process_stats()
        return {name: inst.get_process_stats() for name, inst in running.items()}

    def get_all_model_info(self) -> list[dict[str, object]]:
        """Return status info for all loaded models."""
        now = time.time()
        result = []
        for name, inst in self._instances.items():
            if not inst.is_running:
                continue
            result.append({
                "name": name,
                "loaded_seconds": round(now - inst.loaded_at, 1),
                "idle_seconds": round(inst.idle_seconds, 1),
                "idle_timeout_seconds": self._config.idle_timeout_seconds,
                "active_requests": inst.active_requests,
                "port": inst.port,
            })
        return result

    def record_request(self, model_name: str) -> None:
        """Call on each proxied request to reset the idle timer for a model."""
        inst = self._instances.get(model_name)
        if inst:
            inst.last_request_time = time.monotonic()

    def request_started(self, model_name: str) -> None:
        """Mark a request as in-flight for a model."""
        inst = self._instances.get(model_name)
        if inst:
            inst.active_requests += 1
            inst.last_request_time = time.monotonic()

    def request_finished(self, model_name: str) -> None:
        """Mark a request as complete and reset the idle timer for a model."""
        inst = self._instances.get(model_name)
        if inst:
            inst.active_requests = max(0, inst.active_requests - 1)
            inst.last_request_time = time.monotonic()

    async def ensure_running(self, model_name: str) -> None:
        """Ensure llama-server is running for the requested model.

        If the model is already loaded, this is a no-op.
        Multiple models can run simultaneously.
        """
        async with self._lock:
            inst = self._instances.get(model_name)
            if inst is not None and inst.is_running:
                return
            # Clean up stale instance if process died
            if inst is not None:
                await self._stop_instance(model_name)
            await self._start(model_name)

    async def stop(self) -> None:
        """Gracefully stop all running llama-server processes."""
        async with self._lock:
            names = list(self._instances.keys())
            for name in names:
                await self._stop_instance(name)

    async def stop_model(self, model_name: str) -> None:
        """Gracefully stop a specific model's llama-server."""
        async with self._lock:
            await self._stop_instance(model_name)

    async def _start(self, model_name: str) -> None:
        binary = self._config.llama_server_bin
        if not shutil.which(binary):
            raise FileNotFoundError(
                f"llama-server binary '{binary}' not found on PATH. "
                "Install llama.cpp or set 'llama_server_bin' in config."
            )

        model_path = self._resolve_model(model_name)
        model_cfg = load_model_config(model_path)

        port = _find_free_port()
        args = self._build_args(model_path, model_cfg, port)

        logger.info("Starting llama-server: %s %s", binary, " ".join(args))
        process = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        inst = _ModelInstance(model_name, process, port)
        self._instances[model_name] = inst

        try:
            await self._wait_for_health(inst)
        except (RuntimeError, TimeoutError) as e:
            logger.error("llama-server failed to start for model=%s: %s", model_name, e)
            await self._stop_instance(model_name)
            raise

        self._start_idle_watcher(model_name)
        logger.info("llama-server ready (model=%s, port=%d)", model_name, port)

    async def _stop_instance(self, model_name: str) -> None:
        task = self._idle_tasks.pop(model_name, None)
        if task is not None:
            task.cancel()

        inst = self._instances.pop(model_name, None)
        if inst is None:
            return

        if not inst.is_running:
            return

        logger.info("Stopping llama-server (model=%s, pid=%d)", model_name, inst.process.pid)
        try:
            inst.process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(inst.process.wait(), timeout=10.0)
            except TimeoutError:
                logger.warning("llama-server did not stop in time, sending SIGKILL")
                inst.process.kill()
                await inst.process.wait()
        except ProcessLookupError:
            pass  # already gone

    async def _wait_for_health(self, inst: _ModelInstance) -> None:
        url = f"http://{self._config.llama_server_host}:{inst.port}/health"
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if not inst.is_running:
                    stderr = await self._read_stderr(inst)
                    msg = "llama-server exited unexpectedly during startup"
                    if stderr:
                        msg += f"\nstderr: {stderr}"
                    raise RuntimeError(msg)
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        raise TimeoutError(f"llama-server did not become healthy within {HEALTH_CHECK_TIMEOUT}s")

    async def _read_stderr(self, inst: _ModelInstance) -> str:
        if inst.process.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(inst.process.stderr.read(4096), timeout=1.0)
            return data.decode(errors="replace").strip()
        except TimeoutError, Exception:
            return ""

    def _build_args(self, model_path: Path, model_cfg: ModelConfig, port: int) -> list[str]:
        from vllama.model_config import ModelConfig  # local import to avoid circular

        # Start with global defaults
        global_cfg = ModelConfig(
            n_gpu_layers=self._config.llama_server.n_gpu_layers,
            threads=self._config.llama_server.threads,
        )
        # Model-specific overrides take precedence
        merged = global_cfg.model_copy(update={k: v for k, v in model_cfg.to_dict().items()})

        args = [
            "--model",
            str(model_path),
            "--host",
            self._config.llama_server_host,
            "--port",
            str(port),
        ]
        args.extend(merged.to_llama_args())
        return args

    def _resolve_model(self, model_name: str) -> Path:
        models_dir = self._config.models_dir

        # Single GGUF file
        candidate = models_dir / f"{model_name}.gguf"
        if candidate.exists():
            return candidate

        # Subdirectory (sharded / multi-file) — return first shard
        subdir = models_dir / model_name
        if subdir.is_dir():
            shards = sorted(p for p in subdir.iterdir() if p.suffix.lower() == ".gguf")
            if shards:
                return shards[0]

        raise FileNotFoundError(
            f"Model '{model_name}' not found in {models_dir}. "
            "Run 'vllama models list' to see available models."
        )

    def _start_idle_watcher(self, model_name: str) -> None:
        self._idle_tasks[model_name] = asyncio.create_task(self._idle_watcher(model_name))

    async def _idle_watcher(self, model_name: str) -> None:
        while True:
            await asyncio.sleep(30)  # check every 30 seconds
            inst = self._instances.get(model_name)
            if inst is None or not inst.is_running:
                return
            if inst.active_requests > 0:
                continue  # never shut down while requests are in-flight
            idle = time.monotonic() - inst.last_request_time
            if idle >= self._config.idle_timeout_seconds:
                logger.info(
                    "Idle timeout reached for model=%s (%.0fs), stopping", model_name, idle
                )
                async with self._lock:
                    await self._stop_instance(model_name)
                return


def _find_free_port() -> int:
    """Bind to port 0 to let the OS assign an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
