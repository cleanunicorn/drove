"""Manages the llama-server subprocess lifecycle."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path

import httpx

from vllama.config import Config
from vllama.model_config import load_model_config

logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 0.5  # seconds between health poll attempts
HEALTH_CHECK_TIMEOUT = 60.0  # max seconds to wait for llama-server to be ready


class ServerManager:
    """Manages a single llama-server subprocess.

    Only one model can be loaded at a time. Loading a different model
    stops the current server and starts a new one.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._current_model: str | None = None  # model name (stem)
        self._last_request_time: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def current_model(self) -> str | None:
        return self._current_model

    @property
    def base_url(self) -> str:
        return f"http://{self._config.llama_server_host}:{self._config.llama_server_port}"

    def record_request(self) -> None:
        """Call on each proxied request to reset the idle timer."""
        self._last_request_time = time.monotonic()

    async def ensure_running(self, model_name: str) -> None:
        """Ensure llama-server is running with the requested model.

        If a different model is loaded, the current server is stopped first.
        Thread-safe: concurrent callers wait on the lock.
        """
        async with self._lock:
            if self.is_running and self._current_model == model_name:
                return
            if self.is_running:
                await self._stop()
            await self._start(model_name)

    async def stop(self) -> None:
        """Gracefully stop the running llama-server."""
        async with self._lock:
            await self._stop()

    async def _start(self, model_name: str) -> None:
        model_path = self._resolve_model(model_name)
        model_cfg = load_model_config(model_path)

        # Merge global defaults then model-specific overrides
        args = self._build_args(model_path, model_cfg)

        logger.info("Starting llama-server: %s %s", self._config.llama_server_bin, " ".join(args))
        self._process = await asyncio.create_subprocess_exec(
            self._config.llama_server_bin,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._current_model = model_name
        self._last_request_time = time.monotonic()

        await self._wait_for_health()
        self._start_idle_watcher()
        logger.info("llama-server ready (model=%s)", model_name)

    async def _stop(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

        if self._process is None:
            return

        proc = self._process
        self._process = None
        self._current_model = None

        if proc.returncode is not None:
            return  # already dead

        logger.info("Stopping llama-server (pid=%d)", proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except TimeoutError:
                logger.warning("llama-server did not stop in time, sending SIGKILL")
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass  # already gone

    async def _wait_for_health(self) -> None:
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if not self.is_running:
                    raise RuntimeError("llama-server exited unexpectedly during startup")
                try:
                    resp = await client.get(f"{self.base_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        raise TimeoutError(f"llama-server did not become healthy within {HEALTH_CHECK_TIMEOUT}s")

    def _build_args(self, model_path: Path, model_cfg: "ModelConfig") -> list[str]:  # type: ignore[name-defined]
        from vllama.model_config import ModelConfig  # local import to avoid circular

        # Start with global defaults
        global_cfg = ModelConfig(
            n_gpu_layers=self._config.llama_server.n_gpu_layers,
            threads=self._config.llama_server.threads,
        )
        # Model-specific overrides take precedence
        merged = global_cfg.model_copy(
            update={k: v for k, v in model_cfg.to_dict().items()}
        )

        args = [
            "--model", str(model_path),
            "--host", self._config.llama_server_host,
            "--port", str(self._config.llama_server_port),
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

    def _start_idle_watcher(self) -> None:
        self._idle_task = asyncio.create_task(self._idle_watcher())

    async def _idle_watcher(self) -> None:
        while True:
            await asyncio.sleep(30)  # check every 30 seconds
            idle = time.monotonic() - self._last_request_time
            if idle >= self._config.idle_timeout_seconds:
                logger.info(
                    "Idle timeout reached (%.0fs), stopping llama-server", idle
                )
                async with self._lock:
                    await self._stop()
                return
