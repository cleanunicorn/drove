"""Register drove's models with local coding agents (pi, opencode, Kilo Code).

Each agent keeps its own config file describing OpenAI-compatible providers.
This module builds a description of every drove model — context window,
output budget, vision, tool calling, reasoning — and merges it into that file
under a single provider entry, so the agent can talk to the drove proxy with
no hand-written setup.

Only the provider entry drove owns is touched; everything else in the agent's
config is preserved.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from drove.backend import BACKEND_ASR, detect_backend
from drove.config import Config
from drove.gguf import read_model_metadata
from drove.model_config import ModelConfig, load_global_model_config, load_model_config
from drove.model_store import ModelStore

DEFAULT_PROVIDER_ID = "drove"
PROVIDER_DISPLAY_NAME = "drove (local)"

#: Context assumed when neither the drove config nor the GGUF header gives one.
FALLBACK_CONTEXT = 4096
#: Upper bound on the per-response output budget advertised to agents.
MAX_OUTPUT_TOKENS = 32768
MIN_OUTPUT_TOKENS = 1024

# Chat-template markers. llama-server renders tool calls and reasoning through
# the model's Jinja template, so the template is the best local signal for
# what the model was trained to do.
_TOOL_MARKERS = ("tools", "tool_call")
_REASONING_MARKERS = ("<think>", "enable_thinking", "reasoning_content", "<|channel|>")


@dataclass(frozen=True)
class ModelSpec:
    """Everything an agent needs to know about one drove model."""

    id: str
    name: str
    context_window: int
    max_output: int
    vision: bool
    tool_call: bool
    reasoning: bool


# ── model description ────────────────────────────────────────────────────────


def _effective_model_config(config: Config, primary: Path) -> ModelConfig:
    """Merge model config the way ServerManager does: global → _global.toml → sidecar."""
    merged = ModelConfig(
        n_gpu_layers=config.llama_server.n_gpu_layers,
        threads=config.llama_server.threads,
    )
    for layer in (load_global_model_config(config.models_dir), load_model_config(primary)):
        merged = merged.model_copy(update=layer.to_dict())
    return merged


def _last_flag(args: list[str], flags: tuple[str, ...]) -> tuple[int, str] | None:
    """Return (position, value) of the last ``FLAG VALUE`` / ``FLAG=VALUE`` in *args*.

    llama-server takes the last occurrence of a flag, so the last one wins here too.
    """
    found: tuple[int, str] | None = None
    for i, arg in enumerate(args):
        if arg in flags and i + 1 < len(args):
            found = (i, args[i + 1])
        else:
            name, sep, value = arg.partition("=")
            if sep and name in flags:
                found = (i, value)
    return found


def _last_switch(args: list[str], flags: tuple[str, ...]) -> int | None:
    """Return the position of the last bare switch from *flags* in *args*."""
    positions = [i for i, arg in enumerate(args) if arg in flags]
    return positions[-1] if positions else None


def _int_flag(args: list[str], flags: tuple[str, ...]) -> int | None:
    found = _last_flag(args, flags)
    if found is None:
        return None
    try:
        return int(found[1])
    except ValueError:
        return None


def _context_window(cfg: ModelConfig, training_ctx: int | None) -> int:
    """The context one request can use, following llama-server's slot layout."""
    args = cfg.extra_args or []
    ctx = _int_flag(args, ("-c", "--ctx-size"))
    if ctx is None:
        ctx = cfg.ctx_size
    # 0 (llama-server's default) means "use the model's training context".
    per_slot = ctx or training_ctx or FALLBACK_CONTEXT

    n_parallel = _int_flag(args, ("-np", "--parallel"))
    if n_parallel is None:
        n_parallel = cfg.n_parallel
    auto_slots = n_parallel is None or n_parallel <= 0
    # The KV cache is unified (any slot may use all of it) only when the slot
    # count is left on auto, unless -kvu / -no-kvu says otherwise.
    unified_on = _last_switch(args, ("-kvu", "--kv-unified"))
    unified_off = _last_switch(args, ("-no-kvu", "--no-kv-unified"))
    if unified_on is None and unified_off is None:
        unified = auto_slots
    else:
        unified = unified_off is None or (unified_on is not None and unified_on > unified_off)
    if not unified and not auto_slots and n_parallel is not None and n_parallel > 1:
        per_slot //= n_parallel

    slot_cap = _int_flag(args, ("--kv-unified-per-slot",))
    if slot_cap:
        per_slot = min(per_slot, slot_cap)
    # llama-server caps a slot at the training context.
    if training_ctx:
        per_slot = min(per_slot, training_ctx)
    return max(per_slot, 1)


def _output_budget(context_window: int) -> int:
    """A quarter of the context within [MIN, MAX], but never more than half of it."""
    budget = max(MIN_OUTPUT_TOKENS, min(context_window // 4, MAX_OUTPUT_TOKENS))
    return max(1, min(budget, context_window // 2))


def _has_vision(primary: Path, cfg: ModelConfig) -> bool:
    args = cfg.extra_args or []
    enabled = _last_flag(args, ("-mm", "--mmproj"))
    disabled = _last_switch(args, ("--no-mmproj",))
    if disabled is not None and (enabled is None or disabled > enabled[0]):
        return False
    if enabled is not None or cfg.mmproj:
        return True
    return any(
        f.is_file() and f.suffix.lower() == ".gguf" and "mmproj" in f.name.lower()
        for f in primary.parent.iterdir()
    )


def _chat_template(cfg: ModelConfig, gguf_template: str | None) -> str | None:
    """Return the chat template llama-server will use.

    An empty string means "unknown": a template file that cannot be read, or a
    built-in template referenced by name.
    """
    args = cfg.extra_args or []
    inline = _last_flag(args, ("--chat-template",))
    from_file = _last_flag(args, ("--chat-template-file",))
    if inline is None and from_file is None:
        return gguf_template
    if from_file is not None and (inline is None or from_file[0] > inline[0]):
        try:
            return Path(from_file[1]).expanduser().read_text()
        except OSError:
            return ""
    assert inline is not None
    # --chat-template takes either Jinja source or a built-in template's name.
    return inline[1] if "{%" in inline[1] or "{{" in inline[1] else ""


def describe_model(config: Config, name: str, primary: Path) -> ModelSpec | None:
    """Build a ModelSpec, or None for models agents cannot chat with (speech-to-text)."""
    cfg = _effective_model_config(config, primary)
    if detect_backend(primary, cfg) == BACKEND_ASR:
        return None

    meta = read_model_metadata(primary)
    ctx = _context_window(cfg, meta.context_length if meta else None)

    args = cfg.extra_args or []
    template = _chat_template(cfg, meta.chat_template if meta else None) or ""
    jinja = _last_switch(args, ("--jinja",))
    no_jinja = _last_switch(args, ("--no-jinja",))
    if no_jinja is not None and (jinja is None or no_jinja > jinja):
        # Tool calls and reasoning extraction both need the Jinja engine.
        tool_call = reasoning = False
    else:
        # With no known template llama-server falls back to its generic
        # tool-call format, so only a template that never mentions tools
        # rules them out.
        tool_call = not template or any(m in template for m in _TOOL_MARKERS)
        reasoning = any(m in template for m in _REASONING_MARKERS)
        budget = _last_flag(args, ("--reasoning-budget",))
        if budget is not None and budget[1] == "0":
            reasoning = False

    return ModelSpec(
        id=name,
        name=f"{name} (drove)",
        context_window=ctx,
        max_output=_output_budget(ctx),
        vision=_has_vision(primary, cfg),
        tool_call=tool_call,
        reasoning=reasoning,
    )


def collect_model_specs(config: Config, names: Iterable[str] | None = None) -> list[ModelSpec]:
    """Describe the named models, or every chat-capable model when *names* is None.

    Raises FileNotFoundError for an unknown name and ValueError for a named
    model agents cannot use (speech-to-text).
    """
    store = ModelStore(config.models_dir)
    if names is None:
        entries = [(e.name, e.primary) for e in store.list()]
        specs = [describe_model(config, n, p) for n, p in entries]
        return [s for s in specs if s is not None]

    result: list[ModelSpec] = []
    for name in names:
        spec = describe_model(config, name, store.resolve(name))
        if spec is None:
            raise ValueError(
                f"Model '{name}' is a speech-to-text model; agents cannot chat with it."
            )
        result.append(spec)
    return result


def default_base_url(config: Config) -> str:
    """The OpenAI-compatible base URL of the local drove proxy."""
    host = config.listen_host
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{config.listen_port}/v1"


# ── JSON / JSONC files ───────────────────────────────────────────────────────

# Strings first so comment markers and commas inside them are left alone.
_JSONC_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*.*?\*/', re.DOTALL)
_TRAILING_COMMA = re.compile(r'"(?:\\.|[^"\\])*"|,(\s*[}\]])', re.DOTALL)


def _strip_comments(text: str) -> str:
    # A comment becomes a space so it still separates the tokens around it.
    return _JSONC_TOKEN.sub(lambda m: m.group(0) if m.group(0).startswith('"') else " ", text)


def _strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas from JSONC text."""
    return _TRAILING_COMMA.sub(lambda m: m.group(1) or m.group(0), _strip_comments(text))


def _read_text(path: Path) -> str:
    # utf-8-sig drops a byte-order mark, which json.loads rejects.
    return path.read_text(encoding="utf-8-sig")


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON or JSONC object from *path*; a missing or empty file is ``{}``."""
    if not path.exists():
        return {}
    text = _read_text(path)
    if not text.strip():
        return {}
    data = json.loads(_strip_jsonc(text))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return data


def has_comments(path: Path) -> bool:
    """Whether rewriting *path* as plain JSON would drop comments."""
    if not path.exists():
        return False
    text = _read_text(path)
    return _strip_comments(text) != text


def dump_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _stage(target: Path, text: str) -> Path:
    """Write *text* to a fresh private temp file beside *target* and return it.

    mkstemp creates the file exclusively with mode 0600, so a planted file or
    symlink at a predictable name cannot redirect the write. The original's
    permissions are copied over so a private config stays private.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if target.exists():
            os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _backup(path: Path, target: Path) -> Path:
    backup = path.with_name(path.name + ".bak")
    if backup.is_symlink():
        backup.unlink()  # never write through a link at the backup name
    shutil.copy2(target, backup)
    return backup


def write_configs(changes: list[tuple[Path, dict[str, Any]]]) -> list[Path]:
    """Write every file in *changes*, or none of them.

    Each file is staged beside its target first, so a failure while writing
    (say, pi's ``settings.json`` after its ``models.json``) leaves every
    original in place. A config that is a symlink (into a dotfiles repo, say)
    stays a symlink: its target is updated. Previous contents are kept as
    ``<file>.bak``. Returns the backup paths.
    """
    staged: list[tuple[Path, Path, Path]] = []  # (path, real target, temp file)
    replaced: list[tuple[Path, Path | None]] = []  # (real target, backup)
    try:
        for path, data in changes:
            target = path.resolve() if path.is_symlink() else path
            staged.append((path, target, _stage(target, dump_json(data))))
        for path, target, tmp in staged:
            backup = _backup(path, target) if target.exists() else None
            os.replace(tmp, target)
            replaced.append((target, backup))
    except BaseException as e:
        failed = _roll_back(replaced, [tmp for _, _, tmp in staged])
        if failed:
            raise IncompleteRollbackError(failed) from e
        raise
    return [b for _, b in replaced if b is not None]


class IncompleteRollbackError(OSError):
    """A failed write could not be fully undone; *paths* hold the new contents."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        listed = ", ".join(str(p) for p in paths)
        super().__init__(f"could not restore {listed}; the previous contents are in the .bak files")


def _roll_back(replaced: list[tuple[Path, Path | None]], temps: list[Path]) -> list[Path]:
    """Undo *replaced* writes, newest first; return the targets that could not be restored.

    Restores go through the same staged replacement as writes, so a target
    that is now read-only (its directory still writable) is restored too. Every
    restore and cleanup is attempted even after one fails.
    """
    failed: list[Path] = []
    for target, backup in reversed(replaced):
        try:
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                restore = _stage(target, backup.read_text(encoding="utf-8"))
                temps.append(restore)
                os.replace(restore, target)
        except OSError:
            failed.append(target)
    for tmp in temps:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return failed


class ConfigFiles:
    """Agent config files, loaded once, edited in memory, and written together.

    An integration can touch more than one file (pi keeps its default model in
    ``settings.json`` next to ``models.json``); routing every edit through one
    instance keeps them consistent and writes only the files that changed.
    """

    def __init__(self) -> None:
        self._data: dict[Path, dict[str, Any]] = {}
        self._original: dict[Path, str] = {}

    def get(self, path: Path) -> dict[str, Any]:
        if path not in self._data:
            data = load_json_file(path)
            self._data[path] = data
            self._original[path] = dump_json(data)
        return self._data[path]

    def changed(self) -> list[tuple[Path, dict[str, Any]]]:
        return [(p, d) for p, d in self._data.items() if dump_json(d) != self._original[p]]


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge *source* into *target*, recursing into nested objects.

    Generated values win, while keys the user added at any depth (such as an
    opencode ``limit.input``) survive.
    """
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ── agents ───────────────────────────────────────────────────────────────────


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


#: Removed model ids, or None when the whole provider was removed.
Removed = set[str] | None


def _points_at_removed(ref: Any, removed: Removed, provider_id: str) -> bool:
    """Whether a ``provider/model`` reference names something that was removed."""
    if not isinstance(ref, str) or not ref.startswith(f"{provider_id}/"):
        return False
    return removed is None or ref.split("/", 1)[1] in removed


class Agent(Protocol):
    """A local agent whose provider config drove can manage.

    Methods edit files through *files*; *path* is the agent's main config file.
    """

    key: str
    display_name: str

    def config_path(self) -> Path: ...

    def model_ids(self, files: ConfigFiles, path: Path, *, provider_id: str) -> list[str]: ...

    def apply(
        self,
        files: ConfigFiles,
        path: Path,
        specs: list[ModelSpec],
        *,
        provider_id: str,
        base_url: str,
        prune: bool,
    ) -> None: ...

    def remove(
        self, files: ConfigFiles, path: Path, model_ids: list[str] | None, *, provider_id: str
    ) -> None: ...

    def set_default(
        self, files: ConfigFiles, path: Path, model_id: str, *, provider_id: str
    ) -> None: ...


class _OpenCodeLike:
    """opencode's config schema, which Kilo Code's CLI shares (it is an opencode fork)."""

    key = ""
    display_name = ""
    #: Env var that points at an explicit config file.
    env_var = ""
    #: Directory under the XDG config home, and the file names the agent reads
    #: there in the order drove prefers them; the first is created if none exist.
    config_dir = ""
    candidates: tuple[str, ...] = ()
    #: Top-level keys that select a ``provider/model``.
    selection_keys = ("model", "small_model")

    def config_path(self) -> Path:
        explicit = os.environ.get(self.env_var)
        if explicit:
            return Path(explicit).expanduser()
        directory = _xdg_config_home() / self.config_dir
        for name in self.candidates:
            if (directory / name).exists():
                return directory / name
        return directory / self.candidates[0]

    @staticmethod
    def _model_entry(spec: ModelSpec) -> dict[str, Any]:
        modalities = ["text", "image"] if spec.vision else ["text"]
        return {
            "name": spec.name,
            "tool_call": spec.tool_call,
            "reasoning": spec.reasoning,
            "attachment": spec.vision,
            "temperature": True,
            "modalities": {"input": modalities, "output": ["text"]},
            "limit": {"context": spec.context_window, "output": spec.max_output},
        }

    def model_ids(self, files: ConfigFiles, path: Path, *, provider_id: str) -> list[str]:
        provider = files.get(path).get("provider", {}).get(provider_id, {})
        return list(provider.get("models", {}))

    def _clear_selections(self, data: dict[str, Any], removed: Removed, provider_id: str) -> None:
        for key in self.selection_keys:
            if _points_at_removed(data.get(key), removed, provider_id):
                del data[key]

    def apply(
        self,
        files: ConfigFiles,
        path: Path,
        specs: list[ModelSpec],
        *,
        provider_id: str,
        base_url: str,
        prune: bool,
    ) -> None:
        data = files.get(path)
        data.setdefault("$schema", "https://opencode.ai/config.json")
        provider = data.setdefault("provider", {}).setdefault(provider_id, {})
        provider["npm"] = "@ai-sdk/openai-compatible"
        provider["name"] = PROVIDER_DISPLAY_NAME
        provider.setdefault("options", {})["baseURL"] = base_url
        models: dict[str, Any] = provider.setdefault("models", {})
        if prune:
            wanted = {s.id for s in specs}
            stale = {k for k in models if k not in wanted}
            for mid in stale:
                del models[mid]
            self._clear_selections(data, stale, provider_id)
        for spec in specs:
            _deep_update(models.setdefault(spec.id, {}), self._model_entry(spec))

    def remove(
        self, files: ConfigFiles, path: Path, model_ids: list[str] | None, *, provider_id: str
    ) -> None:
        data = files.get(path)
        providers = data.get("provider", {})
        if model_ids is None:
            providers.pop(provider_id, None)
            self._clear_selections(data, None, provider_id)
            return
        models = providers.get(provider_id, {}).get("models", {})
        for mid in model_ids:
            models.pop(mid, None)
        self._clear_selections(data, set(model_ids), provider_id)

    def set_default(
        self, files: ConfigFiles, path: Path, model_id: str, *, provider_id: str
    ) -> None:
        files.get(path)["model"] = f"{provider_id}/{model_id}"


class OpenCode(_OpenCodeLike):
    key = "opencode"
    display_name = "opencode"
    env_var = "OPENCODE_CONFIG"
    config_dir = "opencode"
    candidates = ("opencode.json", "opencode.jsonc", "config.json")


class KiloCode(_OpenCodeLike):
    key = "kilo"
    display_name = "Kilo Code"
    env_var = "KILO_CONFIG"
    config_dir = "kilo"
    candidates = ("kilo.json", "kilo.jsonc", "opencode.json", "opencode.jsonc", "config.json")


class Pi:
    """pi coding agent: ``models.json`` and ``settings.json`` in its agent directory."""

    key = "pi"
    display_name = "pi"

    def config_path(self) -> Path:
        explicit = os.environ.get("PI_CODING_AGENT_DIR")
        agent_dir = Path(explicit).expanduser() if explicit else Path.home() / ".pi" / "agent"
        return agent_dir / "models.json"

    @staticmethod
    def _settings_path(path: Path) -> Path:
        # pi keeps the default model in settings.json beside models.json, so a
        # --file elsewhere edits the settings that belong with that file.
        return path.with_name("settings.json")

    @staticmethod
    def _model_entry(spec: ModelSpec) -> dict[str, Any]:
        return {
            "id": spec.id,
            "name": spec.name,
            "reasoning": spec.reasoning,
            "input": ["text", "image"] if spec.vision else ["text"],
            "contextWindow": spec.context_window,
            "maxTokens": spec.max_output,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }

    def model_ids(self, files: ConfigFiles, path: Path, *, provider_id: str) -> list[str]:
        provider = files.get(path).get("providers", {}).get(provider_id, {})
        return [m["id"] for m in provider.get("models", []) if isinstance(m, dict) and "id" in m]

    def _clear_selections(
        self, files: ConfigFiles, path: Path, removed: Removed, provider_id: str
    ) -> None:
        settings_path = self._settings_path(path)
        if removed is not None and not removed:
            return
        if not settings_path.exists():
            return
        settings = files.get(settings_path)
        if settings.get("defaultProvider") == provider_id and (
            removed is None or settings.get("defaultModel") in removed
        ):
            settings.pop("defaultProvider", None)
            settings.pop("defaultModel", None)
        enabled = settings.get("enabledModels")
        if isinstance(enabled, list):
            settings["enabledModels"] = [
                m for m in enabled if not _points_at_removed(m, removed, provider_id)
            ]

    def apply(
        self,
        files: ConfigFiles,
        path: Path,
        specs: list[ModelSpec],
        *,
        provider_id: str,
        base_url: str,
        prune: bool,
    ) -> None:
        data = files.get(path)
        data.setdefault("$schema", "https://pi.dev/schemas/models.schema.json")
        provider = data.setdefault("providers", {}).setdefault(provider_id, {})
        provider["name"] = PROVIDER_DISPLAY_NAME
        provider["baseUrl"] = base_url
        provider["api"] = "openai-completions"
        # pi only lists a provider that has a key; drove ignores it.
        provider.setdefault("apiKey", "drove")

        existing = {m.get("id"): m for m in provider.get("models", []) if isinstance(m, dict)}
        if prune:
            wanted = {s.id for s in specs}
            stale = {k for k in existing if k not in wanted}
            existing = {k: v for k, v in existing.items() if k not in stale}
            self._clear_selections(files, path, {str(k) for k in stale}, provider_id)
        for spec in specs:
            _deep_update(existing.setdefault(spec.id, {}), self._model_entry(spec))
        provider["models"] = list(existing.values())

    def remove(
        self, files: ConfigFiles, path: Path, model_ids: list[str] | None, *, provider_id: str
    ) -> None:
        providers = files.get(path).get("providers", {})
        if model_ids is None:
            providers.pop(provider_id, None)
            self._clear_selections(files, path, None, provider_id)
            return
        if provider_id in providers:
            provider = providers[provider_id]
            provider["models"] = [
                m for m in provider.get("models", []) if m.get("id") not in model_ids
            ]
        self._clear_selections(files, path, set(model_ids), provider_id)

    def set_default(
        self, files: ConfigFiles, path: Path, model_id: str, *, provider_id: str
    ) -> None:
        settings = files.get(self._settings_path(path))
        settings["defaultProvider"] = provider_id
        settings["defaultModel"] = model_id


AGENTS: dict[str, Agent] = {a.key: a for a in (Pi(), OpenCode(), KiloCode())}


def get_agent(key: str) -> Agent:
    try:
        return AGENTS[key.lower()]
    except KeyError:
        valid = ", ".join(sorted(AGENTS))
        raise ValueError(f"Unknown agent '{key}'. Supported agents: {valid}") from None
