"""Tests for registering drove models with local agents (`drove integrate`)."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from drove.cli.main import app
from drove.config import Config
from drove.integrations import (
    FALLBACK_CONTEXT,
    ConfigFiles,
    IncompleteRollbackError,
    KiloCode,
    ModelSpec,
    OpenCode,
    Pi,
    collect_model_specs,
    default_base_url,
    get_agent,
    has_comments,
    load_json_file,
    write_configs,
)
from tests.test_gguf import _kv_string, _kv_uint32, write_gguf

TOOL_THINK_TEMPLATE = "{% if tools %}{{ tools }}{% endif %}<think>"


@pytest.fixture(autouse=True)
def _isolated_agent_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every agent's config lookup inside tmp_path."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    monkeypatch.delenv("KILO_CONFIG", raising=False)


def _model(models_dir: Path, name: str, *, ctx: int | None = 32768, template: str = "") -> Path:
    model_dir = models_dir / name
    model_dir.mkdir(parents=True)
    entries = [_kv_string("general.architecture", "qwen3")]
    if ctx is not None:
        entries.append(_kv_uint32("qwen3.context_length", ctx))
    if template:
        entries.append(_kv_string("tokenizer.chat_template", template))
    return write_gguf(model_dir / f"{name}.gguf", entries)


def _config(tmp_path: Path) -> Config:
    return Config(models_dir=tmp_path / "models", listen_host="0.0.0.0", listen_port=8080)


def _spec(model_id: str, *, vision: bool = False) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=f"{model_id} (drove)",
        context_window=32768,
        max_output=8192,
        vision=vision,
        tool_call=True,
        reasoning=False,
    )


# ── model description ────────────────────────────────────────────────────────


def test_spec_uses_training_context_and_template_capabilities(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "coder", ctx=131072, template=TOOL_THINK_TEMPLATE)

    [spec] = collect_model_specs(config)

    assert spec.id == "coder"
    assert spec.context_window == 131072
    assert spec.max_output == 32768  # capped
    assert spec.tool_call and spec.reasoning
    assert not spec.vision


def test_configured_ctx_size_wins_over_training_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text("ctx_size = 16384\n")

    [spec] = collect_model_specs(config)

    assert spec.context_window == 16384
    assert spec.max_output == 4096


def test_global_model_config_ctx_size_applies(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "m", ctx=131072)
    (config.models_dir / "_global.toml").write_text("ctx_size = 8192\n")

    [spec] = collect_model_specs(config)

    assert spec.context_window == 8192


def test_extra_args_ctx_size_wins_like_llama_server(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text('ctx_size = 16384\nextra_args = ["-c", "65536"]\n')

    [spec] = collect_model_specs(config)

    assert spec.context_window == 65536


def test_unknown_context_falls_back(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "m", ctx=None)

    [spec] = collect_model_specs(config)

    assert spec.context_window == FALLBACK_CONTEXT
    assert spec.known_context_window is None


@pytest.mark.parametrize(
    ("training", "sidecar", "known"),
    [
        (131072, "", 131072),
        (131072, "ctx_size = 16384\n", 16384),
        (None, "ctx_size = 16384\n", 16384),
        (131072, 'ctx_size = 16384\nextra_args = ["-c", "65536"]\n', 65536),
        (131072, 'ctx_size = 16384\nextra_args = ["--ctx-size=0"]\n', 131072),
        (None, "ctx_size = 0\n", None),
        (None, 'extra_args = ["-c", "-1"]\n', None),
        (131072, "ctx_size = 32768\nn_parallel = 4\n", 8192),
        (8192, "ctx_size = 65536\n", 8192),
    ],
)
def test_known_context_window_is_only_reported_when_configured_or_trained(
    tmp_path: Path, training: int | None, sidecar: str, known: int | None
) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=training)
    if sidecar:
        primary.with_suffix(".toml").write_text(sidecar)

    [spec] = collect_model_specs(config)

    assert spec.known_context_window == known
    if known is not None:
        assert spec.context_window == known


def test_template_without_tools_disables_tool_calls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "m", template="{{ messages }}")

    [spec] = collect_model_specs(config)

    assert not spec.tool_call
    assert not spec.reasoning


def test_explicit_parallel_slots_split_the_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text("ctx_size = 32768\nn_parallel = 4\n")

    [spec] = collect_model_specs(config)

    assert spec.context_window == 8192


def test_unified_kv_keeps_the_full_context_per_request(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text(
        'ctx_size = 32768\nn_parallel = 4\nextra_args = ["--kv-unified"]\n'
    )

    [spec] = collect_model_specs(config)

    assert spec.context_window == 32768


def test_slot_flags_in_extra_args_split_the_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text(
        'ctx_size = 32768\nextra_args = ["-np", "2", "-no-kvu"]\n'
    )

    [spec] = collect_model_specs(config)

    assert spec.context_window == 16384


def test_per_slot_cap_limits_the_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text('extra_args = ["--kv-unified-per-slot", "20000"]\n')

    [spec] = collect_model_specs(config)

    assert spec.context_window == 20000


def test_context_is_capped_at_the_training_length(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=8192)
    primary.with_suffix(".toml").write_text("ctx_size = 65536\n")

    [spec] = collect_model_specs(config)

    assert spec.context_window == 8192


def test_zero_ctx_in_extra_args_means_training_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", ctx=131072)
    primary.with_suffix(".toml").write_text('ctx_size = 16384\nextra_args = ["--ctx-size=0"]\n')

    [spec] = collect_model_specs(config)

    assert spec.context_window == 131072


@pytest.mark.parametrize(
    ("ctx", "output"),
    [(512, 256), (2048, 1024), (4096, 1024), (8192, 2048), (262144, 32768)],
)
def test_output_budget_always_leaves_room_for_input(tmp_path: Path, ctx: int, output: int) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "m", ctx=ctx)

    [spec] = collect_model_specs(config)

    assert spec.max_output == output
    assert spec.max_output <= spec.context_window // 2


def test_no_mmproj_in_extra_args_disables_vision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m")
    (primary.parent / "mmproj-F16.gguf").write_bytes(b"")
    primary.with_suffix(".toml").write_text(
        'mmproj = "mmproj-F16.gguf"\nextra_args = ["--no-mmproj"]\n'
    )

    [spec] = collect_model_specs(config)

    assert not spec.vision


def test_chat_template_file_override_drives_capabilities(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", template=TOOL_THINK_TEMPLATE)
    plain = tmp_path / "plain.jinja"
    plain.write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    primary.with_suffix(".toml").write_text(f'extra_args = ["--chat-template-file", "{plain}"]\n')

    [spec] = collect_model_specs(config)

    assert not spec.tool_call
    assert not spec.reasoning


def test_named_chat_template_override_makes_capabilities_unknown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", template=TOOL_THINK_TEMPLATE)
    primary.with_suffix(".toml").write_text('extra_args = ["--chat-template", "chatml"]\n')

    [spec] = collect_model_specs(config)

    # Unknown template: generic tool calling still works, reasoning is not claimed.
    assert spec.tool_call
    assert not spec.reasoning


def test_no_jinja_disables_tools_and_reasoning(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m", template=TOOL_THINK_TEMPLATE)
    primary.with_suffix(".toml").write_text('extra_args = ["--no-jinja"]\n')

    [spec] = collect_model_specs(config)

    assert not spec.tool_call
    assert not spec.reasoning


def test_mmproj_file_marks_vision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = _model(config.models_dir, "m")
    (primary.parent / "mmproj-F16.gguf").write_bytes(b"")

    [spec] = collect_model_specs(config)

    assert spec.vision


def test_speech_to_text_models_are_skipped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "chat")
    asr = config.models_dir / "parakeet"
    asr.mkdir()
    (asr / "encoder-model.onnx").write_bytes(b"")

    assert [s.id for s in collect_model_specs(config)] == ["chat"]
    with pytest.raises(ValueError, match="speech-to-text"):
        collect_model_specs(config, ["parakeet"])


def test_text_to_speech_models_are_skipped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "chat")
    tts = config.models_dir / "kokoro"
    tts.mkdir()
    (tts / "model.tts").write_text("")

    assert [s.id for s in collect_model_specs(config)] == ["chat"]
    with pytest.raises(ValueError, match="is a text-to-speech model"):
        collect_model_specs(config, ["kokoro"])


def test_unknown_model_name_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_model_specs(_config(tmp_path), ["missing"])


def test_default_base_url_replaces_wildcard_host(tmp_path: Path) -> None:
    assert default_base_url(_config(tmp_path)) == "http://127.0.0.1:8080/v1"


# ── JSON files ───────────────────────────────────────────────────────────────


def test_load_jsonc_keeps_comment_markers_inside_strings(tmp_path: Path) -> None:
    path = tmp_path / "opencode.jsonc"
    path.write_text(
        '{\n  // a comment\n  "url": "http://x//y", /* block */\n  "list": [1, 2,],\n}\n'
    )

    assert load_json_file(path) == {"url": "http://x//y", "list": [1, 2]}
    assert has_comments(path)


def test_jsonc_comment_still_separates_tokens(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonc"
    path.write_text('{"x": 1/* c */2}')
    with pytest.raises(ValueError):
        load_json_file(path)


def test_jsonc_with_carriage_return_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonc"
    path.write_bytes(b'{\r// comment\r"x": 1\r}')
    assert load_json_file(path) == {"x": 1}


@pytest.mark.parametrize("text", ['{"x": 1}', '{\n  // c\n  "x": 1,\n}'])
def test_byte_order_mark_is_accepted(tmp_path: Path, text: str) -> None:
    path = tmp_path / "c.json"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode())
    assert load_json_file(path) == {"x": 1}


def test_plain_json_has_no_comments(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text('{"url": "http://x//y"}')
    assert not has_comments(path)


def test_write_configs_backs_up_previous_contents(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text('{"old": true}')
    fresh = tmp_path / "sub" / "new.json"

    backups = write_configs([(path, {"new": True}), (fresh, {"x": 1})])

    assert backups == [path.with_name("c.json.bak")]
    assert json.loads(backups[0].read_text()) == {"old": True}
    assert json.loads(path.read_text()) == {"new": True}
    assert json.loads(fresh.read_text()) == {"x": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.json", "c.json.bak", "sub"]


def test_write_configs_rolls_back_when_a_later_file_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models.json"
    settings = tmp_path / "settings.json"
    models.write_text('{"m": "old"}')
    settings.write_text('{"s": "old"}')

    real_replace = os.replace

    def failing_replace(src: Any, dst: Any) -> None:
        if Path(dst) == settings:
            raise PermissionError("read-only")
        real_replace(src, dst)

    monkeypatch.setattr("drove.integrations.os.replace", failing_replace)
    with pytest.raises(PermissionError):
        write_configs([(models, {"m": "new"}), (settings, {"s": "new"})])

    assert json.loads(models.read_text()) == {"m": "old"}
    assert json.loads(settings.read_text()) == {"s": "old"}
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_write_configs_removes_files_it_created_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = tmp_path / "new.json"
    other = tmp_path / "other.json"
    real_replace = os.replace

    def failing_replace(src: Any, dst: Any) -> None:
        if Path(dst) == other:
            raise PermissionError("read-only")
        real_replace(src, dst)

    monkeypatch.setattr("drove.integrations.os.replace", failing_replace)
    with pytest.raises(PermissionError):
        write_configs([(created, {"a": 1}), (other, {"b": 2})])

    assert not created.exists()


def test_write_configs_rolls_back_a_read_only_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models.json"
    settings = tmp_path / "settings.json"
    models.write_text('{"m": "old"}')
    models.chmod(0o400)
    settings.write_text('{"s": "old"}')
    real_copy = shutil.copy2

    def failing_copy(src: Any, dst: Any) -> Any:
        if Path(dst) == settings.with_name("settings.json.bak"):
            raise PermissionError("read-only backup")
        return real_copy(src, dst)

    monkeypatch.setattr("drove.integrations.shutil.copy2", failing_copy)
    with pytest.raises(PermissionError):
        write_configs([(models, {"m": "new"}), (settings, {"s": "new"})])

    assert json.loads(models.read_text()) == {"m": "old"}
    assert stat.S_IMODE(models.stat().st_mode) == 0o400
    assert json.loads(settings.read_text()) == {"s": "old"}


def test_write_configs_reports_a_rollback_it_could_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models.json"
    settings = tmp_path / "settings.json"
    models.write_text('{"m": "old"}')
    settings.write_text('{"s": "old"}')
    real_replace = os.replace
    calls = {"models": 0}

    def flaky_replace(src: Any, dst: Any) -> None:
        if Path(dst) == settings:
            raise PermissionError("settings read-only")
        if Path(dst) == models:
            calls["models"] += 1
            if calls["models"] > 1:  # the restore
                raise PermissionError("models restore blocked")
        real_replace(src, dst)

    monkeypatch.setattr("drove.integrations.os.replace", flaky_replace)
    with pytest.raises(IncompleteRollbackError) as excinfo:
        write_configs([(models, {"m": "new"}), (settings, {"s": "new"})])

    assert excinfo.value.paths == [models]
    assert json.loads(models.with_name("models.json.bak").read_text()) == {"m": "old"}
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_write_configs_keeps_a_private_config_private(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{}")
    path.chmod(0o600)

    write_configs([(path, {"apiKey": "secret"})])

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_configs_ignores_planted_files_at_staging_names(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    for name in (".c.json.drove-tmp", ".c.json.tmp", "c.json.bak"):
        (tmp_path / name).symlink_to(victim)
    path = tmp_path / "c.json"
    path.write_text("{}")

    write_configs([(path, {"x": 1})])

    assert victim.read_text() == "keep me"
    assert json.loads(path.read_text()) == {"x": 1}


def test_write_configs_updates_a_symlinked_config_in_place(tmp_path: Path) -> None:
    dotfiles = tmp_path / "dotfiles" / "opencode.json"
    dotfiles.parent.mkdir()
    dotfiles.write_text('{"old": true}')
    link = tmp_path / "opencode.json"
    link.symlink_to(dotfiles)

    write_configs([(link, {"new": True})])

    assert link.is_symlink()
    assert json.loads(dotfiles.read_text()) == {"new": True}
    assert json.loads(link.with_name("opencode.json.bak").read_text()) == {"old": True}


# ── opencode / Kilo Code ─────────────────────────────────────────────────────


def _files(path: Path, data: dict[str, Any] | None = None) -> ConfigFiles:
    """A ConfigFiles whose *path* starts out holding *data* (absent if None)."""
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return ConfigFiles()


def test_opencode_entry_has_full_model_setup(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    files = _files(path)

    OpenCode().apply(
        files,
        path,
        [_spec("vis", vision=True)],
        provider_id="drove",
        base_url="http://127.0.0.1:8080/v1",
        prune=False,
    )

    provider = files.get(path)["provider"]["drove"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert provider["models"]["vis"] == {
        "name": "vis (drove)",
        "tool_call": True,
        "reasoning": False,
        "attachment": True,
        "temperature": True,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "limit": {"context": 32768, "output": 8192},
    }


def test_opencode_merge_preserves_other_config_and_user_keys(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    files = _files(
        path,
        {
            "model": "anthropic/claude",
            "provider": {
                "other": {"npm": "x"},
                "drove": {
                    "options": {"apiKey": "k"},
                    "models": {
                        "a": {"options": {"top_k": 20}, "limit": {"input": 24576}},
                        "old": {},
                    },
                },
            },
        },
    )

    OpenCode().apply(files, path, [_spec("a")], provider_id="drove", base_url="u", prune=False)

    data = files.get(path)
    assert data["model"] == "anthropic/claude"
    assert data["provider"]["other"] == {"npm": "x"}
    drove = data["provider"]["drove"]
    assert drove["options"] == {"apiKey": "k", "baseURL": "u"}
    assert drove["models"]["a"]["options"] == {"top_k": 20}
    # Nested keys the user set by hand survive next to the generated ones.
    assert drove["models"]["a"]["limit"] == {"input": 24576, "context": 32768, "output": 8192}
    assert "old" in drove["models"]


def test_opencode_prune_drops_stale_models_and_their_selections(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    files = _files(
        path,
        {
            "model": "drove/gone",
            "small_model": "drove/a",
            "provider": {"drove": {"models": {"gone": {}, "a": {}}}},
        },
    )

    OpenCode().apply(files, path, [_spec("a")], provider_id="drove", base_url="u", prune=True)

    data = files.get(path)
    assert list(data["provider"]["drove"]["models"]) == ["a"]
    assert "model" not in data
    assert data["small_model"] == "drove/a"


def test_opencode_remove_models_and_provider(tmp_path: Path) -> None:
    agent = OpenCode()
    path = tmp_path / "opencode.json"
    files = _files(
        path,
        {
            "model": "drove/a",
            "small_model": "drove/b",
            "provider": {"drove": {"models": {"a": {}, "b": {}}}},
        },
    )

    agent.remove(files, path, ["a"], provider_id="drove")
    data = files.get(path)
    assert list(data["provider"]["drove"]["models"]) == ["b"]
    assert "model" not in data
    assert data["small_model"] == "drove/b"

    agent.remove(files, path, None, provider_id="drove")
    assert data["provider"] == {}
    assert "small_model" not in data


def test_opencode_remove_keeps_unrelated_selections(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    files = _files(path, {"model": "anthropic/claude", "provider": {"drove": {"models": {}}}})

    OpenCode().remove(files, path, None, provider_id="drove")

    assert files.get(path)["model"] == "anthropic/claude"


def test_opencode_default_model_is_provider_qualified(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    files = _files(path)
    OpenCode().set_default(files, path, "org/m", provider_id="drove")
    assert files.get(path)["model"] == "drove/org/m"


def test_opencode_prefers_an_existing_config_file(tmp_path: Path) -> None:
    directory = tmp_path / "xdg" / "opencode"
    assert OpenCode().config_path() == directory / "opencode.json"

    directory.mkdir(parents=True)
    (directory / "opencode.jsonc").write_text("{}")
    assert OpenCode().config_path() == directory / "opencode.jsonc"


def test_kilo_config_path_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert KiloCode().config_path() == tmp_path / "xdg" / "kilo" / "kilo.json"
    monkeypatch.setenv("KILO_CONFIG", str(tmp_path / "custom.json"))
    assert KiloCode().config_path() == tmp_path / "custom.json"


def test_kilo_finds_a_legacy_config_json(tmp_path: Path) -> None:
    directory = tmp_path / "xdg" / "kilo"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"provider": {"drove": {"models": {"a": {}}}}})
    )
    agent = KiloCode()
    path = agent.config_path()
    assert path == directory / "config.json"

    files = ConfigFiles()
    agent.remove(files, path, None, provider_id="drove")
    assert files.changed() == [(path, {"provider": {}})]


# ── pi ───────────────────────────────────────────────────────────────────────


def test_pi_entry_has_full_model_setup(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    files = _files(path)

    Pi().apply(
        files, path, [_spec("vis", vision=True)], provider_id="drove", base_url="u", prune=False
    )

    provider = files.get(path)["providers"]["drove"]
    assert provider["baseUrl"] == "u"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"]
    assert provider["models"] == [
        {
            "id": "vis",
            "name": "vis (drove)",
            "reasoning": False,
            "input": ["text", "image"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
    ]


def test_pi_merge_updates_by_id_and_keeps_user_keys(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    files = _files(
        path,
        {
            "providers": {
                "ollama": {"baseUrl": "x"},
                "drove": {
                    "apiKey": "mine",
                    "models": [{"id": "a", "thinkingLevelMap": {}, "cost": {"tiers": []}}],
                },
            }
        },
    )

    Pi().apply(
        files, path, [_spec("a"), _spec("b")], provider_id="drove", base_url="u", prune=False
    )

    data = files.get(path)
    drove = data["providers"]["drove"]
    assert data["providers"]["ollama"] == {"baseUrl": "x"}
    assert drove["apiKey"] == "mine"
    assert [m["id"] for m in drove["models"]] == ["a", "b"]
    assert drove["models"][0]["thinkingLevelMap"] == {}
    assert drove["models"][0]["cost"]["tiers"] == []
    assert drove["models"][0]["contextWindow"] == 32768


def test_pi_remove_models(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    files = _files(path, {"providers": {"drove": {"models": [{"id": "a"}, {"id": "b"}]}}})
    Pi().remove(files, path, ["a"], provider_id="drove")
    assert files.get(path)["providers"]["drove"]["models"] == [{"id": "b"}]


def test_pi_default_model_goes_to_settings_beside_models_json(tmp_path: Path) -> None:
    path = tmp_path / "elsewhere" / "models.json"
    settings = path.with_name("settings.json")
    files = _files(settings, {"theme": "dark"})

    Pi().set_default(files, path, "m", provider_id="drove")

    assert files.get(settings) == {"theme": "dark", "defaultProvider": "drove", "defaultModel": "m"}
    # The active pi agent directory is left alone.
    assert [p for p, _ in files.changed()] == [settings]


def test_pi_prune_and_remove_clear_defaults_and_enabled_models(tmp_path: Path) -> None:
    agent = Pi()
    path = tmp_path / "models.json"
    settings = path.with_name("settings.json")
    _files(
        settings,
        {
            "defaultProvider": "drove",
            "defaultModel": "gone",
            "enabledModels": ["drove/gone", "drove/a", "anthropic/*"],
        },
    )
    files = _files(path, {"providers": {"drove": {"models": [{"id": "gone"}, {"id": "a"}]}}})

    agent.apply(files, path, [_spec("a")], provider_id="drove", base_url="u", prune=True)

    assert files.get(settings) == {"enabledModels": ["drove/a", "anthropic/*"]}

    agent.remove(files, path, None, provider_id="drove")
    assert files.get(settings) == {"enabledModels": ["anthropic/*"]}


def test_pi_remove_keeps_a_default_on_another_provider(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    settings = path.with_name("settings.json")
    _files(settings, {"defaultProvider": "anthropic", "defaultModel": "a"})
    files = _files(path, {"providers": {"drove": {"models": [{"id": "a"}]}}})

    Pi().remove(files, path, ["a"], provider_id="drove")

    assert files.get(settings) == {"defaultProvider": "anthropic", "defaultModel": "a"}


def test_get_agent_rejects_unknown() -> None:
    assert get_agent("PI").key == "pi"
    with pytest.raises(ValueError, match="Supported agents"):
        get_agent("cursor")


def test_config_files_reports_only_changed_files(tmp_path: Path) -> None:
    kept = tmp_path / "kept.json"
    files = _files(kept, {"a": 1})
    files.get(kept)
    files.get(tmp_path / "missing.json")
    assert files.changed() == []

    files.get(kept)["a"] = 2
    assert files.changed() == [(kept, {"a": 2})]


# ── CLI ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    _model(models_dir, "coder", ctx=65536, template=TOOL_THINK_TEMPLATE)
    _model(models_dir, "small", ctx=8192)
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'models_dir = "{models_dir}"\nlisten_port = 9999\n')
    return cfg


def test_cli_writes_opencode_config(tmp_path: Path, cli_config: Path) -> None:
    result = CliRunner().invoke(
        app, ["-c", str(cli_config), "integrate", "opencode", "--default", "coder"]
    )

    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "xdg" / "opencode" / "opencode.json").read_text())
    provider = data["provider"]["drove"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
    assert set(provider["models"]) == {"coder", "small"}
    assert provider["models"]["coder"]["limit"]["context"] == 65536
    assert data["model"] == "drove/coder"


def test_cli_writes_pi_models_and_settings(tmp_path: Path, cli_config: Path) -> None:
    result = CliRunner().invoke(
        app, ["-c", str(cli_config), "integrate", "pi", "small", "-d", "small"]
    )

    assert result.exit_code == 0, result.output
    models = json.loads((tmp_path / "pi" / "models.json").read_text())
    assert [m["id"] for m in models["providers"]["drove"]["models"]] == ["small"]
    settings = json.loads((tmp_path / "pi" / "settings.json").read_text())
    assert settings["defaultModel"] == "small"


def test_cli_dry_run_writes_nothing(tmp_path: Path, cli_config: Path) -> None:
    result = CliRunner().invoke(app, ["-c", str(cli_config), "integrate", "kilo", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert '"@ai-sdk/openai-compatible"' in result.output
    assert not (tmp_path / "xdg" / "kilo").exists()


def test_cli_remove_and_backup(tmp_path: Path, cli_config: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["-c", str(cli_config), "integrate", "opencode"])
    result = runner.invoke(app, ["-c", str(cli_config), "integrate", "opencode", "--remove"])

    assert result.exit_code == 0, result.output
    path = tmp_path / "xdg" / "opencode" / "opencode.json"
    assert json.loads(path.read_text())["provider"] == {}
    assert "drove" in json.loads(path.with_name("opencode.json.bak").read_text())["provider"]


def test_cli_rejects_default_outside_selection(cli_config: Path) -> None:
    result = CliRunner().invoke(
        app, ["-c", str(cli_config), "integrate", "pi", "small", "-d", "coder"]
    )
    assert result.exit_code == 1


def test_cli_rejects_unknown_agent(cli_config: Path) -> None:
    result = CliRunner().invoke(app, ["-c", str(cli_config), "integrate", "cursor"])
    assert result.exit_code == 1


def test_cli_pi_file_override_keeps_active_settings(tmp_path: Path, cli_config: Path) -> None:
    other = tmp_path / "other" / "models.json"
    result = CliRunner().invoke(
        app, ["-c", str(cli_config), "integrate", "pi", "small", "-d", "small", "-f", str(other)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(other.with_name("settings.json").read_text())["defaultModel"] == "small"
    assert not (tmp_path / "pi").exists()


def test_cli_full_sync_with_no_models_left_prunes_stale_entries(
    tmp_path: Path, cli_config: Path
) -> None:
    runner = CliRunner()
    runner.invoke(app, ["-c", str(cli_config), "integrate", "opencode", "-d", "coder"])
    for name in ("coder", "small"):
        for f in (tmp_path / "models" / name).iterdir():
            f.unlink()
        (tmp_path / "models" / name).rmdir()

    result = runner.invoke(app, ["-c", str(cli_config), "integrate", "opencode"])

    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "xdg" / "opencode" / "opencode.json").read_text())
    assert data["provider"]["drove"]["models"] == {}
    assert "model" not in data


def test_cli_no_models_and_nothing_registered_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'models_dir = "{tmp_path / "models"}"\n')
    result = CliRunner().invoke(app, ["-c", str(cfg), "integrate", "opencode"])
    assert result.exit_code == 1
    assert not (tmp_path / "xdg").exists()


def test_cli_rerun_reports_up_to_date_without_backup(tmp_path: Path, cli_config: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["-c", str(cli_config), "integrate", "kilo"])
    (tmp_path / "xdg" / "kilo" / "kilo.json.bak").unlink(missing_ok=True)

    result = runner.invoke(app, ["-c", str(cli_config), "integrate", "kilo"])

    assert result.exit_code == 0, result.output
    assert "already up to date" in result.output
    assert not (tmp_path / "xdg" / "kilo" / "kilo.json.bak").exists()


def test_cli_reports_unparseable_config(tmp_path: Path, cli_config: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("[1, 2]")
    result = CliRunner().invoke(
        app, ["-c", str(cli_config), "integrate", "opencode", "-f", str(path)]
    )
    assert result.exit_code == 1
    assert path.read_text() == "[1, 2]"


def test_cli_feedback_reports_capabilities_written_paths_and_real_backups(
    tmp_path: Path, cli_config: Path
) -> None:
    runner = CliRunner()
    first = runner.invoke(app, ["-c", str(cli_config), "integrate", "pi", "-d", "coder"])
    assert first.exit_code == 0, first.output
    for value in ("MODEL", "CONTEXT", "OUTPUT", "VISION", "TOOLS", "REASONING", "coder", "small"):
        assert value in first.stdout
    assert "65,536" in first.stdout and "8,192" in first.stdout
    assert "yes" in first.stdout and "no" in first.stdout
    assert "Written" in first.stdout and "Success:" in first.stdout and "Next:" in first.stdout
    assert "Backup" not in first.stdout
    unchanged = runner.invoke(app, ["-c", str(cli_config), "integrate", "pi", "-d", "coder"])
    assert unchanged.exit_code == 0 and "already up to date" in unchanged.stdout
    assert "Backup" not in unchanged.stdout and "Written" not in unchanged.stdout
    removed = runner.invoke(app, ["-c", str(cli_config), "integrate", "pi", "--remove"])
    assert removed.exit_code == 0 and "Removed" in removed.stdout and "Backup" in removed.stdout
    assert (tmp_path / "pi" / "models.json.bak").exists()
    assert (tmp_path / "pi" / "settings.json.bak").exists()


@pytest.mark.parametrize("args", [["codr"], ["--default", "codr"]])
def test_cli_feedback_unknown_integration_model_suggests_match(
    tmp_path: Path, cli_config: Path, args: list[str]
) -> None:
    result = CliRunner().invoke(app, ["-c", str(cli_config), "integrate", "opencode", *args])
    assert result.exit_code == 1
    assert "Error:" in result.stderr and "Did you mean 'coder'" in result.stderr
    assert "drove models list" in result.stderr
    assert not (tmp_path / "xdg" / "opencode").exists()


def test_cli_feedback_dry_run_exact_legacy_transcript(tmp_path: Path, cli_config: Path) -> None:
    from drove.integrations import dump_json

    path = tmp_path / "xdg" / "opencode" / "opencode.json"
    conf = Config(models_dir=tmp_path / "models", listen_port=9999)
    specs = collect_model_specs(conf)
    files = ConfigFiles()
    OpenCode().apply(
        files, path, specs, base_url=default_base_url(conf), provider_id="drove", prune=True
    )
    preamble = ""
    for spec in specs:
        caps = [
            name
            for name, enabled in (
                ("vision", spec.vision),
                ("tools", spec.tool_call),
                ("reasoning", spec.reasoning),
            )
            if enabled
        ]
        preamble += (
            f"  {spec.id}: context {spec.context_window:,}, output {spec.max_output:,}"
            + (f", {', '.join(caps)}" if caps else "")
            + "\n"
        )
    expected = preamble + f"\n# {path}\n" + dump_json(files.get(path))
    result = CliRunner().invoke(
        app, ["-c", str(cli_config), "integrate", "opencode", "--dry-run"], color=True
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == expected and result.stderr == ""
    assert not path.exists() and not path.with_suffix(".json.bak").exists()


@pytest.mark.parametrize("incomplete", [False, True])
def test_cli_feedback_write_failure_never_claims_registration(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, incomplete: bool
) -> None:
    import importlib

    module = importlib.import_module("drove.cli.integrate")

    def fail(changed):
        if incomplete:
            raise IncompleteRollbackError("rollback failed")
        raise OSError("write failed")

    monkeypatch.setattr(module, "write_configs", fail)
    result = CliRunner().invoke(app, ["-c", str(cli_config), "integrate", "opencode"])
    assert result.exit_code == 1 and "Error:" in result.stderr
    assert "Success:" not in result.output and "MODEL" not in result.stdout
    assert (
        ("rollback failed" in result.stderr)
        if incomplete
        else ("nothing was changed" in result.stderr)
    )


def test_cli_feedback_pruning_and_unchanged_dry_run(tmp_path: Path, cli_config: Path) -> None:
    runner = CliRunner()
    prefix = ["-c", str(cli_config), "integrate", "opencode"]
    assert runner.invoke(app, prefix).exit_code == 0
    for primary in (tmp_path / "models").rglob("*.gguf"):
        primary.unlink()
    dry = runner.invoke(app, [*prefix, "--dry-run"])
    assert dry.exit_code == 0 and dry.stdout.startswith(
        "No chat models left; removing 2 stale model(s).\n"
    )
    assert "Success:" not in dry.stdout
    result = runner.invoke(app, prefix)
    assert result.exit_code == 0 and "Removed 2 stale model(s)" in result.stdout
    assert "Backup" in result.stdout
    # Explicit removal also removes the now-empty provider container.
    assert runner.invoke(app, [*prefix, "--remove"]).exit_code == 0
    unchanged = runner.invoke(app, [*prefix, "--remove", "--dry-run"])
    path = tmp_path / "xdg" / "opencode" / "opencode.json"
    assert unchanged.stdout == f"opencode is already up to date: {path}\n"


def test_unservable_and_incomplete_downloads_are_never_registered(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _model(config.models_dir, "coder", ctx=4096, template=TOOL_THINK_TEMPLATE)
    models = config.models_dir
    (models / "org" / "partial" / ".cache" / "huggingface").mkdir(parents=True)
    (models / "org" / "partial" / ".cache" / "huggingface" / "x.incomplete").write_bytes(b"")
    (models / "raw").mkdir()
    (models / "raw" / "model.safetensors").write_bytes(b"weights")
    (models / "split").mkdir()
    (models / "split" / "split-00001-of-00002.gguf").write_bytes(b"GGUF")

    assert [spec.id for spec in collect_model_specs(config)] == ["coder"]
    for name in ("org/partial", "raw", "split"):
        with pytest.raises(FileNotFoundError):
            collect_model_specs(config, [name])
