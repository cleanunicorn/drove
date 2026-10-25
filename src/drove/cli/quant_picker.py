"""Read-only quantization previews and the interactive download picker."""

from __future__ import annotations

import base64
import hashlib
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import psutil
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from drove.downloader import (
    DownloadPlan,
    FileStatus,
    available_onnx_quants,
    available_quants,
    filter_by_quant,
    filter_onnx_quant,
    infer_local_name,
    is_sharded,
)

ALL_VARIANTS = "__all_variants__"


def _files(plan: DownloadPlan) -> dict[str, int]:
    return plan.files | plan.mmproj_files | plan.extra_files


def _size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    return f"{size:,} bytes"


def _has_partial_cache(destination: Path, repo_file: str) -> bool:
    """Recognize this file's Hub 1.x partial artifacts without creating folders.

    Hub hashes the metadata basename, then adds the ETag and `.incomplete`.
    Its get_local_download_paths helper creates directories, so preview must
    not call it. Unknown cache layouts simply don't produce a partial badge.
    """
    metadata = Path(repo_file + ".metadata")
    cache = destination / ".cache" / "huggingface" / "download" / metadata.parent
    prefix = base64.urlsafe_b64encode(hashlib.sha1(metadata.name.encode()).digest()).decode()
    try:
        return any(
            path.name.startswith(prefix + ".")
            and path.name.endswith(".incomplete")
            and path.is_file()
            and path.stat().st_size > 0
            for path in cache.iterdir()
        )
    except OSError:
        return False


def _local_status(plan: DownloadPlan, models_dir: Path) -> str:
    try:
        statuses = plan.check_local_files(models_dir)
        sizes = _files(plan)
        if statuses and all(
            state == FileStatus.COMPLETE and sizes[name] > 0
            for name, (state, _) in statuses.items()
        ):
            return "fully downloaded"
        if any(state != FileStatus.MISSING for state, _ in statuses.values()):
            return "partly downloaded"  # Unknown-size existing files are unverified.
        if any(_has_partial_cache(plan.destination(models_dir), name) for name in sizes):
            return "partly downloaded"
    except OSError:
        return "local status unavailable"
    return ""


@dataclass(frozen=True)
class QuantCandidate:
    tag: str
    plan: DownloadPlan
    quant: str | None
    local_status: str

    @property
    def size_label(self) -> str:
        sizes = _files(self.plan).values()
        known = sum(size for size in sizes if size > 0)
        if any(size <= 0 for size in sizes):
            return f"at least {_size(known)} (total unknown)" if known else "size unknown"
        return _size(known)

    @property
    def weight_bytes(self) -> int | None:
        sizes = (self.plan.files | self.plan.mmproj_files).values()
        return sum(sizes) if sizes and all(size > 0 for size in sizes) else None

    def memory_note(self, available: int | None) -> str:
        if self.weight_bytes is None:
            return "memory estimate unavailable"
        if available is None:
            return "available RAM unknown"
        if self.weight_bytes > available * 4 // 5:
            return "likely too large for available RAM"
        return "within estimated RAM budget"

    def label(self, available: int | None) -> Text:
        title = "Download every variant" if self.tag == ALL_VARIANTS else self.tag
        parts = [title, self.size_label]
        if self.tag != ALL_VARIANTS:
            parts.append(self.memory_note(available))
        if self.local_status:
            parts.append(self.local_status)
        return Text(" · ".join(parts))


def build_candidates(
    plan: DownloadPlan, name_override: str | None, models_dir: Path
) -> list[QuantCandidate]:
    """Preview equivalent plans; don't mutate the plan or touch the filesystem."""
    if plan.is_tts:
        return []
    variants = (
        available_onnx_quants(plan.onnx_files) if plan.is_asr else available_quants(plan.files)
    )
    if len(variants) < 2:
        return []

    candidates: list[QuantCandidate] = []
    for tag in [*variants, ALL_VARIANTS]:
        candidate = copy(plan)
        quant: str | None = tag
        if tag == ALL_VARIANTS:
            candidate.files = dict(plan.onnx_files if plan.is_asr else plan.files)
            quant = None
        elif plan.is_asr and tag == "default":
            candidate.files = dict(plan.files)
            quant = None
        else:
            candidate.files = (
                filter_onnx_quant(plan.onnx_files, tag)
                if plan.is_asr
                else filter_by_quant(plan.files, tag)
            )
            candidate.sharded = is_sharded(list(candidate.files))
            if name_override is None:
                candidate.local_name = infer_local_name(plan.repo_id, list(candidate.files), tag)
        status = _local_status(candidate, models_dir)
        if candidate.local_name != plan.local_name:
            base_candidate = copy(candidate)
            base_candidate.local_name = plan.local_name
            base_status = _local_status(base_candidate, models_dir)
            if base_status:
                location = f"{base_status} at {plan.destination(models_dir)} (separate location)"
                status = f"{status}; {location}" if status else location
        candidates.append(QuantCandidate(tag, candidate, quant, status))
    return candidates


def available_memory() -> int | None:
    try:
        return max(0, int(psutil.virtual_memory().available))
    except OSError, AttributeError, RuntimeError, NotImplementedError:
        return None


def recommended_tag(candidates: list[QuantCandidate], available: int | None) -> str:
    variants = [candidate for candidate in candidates if candidate.tag != ALL_VARIANTS]
    known = [candidate for candidate in variants if candidate.weight_bytes is not None]
    fitting = [
        candidate
        for candidate in known
        if available is None or (candidate.weight_bytes or 0) <= available * 4 // 5
    ]
    for candidate in fitting:
        if candidate.tag in {"Q4_K_M", "int8"}:
            return candidate.tag
    pool = fitting or known or variants
    return min(pool, key=lambda candidate: candidate.weight_bytes or 0).tag


class QuantPicker(App[str | None]):
    """Choose a variant; None always means cancel, never all."""

    CSS = """
    #guidance { height: auto; margin: 1 2; }
    #variants { height: 1fr; margin: 0 2 1 2; border: round $accent; }
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self, candidates: list[QuantCandidate], available: int | None, theme: str = "textual-dark"
    ) -> None:
        super().__init__()
        self.candidates = candidates
        self.available = available
        self.theme = theme
        self.title = "drove download"

    def compose(self) -> ComposeResult:
        yield Header()
        memory = _size(self.available) if self.available is not None else "unknown"
        yield Static(
            Text(
                f"Choose a quantization · available RAM: {memory}\n"
                "RAM estimates leave 20% headroom; context, KV cache and GPU placement "
                "affect actual needs. Use ↑/↓ and Enter; Esc cancels."
            ),
            id="guidance",
        )
        options: list[Option | None] = []
        for candidate in self.candidates:
            if candidate.tag == ALL_VARIANTS:
                options.append(None)
            options.append(Option(candidate.label(self.available), id=candidate.tag))
        yield OptionList(*options, id="variants")
        yield Footer()

    def on_mount(self) -> None:
        options = self.query_one("#variants", OptionList)
        options.highlighted = options.get_option_index(
            recommended_tag(self.candidates, self.available)
        )
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)

    def action_cancel(self) -> None:
        self.exit(None)
