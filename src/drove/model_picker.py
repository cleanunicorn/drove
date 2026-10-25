"""A searchable, themed model picker shared by startup and chat."""

from __future__ import annotations

from rich.filesize import decimal
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from drove.chat_models import ChatModel


def model_label(model: ChatModel) -> Text:
    loaded = "loaded" if model.loaded else "not loaded"
    if model.loaded is None:
        loaded = "load state unknown"
    size = decimal(model.size_bytes) if model.size_bytes is not None else "size unknown"
    capabilities = ", ".join(model.capabilities) if model.capabilities else "capabilities unknown"
    return Text(f"{model.id}\n  {loaded} · {size} · {capabilities}")


class ModelPicker(ModalScreen[str | None]):
    DEFAULT_CSS = """
    ModelPicker { align: center middle; }
    ModelPicker #model-dialog {
        width: 90%; max-width: 100; height: 90%;
        border: round $accent; background: $surface; padding: 0 1;
    }
    ModelPicker #model-title { text-style: bold; }
    ModelPicker #model-options { height: 1fr; border: none; }
    ModelPicker #model-hint { color: $text-muted; }
    """
    BINDINGS = [
        Binding("escape,ctrl+c", "cancel", "Cancel", priority=True),
        Binding("up", "previous", "Previous", priority=True),
        Binding("down", "next", "Next", priority=True),
        Binding("enter", "choose", "Choose", priority=True),
    ]

    def __init__(self, models: list[ChatModel], current: str | None = None) -> None:
        super().__init__()
        self._models = models
        self._visible: list[ChatModel] = []
        self._preferred = current or next((m.id for m in models if m.loaded), None)
        self._switching = current is not None

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            title = (
                "Switch model — starts a new conversation" if self._switching else "Choose a model"
            )
            yield Label(title, id="model-title")
            yield Input(placeholder="Type to filter models", id="model-search")
            yield OptionList(id="model-options")
            yield Label("↑/↓ choose · Enter select · Esc/Ctrl+C cancel", id="model-hint")

    def on_mount(self) -> None:
        self._filter("")
        self.query_one(Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter(event.value)

    def _filter(self, query: str) -> None:
        options = self.query_one(OptionList)
        highlighted = options.highlighted
        selected = self._preferred
        if highlighted is not None and highlighted < len(self._visible):
            selected = self._visible[highlighted].id
        self._visible = [m for m in self._models if query.casefold() in m.id.casefold()]
        options.clear_options()
        options.add_options(Option(model_label(m)) for m in self._visible)
        if self._visible:
            options.highlighted = next(
                (i for i, model in enumerate(self._visible) if model.id == selected), 0
            )
        self.query_one("#model-hint", Label).update(
            "↑/↓ choose · Enter select · Esc/Ctrl+C cancel"
            if self._visible
            else "No matching models · Esc/Ctrl+C cancel"
        )

    def action_previous(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_next(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_choose(self) -> None:
        index = self.query_one(OptionList).highlighted
        if index is not None and 0 <= index < len(self._visible):
            self.dismiss(self._visible[index].id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_choose()

    def action_cancel(self) -> None:
        self.dismiss(None)


class ModelPickerApp(App[str | None]):
    def __init__(self, models: list[ChatModel], theme: str = "textual-dark") -> None:
        super().__init__()
        self._models = models
        self.theme = theme

    def on_mount(self) -> None:
        self.push_screen(ModelPicker(self._models), self.exit)
