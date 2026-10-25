"""Exercise real Textual keyboard handling, including focused search input."""

import pytest
from textual.widgets import Input, OptionList

from drove.chat_models import ChatModel
from drove.model_picker import ModelPicker, ModelPickerApp, model_label


async def test_arrows_select_loaded_default_then_next() -> None:
    app = ModelPickerApp([ChatModel("a"), ChatModel("b", loaded=True), ChatModel("c")])
    async with app.run_test() as pilot:
        assert app.screen.query_one(OptionList).highlighted == 1
        assert isinstance(app.focused, Input)
        await pilot.press("down", "enter")
    assert app.return_value == "c"


async def test_filter_no_match_and_restore_does_not_select_stale_row() -> None:
    app = ModelPickerApp([ChatModel("Alpha"), ChatModel("beta")])
    async with app.run_test() as pilot:
        await pilot.press("z", "enter")
        assert isinstance(app.screen, ModelPicker)
        assert app.screen.query_one(OptionList).option_count == 0
        await pilot.press("backspace", "B", "E")
        assert app.screen.query_one(OptionList).option_count == 1
        await pilot.press("enter")
    assert app.return_value == "beta"


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_single_row_is_visible_and_can_cancel(key: str) -> None:
    app = ModelPickerApp([ChatModel("only")])
    async with app.run_test() as pilot:
        assert app.screen.query_one(OptionList).option_count == 1
        await pilot.press(key)
    assert app.return_value is None


async def test_scroll_and_current_preselection() -> None:
    app = ModelPickerApp([ChatModel(f"model-{i}") for i in range(40)])
    async with app.run_test(size=(55, 15)) as pilot:
        await pilot.press(*(["down"] * 30))
        options = app.screen.query_one(OptionList)
        assert options.highlighted == 30
        assert options.scroll_y > 0
        await pilot.press("enter")
    assert app.return_value == "model-30"


def test_labels_render_literal_names_and_unknown_values() -> None:
    label = model_label(ChatModel("[red]literal[/red]"))
    assert label.plain.startswith("[red]literal[/red]")
    assert not label.spans
    assert "size unknown" in label.plain
    assert "load state unknown" in label.plain
    assert "capabilities unknown" in label.plain
    rich = model_label(ChatModel("a", 1000, True, ("text", "vision"))).plain
    assert "1.0 kB" in rich and "loaded" in rich and "vision" in rich
