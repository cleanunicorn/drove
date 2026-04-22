
import pytest
from textual.app import App, ComposeResult
from vllama.tui import ChatInput

class ChatInputApp(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = False

    def compose(self) -> ComposeResult:
        yield ChatInput(id="input")

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        self.submitted = True

@pytest.mark.asyncio
async def test_chat_input_typing():
    app = ChatInputApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "b", "c")
        input_widget = app.query_one(ChatInput)
        assert input_widget.text == "abc"

@pytest.mark.asyncio
async def test_chat_input_shift_enter():
    app = ChatInputApp()
    async with app.run_test() as pilot:
        input_widget = app.query_one(ChatInput)
        await pilot.press("a", "shift+enter", "b")
        assert input_widget.text == "a\nb"

@pytest.mark.asyncio
async def test_chat_input_enter_submits():
    app = ChatInputApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "enter")
        assert app.submitted == True
        # Verify no newline was inserted by enter
        input_widget = app.query_one(ChatInput)
        assert input_widget.text == "a"

@pytest.mark.asyncio
async def test_chat_input_arrows():
    app = ChatInputApp()
    async with app.run_test() as pilot:
        input_widget = app.query_one(ChatInput)
        await pilot.press("a", "b", "left", "c")
        assert input_widget.text == "acb"

@pytest.mark.asyncio
async def test_chat_input_backspace():
    app = ChatInputApp()
    async with app.run_test() as pilot:
        input_widget = app.query_one(ChatInput)
        await pilot.press("a", "b", "backspace")
        assert input_widget.text == "a"
