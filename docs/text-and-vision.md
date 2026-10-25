# Text and vision

[Docs](./README.md) / Text and vision

Use a GGUF model through an OpenAI client, terminal chat, or a coding agent.
These examples assume [Getting started](./getting-started.md) is complete,
`llama-server` is on your `PATH`, and drove is running on `127.0.0.1:8080`.

## Generate and stream text

Install `openai` in your client project's environment. Use the exact local model
name from `drove models list`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="drove")
stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain lazy loading in one sentence."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

For a single JSON response, omit `stream=True` and read
`response.choices[0].message.content`, as in the tutorial. The SDK requires a
non-empty API key; drove does not authenticate it. Chat-completions parameters,
tool calls, and streaming are forwarded to llama-server. Available capabilities
depend on your model and llama.cpp version.

## Send an image

Download a multimodal model and confirm it has the `vision` capability:

```bash
drove models download unsloth/gemma-3-12b-it-GGUF:Q4_K_M --name gemma-vision
drove models list
drove models info gemma-vision
```

When the repository provides an `mmproj` projector, drove downloads it and
records it in the sidecar. The backend uses that projector when it starts.
Use standard OpenAI image content parts; this example sends a local JPEG as a
data URL:

```python
import base64
from pathlib import Path

from openai import OpenAI

image = base64.b64encode(Path("photo.jpg").read_bytes()).decode("ascii")
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="drove")
response = client.chat.completions.create(
    model="gemma-vision",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
        ],
    }],
)
print(response.choices[0].message.content)
```

Use the MIME type that matches your image. If vision is not detected, check the
model's `mmproj` setting and its [effective configuration](./models.md#set-runtime-parameters).

## Terminal chat

```bash
drove chat                               # use the default model, or pick one
drove chat qwen                          # use a specific local model
drove chat --resume                      # resume the latest saved session
drove chat -s "You are a terse assistant."
```

Without a model argument or configured default, the picker shows loaded state, disk size and inferred
capabilities. Type to filter names, move with ↑/↓, and press Enter to select.
Esc or Ctrl+C cancels. A loaded chat model is preselected when available;
known speech models and models with unreadable configuration are excluded.
Pass a model name explicitly if stdin or stdout is redirected.

Inside chat, `/model` opens the same picker with the current model selected.
Choosing another saves the current conversation and starts a separate one,
preserving its system prompt. Wait for pending replies to finish before switching.
Cancel or select the current model to keep the conversation unchanged. Saved
conversations can be reopened through `/sessions` when using their model.
Type `/help` for all commands, including `/theme`.



Set `default_model` to skip the picker when no model argument is supplied.
An explicit model always wins, including with a remote endpoint. See
[Default model](./configuration.md#default-model) for setup and clearing.

Type `/help` inside chat for commands such as `/sessions` and `/theme`.
The TUI can also use a remote OpenAI-compatible endpoint:

```bash
drove chat -e https://api.openai.com/v1 -k "$OPENAI_API_KEY"
```

Remote chat uses that provider's models and credentials; it does not require a
local drove server. Older servers and third-party endpoints may not supply model
metadata; their entries remain selectable with unknown labels. drove cannot
verify chat capability from a name alone. See [Chat model selection](./cli.md#chat-model-selection)
for the complete CLI behavior.

## Connect a coding agent

Preview a provider configuration, then write it:

```bash
drove integrate opencode --dry-run
drove integrate opencode
```

Supported agents are `pi`, `opencode`, and `kilo`. To register just one model
and make it the default:

```bash
drove integrate pi qwen -d qwen
```

drove infers usable context, output limits, vision, tools, and reasoning from
model settings and GGUF metadata. Previous files are saved as `<file>.bak`.
Rerun integration after downloading, deleting, or reconfiguring models. See
[the CLI reference](./cli.md#coding-agent-integration) for file paths, merge
behavior, and removal options.

[Docs](./README.md) · [HTTP API](./api.md) · [Troubleshooting](./troubleshooting.md)
