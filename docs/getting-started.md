# Getting started

## Install

### Option 1: make install

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
make install
```

### Option 2: uv tool install

```bash
uv tool install 'drove[asr] @ git+https://github.com/cleanunicorn/drove'
```

## Initialize local config

```bash
drove init
```

This creates a config file at `~/.config/drove/config.toml` with defaults.

## Download a model

```bash
drove models download unsloth/Qwen3-8B-GGUF
```

## Run the server

```bash
drove serve
```

The OpenAI-compatible API is then available at:

- `http://127.0.0.1:8080/v1/models`
- `http://127.0.0.1:8080/v1/chat/completions`

## Open a local TUI chat

```bash
drove chat
```

## Use with OpenAI SDKs

Set your base URL to drove and any non-empty API key.

Python example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="drove")

resp = client.chat.completions.create(
    model="unsloth/Qwen3-8B-GGUF",
    messages=[{"role": "user", "content": "Write a haiku about inference latency."}],
)

print(resp.choices[0].message.content)
```
