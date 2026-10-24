# Troubleshooting

[Docs](./README.md) / Troubleshooting

Start with the proxy, then the model name, then the backend. These checks do
not need a GPU or a loaded model:

```bash
drove --version
drove config
drove models list
curl --fail http://127.0.0.1:8080/health
```

## The shell cannot find drove

Run `uv tool update-shell` and open a new terminal. Check `uv tool dir --bin`
to locate the installed executable. If installation failed because of the
Python version, reinstall with the [Python 3.14 command](./getting-started.md#1-install).

## Connection refused

Start the server in another terminal with `drove serve --host 127.0.0.1`.
Check `drove config listen_port` and any `--port` override. Clients should use
`127.0.0.1` locally or the server's reachable address remotely, never `0.0.0.0`.
For a user service, inspect `make service-status` and `make service-logs` from a
checkout. See [Running drove](./operations.md).

## HTTP 404: model not found

Copy the exact local name from `drove models list` into the request's `model`
field. A quantization can be part of that name. Check that the server and CLI
use the same `--config` / `DROVE_CONFIG` and `models_dir`.
For predictable examples, download with an explicit name:

```bash
drove models download unsloth/Qwen3-8B-GGUF:Q4_K_M --name qwen
```

Then use `"model":"qwen"`. See [Model naming](./models.md#download-and-name-a-model).

## HTTP 503: backend startup failed

Run `drove -v serve --host 127.0.0.1` in the foreground after stopping any
existing proxy on that port, or inspect your service logs. Check the relevant
backend:

| Backend | Check |
| --- | --- |
| Text / vision | `llama-server` is on the server's `PATH`, its flags are supported, weights fit in memory, and the vision projector is configured |
| Speech-to-text | drove was installed with `[asr]`, weights are ONNX exports, and `asr_model` matches the export |
| Text-to-speech | uv and required system dependencies are on the server's `PATH`; first-use environment and weight downloads can finish |

TTS first startup can take minutes. Increase its health wait before retrying:

```bash
drove config startup_timeout_seconds 1800
```

See [ASR setup](./speech-to-text.md#install) or
[TTS setup](./text-to-speech.md#quick-start) for backend-specific dependencies.

## A request times out or returns HTTP 502

Separate model startup from inference. `startup_timeout_seconds` controls
backend readiness. `speech_timeout_seconds` controls reading buffered TTS
responses and defaults to 600 seconds:

```bash
drove config speech_timeout_seconds 3600
```

Increase the client's timeout too. Other backend endpoints use a 300-second
proxy read timeout. For long CPU speech jobs, try a shorter input or a faster
device. Check backend logs if the process exited during a request.

## A configuration change has no effect

- Run `drove config` to check the selected file and effective global values.
  `DROVE_*` environment variables override TOML values.
- Put root options before the command: `drove --config /path/to/config.toml config`.
- Run `drove models config qwen` to see each model setting and its source.
  The key is `ctx_size`, not `context_size`; unknown TOML model keys are ignored.
- Restart the proxy for listen address/port changes. Model arguments take
  effect on the next backend start. See [Reload behavior](./configuration.md#reload-behavior).

## Audio formats or voices fail

Without `ffmpeg`, upload WAV for transcription and use WAV for speech output.
Check engine-specific voice support and dependencies in
[Speech-to-text](./speech-to-text.md) and [Text-to-speech](./text-to-speech.md).
Speech output is buffered; silence while synthesis runs is expected.

## Collect useful diagnostics

Include the failed command, status and response body, `drove --version`,
`drove server status -V`, and relevant server logs in a
[bug report](../.github/ISSUE_TEMPLATE/bug_report.md). Remove private prompts,
local paths, or credentials before sharing logs. Request logging can help
reproduce a problem; see [Inspect request logs](./operations.md#inspect-request-logs).

[Docs](./README.md) · [Running drove](./operations.md) · [Configuration](./configuration.md)
