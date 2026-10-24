# HTTP API

[Docs](./README.md) / HTTP API

The client base URL is `http://127.0.0.1:8080/v1` for a local proxy on its
default port. SDKs expect the `/v1` suffix; health and status routes live at the
server root. drove does not authenticate API keys. SDKs that require a key can
use any non-empty string, such as `drove`.

## Routes

| Method | Path | Purpose | Request |
| --- | --- | --- | --- |
| `GET` | `/health` | Check the proxy is responding | No body |
| `GET` | `/status` | Read server, process, model, request, token, and audio metrics | No body |
| `GET` | `/v1/models` | List locally installed model IDs | No body |
| `POST` | `/v1/chat/completions` | Generate text, use tools, or send images via llama-server | JSON with `model` and `messages` |
| `POST` | `/v1/audio/transcriptions` | Transcribe an audio file | Multipart form with `model` and `file` |
| `POST` | `/v1/audio/speech` | Synthesize audio with a TTS engine | JSON with `model`, `input`, and optional speech fields |

Health, status, and model listing are handled by drove without starting a
backend. Other routes are proxied to the selected backend; support depends on
that backend's API. The table lists the primary workflows, not every possible
llama-server route.

## Model IDs and lifecycle

Use the exact name from `drove models list` or `GET /v1/models`:

```bash
curl --fail http://127.0.0.1:8080/v1/models
```

A model downloaded with `--name qwen` appears with `"id":"qwen"`. The inventory
includes unloaded models. A request naming one starts it if necessary, waits
for backend health, and then forwards the request. The first request may take
longer than later requests. See [Managing models](./models.md) for naming and
[Architecture](./architecture.md) for the lifecycle.

Always send `model` explicitly. When it is absent, drove can fall back to a
loaded model; if none is loaded, the request fails with HTTP 400.

## Request examples

- [First chat request with curl and Python](./getting-started.md#5-send-your-first-request).
- [Streaming text and image input](./text-and-vision.md).
- [Transcription with multipart uploads](./speech-to-text.md#transcribe).
- [Speech fields, voices, formats, and SDK examples](./text-to-speech.md#api-contract).

Chat streaming is passed through from llama-server. TTS workers finish
synthesis before returning audio; a streaming SDK response does not imply
incremental synthesis or speech SSE support.

## Timeouts

| Stage | Setting | Default |
| --- | --- | --- |
| Waiting for a backend to become healthy | `startup_timeout_seconds` | 300 seconds |
| Reading a speech synthesis response | `speech_timeout_seconds` | 600 seconds |
| Reading other backend responses | Proxy HTTP client timeout | 300 seconds |

The speech timeout is a read timeout. Configure your client to allow enough
time for both a cold start and inference. See
[Speech request timeout](./configuration.md#speech-request-timeout).

## Errors

| Status | Typical cause | Next check |
| --- | --- | --- |
| `400` | No model specified and none loaded, or invalid backend input | Send a model ID and validate the request fields |
| `404` | Requested model is not installed | Compare with `/v1/models` |
| `502` | Proxy could not complete the backend connection or request | Inspect server logs and timeouts |
| `503` | Backend failed to start or become healthy | Check dependencies, memory, and startup timeout |

Backend error responses are also forwarded, so inspect the response body along
with the status. See [Troubleshooting](./troubleshooting.md) for concrete steps.

[Docs](./README.md) · [CLI](./cli.md) · [Configuration](./configuration.md)
