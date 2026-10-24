# drove documentation

Run text, vision, and speech models behind one OpenAI-compatible endpoint.
drove starts each backend when a request needs it and releases resources after
it goes idle.

**New here?** [Install drove and send your first request](./getting-started.md).
**Something failed?** [Find a fix in troubleshooting](./troubleshooting.md).

## Choose a task

| Goal | Guide | What you will do |
| --- | --- | --- |
| Serve a first model | [Getting started](./getting-started.md) | Install, download a named model, start the proxy, verify a response |
| Manage local models | [Managing models](./models.md) | Choose quantizations, inspect names, set defaults, remove models |
| Chat or send images | [Text and vision](./text-and-vision.md) | Use an SDK, stream text, send images, connect coding agents |
| Transcribe audio | [Speech-to-text](./speech-to-text.md) | Install ASR support, download ONNX weights, upload audio |
| Generate speech | [Text-to-speech](./text-to-speech.md) | Pick an engine and voice, synthesize audio, configure voice cloning |
| Keep drove running | [Running drove](./operations.md) | Set the bind address, install a user service, watch status and request logs |
| Resolve an error | [Troubleshooting](./troubleshooting.md) | Check names, dependencies, startup failures, timeouts, and config |

## Look up a detail

| Reference | Covers |
| --- | --- |
| [CLI](./cli.md) | Commands, flags, output conventions, agent integration |
| [Configuration](./configuration.md) | Global defaults, precedence, timeouts, resource limits, model parameters |
| [HTTP API](./api.md) | Base URL, model IDs, routes, request formats, errors |

## Understand and contribute

- [Architecture](./architecture.md) — request flow, modules, subprocess lifecycle.
- [Contributing](./contributing.md) — development checks and the pull request workflow.
- [Release process](./deploy.md) — Conventional Commits, release automation, maintainer checklist.
- [Design proposals](./proposals/README.md) — proposed changes, separate from current behavior.
- [Changelog](../CHANGELOG.md) — released changes and pending updates.

## About these docs

These Markdown files are the canonical documentation and can be read directly
on GitHub or in a checkout. The hosted site at `drove.dev/docs` is planned.
Examples use `127.0.0.1:8080` for clients. New text examples use explicit local
names such as `qwen`; use the exact name from `drove models list` for your own
models.

[Project overview](../README.md) · [Start the tutorial →](./getting-started.md)
