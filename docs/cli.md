# CLI reference

## Core commands

```bash
drove --help
drove init
drove serve
drove chat
```

## Config management

```bash
drove config
drove config idle_timeout_seconds
drove config idle_timeout_seconds 3600
drove config llama_server.n_gpu_layers -1
```

## Models management

```bash
drove models --help
drove models list
drove models download <repo>
drove models delete <name>
drove models info <name>
drove models config <name>
```

The `CAPS` column in `drove models list` shows model capabilities:
`vision` (multimodal projection) and `stt` (speech-to-text).

## Shell completions

```bash
drove completions --help
```
