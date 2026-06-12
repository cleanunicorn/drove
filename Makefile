.PHONY: install completions test lint fmt typecheck \
        service-install service-uninstall service-start service-stop service-status service-logs

DROVE_BIN := $(HOME)/.local/share/uv/tools/drove/bin/drove
SERVICE_DIR := $(HOME)/.config/systemd/user
SERVICE_FILE := $(SERVICE_DIR)/drove.service

DROVE_PYTHON ?= 3.14
# Extras installed by default (speech-to-text). Set DROVE_EXTRAS= for a
# minimal text-generation-only install.
DROVE_EXTRAS ?= asr

# ── Install ────────────────────────────────────────────────────────────────────

install:
	@set -eu; \
	case "$$(uname -s)" in \
		Linux|Darwin) ;; \
		*) echo "[drove] error: unsupported OS $$(uname -s) — only Linux and macOS are supported" >&2; exit 1 ;; \
	esac; \
	if ! command -v uv >/dev/null 2>&1; then \
		echo "[drove] uv not found — installing uv..."; \
		command -v curl >/dev/null 2>&1 || { echo "[drove] error: curl is required to install uv. Install curl, then re-run make install." >&2; exit 1; }; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		PATH="$$HOME/.local/bin:$$PATH"; \
		command -v uv >/dev/null 2>&1 || { echo "[drove] error: uv installation failed. Install uv manually and re-run: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }; \
		echo "[drove] uv installed."; \
	fi; \
	if [ -n "$(DROVE_EXTRAS)" ]; then spec='.[$(DROVE_EXTRAS)]'; else spec='.'; fi; \
	echo "[drove] installing drove ($$spec) with Python $(DROVE_PYTHON) ..."; \
	uv tool install --force --reinstall-package drove --python "$(DROVE_PYTHON)" "$$spec"; \
	uv_bin="$$(uv tool dir --bin 2>/dev/null || echo "$$HOME/.local/bin")"; \
	case ":$$PATH:" in \
		*":$$uv_bin:"*) echo "[drove] drove installed: $$($$uv_bin/drove --version 2>/dev/null || echo ok)" ;; \
		*) echo "[drove] warning: drove was installed but $$uv_bin is not on your PATH."; \
		   echo "[drove] add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"; \
		   echo "  export PATH=\"$$uv_bin:\$$PATH\"" ;; \
	esac; \
	if ! command -v llama-server >/dev/null 2>&1; then \
		echo "[drove] warning: llama-server not found in PATH."; \
		echo "[drove] drove requires llama.cpp — install it before running the server:"; \
		case "$$(uname -s)" in \
			Darwin) echo "  brew install llama.cpp" ;; \
			*) echo "  see https://github.com/ggml-org/llama.cpp#build"; \
			   echo "  or download a release binary: https://github.com/ggml-org/llama.cpp/releases" ;; \
		esac; \
	fi; \
	echo ""; \
	echo "[drove] installation complete. Quick start:"; \
	echo "  drove init                                    # Create config file"; \
	echo "  drove models download unsloth/Qwen3-8B-GGUF  # Download a model"; \
	echo "  drove serve                                   # Start the proxy"; \
	echo "  drove chat                                    # Interactive chat"; \
	echo ""; \
	echo "  Speech-to-text:"; \
	echo "  drove models download istupakov/parakeet-tdt-0.6b-v3-onnx"

# ── Development ────────────────────────────────────────────────────────────────

completions: install
	drove completions install

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src/

# ── Systemd user service ───────────────────────────────────────────────────────

service-install: install
	mkdir -p $(SERVICE_DIR)
	@printf '[Unit]\nDescription=drove llama.cpp server manager and proxy\nAfter=network.target\n\n[Service]\nType=simple\nExecStart=%s server\nEnvironment=PATH=%s\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n' \
		"$(DROVE_BIN)" "$(PATH)" > $(SERVICE_FILE)
	systemctl --user daemon-reload
	systemctl --user enable drove.service
	-pkill -f "drove server" 2>/dev/null || true
	systemctl --user start drove.service
	@echo "Service installed and started."
	@echo "To start at boot without login: loginctl enable-linger $$USER"

service-uninstall:
	-systemctl --user stop drove.service
	-systemctl --user disable drove.service
	rm -f $(SERVICE_FILE)
	systemctl --user daemon-reload
	@echo "Service removed."

service-start:
	systemctl --user start drove.service

service-stop:
	systemctl --user stop drove.service

service-restart:
	systemctl --user restart drove.service	

service-status:
	systemctl --user status drove.service

service-logs:
	journalctl --user -u drove.service -f
