.PHONY: install completions test lint fmt typecheck \
        service-install service-uninstall service-start service-stop service-status service-logs

VLLAMA_BIN := $(HOME)/.local/share/uv/tools/vllama/bin/vllama
SERVICE_DIR := $(HOME)/.config/systemd/user
SERVICE_FILE := $(SERVICE_DIR)/vllama.service

# ── Development ────────────────────────────────────────────────────────────────

install:
	uv tool install . --force --reinstall-package vllama

completions: install
	vllama completions install

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
	@printf '[Unit]\nDescription=vllama llama.cpp server manager and proxy\nAfter=network.target\n\n[Service]\nType=simple\nExecStart=%s serve\nEnvironment=PATH=%s\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n' \
		"$(VLLAMA_BIN)" "$(PATH)" > $(SERVICE_FILE)
	systemctl --user daemon-reload
	systemctl --user enable vllama.service
	-pkill -f "vllama serve" 2>/dev/null || true
	systemctl --user start vllama.service
	@echo "Service installed and started."
	@echo "To start at boot without login: loginctl enable-linger $$USER"

service-uninstall:
	-systemctl --user stop vllama.service
	-systemctl --user disable vllama.service
	rm -f $(SERVICE_FILE)
	systemctl --user daemon-reload
	@echo "Service removed."

service-start:
	systemctl --user start vllama.service

service-stop:
	systemctl --user stop vllama.service

service-restart:
	systemctl --user restart vllama.service	

service-status:
	systemctl --user status vllama.service

service-logs:
	journalctl --user -u vllama.service -f
