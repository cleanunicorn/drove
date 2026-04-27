.PHONY: install completions test lint fmt typecheck \
        service-install service-uninstall service-start service-stop service-status service-logs

DROVE_BIN := $(HOME)/.local/share/uv/tools/drove/bin/drove
SERVICE_DIR := $(HOME)/.config/systemd/user
SERVICE_FILE := $(SERVICE_DIR)/drove.service

# ── Development ────────────────────────────────────────────────────────────────

install:
	uv tool install . --force --reinstall-package drove

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
