#!/usr/bin/env bash
set -euo pipefail

REPO="cleanunicorn/vllama"
GITHUB_URL="https://github.com/${REPO}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[vllama]${NC} $*"; }
warn()  { echo -e "${YELLOW}[vllama] warning:${NC} $*"; }
error() { echo -e "${RED}[vllama] error:${NC} $*" >&2; exit 1; }
bold()  { echo -e "${BOLD}$*${NC}"; }

echo ""
bold "  vllama installer"
echo "  llama.cpp server manager and proxy"
echo ""

# Check OS
OS="$(uname -s)"
case "$OS" in
    Linux|Darwin) ;;
    *) error "Unsupported OS: $OS. Only Linux and macOS are supported." ;;
esac

# Install uv if not present
if ! command -v uv &>/dev/null; then
    info "uv not found — installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &>/dev/null; then
        error "uv installation failed. Install uv manually and re-run this script: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    info "uv installed."
else
    info "uv found: $(uv --version)"
fi

# Install vllama via uv tool (uv will fetch Python 3.14 automatically if needed)
info "Installing vllama from ${GITHUB_URL} ..."
uv tool install "git+${GITHUB_URL}"

# Ensure uv tool bin dir is on PATH
UV_TOOL_BIN="$(uv tool dir --bin 2>/dev/null || echo "${HOME}/.local/bin")"
if [[ ":${PATH}:" != *":${UV_TOOL_BIN}:"* ]]; then
    export PATH="${UV_TOOL_BIN}:${PATH}"
fi

if ! command -v vllama &>/dev/null; then
    warn "vllama was installed but is not in your PATH."
    warn "Add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    warn "  export PATH=\"${UV_TOOL_BIN}:\$PATH\""
else
    info "vllama installed: $(vllama --version 2>/dev/null || echo 'ok')"
fi

echo ""
bold "  Installation complete!"
echo ""
echo "  Quick start:"
echo "    vllama init                                    # Create config file"
echo "    vllama models download unsloth/Qwen3-8B-GGUF  # Download a model"
echo "    vllama server                                  # Start the proxy"
echo "    vllama chat                                    # Interactive chat"
echo ""

# Warn if llama-server is missing (required runtime dependency)
if ! command -v llama-server &>/dev/null; then
    echo ""
    warn "llama-server not found in PATH."
    warn "vllama requires llama.cpp — install it before running the server:"
    case "$OS" in
        Darwin)
            warn "  brew install llama.cpp"
            ;;
        Linux)
            warn "  See: https://github.com/ggml-org/llama.cpp#build"
            warn "  Or download a release binary: https://github.com/ggml-org/llama.cpp/releases"
            ;;
    esac
    echo ""
fi
