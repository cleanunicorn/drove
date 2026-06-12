#!/usr/bin/env bash
set -euo pipefail

REPO="${DROVE_REPO:-cleanunicorn/drove}"
GITHUB_URL="https://github.com/${REPO}"
DEFAULT_INSTALL_SOURCE="git+${GITHUB_URL}"
INSTALL_SOURCE="${1:-${DROVE_INSTALL_SOURCE:-${DEFAULT_INSTALL_SOURCE}}}"
PYTHON_REQUEST="${DROVE_PYTHON:-3.14}"
# Extras installed by default (speech-to-text). Set DROVE_EXTRAS="" for a
# minimal text-generation-only install.
DROVE_EXTRAS="${DROVE_EXTRAS-asr}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info() { echo -e "${GREEN}[drove]${NC} $*"; }
warn() { echo -e "${YELLOW}[drove] warning:${NC} $*"; }
error() { echo -e "${RED}[drove] error:${NC} $*" >&2; exit 1; }
bold() { echo -e "${BOLD}$*${NC}"; }

require_command() {
    local cmd="$1"
    local message="$2"
    if ! command -v "${cmd}" &>/dev/null; then
        error "${message}"
    fi
}

echo ""
bold "  drove installer"
echo "  llama.cpp server manager and proxy"
echo ""

# Check OS
OS="$(uname -s)"
case "${OS}" in
    Linux|Darwin) ;;
    *) error "Unsupported OS: ${OS}. Only Linux and macOS are supported." ;;
esac

# Install uv if not present
if ! command -v uv &>/dev/null; then
    info "uv not found — installing uv..."
    require_command curl "curl is required to install uv. Install curl, then re-run this script."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session.
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &>/dev/null; then
        error "uv installation failed. Install uv manually and re-run this script: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    info "uv installed."
else
    info "uv found: $(uv --version)"
fi

# Install drove via uv tool. By default this installs from GitHub, but CI and
# contributors can pass a local checkout path as the first argument or via
# DROVE_INSTALL_SOURCE.
if [[ -n "${DROVE_EXTRAS}" ]]; then
    INSTALL_SPEC="drove[${DROVE_EXTRAS}] @ ${INSTALL_SOURCE}"
else
    INSTALL_SPEC="${INSTALL_SOURCE}"
fi
info "Installing drove from ${INSTALL_SOURCE} with Python ${PYTHON_REQUEST} ..."
uv tool install --force --python "${PYTHON_REQUEST}" "${INSTALL_SPEC}"

# Ensure uv tool bin dir is on PATH for post-install checks in this session.
UV_TOOL_BIN="$(uv tool dir --bin 2>/dev/null || echo "${HOME}/.local/bin")"
if [[ ":${PATH}:" != *":${UV_TOOL_BIN}:"* ]]; then
    export PATH="${UV_TOOL_BIN}:${PATH}"
fi

if ! command -v drove &>/dev/null; then
    warn "drove was installed but is not in your PATH."
    warn "Add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    warn "  export PATH=\"${UV_TOOL_BIN}:\$PATH\""
else
    info "drove installed: $(drove --version 2>/dev/null || echo 'ok')"
fi

echo ""
bold "  Installation complete!"
echo ""
echo "  Quick start:"
echo "    drove init                                    # Create config file"
echo "    drove models download unsloth/Qwen3-8B-GGUF  # Download a model"
echo "    drove serve                                   # Start the proxy"
echo "    drove chat                                    # Interactive chat"
echo ""
echo "  Speech-to-text:"
echo "    drove models download istupakov/parakeet-tdt-0.6b-v3-onnx"
echo ""

# Warn if llama-server is missing (required runtime dependency).
if ! command -v llama-server &>/dev/null; then
    echo ""
    warn "llama-server not found in PATH."
    warn "drove requires llama.cpp — install it before running the server:"
    case "${OS}" in
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
