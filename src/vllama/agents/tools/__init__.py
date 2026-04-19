"""Tool registry. Importing this package populates all tool specs."""

# Import for registration side-effects.
from vllama.agents.tools import edit as _edit  # noqa: F401
from vllama.agents.tools import glob as _glob  # noqa: F401
from vllama.agents.tools import grep as _grep  # noqa: F401
from vllama.agents.tools import list as _list  # noqa: F401
from vllama.agents.tools import read as _read  # noqa: F401
from vllama.agents.tools import write as _write  # noqa: F401
from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    all_specs,
    clear_registry,
    get_spec,
    register,
)

__all__ = [
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "all_specs",
    "clear_registry",
    "get_spec",
    "register",
]
