"""Tool definitions and execution for TUI function calling."""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import threading
from pathlib import Path

import httpx

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file from disk."
                " For large files, use offset and limit to read a range of lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line to start from (0-based, default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to read (default: all).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file on disk. Creates parent directories automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (default: current directory).",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list subdirectories recursively.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_execute",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run command in background (returns task ID).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the content of a URL (HTTP GET).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name pattern or content string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory to search in (default: current directory).",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Glob pattern for filenames (e.g. '*.py').",
                    },
                    "content_query": {
                        "type": "string",
                        "description": "Substring to search for within file contents.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_background_task",
            "description": "Check the status and output of a background task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to check.",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
]


# Global storage for background tasks
_background_tasks: dict[str, dict] = {}


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return f"Error: invalid JSON arguments: {arguments}"

    if name == "read_file":
        return _read_file(args)
    elif name == "write_file":
        return _write_file(args)
    elif name == "list_files":
        return _list_files(args)
    elif name == "shell_execute":
        return _shell_execute(args)
    elif name == "fetch_url":
        return _fetch_url(args)
    elif name == "search_files":
        return _search_files(args)
    elif name == "check_background_task":
        return _check_background_task(args)
    else:
        return f"Error: unknown tool '{name}'"


def _read_file(args: dict) -> str:
    path = Path(args.get("path", "")).expanduser()
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"

    lines = text.splitlines(keepends=True)
    offset = args.get("offset", 0)
    limit = args.get("limit")
    if limit is not None:
        lines = lines[offset : offset + limit]
    elif offset:
        lines = lines[offset:]

    return "".join(lines)


def _write_file(args: dict) -> str:
    path = Path(args.get("path", "")).expanduser()
    content = args.get("content", "")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Error writing file: {e}"
    return f"Successfully wrote {len(content)} bytes to {path}"


def _list_files(args: dict) -> str:
    path_str = args.get("path", ".")
    recursive = args.get("recursive", False)
    path = Path(path_str).expanduser()

    if not path.exists():
        return f"Error: path not found: {path}"
    if not path.is_dir():
        return f"Error: not a directory: {path}"

    try:
        if recursive:
            entries = []
            for root, dirs, files in os.walk(path):
                root_path = Path(root)
                for d in dirs:
                    entries.append(str((root_path / d).relative_to(path)) + "/")
                for f in files:
                    entries.append(str((root_path / f).relative_to(path)))
        else:
            entries = []
            for entry in path.iterdir():
                suffix = "/" if entry.is_dir() else ""
                entries.append(entry.name + suffix)

        return "\n".join(sorted(entries))
    except Exception as e:
        return f"Error listing files: {e}"


def _shell_execute(args: dict) -> str:
    command = args.get("command", "")
    background = args.get("background", False)
    timeout = args.get("timeout", 30)

    if not command:
        return "Error: no command provided"

    if background:
        import uuid

        task_id = str(uuid.uuid4())[:8]
        _background_tasks[task_id] = {
            "command": command,
            "status": "running",
            "output": "",
            "error": "",
        }

        def run():
            try:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = proc.communicate()
                _background_tasks[task_id]["status"] = "finished"
                _background_tasks[task_id]["output"] = stdout
                _background_tasks[task_id]["error"] = stderr
                _background_tasks[task_id]["returncode"] = proc.returncode
            except Exception as e:
                _background_tasks[task_id]["status"] = "failed"
                _background_tasks[task_id]["error"] = str(e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return f"Started background task with ID: {task_id}"

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = f"STDOUT:\n{proc.stdout}"
        if proc.stderr:
            output += f"\nSTDERR:\n{proc.stderr}"
        output += f"\nReturn code: {proc.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


def _fetch_url(args: dict) -> str:
    url = args.get("url", "")
    if not url:
        return "Error: no URL provided"

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        return f"Error fetching URL: {e}"


def _search_files(args: dict) -> str:
    path_str = args.get("path", ".")
    name_pattern = args.get("name_pattern")
    content_query = args.get("content_query")
    recursive = args.get("recursive", True)
    path = Path(path_str).expanduser()

    if not path.exists():
        return f"Error: path not found: {path}"

    results = []
    try:
        search_path = path.rglob("*") if recursive else path.iterdir()
        for p in search_path:
            if not p.is_file():
                continue

            matches_name = True
            if name_pattern:
                matches_name = fnmatch.fnmatch(p.name, name_pattern)

            matches_content = True
            if content_query:
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    matches_content = content_query in content
                except Exception:
                    matches_content = False

            if matches_name and matches_content:
                results.append(str(p.relative_to(path) if path_str != "." else p))

        if not results:
            return "No matches found."
        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {e}"


def _check_background_task(args: dict) -> str:
    task_id = args.get("task_id", "")
    if task_id not in _background_tasks:
        return f"Error: task ID '{task_id}' not found"

    task = _background_tasks[task_id]
    status = task["status"]
    res = f"Task {task_id} status: {status}\nCommand: {task['command']}\n"
    if status in ("finished", "failed"):
        res += f"Output:\n{task['output']}\n"
        if task["error"]:
            res += f"Error:\n{task['error']}\n"
        if "returncode" in task:
            res += f"Return code: {task['returncode']}\n"
    return res
