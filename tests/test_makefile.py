"""Regression checks for developer commands and isolated service installation."""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def test_service_restart_runs_despite_same_named_file(tmp_path: Path) -> None:
    (tmp_path / "service-restart").touch()
    result = subprocess.run(
        ["make", "-n", "-f", str(MAKEFILE), "service-restart"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "systemctl --user restart drove.service" in result.stdout


def test_commit_types_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = set(project["tool"]["semantic_release"]["commit_parser_options"]["allowed_tags"])
    workflow = (ROOT / ".github/workflows/pr-title.yml").read_text()
    types = workflow.split("types: |", 1)[1].split("requireScope:", 1)[0]
    assert set(types.split()) == expected
    guidance = (ROOT / "AGENTS.md").read_text()
    allowed = guidance.split("Allowed types:", 1)[1].split(". Add", 1)[0]
    assert set(re.findall(r"`([a-z]+)`", allowed)) == expected
    deploy = (ROOT / "docs/deploy.md").read_text()
    deploy_types = deploy.split("Allowed types", 1)[1].split("```", 2)[1]
    assert set(re.findall(r"[a-z]+", deploy_types)) == expected


@pytest.mark.skipif(sys.platform != "linux", reason="pkill session isolation uses Linux -s")
def test_service_install_stops_alias_before_start(tmp_path: Path) -> None:
    """Run the actual recipe; keep all process matching in a disposable session."""
    pkill = shutil.which("pkill")
    bash = shutil.which("bash")
    assert pkill and bash
    makefile = tmp_path / "Makefile"
    shutil.copyfile(MAKEFILE, makefile)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    # Files, rather than inline shell text, keep the harness command lines from
    # containing server names that the real pkill could match.
    for name, body in {
        "systemctl": f"#!/bin/sh\necho systemctl \"$@\" >> '{events}'\n",
        "pkill": (
            f'#!/bin/sh\necho pkill >> \'{events}\'\nexec "{pkill}" -s "$TEST_SESSION" "$@"\n'
        ),
    }.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import sys\nprint('ready', flush=True)\nsys.stdin.read()\n")
    runner = tmp_path / "runner.py"
    runner.write_text("""import json, os, subprocess, sys
from pathlib import Path
base, makefile, bash, sleeper = sys.argv[1:]
base = Path(base)
os.environ['TEST_SESSION'] = str(os.getsid(0))
children = []
try:
    for name in ['drove serve', 'drove server', 'drove chat']:
        child = subprocess.Popen(
            [bash, '-c', 'exec -a "$1" "$2" "$3"', 'stub', name, sys.executable, sleeper],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        children.append(child)
        assert child.stdout.readline() == 'ready\\n'
    run = subprocess.run(
        ['make', '-f', makefile, '-o', 'install', 'service-install',
         'SERVICE_DIR='+str(base/'service'),
         'SERVICE_FILE='+str(base/'service/drove.service')],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    assert run.returncode == 0, (run.stdout, run.stderr)
    for child in children[:2]:
        assert child.wait(timeout=5) == -15
    assert children[2].poll() is None
    events = (base/'events').read_text().splitlines()
    assert events.index('pkill') < events.index('systemctl --user start drove.service')
    print(json.dumps({'killed': 2, 'chat_alive': True}))
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
""")
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        [sys.executable, str(runner), str(tmp_path), str(makefile), bash, str(sleeper)],
        env=env,
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"killed": 2, "chat_alive": True}
