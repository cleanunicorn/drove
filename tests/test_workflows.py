"""Contract tests for the CI quality gate in .github/workflows/.

The workflows themselves only run on GitHub, so these tests pin the parts of
their shape that the quality gate depends on: which commands the Check job
runs, in what order, and that semantic-release cannot start without it.
"""

from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

# Order matters: `uv lock --check` must run before any `uv run` (behind every
# make target), which would otherwise silently rewrite a stale uv.lock.
CHECK_COMMANDS = ["uv lock --check", "make lint", "make typecheck", "make test"]


def load(name: str) -> dict[Any, Any]:
    data = yaml.safe_load((WORKFLOWS / name).read_text())
    assert isinstance(data, dict)
    return data


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # PyYAML reads the bare `on:` key as the boolean True (YAML 1.1).
    on = workflow.get("on", workflow.get(True))
    assert isinstance(on, dict)
    return on


def test_check_runs_on_pull_requests_and_when_called() -> None:
    on = triggers(load("check.yml"))
    assert "pull_request" in on
    assert "workflow_call" in on


def test_check_job_runs_every_quality_command_in_order() -> None:
    job = load("check.yml")["jobs"]["check"]
    runs = [step["run"].strip() for step in job["steps"] if "run" in step]
    assert runs == CHECK_COMMANDS


def test_check_job_steps_are_blocking() -> None:
    job = load("check.yml")["jobs"]["check"]
    assert not job.get("continue-on-error", False)
    for step in job["steps"]:
        assert not step.get("continue-on-error", False), step.get("name")
        assert "if" not in step, step.get("name")


def test_check_job_uses_project_python() -> None:
    steps = load("check.yml")["jobs"]["check"]["steps"]
    setup = next(s for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv@"))
    assert str(setup["with"]["python-version"]) == "3.14"


def test_check_workflow_is_read_only() -> None:
    assert load("check.yml")["permissions"] == {"contents": "read"}


def test_release_calls_the_check_workflow() -> None:
    jobs = load("release.yml")["jobs"]
    assert jobs["check"]["uses"] == "./.github/workflows/check.yml"


def test_every_release_job_waits_for_check() -> None:
    jobs = load("release.yml")["jobs"]
    assert "release" in jobs
    for name, job in jobs.items():
        if name == "check":
            continue
        needs = job.get("needs", [])
        assert "check" in ([needs] if isinstance(needs, str) else needs), name
        # An `if: always()` (or similar) would run the job despite a red check.
        assert "if" not in job, name
