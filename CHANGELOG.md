# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added GitHub issue templates for bug reports and feature requests to standardize triage-ready submissions.
- Added test coverage for `ServerManager` startup health checks (success, unexpected exit, and timeout scenarios).
- Documented the repository install script in the README install instructions.
- Added an install workflow that runs the installer from a checkout and smoke tests the installed CLI.
- Allowed `install.sh` to install from an explicit source path and request Python 3.14 for `uv tool install`.
- `CHANGELOG.md` tracking notable changes per release.
- Automated release pipeline via `python-semantic-release`: merges to `main` bump the version in `pyproject.toml`, move `[Unreleased]` entries into a versioned section in `CHANGELOG.md`, tag (`vX.Y.Z`), and create a GitHub Release.
- PR-title lint (`.github/workflows/pr-title.yml`) enforcing Conventional Commits so that the squash-merged title drives the correct semver bump.
- `docs/deploy.md` documenting the end-to-end release process for repository maintainers, with worked example and maintainer checklist.

### Fixed
- Fixed model-store test lint issues so full-repository Ruff checks pass.

### Changed
- Updated `drove observe web` to paginate request history and expose a **Load more** button so initial page loads stay fast with large logs.
- Updated the observe web **Load more** control to disable during pagination requests so rapid repeated clicks no longer append duplicate records.


## [0.1.0]

### Added
- Initial release: lazy `llama-server` lifecycle manager, OpenAI-compatible FastAPI proxy, and Typer CLI (`serve`, `models list/download/delete/info/config`).
