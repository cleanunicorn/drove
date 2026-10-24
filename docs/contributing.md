# Contributing

[Docs](./README.md) / Contributing

Use short-lived branches and pull requests targeting `main`. Read
[AGENTS.md](../AGENTS.md) for the full repository conventions and
[Architecture](./architecture.md) for the code layout.

## Set up a checkout

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
git switch -c docs/your-change
```

Use a branch prefix matching the work: `feat/`, `fix/`, `docs/`, `refactor/`,
`perf/`, or `chore/`. The make targets sync the project environment and download
Python 3.14 if needed. `make install` installs a global CLI and is optional for
development. Tests fake backend processes, so they need no model downloads,
GPU, llama-server, or speech engine setup.

## Make and verify a change

Keep each PR focused and add an entry under `## [Unreleased]` in
[CHANGELOG.md](../CHANGELOG.md), including documentation-only changes. Use the
appropriate Keep-a-Changelog group, such as `### Changed`, with a past-tense
bullet describing the user-visible impact.

```bash
make lint
make typecheck
make test
```

For Python formatting, run `make fmt` before lint. Use the make targets rather
than invoking their underlying tools directly. The sanctioned exception for
a focused test is `uv run pytest tests/test_proxy.py` (or a test node within it).

For documentation changes, also check relative links, heading anchors, and
example commands against the CLI. Put each topic's full instructions in its
guide or reference, and link to them from onboarding. Keep current behavior
separate from [design proposals](./proposals/README.md).

## Open a pull request

Commit with a Conventional Commits subject, push the branch, and open a PR
against `main`. Include what changed, why, and the checks you ran. Every PR
title must be a valid Conventional Commits string, for example:

```text
docs: clarify model naming in the quickstart
fix(proxy): handle a failed backend connection
```

`docs:` does not trigger a release. `feat:` triggers a minor release; `fix:`
and `perf:` trigger patches. Breaking-change titles or footers trigger a major
release. See [Release process](./deploy.md) before choosing a type.

The PR title and Install CI checks must pass. CI does not run lint, type
checking, or tests; run those locally. Maintainers squash-merge with the PR
title as the commit message. Never push directly to `main`, edit the project
version, or create release tags by hand.

[Docs](./README.md) · [Architecture](./architecture.md) · [Release process](./deploy.md)
