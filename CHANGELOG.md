# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-13

First tagged release (alpha).

### Added
- Compute-to-data MCP server: agents run SQL and Python against an encrypted vault without the raw rows leaving it.
- SQL AST guard, k-anonymity / differential-privacy egress gate (aggregates, STDDEV/VARIANCE, CASE bucketing), session-wide leak accounting with winnowed fingerprints.
- Namespaced subprocess sandbox for agent Python (Linux `unshare`; rlimits + audit hook elsewhere, with a warning).
- Red-team eval (56 attacks / 0 leaks in the committed results), mutation-hardened tests, Dockerfile, Claude Desktop config, SECURITY.md.

### Changed
- CI runs lint/type/tests on pushes to `main` and on pull requests; the Docker smoke job stays manual.

### Known limitations
- `require_network_isolation` defaults to `False`: where `unshare -rn` is unavailable (macOS, Windows, most containers) agent code runs without a namespace boundary (a warning is logged). Set it to `True` in production.
