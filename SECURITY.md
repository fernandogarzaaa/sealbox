# Security policy

sealbox is alpha software (v0.1). It reduces and meters what an AI agent can learn about sensitive data; it is
not a guarantee. Read the threat model in the README before relying on it.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** via GitHub's "Report a vulnerability" (Security
Advisories) on https://github.com/fernandogarzaaa/sealbox, or by email to fernandogarzaaa@gmail.com with
"sealbox security" in the subject. Do not open a public issue for an unfixed leak.

Useful reports include: the sealbox version/commit, the policy used, the exact tool calls, and what was
recovered. A reproducible attack in the style of `eval/redteam.py` is ideal.

You can expect an acknowledgement within 7 days. Fixes are released as soon as practical and credited unless
you prefer otherwise.

## In scope

* Extracting raw rows, PII, individual values, or protected code/text through the MCP tools.
* Bypassing the SQL guard, the egress gate, DP calibration, the leakage budgets, or the audit log's integrity.
* Escaping the `run_python` sandbox (network, files outside the runtime, the vault or key material).

## Out of scope (known limitations)

* What the model provider sees in outputs the policy allows.
* Timing and other side channels beyond the documented padding.
* Attacks that require control of the host running sealbox (no TEE yet).
* Budget resets by restarting the server process.

## Handling keys

Never commit the master key, vaults, or `.sealbox_master.key`. Prefer `SEALBOX_KEYFILE` pointing at a file
with mode 0600, or a secrets manager that sets `SEALBOX_MASTER_KEY` for the server process only.
