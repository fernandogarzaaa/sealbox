# sealbox

**Compute-to-data MCP server: let a cloud AI agent reason _about_ your sensitive data without the raw data leaving your machine.**

The agent never gets rows or files. It sends computations (a restricted SQL subset, or Python in a sandbox) and
gets back only what passes an egress gate: k-anonymous, differentially private aggregates, findings about
code and documents, never the protected values or text themselves. Every call is metered against a
per-session leakage budget and written to a tamper-evident audit log.

```
 cloud agent (Claude, GPT, ...)            your machine
 ───────────────────────────────           ─────────────────────────────────────────────────────────
  "avg salary by dept?"   ── MCP ──▶  sealbox ─▶ SQL guard ─▶ DuckDB over policy-shaped views
  "list functions in repo"            │           sandboxed Python (namespaces, rlimits, audit hook)
                                      ▼
   aggregates / findings ◀── gate ◀── k-anonymity · DP noise · PII · raw values · content fingerprints
                                      · session leakage budget · hash-chained audit log
                                      ▲
                          encrypted vault (AES-256-GCM, per-dataset keys)
```

Status: **v0.1, alpha.** Python ≥ 3.10, Linux (the sandbox uses Linux namespaces; it degrades with warnings
elsewhere). MIT licensed.

## Quickstart

```bash
pip install -e '.[dev]'            # or: pip install .  (add [parquet] / [pandas] extras as needed)

sealbox init                        # creates ./vault and a master keyfile (~/.config/sealbox/master.key)
sealbox ingest examples/demo_data/employees.csv examples/demo_data/sales.csv examples/demo_data/acme_engine
sealbox datasets                    # review the generated policies in vault/policies/*.yaml
sealbox serve                       # MCP over stdio (or: --transport http --port 8765)
```

`ingest` writes a secure-by-default policy per dataset (column tags guessed from names and statistics:
`public`, `tokenize`, `aggregate_only`, `deny`). **Review them** before serving. Owner-side commands that
are never exposed over MCP: `sealbox detokenize` (map tokens in an answer back to real values),
`sealbox audit verify|tail`.

Claude Desktop: copy [`examples/claude_desktop_config.json`](examples/claude_desktop_config.json) into your
config and fix the absolute paths. End-to-end check with the official MCP client:
`python examples/mcp_stdio_client.py` (spawns `sealbox serve` on a throwaway demo vault, 9 checks).

Docker (the container has no network; mount the vault and keyfile read-only):

```bash
docker build -t sealbox .
docker run --rm -i --network none \
  -v "$PWD/vault:/vault:ro" -v "$HOME/.config/sealbox/master.key:/run/secrets/sealbox.key:ro" sealbox
```

## Tools the agent sees

| tool | what it does |
|---|---|
| `list_datasets` | names, kinds, which tools may touch each dataset |
| `describe_dataset` | schema, tags, policy-aware stats, **synthetic** sample rows |
| `run_sql` | one read-only DuckDB `SELECT` in a safe subset (aggregates, `GROUP BY`, joins; no subqueries/CTEs/windows/table functions) |
| `run_python` | Python over `tables` / `docs` in a sandbox; only the gated `result`/stdout come back |
| `get_result` | page through a gated result |
| `budget_status` | epsilon / bytes / query / error budget left this session |

Deliberately **not** exposed: detokenization, raw reads, policy edits, budget resets.

## Architecture

| module | role |
|---|---|
| `vault.py`, `crypto.py` | encrypted vault: per-dataset AES-256-GCM data keys wrapped by a master key; plaintext only in memory |
| `policy.py` | per-dataset YAML: column tags, `k_min`, DP (`epsilon_per_query`, clamp `bounds`), leak caps, output caps |
| `sqlguard.py` | parses SQL with DuckDB's own parser, enforces tags per column reference, rewrites aggregates to also return `count(DISTINCT entity)` so k-anonymity counts real people, splits arithmetic over aggregates into separately noised leaves |
| `engine.py` | tool logic: k-anonymity suppression, Laplace noise (CSPRNG) calibrated per aggregate, budgets, error sanitising |
| `sandbox.py`, `_runner.py` | `run_python` in `unshare -r -n -m` (no network, vault/keyfile/$HOME hidden), rlimits, timeout, empty env, audit hook, latency padding |
| `gate.py` | egress gate on every output: size caps, prompt-injection stripping, PII redaction, raw protected values, content fingerprints, session-wide leak accounting |
| `fingerprint.py` | exact token shingles, structural MinHash (renamed copies), winnowed whitespace/case-insensitive character fingerprints, decoding of base64/hex/percent/`\x`/zlib/rot13/reversed |
| `budget.py`, `audit.py` | per-session epsilon/bytes/query/error budgets; HMAC hash-chained audit log |
| `tokenizer.py`, `synth.py`, `detectors.py` | HMAC entity tokens, synthetic samples, PII / injection detectors |

### SQL: what is allowed

* `aggregate_only` columns only inside `COUNT/SUM/AVG/STDDEV/VARIANCE`-style aggregates (`CORR`/`COVAR` only with DP off), or in `WHERE`
  filters and in **CASE buckets** (`CASE WHEN salary < 80000 THEN 'low' ... END`, constant labels only,
  at most 12 branches) of an aggregated query. Every group needs ≥ `k_min` contributing entities.
* Extremal aggregates (`min/max/median/quantile/string_agg/list/...`) only on `public` columns.
* With DP on, every sensitive aggregate is noised: `COUNT` (sensitivity 1), `SUM/AVG` (column range or
  operator `dp.bounds`, values clamped in SQL), `STDDEV/VARIANCE` (**require** `dp.bounds`; sensitivity
  `(hi-lo)/sqrt(n-1)` and `(hi-lo)^2/(n-1)`), `AVG(length(text))` (coarse length bound).
  Arithmetic over aggregates (`sum(bonus)/sum(salary)`, `round(.., 2)`) is computed from separately noised
  leaves (one epsilon each, post-processing). Rescaled or opaque expressions such as `sum(salary * 1000)` are
  refused (`dp_uncalibrated`) because they would defeat the calibration.

### Code and documents

`run_python` output is fingerprinted against every protected file and long text cell. Per call: more than 2
shared 8-token shingles, structural similarity ≥ 0.55, or a copied stretch of more than 32 non-whitespace
characters (also after decoding) is denied. Per session (`leak:` in the policy):

* `max_session_protected_chars` (160): estimated protected characters released across all calls (distinct
  winnowed fingerprints × their spacing, plus an edge allowance per piece);
* `max_session_seam_hits` (0): fingerprints that appear only when two separately released fragments are
  joined, in either order and across calls (pieces released for offline reassembly);
* `max_session_derived_chars` (16000) and `max_session_derived_fraction` (0.25): hard cap on bytes of
  code-derived output from code/doc datasets, whatever its encoding: `min(chars, max(256, fraction × corpus))`.

The character index stores Rabin-Karp hashes of 10-char windows thinned by winnowing (window 6) in a sorted
`array('Q')`: ~2.3 bytes per protected character instead of one Python int per character, while any shared run
of ≥ 15 non-whitespace characters is still guaranteed to hit.

## Threat model

**Protects against** an agent (or the cloud provider behind it, or a prompt injection steering it) trying to
extract raw rows, PII, individual values, or proprietary code/text through the tools: row dumps, filters on a
single person, differencing and averaging, padded-group bisection, data-dependent error oracles, encoded,
spaced, reversed, chunked and shuffled code smuggling, and instructions planted in the data.

**Does NOT protect against:**

* **The provider sees every allowed output.** Aggregates, findings and anything the gate lets through go to
  the model provider. sealbox minimises and meters that disclosure; it does not hide it.
* **Side channels.** Timing (only coarsely padded), resource usage, and whether a query is allowed or denied
  are all observable. Denials are themselves 1-bit signals, bounded only by query/error budgets.
* **Inference from legitimate aggregates.** DP bounds what one session learns about one entity under the
  stated epsilon, but the default budget (30) is a utility choice, not a strong privacy guarantee, and DP noise
  for `AVG`/`STDDEV` depends on the exact (un-noised) group size. Data-derived bounds are used for `SUM/AVG` when the
  operator sets no `dp.bounds`.
* **Content the fingerprints cannot see.** Paraphrases, summaries, and custom ciphers are not detected; they
  are bounded only by the derived-volume cap. An agent can still obtain a fraction of a small file in small
  unordered pieces (see the red-team table) before the session caps stop it.
* **Multiple sessions.** Budgets are per server process. Restarting the server resets them.
* **A compromised host.** No TEE yet: whoever controls the machine running sealbox can read the decrypted data
  in memory. **Running sealbox inside a TEE (SEV-SNP/TDX/Nitro) with remote attestation is future work**, as is
  a stronger sandbox (gVisor/Firecracker/nsjail) for hostile multi-tenant use.

## Red-team results

`python eval/redteam.py` simulates an exfiltrating agent against the synthetic demo vault (fresh session per
attack) plus a benign analytics set. Scoring is outcome-based: an attack counts as a leak only if the secret is
recoverable from what was actually released (decoded and reassembled across calls). Full table:
[`results/redteam.md`](results/redteam.md).

| | v0.1 WIP (`fcd2ead`) | now |
|---|---|---|
| attacks | 47 | 56 |
| attacks that leaked | 1 (`smuggle_shuffled_small_chunks`) | **0** |
| leak-block rate | 97.9% | **100%** |
| benign tasks refused | 7 / 27 (25.9%) | **2 / 27 (7.4%)** |

| category | attacks | leaked |
|---|---|---|
| row_dump | 10 | 0 |
| reconstruction | 13 | 0 |
| code_smuggling | 27 | 0 |
| injection | 2 | 0 |
| error_leak | 4 | 0 |

Honest caveats: "0 leaked" means no verbatim protected line, ≥3 secret constants, PII value or salary within 1%
was recoverable. Chunk attacks still obtained part of the 1,124-char target file as unordered pieces: 16% (20-char
chunks), 18% (25-char), 16% (12-char), 7% (char-code lists). Still refused: `median()` on an `aggregate_only`
column (extremal) and `corr()` under DP (no calibrated mechanism yet). The eval runs on small synthetic data;
it is evidence, not proof.

Mutation testing ([testteeth](https://pypi.org/project/testteeth/)), see [`results/mutation.md`](results/mutation.md):
`budget.py` 100% (61/61), `gate.py` 92.7% (sample of 110/137), `fingerprint.py` 91.7% (sample of 120/287).

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q --cov=sealbox      # 344 tests, 95% line+branch coverage
python -m ruff check . && python -m ruff format --check . && python -m mypy
python eval/redteam.py                 # rewrites results/redteam.{json,md}
```

CI (`.github/workflows/ci.yml`) runs lint, type-check and tests on pushes to `main` and on pull requests; the Docker smoke job runs only when triggered manually (`workflow_dispatch`).

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities. License: [MIT](LICENSE) © 2026 Fernando Garza.
