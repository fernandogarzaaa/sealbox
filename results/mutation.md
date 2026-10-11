# Mutation testing

Tool: [testteeth](https://pypi.org/project/testteeth/) 0.2.0 (`pip install testteeth`), run per module against
the tests that target it. Large modules were sampled with `--max-mutants` (testteeth samples evenly and
deterministically, so before/after use the same mutants). `--workers 2`.

| module | test files | mutants | before | after |
|---|---|---|---|---|
| `budget.py` | test_audit_budget (+ test_hardening) | 61 (all) | 85.2% (52/61) | **100%** (61/61) |
| `gate.py` | test_gate, test_session_leak (+ test_hardening) | 110 of 137 | 70.9% (78/110) | **92.7%** (102/110) |
| `fingerprint.py` | test_fingerprint, test_session_leak (+ test_hardening) | 120 of 287 | 62.5% (75/120) | **91.7%** (110/120) |

Commands:

```bash
python -m testteeth run src/sealbox/budget.py --pytest-args "tests/test_audit_budget.py tests/test_hardening.py" --workers 2
python -m testteeth run src/sealbox/gate.py --pytest-args "tests/test_gate.py tests/test_session_leak.py tests/test_hardening.py" --workers 2 --max-mutants 110
python -m testteeth run src/sealbox/fingerprint.py --pytest-args "tests/test_fingerprint.py tests/test_session_leak.py tests/test_hardening.py" --workers 2 --max-mutants 120
```

The new tests live in `tests/test_hardening.py` (boundaries of every cap, golden hash values, decode-depth and
opaque-blob thresholds, join-check limits, budget rounding/tolerance).

## Surviving mutants (after)

Mostly equivalent mutants, i.e. changes with no observable behaviour:

* `winnow`: `w <= 1` → `w < 1` (the general loop selects every position when `w == 1` too); `>=` → `>` on ties
  (the selected hash *values* are identical); `_windows`: `<= _WINDOW` → `< _WINDOW` (one 40-token window either way).
* `CharIndex.__contains__`: `return False` → `return None` (`in` coerces to bool).
* `_try_decompress`: `wbits 31` → `32` (32 auto-detects gzip too); `continue` → `pass` at the end of a loop body.
* `_join_hits`: edge `k - 1` → `k + 1` (`seam_hits` re-slices to `k - 1`).
* LSH query-side band width `4` → `5`, `len(t) >= 8` on decoded base64 (the 16-char run minimum already implies it),
  `sim >= threshold` exact-equality boundary, and a few constant tweaks inside `inspect` that only move an
  estimate by less than its rounding.
