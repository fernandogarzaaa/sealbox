"""End-to-end check: spawn `sealbox serve` over stdio and drive it with the official MCP client.

    python examples/mcp_stdio_client.py

Builds a throwaway vault from examples/demo_data (fresh random key in a temp dir), then calls every
tool the server exposes and prints what a cloud agent would see. Exits non-zero on any surprise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

DEMO = Path(__file__).resolve().parent / "demo_data"


def _payload(res: Any) -> dict[str, Any]:
    return json.loads(res.content[0].text)  # type: ignore[no-any-return]


async def drive(params: StdioServerParameters) -> list[str]:
    checks: list[str] = []

    def check(name: str, ok: bool) -> None:
        checks.append(f"{'PASS' if ok else 'FAIL'}  {name}")

    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        tools = {t.name for t in (await s.list_tools()).tools}
        check(
            "6 tools exposed, no detokenize",
            tools == {"list_datasets", "describe_dataset", "run_sql", "run_python", "get_result", "budget_status"},
        )
        ds = _payload(await s.call_tool("list_datasets", {}))
        check("list_datasets sees 3 demo datasets", len(ds["datasets"]) == 3)
        desc = _payload(await s.call_tool("describe_dataset", {"name": "employees"}))
        real = (DEMO / "employees.csv").read_text()
        samples = desc.get("synthetic_sample_rows") or []
        sensitive = ("employee_id", "name", "email", "performance_note", "ssn")
        vals = [str(r[c]) for r in samples for c in sensitive if c in r]
        check(
            "describe_dataset samples are synthetic (none appear in the real CSV)",
            desc["status"] == "ok" and len(samples) > 0 and not any(len(v) > 6 and v in real for v in vals),
        )
        agg = _payload(
            await s.call_tool(
                "run_sql", {"sql": "SELECT dept, count(*) AS n, avg(salary) AS avg_salary FROM employees GROUP BY dept"}
            )
        )
        check("aggregate SQL allowed (DP-noised)", agg["status"] == "ok" and len(agg["rows"]) == 6)
        print("  aggregate:", json.dumps(agg["rows"][:3]))
        dump = _payload(await s.call_tool("run_sql", {"sql": "SELECT name, ssn, salary FROM employees"}))
        check("raw row dump denied", dump["status"] == "denied")
        print("  dump denial:", dump["reasons"][0])
        code = "result = {p: t.count('def ') for p, t in docs['acme_engine'].items()}"
        py = _payload(await s.call_tool("run_python", {"code": code, "datasets": ["acme_engine"]}))
        check("run_python analysis over proprietary code allowed", py["status"] == "ok")
        print("  python:", py.get("result"))
        leak = _payload(
            await s.call_tool(
                "run_python", {"code": "result = docs['acme_engine']['pricing.py']", "datasets": ["acme_engine"]}
            )
        )
        check("verbatim proprietary code blocked", leak["status"] == "denied")
        page = _payload(await s.call_tool("get_result", {"result_id": agg["result_id"], "offset": 0, "limit": 2}))
        check("get_result pages a gated result", page["status"] == "ok")
        b = _payload(await s.call_tool("budget_status", {}))
        print("  budget:", json.dumps(b))
        check("budget_status tracks queries", b["queries"]["used"] >= 4)
    return checks


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sealbox-mcp-"))
    env = {**os.environ, "SEALBOX_KEYFILE": str(tmp / "k" / "master.key")}
    env.pop("SEALBOX_MASTER_KEY", None)
    base = [sys.executable, "-m", "sealbox.cli", "--vault", str(tmp / "vault")]
    subprocess.run([*base, "init"], check=True, env=env, capture_output=True)
    for p in ("employees.csv", "sales.csv", "acme_engine"):
        subprocess.run([*base, "ingest", str(DEMO / p)], check=True, env=env, capture_output=True)
    params = StdioServerParameters(command=sys.executable, args=[*base[1:], "serve"], env=env)
    checks = anyio.run(drive, params)
    print("\n".join(checks))
    audit = subprocess.run([*base, "audit", "verify"], env=env, capture_output=True, text=True)
    print("audit:", audit.stdout.strip())
    return 0 if all(c.startswith("PASS") for c in checks) and audit.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
