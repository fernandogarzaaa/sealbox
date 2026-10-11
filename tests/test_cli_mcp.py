from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

from conftest import DEMO, write_people
from sealbox import crypto
from sealbox.cli import main
from sealbox.engine import Engine
from sealbox.policy import SessionConfig
from sealbox.server import build_server
from sealbox.vault import Vault


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path]:
    monkeypatch.delenv(crypto.ENV_KEY, raising=False)
    monkeypatch.delenv(crypto.ENV_KEYFILE, raising=False)
    kf = tmp_path / "keys" / "master.key"
    base = ["--vault", str(tmp_path / "vault"), "--keyfile", str(kf)]
    return base, tmp_path


def test_cli_workflow(cli_env: tuple[list[str], Path], capsys: pytest.CaptureFixture[str]) -> None:
    base, tmp = cli_env
    assert main([*base, "init"]) == 0
    assert "created master keyfile" in capsys.readouterr().out
    write_people(tmp / "people.csv")
    assert main([*base, "ingest", str(tmp / "people.csv")]) == 0
    assert "dataset 'people'" in capsys.readouterr().out
    assert main([*base, "ingest", str(tmp / "people.csv")]) == 1  # exists
    assert "already exists" in capsys.readouterr().err
    assert main([*base, "datasets"]) == 0
    assert "salary=aggregate_only" in capsys.readouterr().out
    assert main([*base, "audit", "verify"]) == 0
    assert "chain intact" in capsys.readouterr().out
    keys = crypto.LocalKeyProvider(crypto.load_master_key(Path(base[3])))
    vault = Vault(tmp / "vault", keys)
    eng = Engine(vault, keys, SessionConfig(python_time_quantum_s=0))
    tok = eng.token_map("people")
    some = next(t for t, v in tok.items() if v == "Person Number4")
    assert main([*base, "detokenize", f"winner: {some}", "--dataset", "people"]) == 0
    assert capsys.readouterr().out == "winner: Person Number4\n"
    assert main([*base, "audit", "tail", "-n", "2"]) == 0
    assert "detokenize" in capsys.readouterr().out
    log = vault.audit_path
    lines = log.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["session"] = "forged"
    log.write_text("\n".join([json.dumps(rec), *lines[1:]]) + "\n")
    assert main([*base, "audit", "verify"]) == 2
    assert "TAMPERED" in capsys.readouterr().err


def test_cli_errors(cli_env: tuple[list[str], Path], capsys: pytest.CaptureFixture[str]) -> None:
    base, _ = cli_env
    assert main([*base, "datasets"]) == 1
    assert "error" in capsys.readouterr().err


def test_mcp_inprocess_roundtrip(engine: Engine) -> None:
    from mcp import Client

    server = build_server(engine)

    async def go() -> None:
        async with Client(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == {
                "list_datasets",
                "describe_dataset",
                "run_sql",
                "run_python",
                "get_result",
                "budget_status",
            }
            assert "detokenize" not in tools
            res = await client.call_tool("run_sql", {"sql": "SELECT dept, count(*) FROM people GROUP BY dept"})
            payload = json.loads(res.content[0].text)
            assert payload["status"] == "ok" and payload["columns"][0] == "dept"
            res = await client.call_tool("run_sql", {"sql": "SELECT name, salary FROM people"})
            assert json.loads(res.content[0].text)["status"] == "denied"
            res = await client.call_tool("run_python", {"code": "result = len(rows('people'))"})
            assert json.loads(res.content[0].text)["result"] == 60
            res = await client.call_tool("budget_status", {})
            assert json.loads(res.content[0].text)["queries"]["used"] == 3

    anyio.run(go)


@pytest.mark.slow
def test_mcp_stdio_subprocess_roundtrip(tmp_path: Path) -> None:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    env = {"SEALBOX_KEYFILE": str(tmp_path / "k" / "master.key"), "PATH": "/usr/bin:/bin"}
    sb = [sys.executable, "-m", "sealbox.cli", "--vault", str(tmp_path / "v")]
    subprocess.run([*sb, "init"], check=True, env=env, capture_output=True)
    subprocess.run([*sb, "ingest", str(DEMO / "employees.csv")], check=True, env=env, capture_output=True)
    params = StdioServerParameters(command=sys.executable, args=[*sb[2:], "serve"], env=env)
    params.args = ["-m", "sealbox.cli", "--vault", str(tmp_path / "v"), "serve"]

    async def go() -> None:
        async with Client(params) as client:
            res = await client.call_tool("run_sql", {"sql": "SELECT region, count(*) FROM employees GROUP BY 1"})
            assert json.loads(res.content[0].text)["status"] == "ok"

    anyio.run(go)
    out = subprocess.run([*sb, "audit", "verify"], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "chain intact" in out.stdout
