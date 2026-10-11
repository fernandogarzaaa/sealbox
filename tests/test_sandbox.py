from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sealbox import sandbox
from sealbox.sandbox import isolation_capabilities, pad_latency, run_python

CAPS = isolation_capabilities()
needs_netns = pytest.mark.skipif(not CAPS["netns"], reason="user/network namespaces unavailable")


def run(code: str, **kw: object) -> sandbox.SandboxResult:
    return run_python(
        code, {"t": {"columns": ["a"], "rows": [[1], [2]]}}, {"docs": {"x.py": "print(1)"}}, timeout_s=10, **kw
    )  # type: ignore[arg-type]


def test_result_and_helpers() -> None:
    r = run("print('hi')\nresult = {'sum': sum(x['a'] for x in rows('t')), 'docs': list(docs['docs'])}")
    assert r.ok and r.result == {"sum": 3, "docs": ["x.py"]} and r.stdout == "hi\n"
    assert "rlimits" in r.isolation and "audit-hook" in r.isolation


def test_exception_reports_type_and_line() -> None:
    r = run("x = 1\ny = 2\nraise ValueError('boom')")
    assert not r.ok and r.error_type == "ValueError" and r.lineno == 3 and r.message == "boom"


def test_environment_is_empty_and_master_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEALBOX_MASTER_KEY", "super-secret-value")
    r = run("import os\nresult = dict(os.environ)")
    assert r.ok and "SEALBOX_MASTER_KEY" not in r.result and "super-secret-value" not in str(r.result)


@pytest.mark.parametrize(
    "code",
    [
        "import socket",
        "import subprocess",
        "import ctypes",
        "import os\nos.system('true')",
        "import os\nos.fork()",
        "open('/etc/passwd').read()",
        "open('/tmp/../etc/hostname').read()",
        "import os\nos.listdir('/etc')",
        "open('/usr/lib/evil', 'w')",
        "import os\nos.chmod('/tmp', 0o777)",
        "import os\nos.remove('/tmp/x')",
    ],
)
def test_blocked_operations(code: str) -> None:
    r = run(code)
    assert not r.ok and r.error_type in ("PermissionError", "FileNotFoundError", "OSError")


@needs_netns
def test_network_namespace_blocks_network_even_without_hook() -> None:
    r = run(
        "import _socket\ns = _socket.socket()\ns.settimeout(2)\ntry:\n    s.connect(('1.1.1.1', 80))\n"
        "    result = 'connected'\nexcept OSError as e:\n    result = 'blocked'"
    )
    assert not r.ok or r.result == "blocked"
    assert "netns" in r.isolation


@needs_netns
def test_mount_namespace_hides_paths(tmp_path: Path) -> None:
    if not CAPS["mountns"]:
        pytest.skip("mount namespaces unavailable")
    secret_dir = tmp_path / "keys"
    secret_dir.mkdir()
    (secret_dir / "master.key").write_text("TOPSECRET")
    r = run_python(
        "import os\nresult = os.path.exists({!r})".format(str(secret_dir / "master.key")),
        {},
        {},
        timeout_s=10,
        hide_paths=[secret_dir],
    )
    assert r.ok and r.result is False and "mountns" in r.isolation


def test_writes_inside_workdir_allowed() -> None:
    r = run("open('scratch.txt', 'w').write('x')\nresult = open('scratch.txt').read()")
    assert r.ok and r.result == "x"


def test_timeout_and_cpu_limit() -> None:
    t0 = time.monotonic()
    r = run_python("while True:\n    pass", {}, {}, timeout_s=2)
    assert not r.ok and (r.timed_out or r.error_type == "SandboxExit")
    assert time.monotonic() - t0 < 8


def test_memory_limit() -> None:
    r = run_python("x = bytearray(900 * 1024 * 1024)\nresult = len(x)", {}, {}, timeout_s=10, memory_mb=256)
    assert not r.ok and r.error_type in ("MemoryError", "SandboxExit")


def test_protocol_and_exit_errors() -> None:
    r = run("import os\nos._exit(3)")
    assert not r.ok and r.error_type == "SandboxExit" and "status 3" in r.message
    r = run(
        "import sys\nsys.__stdout__.write('\\n__SEALBOX_RESULT__{not json')\nsys.__stdout__.flush()\n"
        "import os\nos._exit(0)"
    )
    assert not r.ok and r.error_type == "ProtocolError"


def test_require_network_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "isolation_capabilities", lambda: {"netns": False, "mountns": False})
    r = run_python("result = 1", {}, {}, require_network_isolation=True)
    assert not r.ok and r.error_type == "SandboxUnavailable"
    r = run_python("result = 1", {}, {})
    assert r.ok and r.result == 1 and "netns" not in r.isolation and "WARNING" in r.warnings[0]


def test_fallback_without_mountns(monkeypatch: pytest.MonkeyPatch) -> None:
    if not CAPS["netns"]:
        pytest.skip("no netns")
    monkeypatch.setattr(sandbox, "isolation_capabilities", lambda: {"netns": True, "mountns": False})
    r = run_python("result = 2", {}, {})
    assert r.ok and r.result == 2 and "mountns" not in r.isolation and "mount namespace" in r.warnings[0]


def test_probe_cache_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox._ISOLATION_CACHE.pop("definitely-not-a-binary-xyz", None)
    assert sandbox._probe(["definitely-not-a-binary-xyz"]) is False
    assert sandbox._ISOLATION_CACHE["definitely-not-a-binary-xyz"] is False
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    assert isolation_capabilities() == {"netns": False, "mountns": False}


def test_pad_latency() -> None:
    t0 = time.monotonic()
    pad_latency(t0, 0.2)
    assert 0.18 <= time.monotonic() - t0 < 0.4
    t1 = time.monotonic()
    pad_latency(t1, 0)
    assert time.monotonic() - t1 < 0.05


def test_contains() -> None:
    assert sandbox._contains("/a", "/a/b")
    assert not sandbox._contains("/a/b", "/a")
    assert os.path.exists(sandbox.RUNNER)
