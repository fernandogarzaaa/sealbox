"""Run agent-written Python in an isolated subprocess.

Layers (strongest first, each degrades with a clear warning when unavailable):

1. ``unshare -r -n -m``: new user + network namespace (no interfaces but a downed
   loopback, so no network at all) and a private mount namespace in which the vault
   directory, the keyfile directory, ``$HOME`` and ``/proc`` are hidden under empty tmpfs.
2. rlimits: CPU seconds, address space, file size, open files, no new processes.
3. Wall-clock timeout, empty environment (the master key never reaches the child),
   ``python -I`` (no site/user paths from env), throwaway temp working directory.
4. In-process audit hook (network/process/ctypes/file allowlist) - defense in depth only.

For hostile multi-tenant use, run sealbox itself inside gVisor/Firecracker/nsjail or a TEE.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).with_name("_runner.py")
MARKER = "\n__SEALBOX_RESULT__"


@dataclass
class SandboxResult:
    ok: bool
    result: Any = None
    stdout: str = ""
    error_type: str | None = None
    message: str = ""
    lineno: int | None = None
    timed_out: bool = False
    isolation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_ISOLATION_CACHE: dict[str, bool] = {}


def _probe(args: list[str]) -> bool:
    key = " ".join(args)
    if key not in _ISOLATION_CACHE:
        try:
            r = subprocess.run(args, capture_output=True, timeout=10)
            _ISOLATION_CACHE[key] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _ISOLATION_CACHE[key] = False
    return _ISOLATION_CACHE[key]


def isolation_capabilities() -> dict[str, bool]:
    unshare = shutil.which("unshare")
    if not unshare or not sys.platform.startswith("linux"):
        return {"netns": False, "mountns": False}
    netns = _probe([unshare, "-r", "-n", "true"])
    mountns = netns and _probe([unshare, "-r", "-n", "-m", "sh", "-c", "mount -t tmpfs none /tmp && true"])
    return {"netns": netns, "mountns": mountns}


def _allowed_read_paths() -> list[str]:
    paths = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            paths.add(p)
    paths.update(p for p in sys.path if p and os.path.isdir(p))
    paths.update(
        {
            "/usr/lib",
            "/usr/lib64",
            "/lib",
            "/lib64",
            "/usr/share/zoneinfo",
            "/etc/localtime",
            "/dev/null",
            "/dev/urandom",
            "/usr/local/lib",
        }
    )
    return sorted(paths)


def _limits(cpu_s: int, mem_mb: int) -> Any:
    def apply() -> None:  # runs in the child before exec
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    return apply


def run_python(
    code: str,
    tables: dict[str, Any],
    docs: dict[str, Any],
    *,
    timeout_s: float = 20.0,
    memory_mb: int = 1024,
    hide_paths: list[Path] | None = None,
    require_network_isolation: bool = False,
) -> SandboxResult:
    caps = isolation_capabilities()
    warnings: list[str] = []
    isolation = ["rlimits", "timeout", "empty-env", "tempdir", "audit-hook"]
    if require_network_isolation and not caps["netns"]:
        return SandboxResult(
            ok=False,
            error_type="SandboxUnavailable",
            message="network namespace isolation is required by policy but unavailable",
            warnings=["unshare -rn not permitted on this host"],
        )
    workdir = tempfile.mkdtemp(prefix="sealbox-run-")
    os.chmod(workdir, 0o700)
    py = [sys.executable, "-I", "-B", str(RUNNER)]
    cmd: list[str]
    if caps["netns"]:
        unshare = shutil.which("unshare") or "unshare"
        isolation.insert(0, "netns")
        if caps["mountns"]:
            isolation.insert(1, "mountns")
            hide = [str(Path(p).resolve()) for p in (hide_paths or [])] + [str(Path.home()), "/proc"]
            hide = [
                h
                for h in dict.fromkeys(hide)
                if os.path.isdir(h)
                and not _contains(h, sys.prefix)
                and not _contains(h, str(RUNNER))
                and not _contains(h, workdir)
            ]
            script = (
                'for d in "$@"; do [ "$d" = "--" ] && break; mount -t tmpfs -o size=1m,mode=000 none "$d" '
                '|| exit 97; shift; done; shift; exec "$@"'
            )
            cmd = [unshare, "-r", "-n", "-m", "sh", "-c", script, "sh", *hide, "--", *py]
        else:
            cmd = [unshare, "-r", "-n", *py]
            warnings.append("mount namespace unavailable: vault/key paths are hidden only by the audit hook")
    else:
        cmd = py
        warnings.append(
            "WARNING: network namespace isolation unavailable on this host (unshare -rn failed); "
            "agent code is contained only by rlimits + an audit hook, which is NOT a security "
            "boundary. Set session.require_network_isolation: true to refuse instead."
        )
    spec = json.dumps({"code": code, "tables": tables, "docs": docs, "allow_read": _allowed_read_paths()}, default=str)
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        proc = subprocess.run(
            cmd,
            input=spec,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=workdir,
            env=env,
            preexec_fn=_limits(max(1, int(timeout_s)), memory_mb),
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False,
            error_type="Timeout",
            message=f"exceeded {timeout_s}s wall-clock limit",
            timed_out=True,
            isolation=isolation,
            warnings=warnings,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode == 97:
        return SandboxResult(
            ok=False,
            error_type="SandboxSetupError",
            message="could not set up mount isolation",
            isolation=isolation,
            warnings=warnings,
        )
    out = proc.stdout
    pos = out.rfind(MARKER)
    if pos < 0:
        sig = -proc.returncode if proc.returncode < 0 else None
        reason = {9: "killed (CPU limit or memory)", 24: "CPU time limit exceeded", 11: "crashed"}.get(
            sig or 0, f"exited with status {proc.returncode}"
        )
        return SandboxResult(
            ok=False,
            error_type="SandboxExit",
            message=f"sandboxed process {reason}",
            isolation=isolation,
            warnings=warnings,
        )
    try:
        data = json.loads(out[pos + len(MARKER) :])
    except ValueError:
        return SandboxResult(
            ok=False,
            error_type="ProtocolError",
            message="malformed sandbox output",
            isolation=isolation,
            warnings=warnings,
        )
    return SandboxResult(
        ok=bool(data.get("ok")),
        result=data.get("result"),
        stdout=str(data.get("stdout", "")),
        error_type=data.get("error_type"),
        message=str(data.get("message", "")),
        lineno=data.get("lineno"),
        isolation=isolation,
        warnings=warnings,
    )


def _contains(parent: str, child: str) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def pad_latency(started: float, quantum: float) -> None:
    """Sleep so total latency is a multiple of ``quantum`` (blunts timing side channels)."""
    if quantum <= 0:
        return
    elapsed = time.monotonic() - started
    target = (int(elapsed / quantum) + 1) * quantum
    time.sleep(max(0.0, target - elapsed))
