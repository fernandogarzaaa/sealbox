"""Executed inside the sandbox subprocess (``python -I _runner.py``). Do not import sealbox here.

Reads ``{"code", "tables", "docs", "allow_read"}`` JSON from stdin, runs the agent code with
``tables`` / ``docs`` / ``rows()`` / ``df()`` in scope, and writes ``{"ok", "result", "stdout"}``
JSON to the real stdout. Defense in depth on top of namespaces + rlimits: an audit hook
denies network, process creation, ctypes and file access outside an allowlist.
Audit hooks are NOT a security boundary on their own.
"""

import contextlib
import io
import json
import os
import sys
import traceback
from typing import Any

_REAL_STDOUT = sys.stdout
MAX_STDOUT = 20_000


def _preimport() -> None:
    for mod in (
        "math",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "re",
        "json",
        "datetime",
        "decimal",
        "fractions",
        "random",
        "string",
        "textwrap",
        "difflib",
        "hashlib",
        "base64",
        "ast",
        "tokenize",
        "keyword",
        "operator",
        "heapq",
        "bisect",
    ):
        __import__(mod)
    for opt in ("numpy", "pandas"):
        try:
            __import__(opt)
        except Exception:  # noqa: S112
            continue


_BLOCK_PREFIXES = (
    "socket.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.spawn",
    "ctypes.",
    "os.kill",
    "os.killpg",
    "pty.",
    "winreg.",
    "msvcrt.",
    "sys.remote_exec",
    "os.putenv",
    "os.unsetenv",
    "shutil.rmtree",
    "os.chmod",
    "os.chown",
    "os.link",
    "os.symlink",
    "os.rename",
    "os.remove",
    "os.rmdir",
    "os.truncate",
    "urllib.Request",
    "http.client.connect",
    "ftplib.",
    "smtplib.",
    "poplib.",
    "imaplib.",
    "nntplib.",
    "telnetlib.",
    "webbrowser.",
    "sys.addaudithook",
    "sys.setprofile",
    "sys.settrace",
    "code.__new__",
    "marshal.load",
    "function.__new__",
    "object.__setattr__.__code__",
    "cpython.run_",
    "mmap.__new__",
)
_BLOCK_IMPORTS = frozenset(
    {
        "ctypes",
        "_ctypes",
        "socket",
        "_socket",
        "subprocess",
        "_posixsubprocess",
        "multiprocessing",
        "_multiprocessing",
        "urllib.request",
        "http.client",
        "ssl",
        "_ssl",
        "asyncio",
        "selectors",
        "select",
        "pty",
        "fcntl",
        "resource",
        "signal",
        "mmap",
        "_xxsubinterpreters",
        "_interpreters",
        "concurrent.futures.process",
    }
)


def _install_hook(allow_read: list[str], workdir: str) -> None:
    allowed = tuple(os.path.realpath(p) for p in allow_read)
    workdir = os.path.realpath(workdir)

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith(_BLOCK_PREFIXES):
            raise PermissionError(f"sealbox sandbox: {event} is not allowed")
        if (
            event == "import"
            and args
            and (args[0] in _BLOCK_IMPORTS or str(args[0]).split(".")[0] in ("ctypes", "_ctypes", "multiprocessing"))
        ):
            raise PermissionError(f"sealbox sandbox: import of {args[0]} is not allowed")
        if event == "open" and args:
            path = args[0]
            if isinstance(path, int):
                return
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            real = os.path.realpath(os.fsdecode(path))
            writing = any(c in (mode or "") for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
            if real == workdir or real.startswith(workdir + os.sep):
                return
            if writing:
                raise PermissionError("sealbox sandbox: writes outside the temp dir are not allowed")
            if not real.startswith(allowed):
                raise PermissionError("sealbox sandbox: reading files outside the runtime is not allowed")
        if event in ("os.listdir", "os.scandir") and args:
            p = args[0]
            if p is None or isinstance(p, int):
                return
            real = os.path.realpath(os.fsdecode(p))
            if not (real.startswith(allowed) or real.startswith(workdir)):
                raise PermissionError("sealbox sandbox: listing directories outside the runtime is not allowed")

    sys.addaudithook(hook)


def main() -> None:
    spec = json.loads(sys.stdin.read())
    _preimport()
    tables = spec.get("tables", {})
    docs = spec.get("docs", {})

    def rows(name: str) -> list[dict[str, Any]]:
        t = tables[name]
        return [dict(zip(t["columns"], r, strict=False)) for r in t["rows"]]

    def df(name: str) -> Any:
        import pandas as pd  # pre-imported if installed

        t = tables[name]
        return pd.DataFrame(t["rows"], columns=t["columns"])

    ns: dict[str, Any] = {
        "__name__": "__sealbox__",
        "tables": tables,
        "docs": docs,
        "rows": rows,
        "df": df,
        "result": None,
    }
    for opt in ("numpy", "pandas"):
        if opt in sys.modules:
            ns["np" if opt == "numpy" else "pd"] = sys.modules[opt]
    code = compile(spec["code"], "<agent>", "exec")
    buf = io.StringIO()
    _install_hook(spec.get("allow_read", []), os.getcwd())
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, ns)  # noqa: S102 - this is the sandbox
        out = {"ok": True, "result": ns.get("result"), "stdout": buf.getvalue()[:MAX_STDOUT]}
        text = json.dumps(out, default=str)
    except BaseException as exc:
        lineno = None
        for frame in traceback.extract_tb(exc.__traceback__):
            if frame.filename == "<agent>":
                lineno = frame.lineno
        text = json.dumps(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
                "lineno": lineno,
                "stdout": buf.getvalue()[:MAX_STDOUT],
            },
            default=str,
        )
    _REAL_STDOUT.write("\n__SEALBOX_RESULT__" + text)
    _REAL_STDOUT.flush()


if __name__ == "__main__":
    main()
