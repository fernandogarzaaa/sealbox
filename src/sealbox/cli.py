"""``sealbox`` command line: init / ingest / serve / audit / detokenize / datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, crypto
from .audit import verify
from .errors import SealboxError
from .vault import Vault


def _keyfile(args: argparse.Namespace) -> Path | None:
    if args.keyfile:
        return Path(args.keyfile)
    if os.environ.get(crypto.ENV_KEY):
        return None
    if os.environ.get(crypto.ENV_KEYFILE):
        return Path(os.environ[crypto.ENV_KEYFILE])
    return crypto.default_keyfile()


def _keys(args: argparse.Namespace) -> crypto.LocalKeyProvider:
    kf = _keyfile(args)
    if kf is not None:
        os.environ.setdefault(crypto.ENV_KEYFILE, str(kf))
    return crypto.LocalKeyProvider(crypto.load_master_key(kf))


def _vault(args: argparse.Namespace) -> Vault:
    return Vault(Path(args.vault), _keys(args))


def cmd_init(args: argparse.Namespace) -> int:
    kf = _keyfile(args)
    if kf is not None and not kf.exists():
        crypto.write_keyfile(kf)
        print(f"created master keyfile {kf} (mode 600). Back it up: without it the vault is unreadable.")
    vault = Vault.init(Path(args.vault), _keys(args))
    print(f"initialised vault at {vault.root}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    vault = _vault(args)
    for src in args.paths:
        name = vault.ingest(Path(src), name=args.name if len(args.paths) == 1 else None, overwrite=args.overwrite)
        print(f"ingested {src} -> dataset {name!r}; review policy: {vault.policy_path(name)}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    vault = _vault(args)
    for name in vault.names():
        man = vault.manifest(name)
        pol = vault.policy(name)
        tags = ", ".join(f"{c}={t}" for c, t in pol.columns.items())
        print(f"{name:24} {man['kind']:6} k={pol.k_min} dp={'on' if pol.dp.enabled else 'off'}  {tags}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .engine import Engine
    from .server import serve

    keys = _keys(args)
    vault = Vault(Path(args.vault), keys)
    engine = Engine(vault, keys)
    print(
        f"sealbox {__version__}: serving {len(vault.names())} dataset(s) over {args.transport} "
        f"(session {engine.session_id})",
        file=sys.stderr,
    )
    serve(engine, args.transport, args.host, args.port)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    keys = _keys(args)
    vault = Vault(Path(args.vault), keys)
    if args.action == "verify":
        res = verify(vault.audit_path, keys.derive("audit-mac"))
        if res.ok:
            print(f"audit log OK: {res.entries} entries, chain intact")
            return 0
        print(f"audit log TAMPERED: {res.error}", file=sys.stderr)
        return 2
    if not vault.audit_path.exists():
        return 0
    lines = vault.audit_path.read_text().splitlines()[-args.n :]
    for line in lines:
        rec = json.loads(line)
        extra = rec.get("reasons") or rec.get("bytes_released", "")
        print(f"{rec['seq']:>6} {rec['event']:12} {rec.get('tool', ''):16} {rec.get('status', ''):7} {extra}")
    return 0


def cmd_detokenize(args: argparse.Namespace) -> int:
    """Owner-side only: replace <LABEL_xxxxxxxx> tokens with the original values."""
    from .engine import Engine
    from .tokenizer import detokenize_text

    keys = _keys(args)
    vault = Vault(Path(args.vault), keys)
    engine = Engine(vault, keys)
    mapping: dict[str, str] = {}
    for ds in args.dataset or vault.names():
        mapping.update(engine.token_map(ds))
    text = args.text if args.text is not None else sys.stdin.read()
    engine.audit.append("detokenize", session=engine.session_id, datasets=args.dataset or "all", chars=len(text))
    sys.stdout.write(detokenize_text(text, mapping) + ("\n" if args.text is not None else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sealbox", description=__doc__)
    p.add_argument("--version", action="version", version=f"sealbox {__version__}")
    p.add_argument("--vault", default=os.environ.get("SEALBOX_VAULT", "vault"), help="vault directory")
    p.add_argument(
        "--keyfile",
        default=None,
        help="master keyfile (default $SEALBOX_KEYFILE or ~/.config/sealbox/master.key; $SEALBOX_MASTER_KEY wins)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create a vault (and a master keyfile if none exists)").set_defaults(fn=cmd_init)
    ing = sub.add_parser("ingest", help="encrypt CSV/Parquet/JSON files or text/code folders into the vault")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--name")
    ing.add_argument("--overwrite", action="store_true")
    ing.set_defaults(fn=cmd_ingest)
    sub.add_parser("datasets", help="list datasets and their policy tags").set_defaults(fn=cmd_datasets)
    srv = sub.add_parser("serve", help="run the MCP server")
    srv.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    srv.set_defaults(fn=cmd_serve)
    aud = sub.add_parser("audit", help="verify or show the hash-chained audit log")
    aud.add_argument("action", choices=["verify", "tail"])
    aud.add_argument("-n", type=int, default=20)
    aud.set_defaults(fn=cmd_audit)
    det = sub.add_parser("detokenize", help="owner-side: map tokens in TEXT (or stdin) back to raw values")
    det.add_argument("text", nargs="?")
    det.add_argument("--dataset", action="append")
    det.set_defaults(fn=cmd_detokenize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.fn(args)
        return rc
    except SealboxError as exc:
        print(f"sealbox: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
