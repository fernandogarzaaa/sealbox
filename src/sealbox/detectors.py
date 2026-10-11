"""PII and prompt-injection detectors (regex core, optional injectguard hook)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .tokenizer import TOKEN_RE


@dataclass(frozen=True)
class Hit:
    kind: str
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}\b")),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    (
        "API_KEY",
        re.compile(
            r"\b(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|"
            r"xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b"
        ),
    ),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----")),
]


def find_pii(text: str) -> list[Hit]:
    hits: list[Hit] = []
    token_spans = [m.span() for m in TOKEN_RE.finditer(text)]
    for kind, pat in _PII_PATTERNS:
        for m in pat.finditer(text):
            if any(s <= m.start() < e for s, e in token_spans):
                continue
            if kind == "CREDIT_CARD":
                digits = re.sub(r"\D", "", m.group(0))
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)) or len(set(digits)) < 2:
                    continue
            hits.append(Hit(kind, m.start(), m.end()))
    hits.extend(_injectguard_pii(text))
    return _dedupe(hits)


def _dedupe(hits: list[Hit]) -> list[Hit]:
    out: list[Hit] = []
    for h in sorted(hits, key=lambda h: (h.start, -(h.end - h.start))):
        if out and h.start < out[-1].end:
            continue
        out.append(h)
    return out


def redact(text: str, hits: list[Hit]) -> str:
    for h in sorted(hits, key=lambda h: h.start, reverse=True):
        text = text[: h.start] + f"[{h.kind}]" + text[h.end :]
    return text


# --------------------------------------------------------------------------- injection

_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts?|rules)",
    r"\bdisregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system)",
    r"\bforget\s+(?:everything|all|your)\s+(?:instructions|rules|above)",
    r"\byou\s+are\s+now\s+(?:in\s+)?(?:developer|dan|jailbreak|god)\s*mode",
    r"\b(?:system|developer)\s*prompt\b",
    r"\bnew\s+instructions?\s*:",
    r"\b(?:assistant|ai|agent|model|claude|gpt)\s*[,:]?\s*(?:you\s+must|please)\s+(?:now\s+)?"
    r"(?:send|email|post|upload|output|print|reveal|exfiltrate|leak|dump)",
    r"\b(?:exfiltrate|leak|dump)\s+(?:all|the|every)\b",
    r"\b(?:send|post|upload|forward)\s+(?:all|the|every|this)\b.{0,60}\b(?:to|at)\s+(?:https?://|\S+@\S+)",
    r"\b(?:output|print|return|reveal)\s+(?:all|every|the\s+(?:raw|full|entire))\s+(?:rows|records|data|table)",
    r"<\|im_start\|>|<\|system\|>|\[/?INST\]|<<SYS>>|</?system>",
    r"\bcall\s+(?:the\s+)?(?:tool|function)\s+\w+\s+with\b",
]
_INJ_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I | re.S)


def find_injection(text: str) -> list[Hit]:
    hits = [Hit("INJECTION", m.start(), m.end()) for m in _INJ_RE.finditer(text)]
    if not hits and _injectguard_flags(text):
        hits.append(Hit("INJECTION", 0, len(text)))
    return hits


# --------------------------------------------------------------------------- injectguard hook

_IG: Any = None
_IG_TRIED = False


def _injectguard() -> Any:
    """Optional: use injectguard (pip install injectguard) when importable."""
    global _IG, _IG_TRIED
    if not _IG_TRIED:
        _IG_TRIED = True
        try:
            import injectguard

            _IG = injectguard.Guard()
        except Exception:
            _IG = None
    return _IG


def injectguard_available() -> bool:
    return _injectguard() is not None


def _injectguard_flags(text: str) -> bool:
    guard = _injectguard()
    if guard is None or len(text) < 12:
        return False
    try:
        return bool(guard.scan_input(text).flagged)
    except Exception:
        return False


def _injectguard_pii(text: str) -> list[Hit]:
    guard = _injectguard()
    if guard is None:
        return []
    try:
        return [Hit(f.category.upper(), *f.span) for f in guard.pii.findings(text) if f.span]
    except Exception:
        return []
