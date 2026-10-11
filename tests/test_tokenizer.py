from __future__ import annotations

import pytest

from sealbox.tokenizer import TOKEN_RE, Tokenizer, detokenize_text, label_for


def test_tokens_deterministic_and_keyed() -> None:
    t1, t2 = Tokenizer(b"k" * 32), Tokenizer(b"j" * 32)
    a = t1.token("name", "Alice")
    assert a == t1.token("name", "Alice")
    assert a != t1.token("name", "Bob")
    assert a != t2.token("name", "Alice")
    assert a is not None and TOKEN_RE.fullmatch(a) and a.startswith("<PERSON_")
    assert len(a) == len("<PERSON_>") + 8
    assert t1.token("name", None) is None


def test_label_scoping() -> None:
    t = Tokenizer(b"k" * 32)
    assert t.token("email", "x") != t.token("name", "x")


@pytest.mark.parametrize(
    ("col", "label"),
    [
        ("email", "EMAIL"),
        ("phone_number", "PHONE"),
        ("home_address", "ADDRESS"),
        ("customer_name", "PERSON"),
        ("src_ip", "IP"),
        ("iban", "ACCOUNT"),
        ("sku", "SKU"),
        ("9lives", "V9LIVES"),
        ("$$$", "VALUE"),
    ],
)
def test_label_for(col: str, label: str) -> None:
    assert label_for(col) == label


def test_short_key_rejected() -> None:
    with pytest.raises(ValueError):
        Tokenizer(b"short")


def test_detokenize() -> None:
    t = Tokenizer(b"k" * 32)
    m = t.build_map("name", ["Alice", "Bob", None])
    assert len(m) == 2
    tok = t.token("name", "Alice")
    assert detokenize_text(f"top: {tok} and <PERSON_deadbeef>", m) == "top: Alice and <PERSON_deadbeef>"
