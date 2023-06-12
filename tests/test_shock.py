import codecs
from pathlib import Path

import pytest

import hybrid_jp.shock as shk


def b64e(s: str) -> str:
    return codecs.encode(bytes(s, "utf-8"), "base64").decode().strip()[:-1]


def test_load_not_real():
    fake_path = Path(f"./{b64e('test/path')}")
    with pytest.raises(FileNotFoundError):
        shk.load(fake_path)
