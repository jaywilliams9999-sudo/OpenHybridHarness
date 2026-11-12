"""Minimal `.env` loader: no dependency, never overrides the real environment."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Set each key in the file that is not already in ``os.environ``.

    Tolerates a UTF-8 BOM, CRLF, ``#`` comments, blank lines and surrounding quotes.
    A missing file is not an error.
    """
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8").lstrip("﻿")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
