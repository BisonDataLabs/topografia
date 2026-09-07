#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "data", "outputs", "__pycache__"}
PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[ps]_[A-Za-z0-9]{30,}"),
    "Eiwa credential": re.compile(r"EIWA_(?:EMAIL|PASSWORD)\s*=\s*['\"][^'\"]+['\"]"),
}


def main() -> None:
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        if path.name == ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {label}")
    if findings:
        raise SystemExit("\n".join(findings))
    print("Sin secretos reconocibles.")


if __name__ == "__main__":
    main()
