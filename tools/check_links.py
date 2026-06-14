#!/usr/bin/env python3
"""
check_links.py — Validate internal markdown links across the repo.

Usage:
    python tools/check_links.py

Reports broken internal links (relative paths that don't exist on disk).
External links (https://) are not checked.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
MARKDOWN_LINK = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

broken = []

for md_file in ROOT.rglob("*.md"):
    content = md_file.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK.finditer(content):
        text, target = match.group(1), match.group(2)
        # Skip external URLs and anchors
        if target.startswith("http") or target.startswith("#"):
            continue
        # Strip anchor from path
        path_part = target.split("#")[0]
        if not path_part:
            continue
        resolved = (md_file.parent / path_part).resolve()
        if not resolved.exists():
            broken.append({
                "file": str(md_file.relative_to(ROOT)),
                "link_text": text,
                "target": target,
            })

if broken:
    print(f"\n❌ Found {len(broken)} broken internal link(s):\n")
    for b in broken:
        print(f"  File : {b['file']}")
        print(f"  Text : {b['link_text']}")
        print(f"  Target: {b['target']}")
        print()
    raise SystemExit(1)
else:
    print(f"✅ All internal links are valid!")
