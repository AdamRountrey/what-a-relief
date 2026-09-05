#!/usr/bin/env python3
"""Regression test for network-share-compatible Mitsuba progress writes."""

from __future__ import annotations

import errno
import importlib.util
import sys
import tempfile
from pathlib import Path


def load_worker(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("what_a_relief_mitsuba_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Mitsuba worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: test_mitsuba_worker_progress.py <worker.py>", file=sys.stderr)
        return 2

    worker = load_worker(Path(sys.argv[1]).resolve())
    original_atomic_text = worker.atomic_text

    def deny_atomic_replace(*_args, **_kwargs) -> None:
        raise PermissionError(errno.EACCES, "simulated SMB sharing violation")

    with tempfile.TemporaryDirectory(prefix="what-a-relief-progress-test-") as directory:
        path = Path(directory) / "progress.txt"
        progress = worker.Progress(path)
        worker.atomic_text = deny_atomic_replace
        try:
            progress(37, "First network progress update")
            if path.read_text(encoding="utf-8") != "37\nFirst network progress update\n":
                raise RuntimeError("First fallback progress update was incomplete")
            progress(82, "Second network progress update")
            if path.read_text(encoding="utf-8") != "82\nSecond network progress update\n":
                raise RuntimeError("Second fallback progress update was incomplete")
            if path.with_name(path.name + ".part").exists():
                raise RuntimeError("Fallback progress writer retained a partial file")
        finally:
            worker.atomic_text = original_atomic_text

    print("Mitsuba network progress fallback passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
