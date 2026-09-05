#!/usr/bin/env python3
"""Regression test for remote-safe Mitsuba preview PNG writes."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_worker(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("what_a_relief_mitsuba_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Mitsuba worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBitmap:
    def __init__(self) -> None:
        self.written_path: Path | None = None

    def write(self, path: str) -> None:
        self.written_path = Path(path)
        self.written_path.write_bytes(b"encoded-by-mitsuba")


class FakeDecodedBitmap:
    def __array__(self, dtype=None, copy=None):
        values = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        if dtype is not None:
            values = values.astype(dtype)
        return values.copy() if copy else values


class FakeMitsuba:
    def __init__(self) -> None:
        self.read_path: Path | None = None

    def Bitmap(self, path: str):
        self.read_path = Path(path)
        if self.read_path.read_bytes() != b"encoded-by-mitsuba":
            raise RuntimeError("Mitsuba proxy received the wrong input bytes")
        return FakeDecodedBitmap()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: test_mitsuba_worker_png.py <worker.py>", file=sys.stderr)
        return 2

    worker = load_worker(Path(sys.argv[1]).resolve())
    if worker.os.name == "nt" and not worker._is_remote_path(
        Path(r"\\server\share\inverse.work\inverse_height.png")
    ):
        raise RuntimeError("UNC destination was not recognized as remote")
    original_remote_check = worker._is_remote_path
    with tempfile.TemporaryDirectory(prefix="what-a-relief-png-test-") as directory:
        destination = Path(directory) / "remote" / "preview.png"
        bitmap = FakeBitmap()
        fake_mitsuba = FakeMitsuba()
        worker._is_remote_path = lambda _path: True
        try:
            worker._write_bitmap(destination, bitmap)
            decoded = worker.bitmap_array(fake_mitsuba, destination)
        finally:
            worker._is_remote_path = original_remote_check
        if destination.read_bytes() != b"encoded-by-mitsuba":
            raise RuntimeError("Completed Mitsuba bitmap was not copied to its destination")
        if bitmap.written_path is None or bitmap.written_path == destination:
            raise RuntimeError("Mitsuba was given the remote destination instead of a local path")
        if fake_mitsuba.read_path is None or fake_mitsuba.read_path == destination:
            raise RuntimeError("Mitsuba was asked to read the remote destination directly")
        if decoded.shape != (2, 2) or decoded[0, 1] != 1.0:
            raise RuntimeError("Locally proxied bitmap data was decoded incorrectly")
        if list(destination.parent.glob("*.part")):
            raise RuntimeError("Remote-safe preview writer retained a partial file")

    print("Remote-safe Mitsuba preview PNG writer passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
