from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from core.errors import IsolationError


FORBIDDEN_COMPONENTS = {"test"}
FORBIDDEN_TOKENS = (
    "test_alpha",
    "test_result",
    "test_metrics",
    "test_derived",
)


def assert_validation_safe_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_COMPONENTS:
        raise IsolationError(f"ADAPTIVE_TEST_PATH_DENIED: {path}")
    text = path.as_posix().lower()
    if any(token in text for token in FORBIDDEN_TOKENS):
        raise IsolationError(f"ADAPTIVE_TEST_PATH_DENIED: {path}")


def assert_validation_safe_value(value: Any, location: str = "root") -> None:
    """Recursively reject test-derived markers and test directory references."""
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"test", "test_metrics", "test_result", "test_alpha"}:
                raise IsolationError(f"ADAPTIVE_TEST_VALUE_DENIED: {location}.{key}")
            if key_text == "test_derived" and item is not False:
                raise IsolationError(f"ADAPTIVE_TEST_VALUE_DENIED: {location}.{key}")
            assert_validation_safe_value(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_validation_safe_value(item, f"{location}[{index}]")
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if "/test/" in f"/{normalized.strip('/')}/" or any(token in normalized for token in FORBIDDEN_TOKENS):
            raise IsolationError(f"ADAPTIVE_TEST_VALUE_DENIED: {location}")


class AdaptiveReader:
    """Only read validation-safe files beneath explicit roots."""

    def __init__(self, roots: Iterable[Path]):
        self.roots = tuple(path.resolve() for path in roots)

    def _resolve(self, path: Path) -> Path:
        resolved = path.resolve()
        assert_validation_safe_path(resolved)
        if not any(resolved == root or root in resolved.parents for root in self.roots):
            raise IsolationError(f"ADAPTIVE_PATH_OUTSIDE_ALLOWLIST: {resolved}")
        return resolved

    def read_json(self, path: Path) -> Any:
        resolved = self._resolve(path)
        with resolved.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        assert_validation_safe_value(value)
        return value

    def read_text(self, path: Path) -> str:
        return self._resolve(path).read_text(encoding="utf-8")


_ADAPTIVE_GUARD_ROOTS: set[Path] = set()
_ADAPTIVE_GUARD_INSTALLED = False


def install_adaptive_test_path_guard(run_root: Path) -> None:
    """Use the Python audit boundary to deny every adaptive open under test/."""
    global _ADAPTIVE_GUARD_INSTALLED
    _ADAPTIVE_GUARD_ROOTS.add(run_root.resolve())
    if _ADAPTIVE_GUARD_INSTALLED:
        return

    def audit(event: str, args: tuple) -> None:
        if event != "open" or not args:
            return
        target = args[0]
        if not isinstance(target, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(target).resolve()
        except (OSError, TypeError, ValueError):
            return
        if any(root == path or root in path.parents for root in _ADAPTIVE_GUARD_ROOTS):
            if "test" in {part.lower() for part in path.parts}:
                raise IsolationError(f"ADAPTIVE_TEST_PATH_DENIED: {path}")

    sys.addaudithook(audit)
    _ADAPTIVE_GUARD_INSTALLED = True


def install_researcher_write_guard(test_output_root: Path) -> None:
    """Allow a researcher process to write only beneath its own test directory."""
    allowed = test_output_root.resolve()

    def audit(event: str, args: tuple) -> None:
        if event == "open" and len(args) >= 2:
            target, mode = args[0], args[1]
            writing = isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+"))
            if isinstance(mode, int):
                writing = bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and isinstance(target, (str, bytes, os.PathLike)):
                path = Path(target).resolve()
                if path != allowed and allowed not in path.parents:
                    raise IsolationError(f"RESEARCHER_WRITE_OUTSIDE_TEST_DENIED: {path}")
        elif event in {"os.remove", "os.rename"} and args:
            targets = args[:2] if event == "os.rename" else args[:1]
            for target in targets:
                if isinstance(target, (str, bytes, os.PathLike)):
                    path = Path(target).resolve()
                    if path != allowed and allowed not in path.parents:
                        raise IsolationError(f"RESEARCHER_MUTATION_OUTSIDE_TEST_DENIED: {path}")

    sys.addaudithook(audit)
