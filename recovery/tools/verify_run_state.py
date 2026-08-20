#!/usr/bin/env python3
"""Verify preserved experiment artifacts against the recovery state file."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = DEFAULT_ROOT / "recovery" / "run_state.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_artifacts(
    value: Any, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Find every state mapping that explicitly declares path plus SHA-256."""
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield ".".join(prefix) or "artifact", value
        for key, child in value.items():
            yield from _declared_artifacts(child, (*prefix, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _declared_artifacts(child, (*prefix, str(index)))


def _artifact_checks(state: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, artifact in _declared_artifacts(state):
        declared_path = artifact["path"]
        expected = artifact["sha256"]
        if not isinstance(declared_path, str) or not declared_path:
            results.append(
                {
                    "kind": "artifact",
                    "name": name,
                    "path": str(declared_path),
                    "expected_sha256": expected,
                    "actual_sha256": None,
                    "ok": False,
                    "error": "declared path is not non-empty text",
                }
            )
            continue
        path = Path(declared_path).expanduser()
        if not path.is_absolute():
            path = root / path
        valid_digest = (
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
        )
        actual = sha256_file(path) if path.is_file() else None
        results.append(
            {
                "kind": "artifact",
                "name": name,
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "ok": valid_digest and actual == expected,
                **({"error": "declared SHA-256 is invalid"} if not valid_digest else {}),
            }
        )
    return results


def _environment_checks(state: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    environment = state.get("environment")
    if not isinstance(environment, Mapping):
        return []
    checks: list[dict[str, Any]] = []
    version_sources: dict[str, tuple[str | None, str]] = {
        "python": (None, ""),
        "pytorch": ("torch", "__version__"),
        "torchvision": ("torchvision", "__version__"),
        "basicsr": ("basicsr", "__version__"),
        "facexlib": ("facexlib", "__version__"),
        "numpy": ("numpy", "__version__"),
        "opencv": ("cv2", "__version__"),
    }
    for key, (module_name, attribute) in version_sources.items():
        expected = environment.get(key)
        if not isinstance(expected, str):
            continue
        try:
            if module_name is None:
                actual = platform.python_version()
            else:
                module = importlib.import_module(module_name)
                actual = str(getattr(module, attribute, "unknown"))
            error = None
        except Exception as exc:  # pragma: no cover - depends on external environment damage
            actual = None
            error = f"{type(exc).__name__}: {exc}"
        checks.append(
            {
                "kind": "environment",
                "name": f"environment.{key}",
                "expected": expected,
                "actual": actual,
                "ok": actual == expected,
                **({"error": error} if error is not None else {}),
            }
        )

    expected_executable = environment.get("python_executable")
    if isinstance(expected_executable, str):
        checks.append(
            {
                "kind": "environment",
                "name": "environment.python_executable",
                "expected": expected_executable,
                "actual": sys.executable,
                "ok": Path(sys.executable).absolute() == Path(expected_executable).absolute(),
            }
        )

    pip_freeze = environment.get("pip_freeze")
    if isinstance(pip_freeze, Mapping):
        expected = pip_freeze.get("sha256")
        command = pip_freeze.get("command")
        if (
            isinstance(expected, str)
            and isinstance(command, list)
            and command
            and all(isinstance(item, str) and item for item in command)
        ):
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                actual = hashlib.sha256(completed.stdout).hexdigest()
                ok = completed.returncode == 0 and actual == expected
                error = completed.stderr.decode("utf-8", "replace").strip() or None
            except OSError as exc:
                actual = None
                ok = False
                error = str(exc)
            checks.append(
                {
                    "kind": "environment",
                    "name": "environment.pip_freeze.current_output",
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "ok": ok,
                    **({"error": error} if error else {}),
                }
            )

    expected_commit = state.get("repository_commit")
    if isinstance(expected_commit, str):
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        actual_commit = completed.stdout.strip() if completed.returncode == 0 else None
        checks.append(
            {
                "kind": "environment",
                "name": "repository_commit",
                "expected": expected_commit,
                "actual": actual_commit,
                "ok": actual_commit == expected_commit,
                **(
                    {"error": completed.stderr.strip()}
                    if completed.returncode != 0
                    else {}
                ),
            }
        )
    return checks


def verify_state(state_path: Path, root: Path) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, Mapping):
        raise ValueError("run state must be a JSON mapping")
    return [*_artifact_checks(state, root), *_environment_checks(state, root)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = verify_state(args.state.resolve(), args.root.resolve())
    if args.json:
        print(json.dumps({"checks": results}, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "OK" if result["ok"] else "FAIL"
            location = result.get("path", result.get("actual", ""))
            print(f"{status:4} {result['name']}: {location}")
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
