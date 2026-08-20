"""Content-addressed provenance checks for saved-output evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
TREE_DIGEST_ALGORITHM = "sha256(sorted(relative_posix_path_NUL_size_NUL_sha256_LF))"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"completion record field {label!r} must be a mapping")
    return value


def _required_text(mapping: Mapping[str, Any], field: str, label: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"completion record field {label}.{field} must be non-empty text")
    return value


def _required_sha256(mapping: Mapping[str, Any], field: str, label: str) -> str:
    value = _required_text(mapping, field, label)
    if not SHA256.fullmatch(value):
        raise ValueError(f"completion record field {label}.{field} is not a lowercase SHA-256")
    return value


def _resolve_record_path(value: str, record_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = record_path.parent / path
    return path.resolve(strict=False)


def tree_digest(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    """Hash a fixed file set using the recovery inference v3 tree algorithm."""
    lexical_root = root.absolute()
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"tree digest root is not a directory: {resolved_root}")
    lines: list[bytes] = []
    relative_names: set[str] = set()
    for input_path in paths:
        lexical_path = input_path if input_path.is_absolute() else input_path.absolute()
        try:
            relative = lexical_path.relative_to(lexical_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"tree member is outside digest root {lexical_root}: {input_path}"
            ) from exc
        if relative in relative_names:
            raise ValueError(f"duplicate relative path in tree digest: {relative}")
        relative_names.add(relative)
        resolved = input_path.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"tree member is not a regular file: {input_path}")
        try:
            relative_bytes = relative.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:  # pragma: no cover - Python paths are valid Unicode
            raise ValueError(f"tree member cannot be encoded as UTF-8: {relative!r}") from exc
        line = (
            relative_bytes
            + b"\0"
            + str(resolved.stat().st_size).encode("ascii")
            + b"\0"
            + sha256_file(resolved).encode("ascii")
            + b"\n"
        )
        lines.append(line)
    manifest = b"".join(sorted(lines))
    digest = hashlib.sha256(manifest).hexdigest()
    return {
        "algorithm": TREE_DIGEST_ALGORITHM,
        "file_count": len(lines),
        "files_manifest_sha256": digest,
        "tree_sha256": digest,
    }


def validate_completion_record(
    *,
    prediction_name: str,
    prediction_directory: Path,
    prediction_suffix: str,
    prediction_paths: Iterable[Path],
    completion_record_path: Path,
) -> dict[str, Any]:
    """Validate one inference v3 completion record against current bytes."""
    record_path = completion_record_path.resolve(strict=True)
    if not record_path.is_file():
        raise FileNotFoundError(f"completion record is not a file: {record_path}")
    try:
        record_bytes = record_path.read_bytes()
        record = json.loads(record_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid completion record JSON {record_path}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"completion record must contain a JSON mapping: {record_path}")
    if record.get("schema_version") != 3:
        raise ValueError(f"completion record must use schema_version 3: {record_path}")
    if record.get("status") != "complete":
        raise ValueError(f"completion record status is not 'complete': {record_path}")

    model = _required_mapping(record.get("model"), "model")
    if _required_text(model, "name", "model") != prediction_name:
        raise ValueError(
            f"completion model name does not match prediction {prediction_name!r}: {record_path}"
        )
    checkpoint = _required_mapping(model.get("checkpoint"), "model.checkpoint")
    checkpoint_path = _resolve_record_path(
        _required_text(checkpoint, "path", "model.checkpoint"), record_path
    )
    checkpoint_sha256 = _required_sha256(checkpoint, "sha256", "model.checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"completion checkpoint is missing: {checkpoint_path}")
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            f"completion checkpoint hash mismatch for {prediction_name}: "
            f"{actual_checkpoint_sha256} != {checkpoint_sha256}"
        )

    config = _required_mapping(record.get("config"), "config")
    config_path = _resolve_record_path(_required_text(config, "path", "config"), record_path)
    config_sha256 = _required_sha256(config, "sha256", "config")
    if not config_path.is_file():
        raise FileNotFoundError(f"completion config is missing: {config_path}")
    actual_config_sha256 = sha256_file(config_path)
    if actual_config_sha256 != config_sha256:
        raise ValueError(
            f"completion config hash mismatch for {prediction_name}: "
            f"{actual_config_sha256} != {config_sha256}"
        )

    outputs = _required_mapping(record.get("outputs"), "outputs")
    output_root = _resolve_record_path(_required_text(outputs, "root", "outputs"), record_path)
    prediction_root = prediction_directory.resolve(strict=True)
    if output_root != prediction_root:
        raise ValueError(
            f"completion output root does not match prediction directory for {prediction_name}: "
            f"{output_root} != {prediction_root}"
        )
    if outputs.get("prediction_suffix") != prediction_suffix:
        raise ValueError(
            f"completion prediction suffix does not match {prediction_name!r}: "
            f"{outputs.get('prediction_suffix')!r} != {prediction_suffix!r}"
        )
    expected_count = outputs.get("file_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 1:
        raise ValueError("completion record field outputs.file_count must be a positive integer")
    expected_manifest = _required_sha256(outputs, "files_manifest_sha256", "outputs")
    expected_tree = _required_sha256(outputs, "tree_sha256", "outputs")
    if expected_manifest != expected_tree:
        raise ValueError(
            "completion record v3 requires outputs.files_manifest_sha256 and "
            "outputs.tree_sha256 to be identical"
        )
    if outputs.get("tree_sha256_algorithm") != TREE_DIGEST_ALGORITHM:
        raise ValueError(
            "completion record outputs.tree_sha256_algorithm does not match the v3 protocol"
        )
    prediction_files = tuple(prediction_paths)
    expected_entries = {path.absolute() for path in prediction_files}
    actual_entries = {path.absolute() for path in prediction_root.iterdir()}
    if actual_entries != expected_entries or any(
        not path.is_file() or path.parent.absolute() != prediction_root.absolute()
        for path in prediction_files
    ):
        unexpected = sorted(str(path) for path in actual_entries - expected_entries)
        missing = sorted(str(path) for path in expected_entries - actual_entries)
        raise ValueError(
            f"completion output root must contain exactly the indexed prediction files for "
            f"{prediction_name}; unexpected={unexpected[:5]}, missing={missing[:5]}"
        )
    actual_tree = tree_digest(prediction_root, prediction_files)
    if actual_tree["file_count"] != expected_count:
        raise ValueError(
            f"completion output count mismatch for {prediction_name}: "
            f"{actual_tree['file_count']} != {expected_count}"
        )
    if actual_tree["tree_sha256"] != expected_tree:
        raise ValueError(
            f"completion output tree hash mismatch for {prediction_name}: "
            f"{actual_tree['tree_sha256']} != {expected_tree}"
        )

    return {
        "status": "verified",
        "record_path": str(record_path),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "schema_version": 3,
        "checkpoint": {
            "path": str(checkpoint_path),
            "declared_sha256": checkpoint_sha256,
            "verified_sha256": actual_checkpoint_sha256,
        },
        "config": {
            "path": str(config_path),
            "declared_sha256": config_sha256,
            "verified_sha256": actual_config_sha256,
        },
        "outputs": {
            "root": str(output_root),
            **actual_tree,
        },
        "completion_record": dict(record),
    }
