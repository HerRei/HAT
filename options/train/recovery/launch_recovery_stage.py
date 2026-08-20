#!/usr/bin/env python3
"""Preflight and explicitly launch gated Stage B/C recovery training."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CHECKPOINT = REPO_ROOT / 'experiments/pretrained_models/HAT-S_SRx4.pth'
BASE_CHECKPOINT_SHA256 = 'a92f81bd2c0c1aaa371a6e4d6cac69e749fde2e36196885ee47a4a3667542c9a'
GPU_LOCK = REPO_ROOT / 'results/recovery_pilot_matrix/active_gpu.lock'
LAUNCH_ROOT = REPO_ROOT / 'results/recovery_training_launches'
SHA256 = re.compile(r'^[0-9a-f]{64}$')
SAFE_NAME = re.compile(r'^[a-z0-9][a-z0-9_-]{0,95}$')
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
MAX_IDLE_GPU_BUSY_PERCENT = 10
MAX_IDLE_VRAM_BYTES = 1024 * 1024 * 1024

STAGES = {
    'B': {
        'config': Path(__file__).with_name('stage_b_mild_reconstruction_10k.yml'),
        'source_stage': 'A',
        'sentinel': (
            '/home/hermes/hat-face-training/HAT/experiments/'
            '__ACCEPTED_STAGE_A__/models/net_g_5000.pth'
        ),
    },
    'C': {
        'config': Path(__file__).with_name('stage_c_weak_perceptual_5k_OPT_IN.yml'),
        'source_stage': 'B',
        'sentinel': (
            '/home/hermes/hat-face-training/HAT/experiments/'
            '__ACCEPTED_STAGE_B__/models/net_g_10000.pth'
        ),
    },
}


class LaunchPreflightError(RuntimeError):
    """Raised when a recovery launch safety condition is not satisfied."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: UniqueKeyLoader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise LaunchPreflightError(f'Duplicate YAML key: {key!r}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
    except OSError as error:
        raise LaunchPreflightError(f"Cannot hash '{path}': {error}") from error
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + '\n').encode('ascii')


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchPreflightError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise LaunchPreflightError(f'Non-finite JSON constant is forbidden: {value}')


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding='ascii'),
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LaunchPreflightError(f"Cannot read JSON '{path}': {error}") from error
    if not isinstance(value, dict):
        raise LaunchPreflightError(f"Expected a JSON object in '{path}'")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding='ascii'), Loader=UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise LaunchPreflightError(f"Cannot read YAML '{path}': {error}") from error
    if not isinstance(value, dict):
        raise LaunchPreflightError(f"Expected a YAML mapping in '{path}'")
    return value


def _require_regular_file(path: Path, label: str, read_only: bool = False) -> Path:
    if path.is_symlink():
        raise LaunchPreflightError(f'{label} must not be a symlink: {path}')
    try:
        metadata = path.stat()
    except OSError as error:
        raise LaunchPreflightError(f'{label} is not readable: {path}: {error}') from error
    if not stat.S_ISREG(metadata.st_mode):
        raise LaunchPreflightError(f'{label} must be a regular file: {path}')
    if read_only and metadata.st_mode & WRITE_BITS:
        raise LaunchPreflightError(f'{label} must have no write bits (expected mode 0444): {path}')
    return path


def _absolute_record_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or '://' in value:
        raise LaunchPreflightError(f'{label} must be a nonempty local path')
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise LaunchPreflightError(f'{label} must be absolute: {value!r}')
    return path.resolve(strict=False)


def _required_text(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise LaunchPreflightError(f'{label}.{key} must be nonempty text')
    return item


def _required_sha(value: Mapping[str, Any], key: str, label: str) -> str:
    item = _required_text(value, key, label)
    if not SHA256.fullmatch(item):
        raise LaunchPreflightError(f'{label}.{key} must be lowercase SHA-256')
    return item


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LaunchPreflightError(f'{label} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise LaunchPreflightError(f'{label} must be finite')
    return result


def _aware_timestamp(value: str, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise LaunchPreflightError(f'{label} must be ISO-8601: {value!r}') from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LaunchPreflightError(f'{label} must include a timezone offset')
    return parsed


def _verify_hashed_path(path_value: Any, sha_value: Any, label: str) -> dict[str, Any]:
    path = _absolute_record_path(path_value, f'{label}.path')
    _require_regular_file(path, label)
    if not isinstance(sha_value, str) or not SHA256.fullmatch(sha_value):
        raise LaunchPreflightError(f'{label}.sha256 must be lowercase SHA-256')
    actual = _sha256_file(path)
    if actual != sha_value:
        raise LaunchPreflightError(f'{label} hash mismatch: {actual} != {sha_value}')
    return {'path': str(path), 'sha256': actual}


def _verify_hashed_entry(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchPreflightError(f'{label} must be an object')
    return _verify_hashed_path(value.get('path'), value.get('sha256'), label)


def validate_acceptance(path: Path, target_stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable acceptance file and all of its pinned evidence."""
    path = path.expanduser().resolve(strict=False)
    _require_regular_file(path, 'Acceptance JSON', read_only=True)
    sidecar = path.with_name(path.name + '.sha256')
    _require_regular_file(sidecar, 'Acceptance checksum sidecar', read_only=True)
    acceptance_sha = _sha256_file(path)
    expected_sidecar = f'{acceptance_sha}  {path.name}\n'
    try:
        actual_sidecar = sidecar.read_text(encoding='ascii')
    except (OSError, UnicodeError) as error:
        raise LaunchPreflightError(f"Cannot read acceptance sidecar '{sidecar}': {error}") from error
    if actual_sidecar != expected_sidecar:
        raise LaunchPreflightError('Acceptance checksum sidecar content is not canonical or does not match')

    record = _load_json(path)
    if record.get('schema_version') != 1 or record.get('status') != 'accepted':
        raise LaunchPreflightError('Acceptance must have schema_version 1 and status "accepted"')
    expected_source = STAGES[target_stage]['source_stage']
    if record.get('source_stage') != expected_source:
        raise LaunchPreflightError(
            f'Stage {target_stage} requires an accepted Stage {expected_source} checkpoint')

    checkpoint = record.get('checkpoint')
    if not isinstance(checkpoint, Mapping):
        raise LaunchPreflightError('checkpoint must be an object')
    checkpoint_evidence = _verify_hashed_path(
        checkpoint.get('path'), checkpoint.get('sha256'), 'checkpoint')
    if checkpoint.get('param_key') != 'params_ema':
        raise LaunchPreflightError('checkpoint.param_key must be params_ema')
    _required_sha(checkpoint, 'signature_sha256', 'checkpoint')

    gate = record.get('gate')
    if not isinstance(gate, Mapping) or gate.get('passed') is not True:
        raise LaunchPreflightError('gate must be an object with passed=true')
    _required_text(gate, 'name', 'gate')
    metrics = gate.get('metrics')
    if not isinstance(metrics, list) or not metrics:
        raise LaunchPreflightError('gate.metrics must contain at least one numeric constraint')
    normalized_metrics = []
    for index, metric in enumerate(metrics):
        label = f'gate.metrics[{index}]'
        if not isinstance(metric, Mapping) or metric.get('passed') is not True:
            raise LaunchPreflightError(f'{label} must be an object with passed=true')
        for field in ('name', 'metric', 'statistic', 'bucket'):
            _required_text(metric, field, label)
        if metric.get('direction') != 'higher':
            raise LaunchPreflightError(f'{label}.direction must be "higher"')
        value = _finite_number(metric.get('value'), f'{label}.value')
        baseline = _finite_number(metric.get('baseline_value'), f'{label}.baseline_value')
        threshold = _finite_number(metric.get('threshold'), f'{label}.threshold')
        if value < threshold:
            raise LaunchPreflightError(f'{label} fails independently: {value} < {threshold}')
        if 'ci' not in metric:
            raise LaunchPreflightError(f'{label}.ci must be present (null is allowed)')
        normalized_metrics.append(
            {'name': metric['name'], 'value': value, 'baseline_value': baseline, 'threshold': threshold})

    primary_report = _verify_hashed_path(
        gate.get('report_path'), gate.get('report_sha256'), 'gate primary report')
    reports = gate.get('reports')
    if not isinstance(reports, list) or not reports:
        raise LaunchPreflightError('gate.reports must contain hashed bucket reports')
    report_evidence = []
    for index, report in enumerate(reports):
        label = f'gate.reports[{index}]'
        if not isinstance(report, Mapping):
            raise LaunchPreflightError(f'{label} must be an object')
        _required_text(report, 'bucket', label)
        report_evidence.append(_verify_hashed_entry(report, label))

    visual = record.get('visual_attestation')
    if not isinstance(visual, Mapping) or visual.get('attested') is not True:
        raise LaunchPreflightError('visual_attestation must explicitly have attested=true')
    for field in ('reviewer', 'protocol', 'notes'):
        _required_text(visual, field, 'visual_attestation')
    reviewed_at = _required_text(visual, 'reviewed_at', 'visual_attestation')
    _aware_timestamp(reviewed_at, 'visual_attestation.reviewed_at')
    created_at = _required_text(record, 'created_at', 'acceptance')
    _aware_timestamp(created_at, 'created_at')
    if created_at != reviewed_at:
        raise LaunchPreflightError('created_at must exactly match visual_attestation.reviewed_at')
    contact_sheets = visual.get('contact_sheets')
    if not isinstance(contact_sheets, list) or not contact_sheets:
        raise LaunchPreflightError('visual_attestation.contact_sheets must not be empty')
    contact_evidence = []
    for index, sheet in enumerate(contact_sheets):
        label = f'visual_attestation.contact_sheets[{index}]'
        if not isinstance(sheet, Mapping):
            raise LaunchPreflightError(f'{label} must be an object')
        _required_text(sheet, 'bucket', label)
        contact_evidence.append(_verify_hashed_entry(sheet, label))

    provenance = record.get('provenance')
    if not isinstance(provenance, Mapping):
        raise LaunchPreflightError('provenance must be an object')
    source_config = _verify_hashed_path(
        provenance.get('config_path'), provenance.get('config_sha256'), 'source config')
    prepared_manifest = _verify_hashed_path(
        provenance.get('prepared_manifest_path'),
        provenance.get('prepared_manifest_sha256'),
        'prepared inference manifest',
    )
    manifests = provenance.get('manifest_paths')
    if not isinstance(manifests, list) or not manifests:
        raise LaunchPreflightError('provenance.manifest_paths must not be empty')
    manifest_evidence = []
    for index, manifest in enumerate(manifests):
        label = f'provenance.manifest_paths[{index}]'
        if not isinstance(manifest, Mapping):
            raise LaunchPreflightError(f'{label} must be an object')
        _required_text(manifest, 'kind', label)
        manifest_evidence.append(_verify_hashed_entry(manifest, label))

    evidence = {
        'acceptance': {'path': str(path), 'sha256': acceptance_sha, 'sidecar': str(sidecar)},
        'checkpoint': checkpoint_evidence,
        'metrics': normalized_metrics,
        'primary_report': primary_report,
        'reports': report_evidence,
        'contact_sheets': contact_evidence,
        'source_config': source_config,
        'prepared_manifest': prepared_manifest,
        'manifests': manifest_evidence,
    }
    return record, evidence


def _checkpoint_signature(path: Path, param_key: str = 'params_ema') -> tuple[dict[str, Any], str]:
    try:
        import torch
    except (ImportError, RuntimeError) as error:
        raise LaunchPreflightError('Checkpoint inspection requires the preserved PyTorch environment') from error
    try:
        payload = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as error:
        raise LaunchPreflightError(f"Cannot safely load checkpoint '{path}' on CPU: {error}") from error
    if not isinstance(payload, Mapping) or param_key not in payload:
        raise LaunchPreflightError(f"Checkpoint '{path}' lacks top-level key '{param_key}'")
    state = payload[param_key]
    if not isinstance(state, Mapping) or not state:
        raise LaunchPreflightError(f"Checkpoint '{path}' has an empty or invalid '{param_key}' state")
    signature: dict[str, tuple[list[int], str]] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise LaunchPreflightError(f"Checkpoint '{path}' contains a non-tensor parameter")
        if tensor.device.type != 'cpu':
            raise LaunchPreflightError(f"Checkpoint '{path}' was not mapped entirely to CPU")
        signature[key] = ([int(size) for size in tensor.shape], str(tensor.dtype))
    report = {'tensor_count': len(signature)}
    signature_sha = hashlib.sha256(_canonical_json_bytes(signature)).hexdigest()
    del state
    del payload
    gc.collect()
    return report, signature_sha


def validate_checkpoint(
        record: Mapping[str, Any],
        reference_checkpoint: Path = BASE_CHECKPOINT,
        reference_sha256: str | None = None) -> dict[str, Any]:
    checkpoint = record['checkpoint']
    checkpoint_path = Path(str(checkpoint['path'])).resolve()
    reference_checkpoint = reference_checkpoint.resolve()
    if reference_sha256 is None and reference_checkpoint == BASE_CHECKPOINT.resolve():
        reference_sha256 = BASE_CHECKPOINT_SHA256
    if reference_sha256 is not None:
        actual_reference_sha = _sha256_file(reference_checkpoint)
        if actual_reference_sha != reference_sha256:
            raise LaunchPreflightError(
                f'Canonical HAT-S checkpoint hash mismatch: {actual_reference_sha} != {reference_sha256}')
    candidate_report, candidate_signature = _checkpoint_signature(checkpoint_path)
    reference_report, reference_signature = _checkpoint_signature(reference_checkpoint)
    if candidate_signature != reference_signature:
        raise LaunchPreflightError('Accepted checkpoint params_ema signature differs from canonical HAT-S')
    if checkpoint['signature_sha256'] != candidate_signature:
        raise LaunchPreflightError('Accepted checkpoint signature_sha256 does not match its params_ema state')
    return {
        'path': str(checkpoint_path),
        'sha256': checkpoint['sha256'],
        'param_key': 'params_ema',
        'signature_sha256': candidate_signature,
        'tensor_count': candidate_report['tensor_count'],
        'reference_tensor_count': reference_report['tensor_count'],
    }


def validate_stage_c_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Require pinned perceptual metric and genuinely blinded review evidence."""
    metrics = record.get('gate', {}).get('metrics', [])
    perceptual_names = {
        str(metric.get('metric', '')).lower()
        for metric in metrics
        if isinstance(metric, Mapping)
    } & {'lpips', 'dists'}
    if not perceptual_names:
        raise LaunchPreflightError('Stage C requires a passed LPIPS or DISTS numeric gate')
    visual = record.get('visual_attestation', {})
    protocol = str(visual.get('protocol', ''))
    if 'blind' not in protocol.lower():
        raise LaunchPreflightError('Stage C requires a pinned blinded visual-review protocol')
    provenance = record.get('provenance', {})
    metric_pin = provenance.get('perceptual_metric')
    if not isinstance(metric_pin, Mapping):
        raise LaunchPreflightError('Stage C requires provenance.perceptual_metric pins')
    metric_name = _required_text(metric_pin, 'name', 'provenance.perceptual_metric').lower()
    if metric_name not in perceptual_names:
        raise LaunchPreflightError('Pinned perceptual metric does not match the Stage C gate metric')
    implementation = _verify_hashed_path(
        metric_pin.get('implementation_path'),
        metric_pin.get('implementation_sha256'),
        'perceptual metric implementation',
    )
    weights = _verify_hashed_path(
        metric_pin.get('weights_path'),
        metric_pin.get('weights_sha256'),
        'perceptual metric weights',
    )
    return {'name': metric_name, 'implementation': implementation, 'weights': weights, 'protocol': protocol}


def load_stage_config(stage: str) -> tuple[Path, dict[str, Any]]:
    definition = STAGES[stage]
    path = Path(definition['config']).resolve()
    config = _load_yaml(path)
    contract = config.get('recovery_contract')
    if not isinstance(contract, Mapping) or contract.get('stage') != stage:
        raise LaunchPreflightError(f'Stage {stage} config recovery contract is invalid')
    if stage == 'C' and contract.get('evaluation_gate_status') != 'ready':
        blockers = contract.get('blocked_until', [])
        raise LaunchPreflightError(
            'Stage C is blocked until pinned LPIPS/DISTS and a blinded visual protocol exist: '
            + ', '.join(str(item) for item in blockers))
    if config.get('auto_resume') is not False or config.get('path', {}).get('resume_state') is not None:
        raise LaunchPreflightError('Recovery configs must disable auto-resume and have no resume state')
    actual_pretrain = config.get('path', {}).get('pretrain_network_g')
    if actual_pretrain != definition['sentinel']:
        raise LaunchPreflightError(
            f'Stage {stage} config must retain its checked-in checkpoint sentinel')
    name = config.get('name')
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
        raise LaunchPreflightError(f'Stage {stage} experiment name is not safe: {name!r}')
    for dataset_name, dataset in config.get('datasets', {}).items():
        if not isinstance(dataset, Mapping) or not dataset.get('meta_info_file'):
            raise LaunchPreflightError(f'Dataset {dataset_name!r} must use a generated meta_info_file')
    return path, config


def validate_datasets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for name, dataset in config['datasets'].items():
        gt_root = Path(str(dataset['dataroot_gt'])).resolve(strict=False)
        lq_root = Path(str(dataset['dataroot_lq'])).resolve(strict=False)
        if not gt_root.is_dir() or not lq_root.is_dir():
            raise LaunchPreflightError(f'Dataset {name!r} GT/LQ roots are not materialized')
        meta_path = Path(str(dataset['meta_info_file'])).resolve(strict=False)
        _require_regular_file(meta_path, f'Dataset {name!r} meta-info')
        try:
            lines = meta_path.read_text(encoding='ascii').splitlines()
        except (OSError, UnicodeError) as error:
            raise LaunchPreflightError(f"Cannot read meta-info '{meta_path}': {error}") from error
        if not lines or any(not line.strip() for line in lines) or len(lines) != len(set(lines)):
            raise LaunchPreflightError(f'Dataset {name!r} meta-info is empty, blank, or duplicated')
        expected_count = 130000 if name == 'train' else 512
        if len(lines) != expected_count:
            raise LaunchPreflightError(
                f'Dataset {name!r} must have {expected_count} pinned entries, found {len(lines)}')
        reports.append(
            {'name': name, 'meta_info_file': str(meta_path), 'sha256': _sha256_file(meta_path), 'count': len(lines)})
    return reports


def active_hat_processes(proc_root: Path = Path('/proc')) -> list[dict[str, Any]]:
    active = []
    own_pid = os.getpid()
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise LaunchPreflightError(f"Cannot inspect process table '{proc_root}': {error}") from error
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
        except (OSError, PermissionError):
            continue
        if 'hat/train.py' in command or 'hat/test.py' in command:
            active.append({'pid': int(entry.name), 'command': command})
    return sorted(active, key=lambda item: item['pid'])


def _target_gpu_snapshot(sysfs_root: Path = Path('/sys/class/drm')) -> dict[str, int] | None:
    devices = []
    for busy_path in sysfs_root.glob('card*/device/gpu_busy_percent'):
        device = busy_path.parent
        used_path = device / 'mem_info_vram_used'
        total_path = device / 'mem_info_vram_total'
        try:
            devices.append(
                {
                    'busy_percent': int(busy_path.read_text().strip()),
                    'vram_used': int(used_path.read_text().strip()),
                    'vram_total': int(total_path.read_text().strip()),
                })
        except (OSError, ValueError):
            continue
    return max(devices, key=lambda item: item['vram_total']) if devices else None


def _kfd_users() -> list[int]:
    fuser = shutil.which('fuser')
    if fuser is None or not Path('/dev/kfd').exists():
        return []
    completed = subprocess.run(
        [fuser, '/dev/kfd'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    pids = {int(value) for value in re.findall(r'\b\d+\b', completed.stdout + completed.stderr)}
    pids.discard(os.getpid())
    return sorted(pids)


def check_runtime_conflicts(config: Mapping[str, Any]) -> dict[str, Any]:
    if GPU_LOCK.exists() or GPU_LOCK.is_symlink():
        raise LaunchPreflightError(f'Global recovery GPU lock already exists: {GPU_LOCK}')
    active = active_hat_processes()
    if active:
        raise LaunchPreflightError(f'Active HAT process(es) detected: {active!r}')
    experiment = REPO_ROOT / 'experiments' / str(config['name'])
    tensorboard = REPO_ROOT / 'tb_logger' / str(config['name'])
    for label, path in (('experiment', experiment), ('TensorBoard', tensorboard)):
        if path.exists() or path.is_symlink():
            raise LaunchPreflightError(f'Refusing existing {label} path: {path}')
    kfd_users = _kfd_users()
    if kfd_users:
        raise LaunchPreflightError(f'Other process(es) currently hold /dev/kfd: {kfd_users}')
    gpu = _target_gpu_snapshot()
    if gpu is not None:
        if gpu['busy_percent'] > MAX_IDLE_GPU_BUSY_PERCENT:
            raise LaunchPreflightError(f"GPU is busy ({gpu['busy_percent']}%)")
        if gpu['vram_used'] > MAX_IDLE_VRAM_BYTES:
            raise LaunchPreflightError(f"GPU VRAM is already in use ({gpu['vram_used']} bytes)")
    return {
        'active_hat_processes': [],
        'experiment_path': str(experiment),
        'tensorboard_path': str(tensorboard),
        'gpu': gpu,
    }


def build_command(config_path: Path, checkpoint_path: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / 'hat/train.py'),
        '-opt',
        str(config_path),
        '--force_yml',
        f'path:pretrain_network_g={checkpoint_path}',
    ]


def preflight(stage: str, acceptance_path: Path, check_runtime: bool = True) -> dict[str, Any]:
    config_path, config = load_stage_config(stage)
    record, acceptance = validate_acceptance(acceptance_path, stage)
    stage_c_evidence = validate_stage_c_evidence(record) if stage == 'C' else None
    checkpoint = validate_checkpoint(record)
    datasets = validate_datasets(config)
    runtime = check_runtime_conflicts(config) if check_runtime else None
    command = build_command(config_path, Path(checkpoint['path']))
    if '--auto_resume' in command or config['path']['pretrain_network_g'] == checkpoint['path']:
        raise LaunchPreflightError('Internal launch command/sentinel invariant failed')
    return {
        'stage': stage,
        'experiment_name': config['name'],
        'target_config': {'path': str(config_path), 'sha256': _sha256_file(config_path)},
        'acceptance': acceptance,
        'checkpoint': checkpoint,
        'datasets': datasets,
        'stage_c_evidence': stage_c_evidence,
        'runtime': runtime,
        'command': command,
    }


def _acquire_gpu_lock(report: Mapping[str, Any]) -> tuple[int, tuple[int, int]]:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(
        {
            'pid': os.getpid(),
            'operation': 'recovery_training',
            'stage': report['stage'],
            'experiment': report['experiment_name'],
        })
    try:
        descriptor = os.open(GPU_LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise LaunchPreflightError(f'Global recovery GPU lock already exists: {GPU_LOCK}') from error
    os.write(descriptor, content)
    os.fsync(descriptor)
    metadata = os.fstat(descriptor)
    return descriptor, (metadata.st_dev, metadata.st_ino)


def _release_gpu_lock(descriptor: int, identity: tuple[int, int]) -> None:
    try:
        metadata = GPU_LOCK.stat()
        if (metadata.st_dev, metadata.st_ino) == identity:
            GPU_LOCK.unlink()
    finally:
        os.close(descriptor)


def _write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        if path.exists() or path.is_symlink():
            raise LaunchPreflightError(f'Launch evidence path already exists: {path}')
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise LaunchPreflightError(f'Launch evidence path already exists: {path}') from error
    finally:
        temporary.unlink(missing_ok=True)


def launch(stage: str, acceptance_path: Path, confirmed: bool) -> int:
    if not confirmed:
        raise LaunchPreflightError('Launching requires the explicit --confirm-launch flag')
    initial = preflight(stage, acceptance_path, check_runtime=True)
    descriptor, identity = _acquire_gpu_lock(initial)
    try:
        # Revalidate hashes, paths, process state, and GPU state while holding the shared lock.
        report = preflight(stage, acceptance_path, check_runtime=False)
        active = active_hat_processes()
        if active:
            raise LaunchPreflightError(f'Active HAT process appeared before launch: {active!r}')
        runtime = check_runtime_conflicts_without_lock(report['experiment_name'])
        report['runtime'] = runtime
        launch_id = hashlib.sha256(
            (report['acceptance']['acceptance']['sha256'] + report['target_config']['sha256']).encode('ascii')
        ).hexdigest()[:20]
        record_path = LAUNCH_ROOT / f"{report['experiment_name']}_{launch_id}.launch.json"
        _write_exclusive_json(record_path, report)
        print(json.dumps({'launch_record': str(record_path), 'command': report['command']}, indent=2), flush=True)
        experiment = Path(runtime['experiment_path'])
        tensorboard = Path(runtime['tensorboard_path'])
        archived_before = {
            'experiment': _archived_siblings(experiment),
            'tensorboard': _archived_siblings(tensorboard),
        }
        completed = subprocess.run(report['command'], cwd=REPO_ROOT, check=False)
        archived_after = {
            'experiment': _archived_siblings(experiment),
            'tensorboard': _archived_siblings(tensorboard),
        }
        archived_created = {
            key: sorted(set(archived_after[key]) - set(archived_before[key]))
            for key in archived_before
        }
        completion_path = record_path.with_suffix('.completion.json')
        _write_exclusive_json(
            completion_path,
            {
                'launch_record': str(record_path),
                'launch_record_sha256': _sha256_file(record_path),
                'exit_code': completed.returncode,
                'archived_paths_created': archived_created,
            })
        if any(archived_created.values()):
            raise LaunchPreflightError(
                'BasicSR archived a path created during launch, proving a race; preserve all evidence')
        if completed.returncode != 0:
            raise LaunchPreflightError(
                f'HAT training exited with code {completed.returncode}; preserve all partial '
                'evidence and use a new experiment name')
        return 0
    finally:
        _release_gpu_lock(descriptor, identity)


def check_runtime_conflicts_without_lock(experiment_name: str) -> dict[str, Any]:
    """Repeat runtime checks while the caller owns the shared GPU lock."""
    active = active_hat_processes()
    if active:
        raise LaunchPreflightError(f'Active HAT process(es) detected: {active!r}')
    experiment = REPO_ROOT / 'experiments' / experiment_name
    tensorboard = REPO_ROOT / 'tb_logger' / experiment_name
    for label, path in (('experiment', experiment), ('TensorBoard', tensorboard)):
        if path.exists() or path.is_symlink():
            raise LaunchPreflightError(f'Refusing existing {label} path: {path}')
    kfd_users = _kfd_users()
    if kfd_users:
        raise LaunchPreflightError(f'Other process(es) currently hold /dev/kfd: {kfd_users}')
    gpu = _target_gpu_snapshot()
    if gpu is not None and (
            gpu['busy_percent'] > MAX_IDLE_GPU_BUSY_PERCENT or gpu['vram_used'] > MAX_IDLE_VRAM_BYTES):
        raise LaunchPreflightError(f'GPU is not idle: {gpu!r}')
    return {
        'active_hat_processes': [],
        'experiment_path': str(experiment),
        'tensorboard_path': str(tensorboard),
        'gpu': gpu,
    }


def _archived_siblings(path: Path) -> list[str]:
    if not path.parent.is_dir():
        return []
    prefix = f'{path.name}_archived_'
    return [str(item) for item in sorted(path.parent.iterdir()) if item.name.startswith(prefix)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('check', 'launch'):
        child = subparsers.add_parser(command)
        child.add_argument('--stage', choices=sorted(STAGES), required=True)
        child.add_argument('--acceptance', type=Path, required=True)
        if command == 'launch':
            child.add_argument('--confirm-launch', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == 'check':
            print(json.dumps(preflight(args.stage, args.acceptance), indent=2, sort_keys=True))
            return 0
        return launch(args.stage, args.acceptance, args.confirm_launch)
    except LaunchPreflightError as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
