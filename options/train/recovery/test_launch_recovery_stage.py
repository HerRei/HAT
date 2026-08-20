#!/usr/bin/env python3
"""CPU-only stdlib tests for the gated recovery training launcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_recovery_stage as launcher  # noqa: E402


class RecoveryLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / 'candidate.pth'
        self.reference = self.root / 'base.pth'
        state = {'params_ema': {'layer.weight': torch.zeros(2, 3), 'layer.bias': torch.ones(2)}}
        torch.save(state, self.checkpoint)
        torch.save(state, self.reference)
        _, self.signature_sha = launcher._checkpoint_signature(self.checkpoint)
        self.files = {}
        for name in ('report.json', 'clean.json', 'contact.png', 'source.yml', 'prepared.json', 'data.jsonl'):
            path = self.root / name
            path.write_bytes(f'evidence:{name}\n'.encode('ascii'))
            self.files[name] = path

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _record(self) -> dict:
        reviewed = '2026-08-20T16:00:00+02:00'
        return {
            'schema_version': 1,
            'created_at': reviewed,
            'status': 'accepted',
            'source_stage': 'A',
            'checkpoint': {
                'path': str(self.checkpoint),
                'sha256': self._sha(self.checkpoint),
                'param_key': 'params_ema',
                'signature_sha256': self.signature_sha,
            },
            'gate': {
                'name': 'stage_a_clean_gate_v1',
                'passed': True,
                'metrics': [
                    {
                        'name': 'psnr.ci_low',
                        'metric': 'psnr',
                        'statistic': 'ci_low_improvement',
                        'bucket': 'clean',
                        'value': 0.02,
                        'baseline_value': 0.0,
                        'direction': 'higher',
                        'threshold': 0.0,
                        'ci': [0.02, 0.08],
                        'passed': True,
                    }
                ],
                'report_path': str(self.files['report.json']),
                'report_sha256': self._sha(self.files['report.json']),
                'reports': [
                    {
                        'bucket': 'clean',
                        'path': str(self.files['clean.json']),
                        'sha256': self._sha(self.files['clean.json']),
                    }
                ],
            },
            'visual_attestation': {
                'attested': True,
                'reviewer': 'human-reviewer',
                'reviewed_at': reviewed,
                'protocol': 'fixed blinded A/B contact-sheet protocol v1',
                'notes': 'No systematic identity-changing artifacts observed.',
                'contact_sheets': [
                    {
                        'bucket': 'clean',
                        'path': str(self.files['contact.png']),
                        'sha256': self._sha(self.files['contact.png']),
                    }
                ],
            },
            'provenance': {
                'config_path': str(self.files['source.yml']),
                'config_sha256': self._sha(self.files['source.yml']),
                'prepared_manifest_path': str(self.files['prepared.json']),
                'prepared_manifest_sha256': self._sha(self.files['prepared.json']),
                'manifest_paths': [
                    {
                        'kind': 'clean_pilot',
                        'path': str(self.files['data.jsonl']),
                        'sha256': self._sha(self.files['data.jsonl']),
                    }
                ],
            },
        }

    def _write_acceptance(self, record: dict | None = None) -> Path:
        record = self._record() if record is None else record
        path = self.root / 'accepted_checkpoint.json'
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='ascii')
        digest = self._sha(path)
        sidecar = path.with_name(path.name + '.sha256')
        sidecar.write_text(f'{digest}  {path.name}\n', encoding='ascii')
        path.chmod(0o444)
        sidecar.chmod(0o444)
        return path

    def test_valid_acceptance_and_params_ema_signature(self) -> None:
        path = self._write_acceptance()
        record, evidence = launcher.validate_acceptance(path, 'B')
        checkpoint = launcher.validate_checkpoint(record, reference_checkpoint=self.reference)
        self.assertEqual(evidence['acceptance']['sha256'], self._sha(path))
        self.assertEqual(checkpoint['param_key'], 'params_ema')
        self.assertEqual(checkpoint['tensor_count'], 2)

    def test_tampered_evidence_is_rejected(self) -> None:
        path = self._write_acceptance()
        self.files['report.json'].write_text('tampered\n', encoding='ascii')
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'hash mismatch'):
            launcher.validate_acceptance(path, 'B')

    def test_numeric_gate_is_recomputed(self) -> None:
        record = self._record()
        record['gate']['metrics'][0]['value'] = -0.01
        path = self._write_acceptance(record)
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'fails independently'):
            launcher.validate_acceptance(path, 'B')

    def test_missing_human_attestation_is_rejected(self) -> None:
        record = self._record()
        record['visual_attestation']['attested'] = False
        path = self._write_acceptance(record)
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'attested=true'):
            launcher.validate_acceptance(path, 'B')

    def test_wrong_source_stage_is_rejected(self) -> None:
        record = self._record()
        record['source_stage'] = 'B'
        path = self._write_acceptance(record)
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'requires an accepted Stage A'):
            launcher.validate_acceptance(path, 'B')

    def test_writable_acceptance_is_rejected(self) -> None:
        path = self._write_acceptance()
        path.chmod(0o644)
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'no write bits'):
            launcher.validate_acceptance(path, 'B')

    def test_checkpoint_without_params_ema_is_rejected(self) -> None:
        invalid = self.root / 'invalid.pth'
        torch.save({'params': {'weight': torch.zeros(1)}}, invalid)
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'params_ema'):
            launcher._checkpoint_signature(invalid)

    def test_stage_c_remains_machine_blocked(self) -> None:
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'Stage C is blocked'):
            launcher.load_stage_config('C')

    def test_stage_c_requires_perceptual_metric_and_blinded_protocol(self) -> None:
        record = self._record()
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'LPIPS or DISTS'):
            launcher.validate_stage_c_evidence(record)

        record['gate']['metrics'].append(
            {'metric': 'lpips', 'passed': True})
        with self.assertRaisesRegex(launcher.LaunchPreflightError, 'blinded'):
            launcher.validate_stage_c_evidence(record)

    def test_stage_c_accepts_fully_pinned_perceptual_evidence(self) -> None:
        record = self._record()
        record['gate']['metrics'].append({'metric': 'dists', 'passed': True})
        record['visual_attestation']['protocol'] = 'double_blind_fixed_selection_v1'
        record['provenance']['perceptual_metric'] = {
            'name': 'dists',
            'implementation_path': str(self.files['source.yml']),
            'implementation_sha256': self._sha(self.files['source.yml']),
            'weights_path': str(self.files['prepared.json']),
            'weights_sha256': self._sha(self.files['prepared.json']),
        }
        evidence = launcher.validate_stage_c_evidence(record)
        self.assertEqual(evidence['name'], 'dists')

    def test_command_has_only_explicit_checkpoint_override_and_no_resume(self) -> None:
        command = launcher.build_command(Path('/config.yml'), Path('/accepted.pth'))
        self.assertNotIn('--auto_resume', command)
        self.assertIn('path:pretrain_network_g=/accepted.pth', command)
        self.assertEqual(command.count('--force_yml'), 1)

    def test_active_hat_process_scan(self) -> None:
        proc = self.root / 'proc'
        (proc / '123').mkdir(parents=True)
        (proc / '123' / 'cmdline').write_bytes(b'python\0hat/train.py\0-opt\0x.yml\0')
        (proc / 'text').mkdir()
        self.assertEqual(launcher.active_hat_processes(proc)[0]['pid'], 123)

    def test_existing_experiment_path_is_rejected(self) -> None:
        fake_repo = self.root / 'repo'
        experiment = fake_repo / 'experiments' / 'test_experiment'
        experiment.mkdir(parents=True)
        config = {'name': 'test_experiment'}
        with (
            mock.patch.object(launcher, 'REPO_ROOT', fake_repo),
            mock.patch.object(launcher, 'GPU_LOCK', fake_repo / 'gpu.lock'),
            mock.patch.object(launcher, 'active_hat_processes', return_value=[]),
        ):
            with self.assertRaisesRegex(launcher.LaunchPreflightError, 'existing experiment'):
                launcher.check_runtime_conflicts(config)

    def test_launch_requires_confirmation_before_preflight_or_subprocess(self) -> None:
        with (
            mock.patch.object(launcher, 'preflight') as preflight,
            mock.patch.object(launcher.subprocess, 'run') as run,
        ):
            with self.assertRaisesRegex(launcher.LaunchPreflightError, 'confirm-launch'):
                launcher.launch('B', Path('/acceptance.json'), confirmed=False)
            preflight.assert_not_called()
            run.assert_not_called()

    def test_gpu_lock_is_exclusive_and_owner_released(self) -> None:
        lock = self.root / 'active_gpu.lock'
        report = {'stage': 'B', 'experiment_name': 'test'}
        with mock.patch.object(launcher, 'GPU_LOCK', lock):
            descriptor, identity = launcher._acquire_gpu_lock(report)
            try:
                with self.assertRaisesRegex(launcher.LaunchPreflightError, 'already exists'):
                    launcher._acquire_gpu_lock(report)
            finally:
                launcher._release_gpu_lock(descriptor, identity)
        self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
