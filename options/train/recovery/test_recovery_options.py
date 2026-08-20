#!/usr/bin/env python3
"""Static contract tests for recovery training options; no GPU work."""

import os
from pathlib import Path
import unittest

import yaml

import basicsr.data  # noqa: F401 - imports BasicSR dataset registrations
import basicsr.models  # noqa: F401 - imports BasicSR model registrations
import hat.archs  # noqa: F401 - imports HAT architecture registrations
import hat.data  # noqa: F401 - imports HAT dataset registrations
import hat.models  # noqa: F401 - imports HAT model registrations
from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path('/home/hermes/hat-face-training/data/face_sr_recovery')
BASE_CHECKPOINT = Path(
    '/home/hermes/hat-face-training/HAT/experiments/pretrained_models/HAT-S_SRx4.pth')

CONFIGS = {
    'A': HERE / 'stage_a_clean_fidelity_5k.yml',
    'B': HERE / 'stage_b_mild_reconstruction_10k.yml',
    'C': HERE / 'stage_c_weak_perceptual_5k_OPT_IN.yml',
}


def load_config(stage):
    with CONFIGS[stage].open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


class RecoveryOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = {stage: load_config(stage) for stage in CONFIGS}

    def test_yaml_and_registries(self):
        for stage, opt in self.options.items():
            with self.subTest(stage=stage):
                self.assertIsInstance(opt, dict)
                self.assertIsNotNone(MODEL_REGISTRY.get(opt['model_type']))
                self.assertIsNotNone(ARCH_REGISTRY.get(opt['network_g']['type']))
                for dataset in opt['datasets'].values():
                    self.assertIsNotNone(DATASET_REGISTRY.get(dataset['type']))

    def test_common_bounded_training_contract(self):
        limits = {'A': 5000, 'B': 10000, 'C': 5000}
        reference_network = self.options['A']['network_g']
        for stage, opt in self.options.items():
            with self.subTest(stage=stage):
                self.assertEqual(opt['recovery_contract']['stage'], stage)
                self.assertEqual(opt['train']['total_iter'], limits[stage])
                self.assertLessEqual(opt['val']['val_freq'], 1000)
                self.assertFalse(opt['val']['save_img'])
                self.assertFalse(opt['auto_resume'])
                self.assertEqual(opt['datasets']['train']['gt_size'], 256)
                self.assertEqual(opt['datasets']['train']['batch_size_per_gpu'], 2)
                self.assertFalse(opt['datasets']['train']['use_rot'])
                self.assertGreater(opt['train']['ema_decay'], 0)
                self.assertEqual(opt['path']['param_key_g'], 'params_ema')
                self.assertEqual(opt['network_g'], reference_network)
                self.assertLess(
                    max(opt['train']['scheduler']['milestones']),
                    opt['train']['total_iter'])
                for dataset in opt['datasets'].values():
                    self.assertTrue(dataset['meta_info_file'].endswith('/meta_info.txt'))

    def test_stage_a_is_clean_pixel_only_from_base_ema(self):
        opt = self.options['A']
        self.assertEqual(opt['model_type'], 'HATModel')
        self.assertEqual(Path(opt['path']['pretrain_network_g']), BASE_CHECKPOINT)
        self.assertTrue(BASE_CHECKPOINT.is_file())
        self.assertLessEqual(opt['train']['optim_g']['lr'], 1e-5)
        self.assertEqual(opt['train']['pixel_opt']['type'], 'L1Loss')
        self.assertNotIn('perceptual_opt', opt['train'])
        self.assertNotIn('gan_opt', opt['train'])
        self.assertIn('/clean/train/', opt['datasets']['train']['dataroot_lq'])
        self.assertIn('/clean_pilot/', opt['datasets']['val_clean']['dataroot_lq'])
        self.assertEqual(opt['logger']['save_checkpoint_freq'], opt['val']['val_freq'])

    def test_stage_b_is_gated_pixel_only_with_exact_dataset_composition(self):
        opt = self.options['B']
        self.assertEqual(opt['model_type'], 'HATModel')
        self.assertEqual(opt['recovery_contract']['requires_accepted_checkpoint'], 'stage_a')
        self.assertEqual(opt['recovery_contract']['clean_dataset_fraction'], 0.5)
        self.assertIn('__ACCEPTED_STAGE_A__', opt['path']['pretrain_network_g'])
        self.assertIn('/mild_mixed/train/', opt['datasets']['train']['dataroot_lq'])
        self.assertEqual(opt['train']['pixel_opt']['type'], 'L1Loss')
        self.assertNotIn('perceptual_opt', opt['train'])
        self.assertNotIn('gan_opt', opt['train'])
        self.assertSetEqual(
            set(opt['datasets']), {'train', 'val_clean', 'val_mild', 'val_hard'})

    def test_stage_c_is_explicit_opt_in_and_weak(self):
        opt = self.options['C']
        self.assertEqual(opt['model_type'], 'SRGANModel')
        self.assertFalse(opt['recovery_contract']['enabled_by_default'])
        self.assertEqual(opt['recovery_contract']['requires_accepted_checkpoint'], 'stage_b')
        self.assertEqual(opt['recovery_contract']['clean_dataset_fraction'], 0.5)
        self.assertEqual(opt['recovery_contract']['evaluation_gate_status'], 'blocked')
        self.assertSetEqual(
            set(opt['recovery_contract']['blocked_until']),
            {'pinned_lpips_or_dists', 'blinded_visual_protocol'})
        self.assertIn('__ACCEPTED_STAGE_B__', opt['path']['pretrain_network_g'])
        self.assertLessEqual(opt['train']['optim_g']['lr'], 5e-6)
        self.assertLessEqual(opt['train']['perceptual_opt']['perceptual_weight'], 0.1)
        self.assertLessEqual(opt['train']['gan_opt']['loss_weight'], 0.01)
        self.assertEqual(opt['train']['net_d_init_iters'], 0)

    @unittest.skipUnless(
        os.environ.get('RECOVERY_REQUIRE_DATA') == '1',
        'set RECOVERY_REQUIRE_DATA=1 after deterministic data generation')
    def test_configured_data_directories_exist(self):
        self.assertTrue(DATA_ROOT.is_dir())
        for stage, opt in self.options.items():
            for name, dataset in opt['datasets'].items():
                with self.subTest(stage=stage, dataset=name):
                    self.assertTrue(Path(dataset['dataroot_gt']).is_dir())
                    self.assertTrue(Path(dataset['dataroot_lq']).is_dir())
                    self.assertTrue(Path(dataset['meta_info_file']).is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
