from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from recovery.data import build_recovery_data as data_tool


class RecoveryDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train_root = self.root / "source" / "train" / "gt"
        self.val_root = self.root / "source" / "val" / "gt"
        self.train_root.mkdir(parents=True)
        self.val_root.mkdir(parents=True)
        self._write_sources(self.train_root, 0, 2)
        self._write_sources(self.val_root, 10, 4)
        self.output_root = self.root / "output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_sources(root: Path, start: int, count: int) -> None:
        np, cv2, _, _, _ = data_tool._dependencies()
        yy, xx = np.mgrid[0:32, 0:32]
        for index in range(start, start + count):
            image = np.stack(
                (
                    (xx * 7 + index * 13) % 256,
                    (yy * 9 + index * 17) % 256,
                    ((xx + yy) * 5 + index * 19) % 256,
                ),
                axis=2,
            ).astype(np.uint8)
            ok = cv2.imwrite(str(root / f"{index:05d}.png"), image)
            if not ok:
                raise AssertionError("test image write failed")

    def _config(self, **overrides: object) -> data_tool.BuildConfig:
        values = {
            "train_gt_root": self.train_root,
            "val_gt_root": self.val_root,
            "output_root": self.output_root,
            "targets": data_tool.TARGETS,
            "seed": 1234,
            "pilot_seed": 5678,
            "pilot_size": 2,
            "workers": 1,
            "repair": False,
            "dry_run": False,
            "expected_train_count": 2,
            "expected_val_count": 4,
        }
        values.update(overrides)
        return data_tool.BuildConfig(**values)

    @staticmethod
    def _tree_state(root: Path) -> dict[str, tuple[str, object, int]]:
        state = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                state[relative] = ("link", os.readlink(path), path.lstat().st_mtime_ns)
            elif path.is_file():
                state[relative] = ("file", data_tool._sha256_file(path), path.stat().st_mtime_ns)
            else:
                state[relative] = ("dir", 0, path.stat().st_mtime_ns)
        return state

    def test_full_build_exact_clean_mix_pilots_verify_and_idempotence(self) -> None:
        data_tool.build_recovery_data(self._config())

        clean_manifest = data_tool._load_manifest(
            self.output_root / "manifests" / "clean_train.jsonl"
        )
        mixed_manifest = data_tool._load_manifest(
            self.output_root / "manifests" / "mild_mixed_train.jsonl"
        )
        self.assertEqual(len(clean_manifest), 2)
        self.assertEqual(len(mixed_manifest), 4)
        self.assertEqual([record["recipe"] for record in mixed_manifest], ["clean", "mild", "clean", "mild"])
        self.assertEqual(
            (self.output_root / "clean" / "train" / "meta_info.txt").read_text(encoding="ascii"),
            "00000.png (32,32,3)\n00001.png (32,32,3)\n",
        )
        self.assertTrue((self.output_root / data_tool.CONTRACT_FILENAME).is_file())

        source, _ = data_tool._read_image(self.train_root / "00000.png")
        expected = data_tool.matlab_bicubic_x4(source)
        actual, _ = data_tool._read_image(self.output_root / "clean" / "train" / "lq" / "00000.png")
        np, _, _, _, _ = data_tool._dependencies()
        self.assertTrue(np.array_equal(actual, expected))

        clean_gt = self.output_root / "mild_mixed" / "train" / "gt" / "00000_clean.png"
        mild_gt = self.output_root / "mild_mixed" / "train" / "gt" / "00000_mild.png"
        clean_lq = self.output_root / "mild_mixed" / "train" / "lq" / "00000_clean.png"
        self.assertTrue(clean_gt.is_symlink())
        self.assertTrue(mild_gt.is_symlink())
        self.assertTrue(clean_lq.is_symlink())
        self.assertTrue(os.path.samefile(clean_gt, self.train_root / "00000.png"))

        pilot_sets = []
        for recipe in ("clean", "mild", "hard"):
            manifest = data_tool._load_manifest(
                self.output_root / "manifests" / f"{recipe}_pilot.jsonl"
            )
            self.assertEqual(len(manifest), 2)
            pilot_sets.append([record["id"] for record in manifest])
            self.assertTrue(all(Path(record["gt_path"]).is_symlink() for record in manifest))
            self.assertTrue(all(Path(record["lq_path"]).is_symlink() for record in manifest))
        self.assertEqual(pilot_sets[0], pilot_sets[1])
        self.assertEqual(pilot_sets[1], pilot_sets[2])

        data_tool.verify_manifests(self.output_root, recompute=True, workers=1)
        before = self._tree_state(self.output_root)
        data_tool.build_recovery_data(self._config())
        after = self._tree_state(self.output_root)
        self.assertEqual(before, after)

        missing_lq = self.output_root / "hard" / "val" / "lq" / "00010.png"
        missing_lq.unlink()
        data_tool.build_recovery_data(self._config())
        self.assertTrue(missing_lq.is_file())
        data_tool.verify_manifests(self.output_root, recompute=True, workers=1)

    def test_seeded_outputs_match_across_roots(self) -> None:
        data_tool.build_recovery_data(self._config(targets=("benchmarks",)))
        first_records = {
            recipe: data_tool._load_manifest(
                self.output_root / "manifests" / f"{recipe}_val.jsonl"
            )
            for recipe in ("mild", "hard")
        }
        second_root = self.root / "second_output"
        data_tool.build_recovery_data(
            self._config(output_root=second_root, targets=("benchmarks",), workers=2)
        )
        for recipe in ("mild", "hard"):
            second = data_tool._load_manifest(second_root / "manifests" / f"{recipe}_val.jsonl")
            first_signature = [
                (record["id"], record["seed"], record["parameters"], record["lq_sha256"])
                for record in first_records[recipe]
            ]
            second_signature = [
                (record["id"], record["seed"], record["parameters"], record["lq_sha256"])
                for record in second
            ]
            self.assertEqual(first_signature, second_signature)

    def test_dry_run_leakage_guard_and_repair(self) -> None:
        dry_root = self.root / "dry_output"
        data_tool.build_recovery_data(self._config(output_root=dry_root, dry_run=True))
        self.assertFalse(dry_root.exists())

        data_tool.build_recovery_data(self._config(targets=("clean",)))
        lq_path = self.output_root / "clean" / "train" / "lq" / "00000.png"
        lq_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "--repair"):
            data_tool.build_recovery_data(self._config(targets=("clean",)))
        data_tool.build_recovery_data(self._config(targets=("clean",), repair=True))
        data_tool.verify_manifests(self.output_root, recompute=True, workers=1)

        duplicate_val = self.root / "duplicate_val"
        duplicate_val.mkdir()
        for source in sorted(self.val_root.iterdir())[:3]:
            (duplicate_val / source.name).write_bytes(source.read_bytes())
        shutil_source = self.train_root / "00000.png"
        (duplicate_val / "99999.png").write_bytes(shutil_source.read_bytes())
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "potential leakage"):
            data_tool.audit_source_split(
                self._config(val_gt_root=duplicate_val, expected_val_count=4)
            )

        pixel_duplicate_val = self.root / "pixel_duplicate_val"
        pixel_duplicate_val.mkdir()
        for source in sorted(self.val_root.iterdir())[:3]:
            (pixel_duplicate_val / source.name).write_bytes(source.read_bytes())
        np, cv2, _, _, _ = data_tool._dependencies()
        duplicate_pixels, _ = data_tool._read_image(self.train_root / "00000.png")
        reencoded_path = pixel_duplicate_val / "99999.png"
        self.assertTrue(
            cv2.imwrite(
                str(reencoded_path),
                duplicate_pixels,
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )
        )
        self.assertNotEqual(
            data_tool._sha256_file(reencoded_path),
            data_tool._sha256_file(self.train_root / "00000.png"),
        )
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "Exact decoded pixels"):
            data_tool.audit_source_split(
                self._config(val_gt_root=pixel_duplicate_val, expected_val_count=4)
            )

    def test_immutable_contract_rejects_changed_pilot_settings_without_writes(self) -> None:
        data_tool.build_recovery_data(self._config())
        before = self._tree_state(self.output_root)

        with self.assertRaisesRegex(data_tool.RecoveryDataError, "contract mismatch"):
            data_tool.build_recovery_data(self._config(pilot_seed=5679))
        self.assertEqual(before, self._tree_state(self.output_root))

        with self.assertRaisesRegex(data_tool.RecoveryDataError, "contract mismatch"):
            data_tool.build_recovery_data(self._config(pilot_size=1))
        self.assertEqual(before, self._tree_state(self.output_root))

    def test_completed_legacy_build_is_adopted_by_same_default_rerun(self) -> None:
        data_tool.build_recovery_data(self._config())
        (self.output_root / data_tool.CONTRACT_FILENAME).unlink()
        for meta_info in self.output_root.rglob("meta_info.txt"):
            meta_info.unlink()

        before = self._tree_state(self.output_root)
        data_tool.build_recovery_data(self._config(dry_run=True))
        self.assertEqual(before, self._tree_state(self.output_root))

        with self.assertRaisesRegex(data_tool.RecoveryDataError, "Legacy pilot size"):
            data_tool.build_recovery_data(self._config(pilot_size=1))
        self.assertEqual(before, self._tree_state(self.output_root))

        data_tool.build_recovery_data(self._config())
        self.assertTrue((self.output_root / data_tool.CONTRACT_FILENAME).is_file())
        self.assertTrue((self.output_root / "clean" / "train" / "meta_info.txt").is_file())
        data_tool.verify_manifests(self.output_root, recompute=True, workers=1)

    def test_builder_and_verifier_reject_stale_members_and_meta_changes(self) -> None:
        data_tool.build_recovery_data(self._config())
        stale_gt = (
            self.output_root
            / "benchmarks"
            / "clean_pilot"
            / "gt"
            / "stale.png"
        )
        os.symlink(self.val_root / "00010.png", stale_gt)
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "Unexpected member"):
            data_tool.build_recovery_data(self._config())
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "Unexpected member"):
            data_tool.verify_manifests(self.output_root, recompute=False, workers=1)
        stale_gt.unlink()

        stale_lq = self.output_root / "clean" / "train" / "lq" / "stale.png"
        stale_lq.write_bytes(b"not a managed image")
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "Unexpected member"):
            data_tool.verify_manifests(self.output_root, recompute=False, workers=1)
        stale_lq.unlink()

        meta_info = self.output_root / "clean" / "train" / "meta_info.txt"
        meta_info.write_text("stale.png (32,32,3)\n", encoding="ascii")
        with self.assertRaisesRegex(data_tool.RecoveryDataError, "meta-info"):
            data_tool.verify_manifests(self.output_root, recompute=False, workers=1)


if __name__ == "__main__":
    unittest.main()
