from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from recovery.eval import provenance as eval_provenance
from recovery.inference import pilot_matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    content = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    metadata = {
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "record_count": len(records),
        "schema_version": 1,
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="ascii"
    )


def _fake_provenance(final_directory: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    content = b"fake-pip-freeze\n"
    entry = {
        "path": str(final_directory / "pip_freeze.txt"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    return (
        {
            "schema_version": 1,
            "git": {"head": "0" * 40},
            "source": {"runner": {"sha256": "runner"}},
            "runtime": {"torch": "test", "gpu_probe": {"device_count": 1}},
            "pip_freeze": {"path": entry["path"], "sha256": entry["sha256"]},
            "evidence_files": {"pip_freeze.txt": entry},
        },
        {"pip_freeze.txt": content},
    )
class CanonicalInferenceMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix, cls.template = pilot_matrix.load_canonical()

    def test_canonical_matrix_has_expected_cells(self) -> None:
        model_ids = [model["id"] for model in self.matrix["models"]]
        dataset_ids = [dataset["id"] for dataset in self.matrix["datasets"]]
        self.assertEqual(
            model_ids,
            [
                "base",
                "face85k",
                "face95k",
                "face125k",
                "face130k",
                "interp_a0p1",
                "interp_a0p25",
                "interp_a0p5",
            ],
        )
        self.assertEqual(dataset_ids, ["clean", "mild", "hard"])
        self.assertEqual(len({(model, dataset) for model in model_ids for dataset in dataset_ids}), 24)
        self.assertEqual(self.matrix["checkpoint_param_key"], "params_ema")

    def test_base_clean_materialization_excludes_other_buckets(self) -> None:
        model = self.matrix["models"][0]
        dataset = self.matrix["datasets"][0]
        config = pilot_matrix.materialize_config(self.template, "unit_v1", model, dataset)
        rendered = yaml.safe_dump(config, sort_keys=False)

        self.assertEqual(list(config["datasets"]), ["test_clean"])
        self.assertEqual(
            config["datasets"]["test_clean"]["name"], "face_recovery_clean_pilot"
        )
        self.assertNotIn("face_recovery_mild_pilot", rendered)
        self.assertNotIn("face_recovery_hard_pilot", rendered)
        self.assertNotIn("/mild_pilot/", rendered)
        self.assertNotIn("/hard_pilot/", rendered)
        self.assertEqual(config["path"]["param_key_g"], "params_ema")
        self.assertTrue(config["val"]["save_img"])
        self.assertEqual(config["val"]["suffix"], "base")
        self.assertNotIn("auto_resume", set(pilot_matrix._walk_keys(config)))
        self.assertNotIn("resume_state", set(pilot_matrix._walk_keys(config)))

    def test_each_cell_has_a_unique_experiment_and_result_path(self) -> None:
        experiments = set()
        results = set()
        for model in self.matrix["models"]:
            for dataset in self.matrix["datasets"]:
                config = pilot_matrix.materialize_config(
                    self.template, "unit_v1", model, dataset
                )
                experiments.add(config["name"])
                results.add(
                    str(Path(config["path"]["results_root"]) / config["name"])
                )
        self.assertEqual(len(experiments), 24)
        self.assertEqual(len(results), 24)

    def test_subset_prepare_writes_one_machine_readable_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orchestration = root / "orchestration"
            outputs = root / "outputs"
            data_report = {
                "clean": {"snapshot": {"sha256": "data"}, "ids": ["face_a"]},
                "pilot_selection": {
                    "snapshot": {"sha256": "selection"},
                    "ids": ["face_a"],
                },
            }
            checkpoint_report = {
                "base": {
                    "path": "/checkpoint/base.pth",
                    "sha256": "checkpoint",
                    "param_key": "params_ema",
                }
            }
            with (
                mock.patch.object(pilot_matrix, "ORCHESTRATION_ROOT", orchestration),
                mock.patch.object(pilot_matrix, "OUTPUT_ROOT", outputs),
                mock.patch.object(pilot_matrix, "preflight_data", return_value=data_report),
                mock.patch.object(
                    pilot_matrix, "preflight_model_specs", return_value=checkpoint_report
                ),
                mock.patch.object(
                    pilot_matrix, "_collect_provenance", side_effect=_fake_provenance
                ),
            ):
                workspace = pilot_matrix.prepare(
                    "unit_subset", ["base"], ["clean"]
                )
            manifest = json.loads(
                (workspace / "run_manifest.json").read_text(encoding="ascii")
            )
            self.assertEqual(manifest["selection"]["models"], ["base"])
            self.assertEqual(manifest["selection"]["datasets"], ["clean"])
            self.assertEqual(list(manifest["configs"]), ["base__clean"])
            config_path = Path(manifest["configs"]["base__clean"]["path"])
            config = yaml.safe_load(config_path.read_text(encoding="ascii"))
            self.assertEqual(list(config["datasets"]), ["test_clean"])
            self.assertFalse(manifest["configs"]["base__clean"]["auto_resume"])
            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue((workspace / "run_manifest.sha256").is_file())
            self.assertEqual(manifest["provenance"]["git"]["head"], "0" * 40)

    def test_candidate_prepare_freezes_base_and_candidate_without_matrix_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "net_g_5000.pth"
            candidate.write_bytes(b"candidate checkpoint")
            candidate_hash = _sha256(candidate)
            matrix_hash_before = _sha256(pilot_matrix.MATRIX_PATH)
            data_report = {
                "clean": {"snapshot": {"sha256": "data"}, "ids": ["face_a"]},
                "pilot_selection": {
                    "snapshot": {"sha256": "selection"},
                    "ids": ["face_a"],
                },
            }

            def fake_checkpoint_report(
                _matrix: dict[str, object], specs: list[dict[str, str]]
            ) -> dict[str, object]:
                return {
                    spec["id"]: {
                        "path": spec["checkpoint"],
                        "sha256": spec["sha256"],
                        "param_key": "params_ema",
                        "signature_sha256": "signature",
                        "tensor_count": 1,
                        "source": spec["source"],
                    }
                    for spec in specs
                }

            orchestration = root / "orchestration"
            outputs = root / "outputs"
            with (
                mock.patch.object(pilot_matrix, "ORCHESTRATION_ROOT", orchestration),
                mock.patch.object(pilot_matrix, "OUTPUT_ROOT", outputs),
                mock.patch.object(pilot_matrix, "preflight_data", return_value=data_report),
                mock.patch.object(
                    pilot_matrix,
                    "preflight_model_specs",
                    side_effect=fake_checkpoint_report,
                ),
                mock.patch.object(
                    pilot_matrix, "_collect_provenance", side_effect=_fake_provenance
                ),
            ):
                workspace = pilot_matrix.prepare_candidate(
                    "stagea_review_v1",
                    "stagea_5k",
                    str(candidate),
                    candidate_hash,
                    ["clean"],
                )
                validated = pilot_matrix._prepared_manifest("stagea_review_v1")

            self.assertEqual(validated["mode"], "explicit_candidate")
            self.assertEqual(validated["selection"]["models"], ["base", "stagea_5k"])
            self.assertEqual(
                set(validated["configs"]), {"base__clean", "stagea_5k__clean"}
            )
            self.assertEqual(validated["models"][1]["checkpoint"], str(candidate))
            self.assertEqual(validated["models"][1]["sha256"], candidate_hash)
            candidate_config = yaml.safe_load(
                Path(validated["configs"]["stagea_5k__clean"]["path"]).read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(
                candidate_config["path"]["pretrain_network_g"], str(candidate)
            )
            self.assertEqual(list(candidate_config["datasets"]), ["test_clean"])
            self.assertEqual(_sha256(pilot_matrix.MATRIX_PATH), matrix_hash_before)

    def test_candidate_requires_safe_noncanonical_id_and_explicit_hash(self) -> None:
        with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "collides"):
            pilot_matrix._candidate_model_specs(
                self.matrix, "base", "/tmp/model.pth", "0" * 64
            )
        with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "64-digit"):
            pilot_matrix._candidate_model_specs(
                self.matrix, "stagea_5k", "/tmp/model.pth", "guess-me"
            )

    def test_gpu_run_requires_explicit_confirmation_before_preflight(self) -> None:
        with mock.patch.object(
            pilot_matrix, "_prepared_manifest", side_effect=AssertionError("must not preflight")
        ):
            with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "confirm-gpu-run"):
                pilot_matrix.run_prepared(
                    "unit_v1", ["base"], ["clean"], pilot_matrix.DEFAULT_PYTHON, False
                )

    def test_all_selector_must_not_be_mixed_with_ids(self) -> None:
        self.assertEqual(
            pilot_matrix._selected_ids(["base", "face95k"], ["all"], "model"),
            ["base", "face95k"],
        )
        with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "alone"):
            pilot_matrix._selected_ids(["base", "face95k"], ["all", "base"], "model")


class DataAndOutputPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.ids = ["face_a", "face_b"]
        source_root = self.root / "source"
        source_root.mkdir()
        self.sources = {}
        for image_id in self.ids:
            source = source_root / f"{image_id}.png"
            source.write_bytes(f"gt-{image_id}".encode("ascii"))
            self.sources[image_id] = source

        datasets = []
        for dataset_id in ("clean", "mild", "hard"):
            relative_root = Path("benchmarks") / f"{dataset_id}_pilot"
            dataset_root = self.data_root / relative_root
            gt_root, lq_root = dataset_root / "gt", dataset_root / "lq"
            gt_root.mkdir(parents=True)
            lq_root.mkdir()
            records = []
            for image_id in self.ids:
                gt_path = gt_root / f"{image_id}.png"
                lq_path = lq_root / f"{image_id}.png"
                os.symlink(self.sources[image_id], gt_path)
                lq_path.write_bytes(f"{dataset_id}-lq-{image_id}".encode("ascii"))
                records.append(
                    {
                        "bucket": f"{dataset_id}_pilot",
                        "gt_path": str(gt_path),
                        "gt_sha256": _sha256(self.sources[image_id]),
                        "gt_size": [32, 32],
                        "id": image_id,
                        "lq_path": str(lq_path),
                        "lq_sha256": _sha256(lq_path),
                        "lq_size": [8, 8],
                        "sample_id": image_id,
                        "scale": 4,
                        "source_gt_path": str(self.sources[image_id]),
                    }
                )
            manifest = Path("manifests") / f"{dataset_id}_pilot.jsonl"
            _write_manifest(self.data_root / manifest, records)
            datasets.append(
                {
                    "id": dataset_id,
                    "name": f"face_recovery_{dataset_id}_pilot",
                    "root": str(relative_root),
                    "manifest": str(manifest),
                }
            )
        selection = [
            {
                "id": image_id,
                "source_gt_path": str(self.sources[image_id]),
                "source_sha256": _sha256(self.sources[image_id]),
            }
            for image_id in self.ids
        ]
        _write_manifest(self.data_root / "manifests" / "pilot_selection.jsonl", selection)
        self.matrix = {
            "data_root": str(self.data_root),
            "expected_count_per_dataset": 2,
            "datasets": datasets,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_subset_preflight_reads_only_selected_bucket(self) -> None:
        report = pilot_matrix.preflight_data(self.matrix, {"clean"})
        self.assertEqual(set(report), {"clean", "pilot_selection"})
        self.assertEqual(report["clean"]["ids"], self.ids)

        missing_hard = self.data_root / "benchmarks" / "hard_pilot" / "lq"
        for path in missing_hard.iterdir():
            path.unlink()
        missing_hard.rmdir()
        report = pilot_matrix.preflight_data(self.matrix, {"clean"})
        self.assertEqual(report["clean"]["ids"], self.ids)

    def test_preflight_rejects_tampered_lq(self) -> None:
        lq = self.data_root / "benchmarks" / "clean_pilot" / "lq" / "face_a.png"
        lq.write_bytes(b"tampered")
        with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "hash mismatch"):
            pilot_matrix.preflight_data(self.matrix, {"clean"})

    def test_saved_output_suffix_matches_evaluator_contract(self) -> None:
        output_root = self.root / "outputs"
        with mock.patch.object(pilot_matrix, "OUTPUT_ROOT", output_root):
            dataset = self.matrix["datasets"][0]
            output_dir = (
                pilot_matrix.result_dir("unit_v1", "base", "clean")
                / "visualization"
                / dataset["name"]
            )
            output_dir.mkdir(parents=True)
            for image_id in self.ids:
                (output_dir / f"{image_id}_base.png").write_bytes(
                    f"sr-{image_id}".encode("ascii")
                )
            report = pilot_matrix._validate_saved_outputs(
                self.matrix,
                "unit_v1",
                "base",
                "clean",
                {"clean": {"ids": self.ids}},
            )
        self.assertEqual(report["file_count"], 2)
        self.assertEqual(report["prediction_suffix"], "_base")
        self.assertEqual(report["files_manifest_sha256"], report["tree_sha256"])
        self.assertEqual(
            report["tree_sha256_algorithm"], pilot_matrix.TREE_DIGEST_ALGORITHM
        )

    def test_runner_v3_completion_passes_independent_evaluator_validation(self) -> None:
        output_root = self.root / "outputs"
        checkpoint = self.root / "base.pth"
        checkpoint.write_bytes(b"checkpoint")
        config = self.root / "base__clean.yml"
        config.write_text("name: base\n", encoding="ascii")
        with mock.patch.object(pilot_matrix, "OUTPUT_ROOT", output_root):
            dataset = self.matrix["datasets"][0]
            cell_root = pilot_matrix.result_dir("unit_v1", "base", "clean")
            output_dir = cell_root / "visualization" / dataset["name"]
            output_dir.mkdir(parents=True)
            prediction_paths = []
            for image_id in self.ids:
                prediction = output_dir / f"{image_id}_base.png"
                prediction.write_bytes(f"sr-{image_id}".encode("ascii"))
                prediction_paths.append(prediction)
            outputs = pilot_matrix._validate_saved_outputs(
                self.matrix,
                "unit_v1",
                "base",
                "clean",
                {"clean": {"ids": self.ids}},
            )
            record = pilot_matrix._completion_record(
                run_id="unit_v1",
                current_cell="base__clean",
                model_id="base",
                dataset_id="clean",
                checkpoint={
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint),
                    "param_key": "params_ema",
                },
                data_provenance={"sha256": "data"},
                command=["python", "hat/test.py", "-opt", str(config)],
                config_path=config,
                config_sha256=_sha256(config),
                provenance={"git": {"head": "0" * 40}},
                outputs=outputs,
                archive_report={
                    "before": [],
                    "after": [],
                    "created_during_cell": [],
                },
            )
            completion_path = cell_root / "completion.json"
            digest, digest_path = pilot_matrix._write_immutable_completion(
                completion_path, record
            )
            verified = eval_provenance.validate_completion_record(
                prediction_name="base",
                prediction_directory=output_dir,
                prediction_suffix="_base",
                prediction_paths=prediction_paths,
                completion_record_path=completion_path,
            )
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["outputs"]["tree_sha256"], outputs["tree_sha256"])
        self.assertEqual(digest, _sha256(completion_path))
        self.assertEqual(
            digest_path.read_text(encoding="ascii"),
            f"{digest}  completion.json\n",
        )


class CheckpointPreflightTests(unittest.TestCase):
    def test_state_signature_requires_params_ema(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.pth"
            bad = root / "bad.pth"
            torch.save({"params_ema": {"weight": torch.zeros(2, 3)}}, good)
            torch.save({"params": {"weight": torch.zeros(2, 3)}}, bad)
            signature, count = pilot_matrix._state_signature(good, "params_ema")
            self.assertEqual(count, 1)
            self.assertEqual(signature["weight"], ([2, 3], "torch.float32"))
            with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "lacks top-level key"):
                pilot_matrix._state_signature(bad, "params_ema")

    def test_explicit_candidate_must_match_base_params_ema_signature(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.pth"
            candidate = root / "candidate.pth"
            mismatch = root / "mismatch.pth"
            torch.save({"params_ema": {"weight": torch.zeros(2, 3)}}, base)
            torch.save({"params_ema": {"weight": torch.ones(2, 3)}}, candidate)
            torch.save({"params_ema": {"weight": torch.ones(2, 4)}}, mismatch)
            matrix = {
                "checkpoint_param_key": "params_ema",
                "models": [
                    {"id": "base", "checkpoint": "base.pth", "sha256": _sha256(base)}
                ],
            }
            with mock.patch.object(pilot_matrix, "REPO_ROOT", root):
                specs = [
                    *pilot_matrix._canonical_model_specs(matrix, ["base"]),
                    {
                        "id": "stagea_5k",
                        "checkpoint": str(candidate),
                        "sha256": _sha256(candidate),
                        "source": "explicit_candidate",
                    },
                ]
                report = pilot_matrix.preflight_model_specs(matrix, specs)
                self.assertEqual(set(report), {"base", "stagea_5k"})
                specs[1]["checkpoint"] = str(mismatch)
                specs[1]["sha256"] = _sha256(mismatch)
                with self.assertRaisesRegex(
                    pilot_matrix.PilotMatrixError, "differs from canonical base"
                ):
                    pilot_matrix.preflight_model_specs(matrix, specs)


class GpuLaunchSafetyTests(unittest.TestCase):
    def test_existing_target_is_rechecked_without_invoking_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "cell"
            target.mkdir()
            with mock.patch.object(pilot_matrix.subprocess, "run") as mocked_run:
                with self.assertRaisesRegex(
                    pilot_matrix.PilotMatrixError, "immediately before HAT launch"
                ):
                    pilot_matrix._invoke_hat_cell(["fake"], target, {})
            mocked_run.assert_not_called()

    def test_new_basicsr_archive_is_reported_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "cell"
            archive = Path(f"{target}_archived_20260820_150000")

            def fake_run(*_args: object, **_kwargs: object) -> object:
                archive.mkdir()
                target.mkdir()
                return pilot_matrix.subprocess.CompletedProcess(["fake"], 0)

            with mock.patch.object(
                pilot_matrix.subprocess, "run", side_effect=fake_run
            ):
                with self.assertRaisesRegex(
                    pilot_matrix.PilotMatrixError, "result-path race"
                ):
                    pilot_matrix._invoke_hat_cell(["fake"], target, {})
            self.assertTrue(target.is_dir())
            self.assertTrue(archive.is_dir())

    def test_cooperative_lock_is_acquired_before_final_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "active_gpu.lock"
            events: list[str] = []
            manifest = {
                "selection": {"models": ["base"], "datasets": ["clean"]},
                "models": [{"id": "base"}],
            }
            matrix, template = pilot_matrix.load_canonical()

            def acquire(_run_id: str, _cells: list[str]) -> tuple[int, Path]:
                events.append("lock")
                descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                return descriptor, lock_path

            def stop_preflight(*_args: object, **_kwargs: object) -> object:
                events.append("data")
                raise pilot_matrix.PilotMatrixError("intentional stop")

            with (
                mock.patch.object(pilot_matrix, "OUTPUT_ROOT", root / "outputs"),
                mock.patch.object(pilot_matrix, "_prepared_manifest", return_value=manifest),
                mock.patch.object(pilot_matrix, "load_canonical", return_value=(matrix, template)),
                mock.patch.object(pilot_matrix, "_acquire_gpu_lock", side_effect=acquire),
                mock.patch.object(pilot_matrix, "_active_hat_processes", return_value=[]),
                mock.patch.object(pilot_matrix, "preflight_data", side_effect=stop_preflight),
            ):
                with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "intentional stop"):
                    pilot_matrix.run_prepared(
                        "unit_lock", ["base"], ["clean"], Path("/bin/true"), True
                    )
            self.assertEqual(events, ["lock", "data"])
            self.assertFalse(lock_path.exists())

    def test_provenance_drift_fails_under_lock_before_hat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "active_gpu.lock"
            checkpoint = {"path": "/checkpoint", "sha256": "hash"}
            prepared_provenance = {"git": {"head": "0" * 40}}
            manifest = {
                "selection": {"models": ["base"], "datasets": ["clean"]},
                "models": [{"id": "base"}],
                "checkpoints": {"base": checkpoint},
                "data": {"clean": {"sha256": "data"}},
                "provenance": prepared_provenance,
                "configs": {},
            }
            matrix, template = pilot_matrix.load_canonical()
            data_report = {
                "clean": {"snapshot": {"sha256": "data"}, "ids": ["face_a"]},
                "pilot_selection": {"snapshot": {}, "ids": ["face_a"]},
            }

            def acquire(_run_id: str, _cells: list[str]) -> tuple[int, Path]:
                descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                return descriptor, lock_path

            with (
                mock.patch.object(pilot_matrix, "OUTPUT_ROOT", root / "outputs"),
                mock.patch.object(pilot_matrix, "_prepared_manifest", return_value=manifest),
                mock.patch.object(pilot_matrix, "load_canonical", return_value=(matrix, template)),
                mock.patch.object(pilot_matrix, "_acquire_gpu_lock", side_effect=acquire),
                mock.patch.object(pilot_matrix, "_active_hat_processes", return_value=[]),
                mock.patch.object(pilot_matrix, "preflight_data", return_value=data_report),
                mock.patch.object(
                    pilot_matrix, "preflight_model_specs", return_value={"base": checkpoint}
                ),
                mock.patch.object(
                    pilot_matrix,
                    "_collect_provenance",
                    return_value=({"git": {"head": "1" * 40}}, {}),
                ),
                mock.patch.object(pilot_matrix, "_invoke_hat_cell") as invoke,
            ):
                with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "provenance drifted"):
                    pilot_matrix.run_prepared(
                        "unit_drift", ["base"], ["clean"], Path("/bin/true"), True
                    )
            invoke.assert_not_called()
            self.assertFalse(lock_path.exists())


class ProvenanceTests(unittest.TestCase):
    def test_capture_includes_git_source_packages_pip_and_gpu_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "hat").mkdir()
            (root / "hat" / "model.py").write_text("VALUE = 1\n", encoding="ascii")
            template = root / "options" / "test" / "recovery" / "pilot_template.yml"
            template.parent.mkdir(parents=True)
            template.write_text("name: template\n", encoding="ascii")
            matrix_path = root / "recovery" / "inference" / "pilot_matrix.json"
            matrix_path.parent.mkdir(parents=True)
            matrix_path.write_text("{}\n", encoding="ascii")

            def capture(_command: object, label: str) -> bytes:
                values = {
                    "git HEAD": ("0" * 40 + "\n").encode("ascii"),
                    "git status": b"?? file\0",
                    "binary git diff": b"diff --git a/x b/x\n",
                    "pip freeze": b"package==1\n",
                    "Torch GPU visibility probe": b'{"device_count":1,"devices":[{"name":"GPU"}]}\n',
                }
                return values[label]

            final = root / "final" / "provenance"
            with (
                mock.patch.object(pilot_matrix, "REPO_ROOT", root),
                mock.patch.object(pilot_matrix, "MATRIX_PATH", matrix_path),
                mock.patch.object(pilot_matrix, "_capture_command", side_effect=capture),
                mock.patch.object(pilot_matrix.shutil, "which", return_value=None),
            ):
                snapshot, contents = pilot_matrix._collect_provenance(final)
            self.assertEqual(snapshot["git"]["head"], "0" * 40)
            self.assertEqual(snapshot["source"]["hat_python_file_count"], 1)
            self.assertIn("runner", snapshot["source"])
            self.assertIn("template", snapshot["source"])
            self.assertIn("matrix", snapshot["source"])
            self.assertIn("torch", snapshot["runtime"])
            self.assertEqual(snapshot["runtime"]["gpu_probe"]["device_count"], 1)
            self.assertIn("pip_freeze.txt", contents)
            self.assertIn("git_diff_head_binary.patch", contents)

    def test_stored_provenance_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, contents = _fake_provenance(root)
            for name, content in contents.items():
                (root / name).write_bytes(content)
            pilot_matrix._verify_stored_provenance(snapshot)
            (root / "pip_freeze.txt").write_bytes(b"changed\n")
            with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "changed"):
                pilot_matrix._verify_stored_provenance(snapshot)


class EvaluationCommandTests(unittest.TestCase):
    def test_candidate_evaluation_requests_evidence_arcface_ci_and_contact_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arcface = root / "arcface.pth"
            arcface.write_bytes(b"arcface")
            matrix, template = pilot_matrix.load_canonical()
            manifest = {
                "selection": {
                    "models": ["base", "stagea_5k"],
                    "datasets": ["clean"],
                },
                "data": {"clean": {"sha256": "data"}},
            }
            data_report = {
                "clean": {"snapshot": {"sha256": "data"}, "ids": ["face_a"]},
                "pilot_selection": {"snapshot": {}, "ids": ["face_a"]},
            }
            with (
                mock.patch.object(pilot_matrix, "EVALUATION_ROOT", root / "evaluations"),
                mock.patch.object(pilot_matrix, "_prepared_manifest", return_value=manifest),
                mock.patch.object(pilot_matrix, "load_canonical", return_value=(matrix, template)),
                mock.patch.object(pilot_matrix, "preflight_data", return_value=data_report),
                mock.patch.object(pilot_matrix, "_validated_completion"),
            ):
                command = pilot_matrix.evaluation_command(
                    "stagea_review_v1",
                    "clean",
                    ["base", "stagea_5k"],
                    Path("/bin/true"),
                    "facexlib-pth",
                    arcface,
                    "cpu",
                    16,
                    False,
                    24,
                    "arcface_identity_similarity",
                )
            self.assertIn("--bootstrap-samples", command)
            self.assertEqual(command[command.index("--bootstrap-samples") + 1], "5000")
            self.assertEqual(command.count("--completion-record"), 2)
            self.assertIn("--arcface-backend", command)
            self.assertIn(str(arcface), command)
            self.assertEqual(command[command.index("--arcface-device") + 1], "cpu")
            self.assertEqual(
                command[command.index("--selection-metric") + 1],
                "arcface_identity_similarity",
            )

    def test_arcface_gpu_command_requires_and_forwards_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arcface = root / "arcface.pth"
            arcface.write_bytes(b"arcface")
            matrix, template = pilot_matrix.load_canonical()
            manifest = {
                "selection": {"models": ["base", "stagea_5k"], "datasets": ["clean"]},
                "data": {"clean": {"sha256": "data"}},
            }
            data_report = {
                "clean": {"snapshot": {"sha256": "data"}, "ids": ["face_a"]},
                "pilot_selection": {"snapshot": {}, "ids": ["face_a"]},
            }
            patches = (
                mock.patch.object(pilot_matrix, "EVALUATION_ROOT", root / "evaluations"),
                mock.patch.object(pilot_matrix, "_prepared_manifest", return_value=manifest),
                mock.patch.object(pilot_matrix, "load_canonical", return_value=(matrix, template)),
                mock.patch.object(pilot_matrix, "preflight_data", return_value=data_report),
                mock.patch.object(pilot_matrix, "_validated_completion"),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(pilot_matrix.PilotMatrixError, "confirm-arcface-gpu"):
                    pilot_matrix.evaluation_command(
                        "stagea_review_v1",
                        "clean",
                        ["base", "stagea_5k"],
                        Path("/bin/true"),
                        "facexlib-pth",
                        arcface,
                        "cuda",
                        8,
                        False,
                    )
                command = pilot_matrix.evaluation_command(
                    "stagea_review_v1",
                    "clean",
                    ["base", "stagea_5k"],
                    Path("/bin/true"),
                    "facexlib-pth",
                    arcface,
                    "cuda",
                    8,
                    True,
                )
            self.assertIn("--confirm-arcface-gpu", command)
            self.assertEqual(command[command.index("--arcface-batch-size") + 1], "8")

    def test_contact_command_uses_same_comparison_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "selection": {
                    "models": ["base", "stagea_5k"],
                    "datasets": ["clean"],
                }
            }
            with (
                mock.patch.object(pilot_matrix, "EVALUATION_ROOT", root),
                mock.patch.object(pilot_matrix, "_prepared_manifest", return_value=manifest),
            ):
                output = pilot_matrix._comparison_output(
                    "stagea_review_v1", "clean", ["base", "stagea_5k"]
                )
                output.mkdir(parents=True)
                (output / "contact_selection.json").write_text("{}\n", encoding="ascii")
                command = pilot_matrix.contact_sheet_command(
                    "stagea_review_v1",
                    "clean",
                    ["base", "stagea_5k"],
                    Path("/bin/true"),
                    128,
                )
            self.assertIn(str(output / "contact_selection.json"), command)
            self.assertIn(str(output / "contact_sheet.png"), command)


if __name__ == "__main__":
    unittest.main()
