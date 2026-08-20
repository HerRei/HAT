# Recovery Pilot Inference And Acceptance

This directory provides reproducible evaluation-only tooling for the face-SR
recovery workflow. It never trains, resumes, modifies, or silently selects a
model.

`pilot_matrix.json` pins the base, failed 85K/95K/125K/130K checkpoints, and
three base-to-95K interpolations. `prepare-candidate` handles new Stage A/B/C
checkpoints without editing that canonical matrix. A model/bucket pair is one
cell, so base/clean can run independently of mild or hard evaluation.

## Safety And Provenance

Only `pilot_matrix run --confirm-gpu-run` can invoke `hat/test.py`. Preparation
and checking may load checkpoints on CPU and query the ROCm/Torch device
identity, but they perform no neural-network inference or training.

Every preparation freezes and digest-protects:

- Git HEAD, NUL-delimited dirty status, and `git diff --binary HEAD`;
- the path, size, and SHA-256 of every `hat/**/*.py` file;
- the runner, template, and canonical matrix hashes;
- Python, PyTorch/ROCm, BasicSR, and OpenCV versions;
- exact `pip freeze --all` output and its content path;
- ROCm SMI identity and Torch-runtime-visible GPU properties;
- GPU visibility environment variables;
- data manifests, checkpoint bytes, `params_ema` tensor signatures, configs,
  commands, and expected result paths.

`run` acquires the recovery tool's cooperative lock before checking result
paths or starting the lengthy final provenance/data/checkpoint preflight. Any
code, package, git, GPU identity, checkpoint, or data drift fails before HAT.
It then rechecks each exact cell result path immediately before its subprocess.

The lock is cooperative only among `recovery.inference.pilot_matrix` callers.
It cannot stop a user or external controller from launching BasicSR/HAT
directly. BasicSR archives an existing result as `<name>_archived_<timestamp>`;
the runner snapshots those siblings around every cell and fails if a new one
appears. It never deletes the target, archive, or other partial evidence.

Other enforced safeguards:

- no `--auto_resume`, resume key, URL checkpoint, or guessed checksum;
- strict `params_ema` loading and exact HAT-S key/shape/dtype match to base;
- strict 512-image manifest membership, hashes, GT identity, and x4 sizes;
- no reuse of any existing or partial result directory;
- an explicit safe model ID and lowercase 64-hex SHA for every candidate;
- no `completion.json` until all 512 output PNGs and their hashes validate.

## Artifact Contract

A full canonical preparation creates 24 read-only cell configs. A candidate
preparation creates base plus one candidate for each selected bucket:

```text
results/recovery_pilot_matrix/orchestration/<run-id>/
  run_manifest.json
  run_manifest.sha256
  configs/<model-id>__<dataset-id>.yml
  provenance/{git,pip,source,GPU evidence files}
```

BasicSR appends the option `name` to `results_root`:

```text
results/recovery_pilot_matrix/outputs/<run-id>/
  face_sr_pilot_<run-id>_<model-id>_<dataset-id>/
    visualization/face_recovery_<dataset-id>_pilot/<image-id>_<model-id>.png
    completion.json
```

Completion schema v3 contains `status: complete`, model/checkpoint and config
hashes, data and environment provenance, the actual command, and BasicSR archive
observations. `files_manifest_sha256` and `tree_sha256` both hash sorted bytes:

```text
relative_posix_path NUL size NUL sha256 LF
```

The evaluator receives each completion through
`--completion-record NAME=PATH`, recomputes that prediction-tree digest, and
refuses unverified evidence.

## Preserved Matrix

Use the preserved environment:

```bash
cd /home/hermes/hat-face-training/HAT
PY=/home/hermes/hat-face-training/hat-face/bin/python
```

Run checked-in static validation, then the real CPU/runtime preflight only after
data generation finishes:

```bash
$PY -m recovery.inference.pilot_matrix static-check
$PY -m recovery.inference.pilot_matrix check
```

Prepare the full matrix or a bounded base/clean run. Neither command invokes
HAT:

```bash
$PY -m recovery.inference.pilot_matrix prepare --run-id recovery_v1

$PY -m recovery.inference.pilot_matrix prepare \
  --run-id recovery_base_clean_v1 \
  --model base \
  --dataset clean
```

The first bounded GPU action is:

```bash
$PY -m recovery.inference.pilot_matrix run \
  --run-id recovery_base_clean_v1 \
  --model base \
  --dataset clean \
  --confirm-gpu-run
```

Selectors may be repeated. `all` must be used alone and only on pending cells:

```bash
$PY -m recovery.inference.pilot_matrix run \
  --run-id recovery_v1 \
  --model face95k --model interp_a0p1 --model interp_a0p25 \
  --dataset clean \
  --confirm-gpu-run

$PY -m recovery.inference.pilot_matrix status --run-id recovery_v1
```

## New Stage Candidate

Never add a Stage A/B/C checkpoint to `pilot_matrix.json`. First print and
review its exact hash. Do not use a placeholder or inferred latest checkpoint:

```bash
CANDIDATE=/home/hermes/hat-face-training/HAT/experiments/recovery_stage_a_clean_fidelity_5k/models/net_g_5000.pth
sha256sum "$CANDIDATE"
```

Set `CANDIDATE_SHA` to the reviewed 64-hex output, then prepare canonical base
plus this candidate. The command rejects a wrong hash or incompatible state
dict before writing anything:

```bash
CANDIDATE_SHA=<reviewed-lowercase-64-hex-sha256>

$PY -m recovery.inference.pilot_matrix prepare-candidate \
  --run-id stagea_5k_review_v1 \
  --candidate-id stagea_5k \
  --checkpoint "$CANDIDATE" \
  --sha256 "$CANDIDATE_SHA" \
  --dataset clean
```

Run base and candidate cells explicitly:

```bash
$PY -m recovery.inference.pilot_matrix run \
  --run-id stagea_5k_review_v1 \
  --model all \
  --dataset clean \
  --confirm-gpu-run
```

For Stage B, prepare and run clean, mild, and hard. Stage C checkpoints use the
same candidate mechanism but remain separate perceptual specialists:

```bash
$PY -m recovery.inference.pilot_matrix prepare-candidate \
  --run-id stageb_10k_review_v1 \
  --candidate-id stageb_10k \
  --checkpoint /absolute/path/to/reviewed/net_g_10000.pth \
  --sha256 <reviewed-lowercase-64-hex-sha256> \
  --dataset clean --dataset mild --dataset hard
```

## Paired Metrics, ArcFace, And Contact Sheets

ArcFace is never downloaded or guessed. Supply a reviewed local model. The
following prints one exact evaluator command with strict pairing, 5,000 paired
bootstrap samples, completion evidence, ArcFace identity, and deterministic
contact selection:

```bash
ARCFACE=/absolute/path/recognition_arcface_ir_se50.pth

$PY -m recovery.inference.pilot_matrix eval-command \
  --run-id stagea_5k_review_v1 \
  --dataset clean \
  --model base --model stagea_5k \
  --arcface-backend facexlib-pth \
  --arcface-model "$ARCFACE" \
  --arcface-device cpu \
  --arcface-batch-size 16 \
  --selection-metric arcface_identity_similarity
```

CPU is the default identity path. A later explicit ROCm identity pass may use
`--arcface-device cuda --confirm-arcface-gpu`; command generation refuses CUDA
without that confirmation, and the evaluator refuses to initialize it while a
HAT train/test process is active.

Inspect and execute the printed command. It writes aggregate v2 evidence under
`results/recovery_pilot_matrix/evaluations/<run>/<bucket>/<models>/`. Then print
and execute the contact-sheet command:

```bash
$PY -m recovery.inference.pilot_matrix contact-command \
  --run-id stagea_5k_review_v1 \
  --dataset clean \
  --model base --model stagea_5k
```

Repeat evaluation and contact-sheet generation for every Stage B bucket.

## Deterministic Acceptance

Numeric thresholds are versioned in `acceptance_gates.json`. Stage A requires
clean fidelity/structure/identity constraints. Stage B requires bounded clean
regression and declared mild/hard improvements. Each constraint records one
delta statistic with `direction: higher`; it passes exactly when
`value >= threshold`.

Metrics alone cannot approve facial appearance. An operator must inspect the
fixed contact sheet(s) and author a JSON attestation. The tool never creates an
approved attestation. Stage A example:

```json
{
  "schema_version": 1,
  "attested": true,
  "decision": "approved",
  "reviewer": "operator-name",
  "reviewed_at": "2026-08-20T18:00:00Z",
  "protocol": "side_by_side_fixed_selection_v1",
  "notes": "Reviewed all fixed cases at native and zoomed scale.",
  "contact_sheets": [
    {
      "bucket": "clean",
      "path": "/absolute/path/contact_sheet.png",
      "sha256": "<reviewed-contact-sheet-sha256>"
    }
  ]
}
```

Only after aggregate v2 evidence and this attestation exist may the gate run:

```bash
$PY -m recovery.inference.acceptance_gate \
  --run-id stagea_5k_review_v1 \
  --stage A \
  --candidate-id stagea_5k \
  --aggregate clean=/absolute/path/aggregate.json \
  --human-attestation /absolute/path/human_attestation.json \
  --out /absolute/path/accepted_checkpoint.json
```

Stage B repeats `--aggregate` for `clean`, `mild`, and `hard`, and its
attestation must hash all three contact sheets. A failed metric, missing ArcFace
metric, unverified completion, changed artifact, or absent human approval
produces no acceptance record.

On success, `accepted_checkpoint.json` and
`accepted_checkpoint.json.sha256` are mode `0444`. The record binds stage,
checkpoint/config/data/aggregate/completion/contact hashes, all metric deltas and
paired CIs, environment provenance, and the explicit reviewer decision. Stage
B/C launch preflight consumes this record; a checkpoint filename alone is not
acceptance.

## Tests

All tests are stdlib `unittest` and invoke no HAT inference:

```bash
$PY -m unittest -v recovery.tests.test_recovery_inference
$PY -m unittest -v recovery.tests.test_recovery_acceptance_gate
$PY -m unittest discover -v -s recovery/tests -p 'test_*.py'
```
