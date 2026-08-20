# Saved-output evaluator

This tool evaluates PNG/JPEG outputs that already exist. It never loads an HAT
checkpoint, starts inference, or selects a GPU.

The result directory is evidence, not a cache. It must be absent or empty. Any
nonempty directory is rejected, and every result file is created exclusively so
an interrupted or completed evaluation cannot be overwritten by a retry.

## Protocol

- PSNR and SSIM call BasicSR's NumPy functions on decoded `uint8` BGR arrays.
- `--color-space y` uses BasicSR's Matlab-style Y conversion; `rgb` averages
  the three color channels. `--crop-border` is applied exactly as in BasicSR.
- Sharpness is Y-channel Laplacian variance. It is descriptive: more sharpness
  can mean real detail or hallucinated/noisy detail. Sharpness absolute error
  compares it with GT and is treated as lower-is-better.
- Edge correlation is Pearson correlation between Y-channel Sobel magnitudes.
- Comparisons are paired by image ID and report candidate-minus-baseline means,
  win/tie/loss rates, and deterministic percentile-bootstrap confidence intervals.
- Image IDs are case-sensitive basenames after removing only the explicit suffix.
  Duplicate IDs are fatal. Strict pairing is the default. Intersection mode reports
  all exclusions and should only be used for an intentionally incomplete output set.

Run with the existing project environment:

```bash
cd /home/hermes/hat-face-training/HAT
PY=/home/hermes/hat-face-training/hat-face/bin/python
$PY -m recovery.eval.evaluate_saved \
  --gt /home/hermes/hat-face-training/data/ffhq_face_sr_v2/val/gt \
  --prediction base=results/HAT-S_SRx4_base_on_faces/visualization/FFHQ_face_val_base \
  --prediction-suffix base=_base \
  --prediction face125k=experiments/HAT-S_SRx4_face_finetune_v2/visualization \
  --prediction-suffix face125k=_125000 \
  --baseline base \
  --pair-policy intersection \
  --color-space y \
  --crop-border 4 \
  --bootstrap-samples 5000 \
  --out recovery/runs/base_vs_face125k_y \
  --selection-json recovery/runs/base_vs_face125k_y/contact_selection.json
```

The command writes `per_image.jsonl`, `per_image.csv`, and `aggregate.json`.
The example intentionally uses intersection mode because the interrupted base run
saved only a subset. Confirm `pairing.evaluated_common_count` and all missing counts
in `aggregate.json` before interpreting scores.

For recovery-pilot inference, pass one version-3 completion record for every
prediction. The matrix runner's `eval-command` emits these arguments automatically:

```bash
$PY -m recovery.eval.evaluate_saved \
  --gt /absolute/pilot/gt \
  --prediction base=/absolute/base/outputs \
  --prediction-suffix base=_base \
  --completion-record base=/absolute/base/completion.json \
  --prediction candidate=/absolute/candidate/outputs \
  --prediction-suffix candidate=_candidate \
  --completion-record candidate=/absolute/candidate/completion.json \
  --baseline base --out /new/evaluation/path
```

Completion records are all-or-none. The evaluator independently verifies their
`status`, model name, checkpoint bytes, rendered config bytes, output root/suffix,
file count, and deterministic tree digest. The full records and verified digests
are embedded under `aggregate.provenance.predictions`. Historical output sets may
still be evaluated without records, but the aggregate marks them
`completion_record_status: not_supplied` rather than presenting them as verified.

`aggregate.identifiers.protocol_id` binds the metric/statistical settings,
dependencies, optional metric keys, and evaluator source hashes.
`aggregate.identifiers.evaluation_id` additionally binds the GT tree, exact
evaluated IDs, baseline, and every prediction/config/checkpoint/output digest.

Render the fixed selection without recomputing metrics:

```bash
$PY -m recovery.eval.build_contact_sheet \
  --selection recovery/runs/base_vs_face125k_y/contact_selection.json \
  --out recovery/runs/base_vs_face125k_y/contact_sheet.png
```

Run every CPU-only recovery test with the existing environment:

```bash
cd /home/hermes/hat-face-training/HAT
/home/hermes/hat-face-training/hat-face/bin/python -m unittest discover \
  -s recovery/tests -p 'test_*.py' -v
```

## Optional metrics

Optional metrics are off by default and are never silently skipped.

LPIPS requires the separate `lpips` package. Because the upstream constructor may
download torchvision backbone weights, it also requires explicit
`--lpips-allow-model-downloads` consent. An optional local calibration file can be
given with `--lpips-calibration-weights`; its SHA-256 is recorded. Package-bundled
calibration weights are recorded as such when no file is specified.

ArcFace requires both a strict backend selection and an explicit local weight file.
The preferred backend in this environment is facexlib's PyTorch IR-SE50:

```bash
$PY -m recovery.eval.evaluate_saved ... \
  --arcface-backend facexlib-pth \
  --arcface-model /absolute/path/recognition_arcface_ir_se50.pth \
  --arcface-device cpu \
  --arcface-batch-size 16
```

The evaluator constructs `facexlib.recognition.arcface_arch.Backbone` directly on
CPU and loads the file as a strict state dict. It deliberately does not call
`facexlib.init_recognition_model`, because that helper can select CUDA and download
weights. The local file SHA-256 is recorded. An `onnx` backend is also available
when CPU `onnxruntime` is installed. Backend selection is mandatory and is never
guessed from the extension.

Both adapters expose normalized embedding operations. The evaluator caches one
GT embedding per image and reuses it for every compared model. Facexlib runs
bounded native CPU batches; ONNX remains sequential because a local ONNX model's
batch dimension may be fixed. Aggregate metadata records the content-derived
model key, preprocessing key (including crop border), GT-cache key, entry/reuse
counts, and batch policy. On the current host, a real IR-SE50 smoke test returned
finite unit-norm `(2, 512)` embeddings for a two-face CPU batch without selecting
CUDA or downloading anything.

CPU is always the default. GPU identity scoring is never inferred and requires
both `--arcface-device cuda` and `--confirm-arcface-gpu`. Before loading or
transferring the model, the adapter scans `/proc` and refuses to initialize if any
`hat/train.py` or `hat/test.py` process is active. PyTorch's `cuda` device name is
also the API used by ROCm builds. Use a separate evaluation window; this guard is
not permission to compete with training for VRAM.

Both backends apply the metric border crop, resize the complete remaining aligned
face to `112x112`, convert BGR to RGB, and normalize to `[-1, 1]`. This protocol
assumes FFHQ-style aligned inputs; it does not detect or landmark-align arbitrary
real photos. For unaligned benchmarks, perform and freeze face alignment before
using the identity score. Directly resizing an FFHQ frame is a consistency proxy,
not a substitute for the ArcFace five-landmark alignment used in recognition
benchmarks.

LPIPS and ONNX Runtime are not currently installed. Facexlib, PyTorch, and the
SHA-pinned local IR-SE50 file recorded in `recovery/run_state.json` are available.
This evaluator does not install packages or fetch weights.

## Limitations

- Saved 8-bit images cannot recover pre-quantization model outputs.
- PSNR/SSIM on bicubic validation measure fidelity on that degradation, not quality
  on unknown real-world degradation.
- Bootstrap intervals describe uncertainty across this image sample; they do not
  correct dataset leakage, selection bias, or identity imbalance.
- Laplacian sharpness is not evidence of correct facial detail.
- ArcFace scores are only meaningful with a reviewed model/license and consistent
  alignment protocol.
