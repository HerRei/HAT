# Deterministic Recovery Data

This directory contains CPU-only tooling for the face SR recovery workflow. It
uses only the original consistent FFHQ split:

- Train GT: `/home/hermes/hat-face-training/data/ffhq_face_sr/train/gt`
- Validation GT: `/home/hermes/hat-face-training/data/ffhq_face_sr/val/gt`

Generated images are intentionally stored outside Git at
`/home/hermes/hat-face-training/data/face_sr_recovery`. GT entries are symlinks
to the old split; the builder never copies or modifies source GT.

## Data Contract

| Path | Contents |
|---|---|
| `clean/train/{gt,lq}` | Full 65K exact BasicSR/Matlab-style bicubic x4 pairs |
| `clean/val/{gt,lq}` | Full 5K clean validation pairs |
| `mild_mixed/train/{gt,lq}` | 130K pairs: one `_clean` and one `_mild` sample per source, exactly 50/50 |
| `mild/val/{gt,lq}` | Full 5K deterministic one-stage mild validation pairs |
| `hard/val/{gt,lq}` | Full 5K deterministic two-stage hard validation pairs |
| `benchmarks/clean_pilot/{gt,lq}` | Fixed 512-image clean pilot symlink farm |
| `benchmarks/mild_pilot/{gt,lq}` | The same fixed IDs with mild LQ |
| `benchmarks/hard_pilot/{gt,lq}` | The same fixed IDs with hard LQ |
| `manifests/*.jsonl` | Per-image paths, seeds, parameters, sizes, and hashes |
| `*/meta_info.txt` | Deterministic BasicSR paired-file lists derived from the matching manifest |
| `build_contract.json` | Immutable semantic invocation, source fingerprints, versions, targets, and pilot policy |

Clean LQ generation is exactly
`basicsr.utils.matlab_functions.imresize(float32_image, 0.25,
antialiasing=True)`, followed by recorded `numpy.rint` uint8 quantization and
lossless PNG encoding. Mild data uses one blur/downsample/noise/JPEG stage.
Hard data uses two deterministic degradation stages. The precise sampled
parameters and independent noise seeds are stored on every JSONL line.

The Stage B mixed set does not duplicate clean LQ bytes. `_clean` LQ entries
symlink the clean training branch; both GT variants symlink the same source GT.
Each paired dataset root has `meta_info.txt` in BasicSR's
`filename.png (height,width,3)` format. Training configurations should point
`meta_info_file` to this file instead of relying on an unconstrained folder
scan.

## Commands

Activate the working project environment, or call its Python directly:

```bash
PY=/home/hermes/hat-face-training/hat-face/bin/python
```

Audit both source splits and show the complete plan without writing anything:

```bash
$PY recovery/data/build_recovery_data.py build --dry-run
```

Build all clean, mixed-training, and benchmark data:

```bash
$PY recovery/data/build_recovery_data.py build --workers 8
```

Build only selected targets by repeating `--target`:

```bash
$PY recovery/data/build_recovery_data.py build \
  --target clean --target benchmarks --workers 8
```

Verify paths, source-link targets, byte hashes, decoded dimensions, and pixel hashes:

```bash
$PY recovery/data/build_recovery_data.py verify --workers 8
```

The stronger verification regenerates every LQ in memory and checks its sampled
parameters and encoded PNG hash:

```bash
$PY recovery/data/build_recovery_data.py verify --recompute --workers 8
```

Rerunning `build` is idempotent: matching artifacts and unchanged manifests are
left untouched. A mismatched generated artifact causes a hard failure. Use
`--repair` only after reviewing the reported path; real directories are never
removed automatically.

The first build writes `build_contract.json` before any image artifact. Future
runs must match its source inventory, generator environment, global seed,
pilot seed and size, targets, recipes, and output root exactly. Use a new output
root for any semantic change. Interrupted same-contract runs may have missing
members and can resume. Unexpected members, wrong path types, and wrong symlink
targets are rejected before writes and are never deleted automatically.

The initial August 20 build completed with the pre-contract in-memory code. One
same-default idempotent rerun safely adopts its compatible manifest sidecars,
writes the immutable contract and meta-info files, adds decoded-source hashes,
and reuses byte-identical LQ images. Run that rerun before strict verification.

## Reproducibility And Safety

- Per-image seeds depend only on the global seed, split, recipe, and relative
  source path. Worker count, traversal order, retries, and interrupted runs do
  not affect bytes.
- Pilot membership is the first 512 items ranked by SHA-256 of the fixed pilot
  seed and validation-relative path. All three pilot buckets use identical IDs.
- Source roots must contain only regular lowercase `.png` files with unique
  stems. Before any output is written, the builder checks expected split
  counts, disjoint basenames, and both encoded-file and decoded-pixel SHA-256
  collisions across train and val.
- The split audit detects only exact byte or decoded-pixel duplicates. It does
  not detect identity overlap, crops, transformations, or perceptual/near
  duplicates and must not be described as doing so.
- Files and manifests are written with atomic replacement. Manifests contain no
  timestamps, and each has a deterministic metadata sidecar with its own hash.
- Strict verification requires exact GT/LQ directory membership, topology,
  symlink targets, manifest/meta-info agreement, and identical pilot ID order.
- OpenCV, NumPy, BasicSR, Torch, Python, and generator versions are recorded in
  manifest metadata. Reproduce outputs with the preserved project environment.
- This script imports Torch only for BasicSR's CPU `imresize`; it does not create
  a CUDA/ROCm device or run model inference.

## Tests

Tests create tiny synthetic images under temporary directories and need no
pytest dependency:

```bash
$PY -m unittest recovery.tests.test_recovery_data -v
```
