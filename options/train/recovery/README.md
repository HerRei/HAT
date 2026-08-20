# Face SR recovery training

These options replace the failed one-step RealHATGAN fine-tune with three
gated stages. They deliberately use deterministic offline paired data. No
custom dataset or model implementation is required, and the old two-stage
RealESRGAN degradation path is not used.

## Why the stages are separate

1. Stage A tests the smallest useful change: whether low-LR, pixel-only
   training on full aligned faces can improve clean bicubic face fidelity.
2. Stage B starts only from an accepted Stage A checkpoint. Its offline paired
   dataset has exact 50/50 clean/mild composition.
3. Stage C is an optional perceptual specialist. It starts only from an
   accepted Stage B checkpoint and uses losses 20x weaker than the failed run
   for perceptual loss and 20x weaker for GAN loss. It is not part of the
   default recovery path.

More GPU memory would not repair a mismatched objective. This design first
proves reconstruction fidelity on the existing 16 GB GPU, then adds one source
of difficulty at a time.

## Data contract

The generation and audit code lives outside this directory. The configs expect
these runtime roots:

```text
/home/hermes/hat-face-training/data/face_sr_recovery/
  clean/train/{gt,lq}/
  mild_mixed/train/{gt,lq}/
  benchmarks/clean_pilot/{gt,lq}/
  benchmarks/mild_pilot/{gt,lq}/
  benchmarks/hard_pilot/{gt,lq}/
```

All GT images must be aligned 256x256 faces and all LQ images must be 64x64.
GT and LQ basenames must match. `clean/train` uses BasicSR 1.4.2's
MATLAB-style `imresize(float32, 0.25, antialiasing=True)`, then `numpy.rint`
uint8 quantization. This intentionally differs from the old PIL bicubic
validation recipe. Benchmark base HAT-S again on `clean_pilot`; do not compare
the new pilot metrics directly with the old `36.25 dB` result. The offline
`mild_mixed/train` set contains two entries per source:
`<stem>_clean.png` and `<stem>_mild.png`. Their GT files point to the same
source face, while their LQ files contain the clean and deterministic one-stage
mild recipes. This gives an auditable 50/50 dataset composition.

BasicSR does not stratify a bounded iteration prefix by `_clean` and `_mild`.
With `dataset_enlarge_ratio: 1`, `EnlargedSampler` creates an epoch-seeded
`torch.randperm` of all 130K entries. The 10K Stage B pilot consumes only its
first 20K sample indices at batch 2. Consequently, the realized prefix is a
deterministic unstratified sample for this preserved software environment, not
guaranteed exact 50% replay. Do not report an exact realized fraction unless a
future launcher records the actual sampled suffix counts.

Every dataset block points to the generated `meta_info.txt`. This pins ordered
membership to the deterministic builder manifest and avoids unconstrained
folder scans.

The pilot validation manifests must be fixed before the base-model benchmark
is recorded. Do not regenerate a benchmark between model comparisons.

## Configs and gates

### Stage A: `stage_a_clean_fidelity_5k.yml`

- Starts from `HAT-S_SRx4.pth` key `params_ema`.
- Uses `HATModel`, L1 only, LR `5e-6`, batch 2, horizontal flip, no rotation,
  full 256x256 faces, and EMA `0.999`.
- Stops at 5K. Validates the small clean pilot every 500 iterations without
  writing images; checkpoints are written every 500 so every validation point
  can be retained.
- Accept only if the exact clean benchmark beats base HAT-S, or a predeclared
  identity/perceptual benefit is obtained with negligible clean regression.
  Do not extend automatically if it has not matched base by 5K.

### Stage B: `stage_b_mild_reconstruction_10k.yml`

- The checkpoint path is an intentionally nonexistent Stage A sentinel.
- Uses `HATModel`, L1 only, LR `1e-5`, batch 2, EMA `0.999`, and the offline
  50/50 mild/clean paired set.
- Stops at 10K. Validates clean, mild, and hard fixed pilots every 1K.
- Accept only if it beats base and accepted Stage A on the mild benchmark,
  preserves identity, and stays inside the predeclared clean regression budget.

### Stage C: `stage_c_weak_perceptual_5k_OPT_IN.yml`

- The filename, metadata, and nonexistent checkpoint sentinel mark this as
  opt-in. Do not schedule it as part of an automated sequence.
- Uses standard paired `SRGANModel`, not `RealHATGANModel`, so no hidden
  on-the-fly degradation is introduced.
- Keeps L1 weight `1.0`, uses perceptual weight `0.05`, GAN weight `0.005`,
  and generator LR `5e-6`. There is no discriminator-only warm-up: allowing a
  randomly initialized discriminator 500 unopposed steps could make it
  dominant before the first generator adversarial update.
- Accept only if LPIPS/DISTS and blinded comparisons improve without material
  ArcFace/landmark regression. It remains a separate perceptual model even if
  accepted; it does not replace the clean fidelity specialist by default.
- Stage C is currently blocked in the machine-readable config. It may not be
  changed to `ready` until a pinned LPIPS or DISTS implementation and a written
  blinded visual-review protocol exist and are exercised by the gate tool.
  Even after the status changes, the launcher independently requires a passed
  LPIPS/DISTS constraint, hashed implementation and model-weight files, and a
  visual protocol identifier containing `blind`.

## Static validation

Run this from the HAT repository. It parses every YAML file and checks model,
dataset, loss, learning-rate, checkpoint-gate, and pilot-bound invariants. It
does not instantiate a model or use the GPU.

```bash
source /home/hermes/hat-face-training/hat-face/bin/activate
python options/train/recovery/test_recovery_options.py
```

After data generation finishes, additionally verify that all configured data
directories exist:

```bash
RECOVERY_REQUIRE_DATA=1 \
  python options/train/recovery/test_recovery_options.py
```

## Launch protocol

Before any GPU command, confirm that no training or test process is active and
record the config hash, data-manifest hash, command, and acceptance thresholds
in the recovery handoff. Never pass `--auto_resume` to a new stage. BasicSR's
CLI flag overrides the YAML value and may attach to old state with the same
experiment name.

Run Stage A only after its base benchmark and smoke test are complete:

```bash
python hat/train.py \
  -opt options/train/recovery/stage_a_clean_fidelity_5k.yml
```

Do not launch Stage B or C with `hat/train.py` directly, do not edit their
checkpoint sentinels, and do not manually add `--force_yml`. The only supported
entry point is `launch_recovery_stage.py`. It requires an
`accepted_checkpoint.json` emitted by the inference/gate tool with:

- a read-only JSON file and canonical read-only `.sha256` sibling;
- the correct source stage and an accepted checkpoint byte hash;
- `params_ema` with the exact canonical HAT-S tensor signature;
- at least one finite numeric gate constraint that independently satisfies
  `value >= threshold` and is marked passed;
- hashed aggregate reports and dataset manifests;
- explicit human attestation with reviewer, timezone-aware review time,
  protocol, notes, and hashed contact sheets;
- hashed source config and prepared inference provenance.

The preflight recomputes every referenced hash. It also requires generated
meta-info membership, rejects active `hat/train.py` or `hat/test.py` processes,
other `/dev/kfd` users, a busy GPU, the shared inference GPU lock, and existing
experiment/TensorBoard paths. It never passes `--auto_resume` and uses only the
accepted checkpoint as an explicit sentinel override.

CPU-check an accepted Stage A checkpoint for Stage B without launching HAT:

```bash
python options/train/recovery/launch_recovery_stage.py check \
  --stage B \
  --acceptance /absolute/path/to/accepted_checkpoint.json
```

Launch only after reviewing that report; the confirmation flag is mandatory:

```bash
python options/train/recovery/launch_recovery_stage.py launch \
  --stage B \
  --acceptance /absolute/path/to/accepted_checkpoint.json \
  --confirm-launch
```

The launcher is synchronous and holds the same exclusive lock as recovery
inference until HAT exits. It writes immutable launch and completion evidence
under `results/recovery_training_launches`. A partial or failed experiment is
preserved and its experiment name must not be reused.

Stage C uses the same interface with `--stage C`, but the checked-in config
currently refuses preflight because its evaluation gate is blocked.

## Two-iteration smoke override

This still uses the GPU, so run it only after the GPU is free. It bounds the
run to two optimizer steps and makes every event visible:

```bash
python hat/train.py \
  -opt options/train/recovery/stage_a_clean_fidelity_5k.yml \
  --force_yml name=smoke_recovery_stage_a \
  train:total_iter=2 \
  val:val_freq=1 \
  logger:print_freq=1 \
  logger:save_checkpoint_freq=2 \
  datasets:train:num_worker_per_gpu=0 \
  datasets:train:batch_size_per_gpu=1
```

This manual smoke override is for Stage A only. Stage B/C must not bypass their
acceptance wrapper. A smoke test proves only that data loading,
forward/backward, validation, and saving work; it is not evidence that the
model is improving.
