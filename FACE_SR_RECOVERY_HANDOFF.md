# HAT-S Face Super-Resolution Recovery Handoff

Last updated: 2026-08-20

## Purpose

This document is the durable source of truth for recovering the face super-resolution experiment. A new operator or AI agent should be able to continue from this file without relying on chat history.

The target is not simply "sharper faces." The project must distinguish:

1. Clean 4x bicubic face super-resolution with high reconstruction and identity fidelity.
2. Restoration of mildly blurred, noisy, or JPEG-compressed faces.
3. Optional perceptual enhancement that may trade fidelity for plausible detail.

These targets require separate validation buckets and may require separate checkpoints. Do not promote one model as universally better based on a single metric or degradation domain.

## Non-Destructive Rules

- Never overwrite or delete `experiments/HAT-S_SRx4_face_finetune_v2`.
- Preserve checkpoints 85K, 95K, 125K, and 130K until salvage evaluation is complete.
- Never resume the failed run into the same experiment directory when testing a new recipe.
- Do not revert unrelated dirty-worktree changes in `hat/data/imagenet_paired_dataset.py`, `requirements.txt`, or `setup.py`.
- Use a unique experiment name and output directory for every pilot.
- Record the exact config, git commit, environment, dataset manifest, seed, and command for every result.
- Run a bounded smoke test before a pilot and a bounded pilot before a long run.

## Current Workflow State

| Work item | Status | Notes |
|---|---|---|
| Failed-run postmortem | Complete | Evidence summarized below |
| 130K checkpoint preservation | Complete | Generator, discriminator, and state saved at 2026-08-20 14:29 local time |
| Graceful shutdown | Complete | Initial PID 9397 exited; obsolete controller restarted PID 909702, which was stopped; controller PID 7333 was then terminated to prevent further restarts |
| Recovery tooling tests | Complete | 15 stdlib recovery tests pass; checkpoint hashes verified |
| Clean dataset manifest | Building | CPU-only build PID 918016; exact BasicSR bicubic, aligned full faces, leakage audit passed |
| Mild/hard deterministic validation sets | Building | Same build; fixed per-image seeds and recorded degradation parameters |
| Saved-output evaluator | Complete | Fidelity, sharpness, edges, paired bootstrap, contact sheets, optional explicit LPIPS/ArcFace |
| Pilot inference matrix | In progress | Checksum-pinned base, failed checkpoints, and interpolation candidates; no inference launched yet |
| Stage A fidelity config | Complete | Pixel-only HATModel, LR 5e-6, 5K hard cap |
| Stage B restoration config | Complete | Offline paired L1, exact 50% clean replay, LR 1e-5, 10K hard cap |
| Stage C perceptual config | Complete but opt-in | Weak paired SRGAN; sentinel checkpoint prevents accidental launch |
| GPU smoke test | Pending | Run only after CPU/data/config checks pass |
| First 5K pilot | Pending | Do not extend unless acceptance gates pass |

Update this table whenever a step completes or the workflow is blocked.

## Active Long-Running Work

At 2026-08-20 14:47 local time, the deterministic CPU-only data build was started:

```bash
/home/hermes/hat-face-training/hat-face/bin/python \
  recovery/data/build_recovery_data.py build --workers 8
```

- PID: `918016`
- Output root: `/home/hermes/hat-face-training/data/face_sr_recovery`
- Builder SHA-256: `83d64a24f70b4e2b585f8aefbfe0caed0ff142a0a361bd46f1f7f8469f8594ce`
- Safety: source GT is read-only; generated files use atomic replacement; reruns are idempotent.
- If interrupted: rerun the identical command without `--repair`. Matching outputs are reused. Do not delete the output root.
- On completion: run normal verification, then `--recompute`, then rerun the recovery config test with `RECOVERY_REQUIRE_DATA=1`.

No HAT train or test process is active. Do not confuse unrelated `agy` processes on the host with the terminated failed-run controller; identify a controller by its child command and working context before acting.

## Existing Run and Safe Recovery Point

Failed experiment root:

`/home/hermes/hat-face-training/HAT/experiments/HAT-S_SRx4_face_finetune_v2`

Effective configuration:

`/home/hermes/hat-face-training/HAT/experiments/HAT-S_SRx4_face_finetune_v2/train_HAT-S_SRx4_face_v2.yml`

Primary log:

`/home/hermes/hat-face-training/HAT/experiments/HAT-S_SRx4_face_finetune_v2/train_HAT-S_SRx4_face_finetune_v2_20260817_171139.log`

Latest complete recovery point:

- `models/net_g_130000.pth`
- `models/net_d_130000.pth`
- `training_states/130000.state`

The original base model is:

`/home/hermes/hat-face-training/HAT/experiments/pretrained_models/HAT-S_SRx4.pth`

The base checkpoint contains both `params` and `params_ema`. Evaluation and new initialization should normally use `params_ema`.

Artifact checksums and the environment fingerprint are stored in `recovery/run_state.json`. Verify them before salvage, resumption, or deletion decisions.

Generated interpolation candidates are evaluation-only artifacts under `experiments/face_sr_recovery_interpolations`. They blend base `params_ema` with 95K `params_ema` at alpha 0.1, 0.25, and 0.5. Their embedded metadata records both source hashes and the interpolation coefficient.

## Quantitative Postmortem

The base evaluation timed out after producing 971 outputs. That paired subset is still large enough for a decisive clean-domain comparison.

| Checkpoint | PSNR Y, crop 4 | SSIM Y, crop 4 |
|---|---:|---:|
| Base HAT-S | 36.2505 | 0.92253 |
| Face GAN 5K | 31.0939 | 0.85392 |
| Face GAN 85K | 31.1553 | 0.85524 |
| Face GAN 95K | 31.1601 | 0.85423 |
| Face GAN 125K | 31.0977 | 0.85265 |

Every checked fine-tune checkpoint lost to base on all 971 images for both PSNR and SSIM. The full fine-tune validation log agrees with the subset result: best PSNR was 31.1593 at 95K, best SSIM was 0.8559 at 85K, and 125K was 31.1015/0.8533.

The run did learn its synthetic objective:

- Average sampled training L1 fell from roughly 0.0433 to 0.0386.
- Average sampled VGG perceptual loss fell from roughly 10.63 to 9.07.
- Laplacian sharpness on the paired subset rose from 50.4 for base to 72.9 at 125K.
- Ground-truth edge-location correlation fell from 0.868 for base to 0.729 at 125K.

Interpretation: the model added high-frequency detail, but much of it was misplaced or invented. It is conclusively worse for clean bicubic reconstruction. Its value for matched real-world degradation remains unknown because that domain was never validated.

## Root Causes

### 1. The reconstruction stage was skipped

The repository's intended Real-HAT recipe is two-stage:

1. Train `RealHATMSEModel` with pixel reconstruction loss.
2. Initialize `RealHATGANModel` from the trained restoration checkpoint.

The failed run initialized GAN, VGG perceptual loss, and a discriminator directly from the classical bicubic HAT-S checkpoint. This rapidly moved the model away from the strong base mapping before it learned stable blind restoration.

Reference configs:

- `options/train/train_Real_HAT_SRx4_mse_model.yml`
- `options/train/train_Real_HAT_GAN_SRx4_finetune_from_mse_model.yml`

### 2. Training and validation domains did not match

Every training sample received aggressive RealESRGAN-style corruption: blur, resize, noise, JPEG, a second noise/JPEG stage, usually a second blur, and usually a final sinc filter. There was effectively no clean bicubic replay.

Validation used clean PIL bicubic 64x64 inputs. Therefore validation measured a mapping the run was actively forgetting rather than a mapping it was trained to improve.

### 3. Whole-model learning rate was too aggressive

The generator used LR `1e-4` from the beginning. At 125K, EMA weights had moved approximately 27.8% relative L2 from base; the upsampler had moved approximately 62.8%. A face-specialization fine-tune intended to preserve a strong base should start around `5e-6` to `1e-5` and be gated early.

### 4. Scalar loss balance favored perceptual appearance

Logged pixel loss was around 0.04 while logged perceptual loss was around 9. The scalar generator objective was overwhelmingly perceptual. This does not directly prove the same gradient ratio, but it confirms that exact reconstruction was not the dominant optimized scalar objective.

### 5. Data geometry did not consistently teach a full face

The v2 source stores aligned 512x512 FFHQ images. The RealESRGAN dataset and model take random crops before producing the final 256x256 target. Training therefore often sees local face fragments rather than one complete aligned 256x256 face.

The older dataset at `/home/hermes/hat-face-training/data/ffhq_face_sr` contains 65,000 full aligned 256x256 train faces and 5,000 held-out faces. Its existing LQ images include random Gaussian blur before bicubic downsampling, so they are useful for a mild bucket but are not a pure-bicubic benchmark. Generate a separate exact-bicubic LQ branch from the same GT and do not mix source datasets without checking identity overlap.

### 6. Split and evaluation hygiene defects

- v2 training contains 64,997 files; IDs 16524-16526 are missing.
- v2 validation contains 5,002 files; unexpected IDs 00000 and 00001 are exact center crops of training images.
- Validation used PIL bicubic while other SR tooling may use MATLAB-style bicubic. Choose one kernel and record it.
- Saving every one of 5,002 outputs every 5K generated about 12 GB of PNGs and added roughly 14 minutes per validation.

These defects do not explain the 5 dB regression, but future experiments must remove them.

## Hardware and Software Envelope

- GPU: AMD Radeon RX 9060 XT, 16 GB VRAM, `gfx1200`.
- Current GAN workload used about 11.7 GB and kept the GPU at 100% utilization.
- CPU: Intel Core Ultra 7 270K Plus, 24 cores/threads.
- RAM: approximately 31 GB plus approximately 31 GB swap.
- Storage: approximately 594 GB free at postmortem time.
- Python: 3.13.13.
- PyTorch: 2.9.1+rocm6.3.
- TorchVision: 0.24.1+rocm6.3.
- BasicSR: 1.4.2.
- HAT git commit: `1638a9a822581657811867bf670717f8371fc3e5`.

Observed GAN throughput was about 1.9 seconds per iteration at batch 2, plus about 14 minutes for full validation every 5K. A 5K interval cost about 2 hours 53 minutes. Pixel-only training should use less VRAM and be faster, but must be benchmarked rather than assumed.

This hardware is sufficient for controlled HAT-S fine-tuning. It is not the cause of the failed objective. Do not attempt HAT-L or a large diffusion model from scratch before a correctly targeted HAT-S baseline proves model capacity is limiting.

The ROCm/Fedora combination is unconventional but has run stably for days. Preserve the working virtual environment before considering upgrades.

## Required Validation Design

Create immutable manifests under a new recovery directory. Each manifest must contain image ID, GT path, LQ path, degradation bucket, seed, and all degradation parameters.

The implemented builder is `recovery/data/build_recovery_data.py`; its complete contract and recovery commands are in `recovery/data/README.md`. Generated data lives outside Git at `/home/hermes/hat-face-training/data/face_sr_recovery`. It uses the consistent older split with exactly 65,000 train and 5,000 validation GT images. A real-source dry run found zero basename collisions and zero byte-identical cross-split images.

### Clean bucket

- Complete aligned 256x256 GT faces.
- Exact 64x64 bicubic downsampling using one recorded implementation.
- No noise, JPEG, blur, sinc, or random parameters.

The selected implementation is BasicSR 1.4.2's MATLAB-style antialiased `imresize`, followed by `numpy.rint` and lossless PNG encoding. This differs from the old PIL-bicubic validation data, so the old 36.2505 dB base number must not be compared directly with the new clean pilot. Establish a fresh base score on the immutable new pilot.

### Mild bucket

Suggested initial ranges, to be calibrated against deployment inputs:

- Blur sigma or equivalent radius no greater than about 1.5.
- Gaussian noise sigma no greater than about 10.
- JPEG quality at least 60.
- Prefer one degradation stage initially.
- Parameters generated once from a fixed seed and written to the manifest.

### Hard bucket

- Deterministic subset of the existing two-stage degradation family.
- Intended only to measure robustness, not to select the clean-fidelity model.

### Real bucket

- Fixed representative real photos with no GT.
- Store source provenance and licensing information.
- Use no-reference metrics only as supporting evidence; require fixed visual comparisons.

### Metrics

- PSNR and SSIM with explicitly recorded color space and crop.
- LPIPS or DISTS for perceptual distance.
- ArcFace-style embedding cosine similarity for identity preservation.
- Optional landmark error and no-reference NIQE as supporting metrics.
- Per-image paired differences, win rate, median, and confidence interval.
- Fixed contact sheets for eyes, mouth/teeth, hair, glasses, skin texture, and face contour.

The implemented saved-output evaluator is documented in `recovery/eval/README.md`. It never starts HAT inference. The pinned FaceXLib IR-SE50 weight is:

```text
/home/hermes/hat-face-training/models/metrics/recognition_arcface_ir_se50.pth
SHA-256 a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28
source https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth
```

It is loaded as a strict local state dict on CPU. The evaluator does not call FaceXLib's downloader. Directly resizing an aligned FFHQ frame to 112x112 is an identity-consistency proxy, not the standard five-landmark ArcFace protocol for arbitrary photographs.

## Staged Training Plan

### Stage A: clean fidelity specialist

- Initialize from base HAT-S `params_ema`.
- Use `HATModel`, not a GAN model.
- Use complete aligned 256x256 faces and exact 64x64 bicubic pairs.
- Use horizontal flip; disable rotation.
- Use L1 or Charbonnier only for the first baseline.
- Whole-model LR: start in the `5e-6` to `1e-5` range.
- Batch 2 is known safe. Benchmark batch 4 only after smoke testing.
- EMA: 0.999.
- Validate a fixed 256-512 image pilot subset every 500-1,000 iterations.
- First decision point: 5K. Maximum initial pilot: 20K.

Stage A acceptance gate:

- Primary gate: mean clean-pilot PSNR must exceed base and the paired 95% bootstrap confidence interval for candidate minus base must not cross zero.
- Secondary non-inferiority gates: mean SSIM must not fall by more than 0.001 and mean ArcFace similarity must not fall by more than 0.002 versus base.
- Must not introduce systematic eye, mouth, teeth, or contour changes.
- If it has not matched base by 5K, do not extend it automatically.

### Stage B: degradation-aware reconstruction

- Initialize from the best accepted Stage A checkpoint.
- Use the implemented standard paired `HATModel` with L1 only. Degradations are generated offline, so no hidden high-order corruption path runs during training.
- Use the deterministic one-stage mild recipe first.
- Use the implemented exact 50% clean bicubic replay branch.
- Suggested starting LR: about `1e-5`.
- Run a 10K pilot, then extend toward 50K only if mild/hard metrics improve without unacceptable clean/identity loss.

Stage B acceptance gate:

- Mean mild-pilot PSNR must beat both base and accepted Stage A, with a paired 95% bootstrap confidence interval above zero versus the stronger baseline.
- Mean mild ArcFace similarity may not fall by more than 0.002 versus the stronger baseline.
- Clean mean PSNR may regress by at most 0.10 dB, clean mean SSIM by at most 0.001, and clean mean ArcFace similarity by at most 0.002 versus accepted Stage A.
- Hard-bucket improvement is supporting evidence only and cannot override a failed clean or identity gate.

### Stage C: optional perceptual specialist

- Initialize only from an accepted Stage B checkpoint.
- Use a short run and a much weaker perceptual/adversarial objective than the failed run.
- Suggested starting search point: perceptual weight 0.1, GAN weight 0.005-0.01, generator LR about `5e-6`.
- First pilot: 5K-10K.
- Keep a separate checkpoint/model name from the fidelity specialist.

Stage C acceptance gate:

- LPIPS/DISTS and blinded visual comparisons must improve.
- ArcFace identity and landmark behavior must not materially regress.
- Do not promote it as the clean fidelity model if PSNR/identity decline beyond the declared budget.

The initial Stage C non-inferiority budget is the same as Stage B: no more than 0.10 dB clean PSNR, 0.001 clean SSIM, or 0.002 ArcFace similarity regression versus accepted Stage B. These are engineering guardrails declared before results, not established clinical identity thresholds. Tighten them if visual review shows identity-bearing changes.

## Salvage Experiments

These are cheap evaluations, not reasons to resume the failed run:

1. Evaluate base, 85K, 95K, 125K, and 130K on deterministic mild and hard manifests.
2. Interpolate EMA weights between base and 95K at alpha 0.1, 0.25, and 0.5.
3. Optionally test output blending, noting that it requires two inference passes.
4. Consider a short low-LR pixel-only recovery from 95K only if matched-degradation evaluation proves that 95K contains useful robustness.

Never assume interpolated or recovered models are better. Apply the same gates.

## Experiment Economics and Stop Rules

- Never schedule 250K before a 5K pilot passes.
- Use a small fixed validation subset during pilots.
- Disable `save_img` for routine validation; save only fixed comparison IDs or milestone outputs.
- Run full 5K validation only at baseline, accepted milestones, and final evaluation.
- Stop on NaN/Inf, repeated metric regression, identity regression, or failure to match the appropriate baseline by the declared decision point.
- Preserve the best checkpoint by each declared metric rather than assuming the latest is best.

## Next-Agent Startup Checklist

1. Read this file completely.
2. Run `git status --short` and do not revert pre-existing changes.
3. Check for active training/test processes before using the GPU.
4. Confirm the latest preserved failed-run artifacts and checksums.
5. Inspect the workflow-state table and continue the first pending item.
6. Update this document before and after every long-running process.
7. Never launch a long run without recording its command, PID/session, config, manifest hashes, and stop gates here.

## Commands for Read-Only Monitoring

```bash
ps -eo pid,ppid,etimes,%cpu,%mem,cmd | rg 'hat/(train|test)\.py'
cat /sys/class/drm/card0/device/gpu_busy_percent
cat /sys/class/drm/card0/device/mem_info_vram_used
tail -50 experiments/<experiment-name>/train_*.log
```

## Change Log

- 2026-08-20: Initial postmortem and recovery workflow recorded at the 130K safe checkpoint boundary.
- 2026-08-20: Trainer PID 9397 stopped gracefully after the 130K checkpoint; no escalation was required.
- 2026-08-20: The older `agy` controller PID 7333 automatically restarted the failed run as trainer PID 909702 from 130K. The controller was paused, PID 909702 was interrupted cleanly, and the obsolete controller was terminated after ignoring SIGTERM. Parent shell PID 7260 was left intact. GPU returned to 0% with no HAT process active.
- 2026-08-20: Added deterministic data generation, saved-output evaluation, checkpoint interpolation, three gated training configs, and stdlib tests. All 15 recovery tests and six static config tests pass; the data-existence check remains intentionally pending until the external build completes.
- 2026-08-20: Started the full CPU-only recovery-data build as PID 918016. Downloaded the explicit FaceXLib ArcFace IR-SE50 release weight and pinned SHA-256 `a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28`.
