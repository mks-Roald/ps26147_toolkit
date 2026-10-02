# Training Efficiency Report — Synthetic Pretraining, Seed 42

## Scope

Read-only analysis of the saved run record and source implementation. No training, inference, or evaluation was run for this report. The reported peak is the run's saved `torch.cuda.max_memory_allocated` value; estimated batch-size figures below are projections, not measurements.

## Executive summary

- The model is small: **228,701 trainable parameters** (about 0.87 MiB of FP32 weights). Parameter and AdamW state memory are not the main VRAM consumer.
- The saved run used batch 16, AMP, CUDA on an RTX 2050, and recorded **247,409,152 bytes (236 MiB) peak allocated**. This is well below 4 GiB.
- A first-order batch-linear projection suggests batch 64 around **0.92 GiB allocated**, and batch 128 around **1.84 GiB**. Batch 128 is plausible but unverified; it should not be treated as safe without a full-size forward/backward memory preflight.
- **Conservative largest safe recommendation: batch 64** without further profiling. Batch 128 is a candidate for a measured preflight, with 64 as fallback.
- The strongest likely throughput limit is the input pipeline: `num_workers=0`, full-file loads of 2 MiB captures for a 4,096-sample crop, and serial CPU feature/PSD construction. Historical GPU utilization was not recorded, so input-bound behavior is a well-supported suspicion, not a directly measured diagnosis.

## 1. Model parameters and branch shapes

The six-branch model uses three shallow temporal encoders, one IQ CNN, a shallow PSD encoder, the feature MLP, 128-D fusion, and a 7-class classifier. The temporal branches use 1-D convolutions and pooling; they do not flatten the full 4,096-point streams. Global average pooling reduces each branch to a fixed-size embedding.

| Component | Trainable parameters |
|---|---:|
| IQ encoder | 35,712 |
| Polar encoder | 36,288 |
| Nonlinear PSK encoder | 36,288 |
| IF encoder | 35,616 |
| PSD encoder and PSD projection | 26,800 |
| 22-feature encoder | 6,272 |
| Polar/PSK/IF projections | 24,960 |
| Fusion (LayerNorms and scalar logits) | 1,542 |
| Classifier | 25,223 |
| **Total** | **228,701** |

At 4 bytes per parameter, FP32 model weights occupy about **0.873 MiB**. For AdamW training, a rough persistent-state budget is weights + gradients + two FP32 moment buffers, approximately **3.49 MiB**. AMP does not require a second FP32 master copy in this standard PyTorch setup because the model parameters remain FP32.

The batch input tensors are all collated as float32 before autocast:

| Representation | Per-sample shape | Per-sample FP32 size |
|---|---|---:|
| IQ | `[2,4096]` | 32 KiB |
| Polar | `[4,4096]` | 64 KiB |
| Nonlinear PSK | `[4,4096]` | 64 KiB |
| IF | `[1,4096]` | 16 KiB |
| PSD | `[1,2048]` | 8 KiB |
| Features | `[22]` | 88 bytes |
| **Total** | 47,126 float values | **about 0.180 MiB** |

Thus model inputs on GPU are only about 2.88 MiB at batch 16, 11.5 MiB at batch 64, and 23 MiB at batch 128. These are small relative to the recorded peak.

## 2. Activation memory estimate

For an estimate from the actual layer lengths, count one output tensor at each convolution/pooling stage, before autograd's additional saved tensors:

- IQ: 786,432 activation elements per example.
- Polar, PSK, and IF combined: 3 × 442,368 = 1,327,104 elements/example.
- PSD: 69,632 elements/example.
- Total counted stage outputs: **2,183,168 elements/example**.

At 2 bytes/element (FP16), this one-copy count is about **4.16 MiB/example**; at 4 bytes/element it is about **8.33 MiB/example**. This is not the full training activation footprint: convolution/normalization backward saves additional values, gradients are created, and cuDNN may allocate workspaces. The recorded batch-16 peak of 236 MiB is the best available calibration for those effects.

### Estimated peak allocated memory by batch size

The nominal column scales the measured 236 MiB batch-16 peak approximately linearly, subtracting only a negligible fixed parameter/optimizer component. The planning column adds 25% to that projection for workspace and scaling uncertainty. Neither is a benchmark.

| Batch | Nominal projection | 25% planning estimate | Input tensors only (FP32) |
|---:|---:|---:|---:|
| 16 | 236 MiB (measured: 247.4 MB) | 295 MiB | 2.88 MiB |
| 32 | 472 MiB | 590 MiB | 5.75 MiB |
| 64 | 944 MiB | 1,180 MiB (1.15 GiB) | 11.5 MiB |
| 128 | 1,888 MiB (1.84 GiB) | 2,360 MiB (2.30 GiB) | 23.0 MiB |

The RTX 2050 has 4 GiB reported physical VRAM. `nvidia-smi` at analysis time showed about 100 MiB in use while idle, but this is not the run-time peak. A further reserve is needed for the WDDM display driver, other processes, allocator fragmentation, and any workspace algorithm changes. Batch 64 leaves comfortable estimated headroom. Batch 128 has plausible headroom under the projection but less protection against an unusually large workspace or concurrent graphics usage.

The run record has allocated memory only. It does **not** report peak reserved memory, free memory at the time of training, cuDNN workspace details, or activation-level profiling. Therefore it cannot certify batch 128.

## 3. DataLoader and CPU preprocessing

Saved configuration: batch size 16, `num_workers=0`. The trainer sets `pin_memory=True` on CUDA, but with zero workers there is no parallel sample preparation and no worker prefetch. `non_blocking=True` transfers are used, but these cannot hide a serial `__getitem__` bottleneck when the next batch has not yet been assembled.

Each dataset item is lazy, which avoids keeping the full 50 GB collection in RAM. However, `load_capture()` calls the project IQ loader for each item. For the generated files this calls `np.fromfile` on the complete 262,144-sample complex64 recording, then crops to 4,096 samples. Each source file is 2 MiB, so most of the loaded signal is discarded for each model input. The same SigMF metadata is also parsed again through the loader on each read, despite metadata having already been included in the manifest records.

After loading/cropping, CPU work builds all model views: joint normalization, phase-derived Polar/PSK/IF arrays, the existing 22 features, and complex two-sided Welch PSD (including conversion to complex128 and log/standardization). TRAIN phase rotation then causes all derived views to be recomputed, as required for augmentation. The operations run inside `Dataset.__getitem__` in the main process because `num_workers=0`.

There is an additional serial scaler-fit pass over all **20,002 TRAIN files** before the epoch timer starts. It reads each capture and computes the 22 features and PSD for deterministic center crops. Consequently, the recorded 26,002-second training time excludes this preprocessing/calibration pass.

## 4. Likely throughput bottleneck

The timed loop covers 40 epochs over 20,002 TRAIN and 4,998 VALIDATION examples, about one million per-item dataset accesses in total. At 26,002 seconds, the combined mean throughput is about **38.5 sample items/second**, including training and validation passes. The full-file access pattern represents roughly 48.8 GiB of logical IQ reads per epoch (train plus validation), or around 1.91 TiB across 40 epochs, before the one-time TRAIN scaler pass. This is logical read volume; OS caching may reduce physical storage traffic.

The model is only 0.229 million parameters, while each sample performs full-file loading, Python/numpy/scipy preprocessing, and Welch/feature work synchronously. These facts plus zero DataLoader workers make an **input-pipeline-bound run more likely than a GPU-compute-bound run**. However, the historical record does not include GPU utilization, GPU power, CPU utilization, per-stage timings, or reserved VRAM. It is not possible to prove the bottleneck from saved artifacts alone. The current `nvidia-smi` snapshot is idle and cannot be used to infer historical utilization.

## 5. Recommendations

### Batch size

Use **64 as the largest conservative unprofiled batch recommendation** for a 4 GiB RTX 2050. Batch 128 is a plausible candidate from the measured scaling, but should be attempted only after a batch-128 forward/backward memory preflight on the target machine and with no competing GPU workload. If training protocol comparability requires effective batch 16, do not increase effective batch solely because VRAM allows it; use gradient accumulation or explicitly treat a larger effective batch as a changed protocol.

### Improve input delivery before pursuing larger batches

1. Add a modest DataLoader worker count (start around 2–4, then profile), `persistent_workers=True`, and a small `prefetch_factor`, retaining `pin_memory=True`. Verify CPU and disk saturation rather than assuming more workers always help.
2. For validated `cf32_le` SigMF captures, read only the selected complex sample window via memory mapping or byte-range seeking instead of materializing the full 262,144-sample file. A 4,096-sample complex64 crop is 32 KiB versus 2 MiB for the entire file, a 64× reduction in bytes requested per crop. Preserve random crop semantics and use actual metadata.
3. Avoid reparsing SigMF for every item: use the already parsed sample rate/dtype from the record after verifying that manifest metadata matches the sidecar. Keep the IQ data lazy.
4. Time the TRAIN scaler pass separately from epochs, then time IQ load, feature extraction, PSD construction, transfer, and model step separately. Record GPU utilization and both allocated/reserved peaks during a future run.
5. If CPU preprocessing remains dominant, consider a disk-backed cache of crop-independent items only where augmentation semantics remain exact. Do not cache rotated representations while random phase augmentation is enabled; those views must be regenerated from the rotated IQ.

## Conclusion

The 247 MB number is credible for this compact architecture at batch 16 and does not imply a 4 GB GPU was underconfigured. Parameter memory is below 4 MiB including ordinary AdamW state; most peak allocation is transient activation/workspace memory. Static shape analysis and the measured peak support batch 64 as conservative and batch 128 as plausible but not verified. The larger efficiency concern is likely CPU/input starvation caused by serial full-file loading and per-sample representation construction. Optimize and profile that pipeline before assuming a larger batch will materially shorten the run.
