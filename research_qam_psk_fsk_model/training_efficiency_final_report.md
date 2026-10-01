# Training Efficiency and Memory Preflight

## Scope

This is a pipeline and memory-preflight report only. No training, fine-tuning, model evaluation, or IQ preprocessing was run for this report. The four batch-size probes used zero-valued tensors of the model's expected shapes, executed forward and backward to measure activation memory, and did not call an optimizer step or save a checkpoint.

## Existing pipeline and measured history

The completed synthetic-pretraining hardware record reports 228,701 model parameters, CUDA AMP on an NVIDIA GeForce RTX 2050 with 4 GiB VRAM, batch size 16, and peak allocated memory of 247,409,152 bytes (about 236 MiB). That historical record used `num_workers=0`; historical GPU utilization and reserved-memory peaks were not recorded, so they cannot be reconstructed from that run.

Each dataset item lazily opens one IQ file, reads its SigMF sample rate, crops 4096 samples, and computes the same six branch inputs, including the existing 22 features and Welch PSD. It loads the full source recording into a per-worker NumPy array before cropping. The source IQ capture size is 2 MiB, so a 546-item TRAIN set plus 119-item VALIDATION set iterated for 40 epochs represents about 1.91 TiB of logical file reads if every full capture is reread each epoch; this is a volume estimate, not measured disk traffic, and excludes scaler calibration. The feature and PSD computations are CPU-side and happen per item. The old run's 26,002-second training time and absent GPU telemetry make input-pipeline limitation plausible, but not proven.

The pipeline now supports `num_workers`, `persistent_workers`, `prefetch_factor`, and CUDA-conditional `pin_memory`. The prepared real fine-tuning config uses four workers, persistent workers, prefetch factor 2, and pinned batches. Shared epoch state ensures persistent workers see updated epochs, preserving changing deterministic crops and phase rotations. Optional epoch timing separates DataLoader wait, worker file-load time, worker preprocessing time, GPU event compute time, validation, and total epoch time. Optional `nvidia-smi` snapshots record GPU utilization and memory once per epoch.

## Zero-data memory probe

RTX 2050 measured total VRAM: 4,294,443,008 bytes. Numbers below are the observed peak allocated/reserved during a zero-input forward/backward probe; this did not include AdamW's lazily allocated state. The training estimate adds parameter/gradient/optimizer-state allowance and a 25% planning margin. It is not a guarantee for every driver/workspace state.

| Batch | Probe allocated | Probe reserved | FP16 conv/pool activation floor | Estimated training peak + 25% margin | Probe OOM |
|---:|---:|---:|---:|---:|:---:|
| 16 | 244.7 MB | 255.9 MB | 66.6 MiB | 308.2 MB | No |
| 32 | 470.4 MB | 478.2 MB | 133.2 MiB | 590.3 MB | No |
| 64 | 911.9 MB | 933.2 MB | 266.5 MiB | 1,142.2 MB | No |
| 128 | 1,803.6 MB | 1,849.7 MB | 533.0 MiB | 2,256.7 MB | No |

The stage activation floor counts the specified intermediate feature-map sizes at two bytes per element for FP16. It is a lower-bound accounting of those stage outputs; autograd-saved tensors, convolution workspaces, allocator fragmentation, and concurrent desktop GPU use are not fully represented. The direct memory probe captures substantially more than that simple floor, but still lacks optimizer state and real-data variability.

## Assessment and recommendation

- **Memory-bound:** No at batch 16, 32, or 64 in the probe. Batch 128 fits the measured 4 GiB device with roughly 2.04 GiB free after the conservative estimate, but the probe is not a complete live-run guarantee.
- **Input-pipeline-bound vs compute-bound:** Historical evidence is insufficient for a categorical diagnosis. Full-file reads plus CPU feature/PSD extraction at every item make an input bottleneck plausible. Use the newly added timings and GPU snapshots on the eventual run to verify. Worker preprocessing sums may exceed epoch wall time because workers overlap.
- **Fastest safe starting configuration:** four workers, persistent workers, prefetch factor 2, pinned memory, AMP, and batch 16 for the 546-example fine-tuning set. Batch 16 is intentionally retained for small-data optimization and protocol consistency; a larger batch may improve throughput only if GPU compute is currently limiting. For a future controlled throughput run, batch 64 is a cautious candidate; batch 128 is the largest batch that passed this zero-input memory probe, not an unconditional recommendation for production use.
- **Next tuning evidence:** Compare batch-wait, worker preprocessing, GPU compute, epoch wall time, GPU utilization, and memory on the next authorized training run. Do not change scientific model behavior based solely on the current estimate.

## Machine-readable source

The exact probe readings are in `results/real_finetune/memory_preflight.json`. The new DataLoader and timing controls are implemented in `training/train_multimodal.py` and `data/multimodal_dataset.py`.
