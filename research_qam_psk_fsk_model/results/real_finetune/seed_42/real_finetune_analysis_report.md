# Real Fine-Tuning Run Analysis — Seed 42

## Scope and data boundary

This report analyzes saved artifacts only. No training, inference, evaluation run, checkpoint creation, or TEST access was performed. The stored validation predictions were recomputed into descriptive counts/metrics solely to verify their consistency with the saved reports. The reserved TEST manifest was not read.

The run used 546 TRAIN examples (78/class) and 119 VALIDATION examples (17/class), with the split hashes recorded in `config.json`. Validation was used for checkpoint selection, so all results below are development results and should not be treated as an independent final estimate.

## Artifacts inspected

The run directory contains and was checked for `metrics.json`, `best_metrics.json`, `classification_report.json`, `confusion_matrix.csv`, `predictions.csv`, `training_history.csv`, `training.log`, `config.json`, `scaler_audit.json`, `training_hardware.json`, `best.pt`, and `final.pt`. For the two checkpoints, only serialized metadata fields and the count of state-dict entries were inspected; tensor values were not read for inference or comparison.

## Consistency and checkpoint selection

- `predictions.csv` contains 119 rows. The row counts by true label are 17 for each of the seven classes.
- Reconstructing counts from the saved prediction labels gives 119 total and exactly matches both the saved `confusion_matrix.csv` and the matrix in `metrics.json`.
- Accuracy, macro precision, macro recall, macro-F1, and every per-class precision/recall/F1 recomputed from those saved rows match `metrics.json` and `classification_report.json` (floating-point reporting precision aside).
- `best_metrics.json` matches `metrics.json` for the selected validation predictions and metrics.
- `metrics.json`, `training.log`, and `training_history.csv` all identify epoch 18 as best. Epoch 18 has the highest selection tuple `(macro-F1, 64QAM recall, accuracy)`. Epoch 21 ties the metrics tuple but has worse validation loss, so it does not replace the earlier best checkpoint. The `best.pt` and `best_metrics.json` timestamps align; `final.pt` was written later after training ended. The checkpoint's serialized metadata does not itself store the epoch number, so correspondence is verified from the run log/history and artifact timing rather than an epoch field inside the checkpoint.
- `config.json` records the expected TRAIN/VALIDATION manifest hashes and counts, `test_manifest_opened=false`, `test_iq_opened=false`, and `validation_used_for_scaler_fit=false`. The saved scaler audit reports synthetic TRAIN as the scaler-fit source, consistent with the configured pretrained-checkpoint scaler policy.

## Training dynamics

Training completed 26 of the configured maximum 40 epochs, with best validation macro-F1 at epoch 18 and patience 8. Epoch 18 achieved validation loss 0.13258, accuracy 0.86555, macro-F1 0.86831, macro precision 0.88520, and macro recall 0.86555. At epoch 26, validation loss was 0.14837, accuracy 0.84034, and macro-F1 0.84293. Thus the selected epoch exceeds the final epoch by 2.52 percentage points in accuracy and 2.54 points in macro-F1; validation loss is 0.01579 lower.

The epoch-18 selection is consistent with the configured lexicographic rule: macro-F1 first, then 64QAM recall, then accuracy. Epoch 21 ties the macro-F1/64QAM/accuracy values but has a slightly higher validation loss; validation loss is not a selection criterion. Epochs 19–26 do not beat the epoch-18 selection tuple, so the eight-epoch patience stop occurs at epoch 26 as configured.

There is some validation fluctuation after epoch 18, particularly in epochs 20 and 24–26, and the selected best is not the last epoch. This is compatible with mild late-run overfitting or ordinary validation noise on 119 samples, but the saved history does not show a large sustained train/validation divergence. The training objective includes phase rotation and low-SNR QAM sample weighting, while validation is unaugmented and unweighted; therefore train and validation focal-loss values are not directly comparable as a conventional generalization gap.

## Synthetic pretraining versus real fine-tuning

The synthetic-pretraining metrics are measured on a large synthetic validation set (714 examples/class, 4,998 total); real fine-tuning metrics are on 119 real validation examples (17/class). The two scores are from different domains and sample counts, so the differences are descriptive and cannot isolate transfer-learning gain.

| Metric | Synthetic validation | Real validation after fine-tuning | Difference | Relative difference vs synthetic |
|---|---:|---:|---:|---:|
| Accuracy | 78.33% | 86.55% | +8.22 pp | +10.50% |
| Macro-F1 | 0.7816 | 0.8683 | +0.0867 | +11.10% |
| Macro precision | 0.7921 | 0.8852 | +0.0931 | +11.76% |
| Macro recall | 78.33% | 86.55% | +8.22 pp | +10.50% |

These comparisons show that the final model performs well on the saved real validation split and that scores are higher than on synthetic validation. They do **not** prove synthetic pretraining caused the difference. A same-architecture, same-protocol real-data-from-scratch run would be required to estimate that counterfactual; it is not available here. Whether scratch would likely be worse is therefore unknown. The run demonstrates that the transfer procedure completed and adapted to real data; the causal usefulness of pretraining remains unquantified.

## Class-wise results

| Class | Synthetic P / R / F1 | Fine-tuned real P / R / F1 | Real minus synthetic F1 |
|---|---:|---:|---:|
| BPSK | 87.3 / 91.5 / 89.3% | 100.0 / 100.0 / 100.0% | +10.7 pp |
| QPSK | 56.9 / 80.0 / 66.5% | 60.0 / 88.2 / 71.4% | +4.9 pp |
| 8PSK | 76.4 / 70.3 / 73.2% | 92.3 / 70.6 / 80.0% | +6.8 pp |
| 16QAM | 76.9 / 55.6 / 64.6% | 85.7 / 70.6 / 77.4% | +12.9 pp |
| 64QAM | 72.2 / 61.6 / 66.5% | 87.5 / 82.4 / 84.8% | +18.3 pp |
| 2FSK | 95.7 / 97.6 / 96.7% | 100.0 / 100.0 / 100.0% | +3.3 pp |
| 4FSK | 88.9 / 91.7 / 90.3% | 94.1 / 94.1 / 94.1% | +3.8 pp |

Again, this is a cross-domain descriptive comparison, not a matched test of per-class transfer improvement. Within the real validation set, BPSK and 2FSK are strongest (17/17 correct each); 8PSK is weakest by recall (12/17), while QPSK has the lowest precision (0.60) and F1 (0.714).

### Remaining confusion patterns

The largest individual error is 8PSK→QPSK (5/17, 29.4%). QPSK has one error to 8PSK and one to 4FSK. Three 16QAM captures map to QPSK and two to 64QAM. Two 64QAM captures map to 16QAM and one to QPSK. 4FSK has one error to QPSK. No errors leave BPSK or 2FSK.

## QAM

The real validation matrix contains two 16QAM→64QAM errors and two 64QAM→16QAM errors. Thus 4/34 QAM captures (11.8%) are confused directly between the two QAM orders. Synthetic validation had 164/714 (23.0%) 16QAM→64QAM and 111/714 (15.5%) 64QAM→16QAM; combined, 275/1,428 (19.3%). The observed real split has a lower combined cross-confusion fraction by 7.5 percentage points, but the domains and sample sizes differ. Real validation recalls are 12/17 (70.6%) for 16QAM and 14/17 (82.4%) for 64QAM. This is a noticeable improvement in the held development split relative to synthetic-validation class scores, not a controlled transfer effect.

## PSK

Real validation has 5/17 8PSK→QPSK errors, versus 1/17 QPSK→8PSK. Synthetic validation had 167/714 8PSK→QPSK and 97/714 QPSK→8PSK. The asymmetric direction remains: 8PSK is more often absorbed into QPSK. The combined real cross-confusion share is 6/34 (17.6%) versus 264/1,428 (18.5%) synthetic, but class-specific rates and different validation distributions should be considered; this small real sample does not establish a robust change. BPSK is perfectly classified on the real validation set.

## FSK

Real validation has 17/17 correct 2FSK and 16/17 correct 4FSK. There are zero 4FSK→2FSK errors and zero 2FSK→4FSK errors; the single 4FSK error is 4FSK→QPSK. Synthetic validation had 13/714 2FSK→4FSK and 20/714 4FSK→2FSK. This real validation outcome looks like near-ceiling performance for these 34 captures, but it does not demonstrate population-level saturation: each class has only 17 validation examples, and FSK subgroup counts by condition are smaller. The historical frozen baseline's much lower 4FSK recall is not a matched architecture/protocol comparison.

## Transfer-learning assessment

The transfer run is operationally successful: it initialized from the synthetic checkpoint, used the configured frozen warm-up/selective unfreezing policy, completed, selected an epoch, and achieved 86.6% accuracy and 0.868 macro-F1 on real validation. The configuration preserved the pretrained feature and PSD scalers; no validation fitting is recorded. The final learned fusion weights remain distributed across branches, with IQ largest at 0.302; features 0.161, polar 0.158, PSK 0.142, IF 0.120, and PSD 0.118.

However, transfer benefit versus real-data training from scratch is **not estimable** from this run alone. The synthetic validation score is not a pre-fine-tuning score on the same real validation examples. Therefore the evidence supports “the transferred model performs strongly on this validation set,” not “pretraining improved performance by X.”

## Hardware and throughput

The run used CUDA AMP on one RTX 2050 (4,294,443,008 bytes reported VRAM), batch 16, four DataLoader workers, persistent workers, prefetch 2, and pinned memory. Training took 98.44 seconds for 26 epochs; history sums to 95.96 seconds of epoch time, leaving about 2.49 seconds of run-level overhead. The first epoch was 31.4 seconds, while epochs 2–26 averaged about 2.58 seconds, consistent with a slower first-epoch startup/warm-up. The hardware record reports peak training allocated/reserved memory of 71.6/88.1 MB.

Across the run, recorded batch-wait time sums to 24.0 seconds, CUDA-event compute to 29.6 seconds, and validation to 20.3 seconds. These timers can overlap due asynchronous GPU work and worker prefetch, so they should not be interpreted as additive exclusive wall-time shares. Worker-side file-load and preprocessing sums are 64.1 and 128.0 seconds respectively; these are per-worker sample durations summed across workers, not elapsed wall time. That indicates substantial CPU work, but GPU-utilization snapshots are intermittent point samples rather than an integrated utilization trace.

The 26 epoch utilization snapshots average 30.4% (median 24%; 16/26 nonzero). This suggests the GPU was not continuously saturated, but the sampling method is too coarse to quantify utilization efficiency. Overall, the run appears **mixed/input-sensitive rather than demonstrably GPU-compute-bound**: batch wait and CPU-derived-view computation are material, while CUDA compute time is also substantial. The synthetic run took 26,002 seconds for 40 epochs and reported 247.4 MB peak allocated VRAM, but had no timing breakdown, DataLoader telemetry, or GPU-utilization history. Its much larger sample workload makes direct epoch-time comparison misleading.

The hardware `parameter_counts.total=179,949` is the trainable-parameter count at the selected epoch, not the architecture's full parameter count. The checkpoint config is the same fixed 228,701-parameter model; about 48,752 parameters remained frozen at the selected epoch.

## Readiness and recommendation

- **Held-out TEST readiness:** Proceed to a one-time, separately authorized TEST evaluation using `best.pt`, provided the test protocol is fixed before opening the split. This run has a coherent frozen checkpoint and consistent validation artifacts; validation has already served selection.
- **Additional fine-tuning:** Do not extend this run based on the current validation alone. The selected checkpoint beats the final epoch, and validation is small. More tuning on this same split risks adapting to its 119 examples.
- **Overfitting:** There is late validation regression/noise after epoch 18, but no strong sustained divergence is visible. Report as mild possible overfit/validation variance, not a definitive diagnosis.
- **Early stopping:** The run stopped at epoch 26 after eight epochs without improvement over epoch 18's selection tuple; the saved artifacts are consistent with patience 8.
- **Transfer conclusion:** Strong real-validation result after transfer; incremental benefit over same-architecture training from scratch is unknown.

## Limitations

The real validation set has 17 samples/class; one changed prediction shifts recall by 5.9 percentage points. Metrics were also used for checkpoint selection. Synthetic and real validation differ in sample count and domain. No matched real-data scratch control is present. Hardware utilization is sampled once per epoch and worker timing accumulates sample durations, limiting bottleneck precision. No TEST result is included here.
