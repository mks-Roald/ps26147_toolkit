# Synthetic Pretraining Run Analysis — Seed 42

## Scope and artifact inventory

This is a read-only analysis of the completed `synthetic_pretrain/seed_42` artifacts. No training, model inference, evaluation, or TEST access was performed for this analysis. No TEST artifact was present or opened. The run directory did **not** contain a `report.md`; this document is a separate analysis artifact and does not replace any run output.

Reviewed artifacts: `config.json`, `training_history.csv`, `training.log`, `metrics.json`, `best_metrics.json`, `classification_report.json`, `confusion_matrix.csv`, `predictions.csv`, `best_validation_outputs.npz` (array keys/shapes and consistency only), `scaler_audit.json`, `training_hardware.json`, and metadata/state keys from `best.pt` and `final.pt`. The saved predictions, saved NPZ vectors, and confusion matrix are aligned across all 4,998 validation rows. Checkpoint weights were not run through inference.

## Run setup and headline results

- Stage: synthetic pretraining, seed 42; class order: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK.
- Split: 20,002 TRAIN and 4,998 VALIDATION, 714 validation samples per class. The scaler audit reports TRAIN-only fitting on 20,002 captures; validation was not used for scaler fitting.
- Optimizer/loss: AdamW, learning rate 0.0005, weight decay 0.0001, effective batch 16, focal gamma 2.0, low-SNR QAM weight 0.25, maximum 40 epochs, patience 8. AMP was enabled.
- Hardware: NVIDIA GeForce RTX 2050 (4,294,443,008 reported VRAM bytes), CUDA 13.0, PyTorch 2.14.0+cu130. Peak allocated VRAM reported as 247,409,152 bytes. Elapsed training time was 26,002.3 s (about 7 h 13 min; about 10.8 min/epoch across 40 epochs).
- Selected checkpoint: `best.pt`, epoch 38. `final.pt` is the epoch-40 state. Final `metrics.json` was produced after reloading the best checkpoint, so its reported predictions/metrics are epoch 38, not epoch 40.
- Best validation metrics: accuracy 78.33%, macro-F1 0.7816, macro precision 0.7921, macro recall 0.7833, focal validation loss 0.26485.

## 1. Training dynamics

Training loss declined from 0.5532 at epoch 1 to 0.2041 at epoch 40. Validation loss was variable early (0.5082 at epoch 1, with several increases) and trended down to 0.2612 by epoch 40. Validation accuracy rose from 61.16% to 77.99% at epoch 40; macro-F1 rose from 0.6006 to 0.7791. This is evidence of useful learning and late-stage stabilization on this synthetic split, not evidence of real-world generalization.

Validation metrics remained noisy. For example, macro-F1 fell from 0.7728 at epoch 37 to 0.7476 at epoch 39, then recovered to 0.7791 at epoch 40. Class-level 64QAM recall also fluctuated materially. There are no NaN/Inf reports in the history/log artifacts, and all 40 epochs completed, so the recorded run appears numerically stable. The log does not contain per-batch diagnostics or gradient norms, so this assessment is limited to saved epoch summaries.

There is no clean train-vs-validation generalization-gap estimate here. The training loss uses random crops and phase rotation and applies 0.25 sample weighting to QAM below 0 dB; validation uses deterministic unaugmented crops and unweighted loss. Their losses are therefore not directly like-for-like. The decreasing training loss alone does not establish overfitting. The validation curve continues to improve late, with no sustained late collapse; strong overfitting is not evident from the saved trajectory. Modest remaining underfit cannot be ruled out, especially for QAM, but the QAM errors may reflect overlap/difficulty rather than optimization alone.

### Focal-loss interpretation

The configured focal factor is `(1 - p_t)^2`, multiplying per-sample cross-entropy. It reduces the relative contribution of easy examples and emphasizes harder examples. The training code additionally multiplies the loss for 16QAM/64QAM at SNR below 0 dB by 0.25. This means the challenging low-SNR QAM examples are explicitly down-weighted, despite focal loss, and may receive substantially less total contribution than equally hard non-QAM samples. Validation loss is not subject to this special weight.

The run has no cross-entropy or weight-1 ablation, so the measured effects of focal loss and low-SNR weighting cannot be separated. The especially poor low-SNR QAM recall is consistent with the weighting being a risk to investigate, but it does not prove that the weight caused those errors.

## 2. Best epoch and epoch 40

The checkpoint-selection priority is macro-F1, then 64QAM recall, then accuracy. Epoch 38 reached the maximum saved macro-F1: **0.78156**, with accuracy **0.78331**, macro precision **0.79206**, validation loss **0.26485**, and 64QAM recall **0.61625**. Epoch 40 had macro-F1 **0.77907**, accuracy **0.77991**, validation loss **0.26122**, and 64QAM recall **0.64426**.

Epoch 38 was selected because macro-F1 is the primary key and was higher by 0.00250 than epoch 40. Epoch 40's lower loss and higher 64QAM recall do not override that primary criterion. Accuracy was also 0.34 percentage points higher at epoch 38. Epoch 39 was weaker (macro-F1 0.74758, accuracy 0.75970), followed by recovery in epoch 40. Thus epoch 40 is not a simple monotonic degradation; it traded a small amount of macro-F1/accuracy for better 64QAM recall and lower focal validation loss. Per-class epoch-40 metrics and confusion counts are not saved separately. Producing them would require evaluation, which was not done.

## 3. Class-wise results

All classes have support 714.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| BPSK | 0.8730 | 0.9146 | 0.8933 |
| QPSK | 0.5693 | 0.7997 | 0.6651 |
| 8PSK | 0.7641 | 0.7031 | 0.7323 |
| 16QAM | 0.7694 | 0.5560 | 0.6455 |
| 64QAM | 0.7225 | 0.6162 | 0.6652 |
| 2FSK | 0.9574 | 0.9762 | 0.9667 |
| 4FSK | 0.8887 | 0.9174 | 0.9028 |

The strongest class is 2FSK, followed by 4FSK and BPSK. The weakest recall is 16QAM (55.60%); QPSK has the weakest precision (56.93%), reflecting many false positives. FSK recognition is strong on this synthetic held-out split: 2FSK→4FSK is 13/714 (1.82%), 4FSK→2FSK is 20/714 (2.80%), and 4FSK→QPSK is 33/714 (4.62%).

### PSK group

BPSK is relatively well separated. QPSK and 8PSK are the main PSK ambiguity: 97/714 QPSK examples go to 8PSK (13.59%), while 167/714 8PSK examples go to QPSK (23.39%). QPSK recall is higher than 8PSK recall, but its precision is lower because 8PSK is frequently predicted as QPSK. Smaller confusions include QPSK→BPSK (22) and 8PSK→BPSK (17).

### QAM group

16QAM→64QAM occurs 164/714 times (22.97%); 64QAM→16QAM occurs 111/714 times (15.55%). The total pair confusion is 275/1,428 QAM validation cases (19.26%). A further 95 16QAM and 101 64QAM examples are classified as QPSK. Per-class saved-prediction breakdown shows a steep SNR association:

| SNR | 16QAM recall | 64QAM recall |
|---:|---:|---:|
| -10 dB | 1/93 (1.1%) | 0/97 (0.0%) |
| -5 dB | 43/103 (41.7%) | 16/120 (13.3%) |
| 0 dB | 44/103 (42.7%) | 55/92 (59.8%) |
| 5 dB | 65/108 (60.2%) | 92/107 (86.0%) |
| 10 dB | 67/96 (69.8%) | 86/98 (87.8%) |
| 15 dB | 90/108 (83.3%) | 97/101 (96.0%) |
| 20 dB | 87/103 (84.5%) | 94/99 (94.9%) |

This pattern makes low-SNR QAM the highest-priority transfer risk. It is descriptive, and the SNR-dependent training weight is a plausible contributing training choice but not a demonstrated cause.

### FSK group

2FSK recall is 97.62%; 4FSK recall is 91.74%. Only 33/1,428 total FSK cases cross directly between 2FSK and 4FSK (13 one direction, 20 the other). The larger 4FSK error destination is QPSK (33). The saved conditional summary shows 4FSK recall of 60.9% at -10 dB, versus roughly 91–100% at other SNR bins. This again points to a synthetic low-SNR challenge rather than a broad FSK-family failure on the generated distribution.

## 4. Confusion matrix

Rows are true class; columns are predicted class, in canonical order.

| True \ Pred | BPSK | QPSK | 8PSK | 16QAM | 64QAM | 2FSK | 4FSK |
|---|---:|---:|---:|---:|---:|---:|---:|
| BPSK | 653 | 33 | 11 | 0 | 5 | 2 | 10 |
| QPSK | 22 | 571 | 97 | 4 | 0 | 1 | 19 |
| 8PSK | 17 | 167 | 502 | 4 | 0 | 2 | 22 |
| 16QAM | 28 | 95 | 19 | 397 | 164 | 4 | 7 |
| 64QAM | 23 | 101 | 26 | 111 | 440 | 2 | 11 |
| 2FSK | 1 | 3 | 0 | 0 | 0 | 697 | 13 |
| 4FSK | 4 | 33 | 2 | 0 | 0 | 20 | 655 |

## 5. Architecture and fusion

The six branch encoders are IQ, Polar, nonlinear PSK, IF, PSD, and the handcrafted 22-feature branch. The learned best-checkpoint fusion weights are:

| Branch | Weight |
|---|---:|
| IQ | 0.3018 (30.18%) |
| Handcrafted 22 features | 0.1606 (16.06%) |
| Polar | 0.1576 (15.76%) |
| Nonlinear PSK | 0.1417 (14.17%) |
| IF | 0.1203 (12.03%) |
| PSD | 0.1180 (11.80%) |

IQ has the largest learned scalar coefficient, about twice each auxiliary branch coefficient. The 22-feature branch is the second largest. No single branch exceeds 31%; the weights are relatively distributed across all six branches. The scalar weights are shared globally and applied after branch LayerNorm; they are not per-sample gates. They do not measure causal contribution or prove that a branch is useful. No branch-ablation results are part of this run, so branch-specific performance effects cannot be isolated from these artifacts.

The checkpoint stores the six encoders, branch/fusion parameters, class and feature order, architecture configuration, representations, feature scaler, PSD scaler, focal/low-SNR settings, seed, and stage. Best-checkpoint fusion values match the values in `metrics.json`. The final checkpoint has slightly different fusion coefficients (IQ 0.2997, features 0.1610, Polar 0.1592, PSK 0.1422, IF 0.1200, PSD 0.1179), consistent with its being a later epoch state; no final-checkpoint inference was performed.

## 6. Synthetic-data and model assessment

The held-out synthetic validation set is substantial (4,998 examples, 714/class) and metrics are well above chance. The model learned the generator's FSK patterns well and learned BPSK well; the remaining errors concentrate in adjacent PSK/QAM distinctions and low-SNR QAM. The validation performance and improving late training trajectory do not indicate a simple capacity-wide failure or obvious general underfit.

This does **not** distinguish architecture limitations from generator/data limitations. TRAIN and VALIDATION are drawn from the same synthetic generation source and one random seed/run was analyzed. No real-data performance is present in these artifacts. For the synthetic task, sample count alone does not look like the leading limitation; low-SNR QAM geometry, representation/decision boundaries, or distribution construction are plausible areas, but none can be singled out causally. Do not treat the 78.3% synthetic score as a real-capture forecast.

## 7. Fine-tuning readiness

### Checkpoint status

`best.pt` is a valid synthetic-pretraining initialization candidate with selected epoch-38 weights, the 7-class ordering, all 22 feature names, both fitted scalers, architecture and preprocessing metadata, seed 42, and synthetic stage metadata. No reserved TEST information is present in the saved config (`test_manifest_opened=false`, `test_iq_opened=false`). The saved validation has not been used as evidence about real-domain performance.

### Readiness verdict

**The weights are ready as a candidate for real-data fine-tuning; the current real fine-tuning configuration is not yet ready to launch without decisions/fixes.** The saved config leaves the `real_finetuning_*` paths empty. In addition, the current trainer fits new feature and PSD scalers from whichever stage's TRAIN manifest it receives, then loads the model weights. Thus the pretrained encoders would receive real-stage standardized features/PSD using new real scalers rather than the scaler transforms saved in `best.pt`. That may be a deliberate domain adaptation, but it changes the inputs seen by the pretrained feature and PSD branches. It should be made explicit and recorded before the first transfer run; the validation data must not be used to fit those scalers.

The existing trainer also exposes one global learning rate and no staged freeze/unfreeze or discriminative learning-rate groups. The concrete schedule below therefore requires a small future training-pipeline/config enhancement, or a simpler all-trainable run at a low global learning rate. This analysis does not make that code change.

### Recommended conservative plan (not executed)

1. **Set the target paths** to the already-established real TRAIN and VALIDATION manifests only. Keep the existing split unchanged; do not open or load TEST.
2. **Resolve normalization explicitly before training.** For the first clean transfer baseline, prefer reusing the synthetic feature and PSD scalers stored in `best.pt`, so the branch inputs remain in the coordinates used during pretraining. Record that policy. If real-TRAIN-fitted scalers are preferred to adapt normalization, treat that as a separate, explicitly named transfer condition; fit only on real TRAIN and recognize that the MLP inputs shift. Never fit on real VALIDATION.
3. **Warm-up:** freeze IQ, Polar, PSK, IF, PSD, and feature encoders. Train only branch projections, fusion LayerNorm/weights, and classifier for 3–5 epochs at LR **1e-4**, AdamW with weight decay **1e-4**.
4. **Selective fine-tune:** unfreeze the final convolution block of each signal encoder and the final feature-MLP layer. Use discriminative rates of **1e-5** for these unfrozen encoder parameters and **3e-5** for fusion/projection/classifier parameters. Keep the early encoder blocks frozen initially. Use macro-F1 → 64QAM recall → accuracy selection, early stopping (patience about 5), and no threshold tuning.
5. **Loss:** keep focal gamma 2.0 for the first transfer run, but set low-SNR QAM weighting to **1.0** initially so low-SNR QAM examples are not further down-weighted on the target domain. This is a conservative proposal, not a result; if retaining 0.25 is scientifically required, compare it as a separately prespecified condition rather than changing it after looking at validation outcomes.
6. **Small-validation safeguards:** use the fixed real VALIDATION split only for checkpoint selection and report per-class counts/metrics alongside macro-F1. With 17/class, small changes are noisy. Avoid broad hyperparameter sweeps and do not optimize on the historical nine-error subset.
7. **Save an immutable transfer copy** under a new real-finetune run path; preserve `best.pt` from synthetic pretraining. Log scaler source/hashes, frozen layers, per-group learning rates, seed, selected epoch, per-class metrics, and confusion counts.

The existing trainer cannot currently express the proposed warm-up and differential rates, and its real stage currently refits scalers. Do not launch the above plan until those behaviors are either added/configured or a deliberate single-rate/scaler policy is selected.

## Limitations and conclusion

This is one seed on a synthetic held-out split from the same generation source. It provides no real-validation evaluation and no test evidence. The run suggests the model has learned strong synthetic FSK/BPSK discrimination, while QPSK↔8PSK and especially low-SNR/adjacent QAM remain weak. Focal loss did not prevent those low-SNR QAM errors, but the 0.25 training weighting and absence of a loss ablation prevent attribution. Learned fusion weights show IQ as the largest global scalar branch, not proof that IQ alone drives decisions.

**Recommendation:** proceed to a carefully controlled real TRAIN/VALIDATION fine-tuning experiment only after normalizer behavior and staged-freezing support are decided. Treat the synthetic checkpoint as a transfer initialization, not as evidence that real-domain performance is solved. No training or evaluation was run for this report.
