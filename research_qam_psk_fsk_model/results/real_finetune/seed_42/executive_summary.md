# Executive Summary — Real Fine-Tuning, Seed 42

## Result

- **Best epoch:** 18 of 26 completed
- **Real validation accuracy:** 86.55% (103/119)
- **Macro-F1:** 0.8683
- **Macro precision / recall:** 0.8852 / 0.8655
- **Compared with synthetic validation:** accuracy +8.22 percentage points, macro-F1 +0.0867, macro precision +0.0931, macro recall +8.22 points. These are descriptive cross-domain differences, not an estimate of transfer-learning gain.

## Strengths

- BPSK and 2FSK: 17/17 correct each.
- 4FSK: 16/17 correct; no 4FSK→2FSK mistakes in this validation set.
- 64QAM: 14/17 correct, recall 82.4%.
- Saved predictions, confusion matrix, classification report, and aggregate metrics are consistent.
- Patience-based stopping selected epoch 18 and stopped after eight subsequent non-improving epochs.

## Weaknesses

- 8PSK recall is 70.6%; five 8PSK samples were classified as QPSK.
- QPSK precision is 60.0%, with errors including incoming 8PSK and 16QAM samples.
- 16QAM recall is 70.6%; two 16QAM↔64QAM confusions occur in each direction.
- Only 17 validation examples per class; results are development estimates and were used for checkpoint selection.
- There is no same-architecture real-data-from-scratch control, so the gain attributable specifically to pretraining cannot be quantified.

## Recommendation

**Proceed to a one-time held-out TEST evaluation** with the selected `best.pt`, under a fixed evaluation protocol and without further tuning on the current validation set. This recommendation is based on internally consistent artifacts and strong validation results, not on a claim of generalization or proof that pretraining outperforms scratch training.

No TEST data was accessed while preparing this analysis. No training, inference, evaluation run, or checkpoint modification was performed.
