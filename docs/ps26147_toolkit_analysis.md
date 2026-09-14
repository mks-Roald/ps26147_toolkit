# PS26147 Toolkit — Code Analysis, Tasks & Solutions

## 1. What the toolkit does
RF signal analysis tool (`.iq` / `.wav` input) that estimates center frequency, bandwidth,
SNR, baud rate, and classifies modulation type (BPSK/QPSK/8PSK/16QAM/64QAM/2FSK/4FSK/AM)
using a Random Forest over 12 hand-engineered features.

## 2. Sample frequency handling
| File type | Source of `fs` |
|---|---|
| `.wav` | Read from file's own header (`scipy.io.wavfile.read`) — real metadata |
| `.iq`  | **No metadata in file at all** (raw interleaved float32, no header). Comes from `--fs` CLI flag, silently defaults to 1,000,000 Hz if not passed |

## 3. Core algorithms (as implemented)
- **Center frequency**: Welch PSD → Savitzky-Golay smoothing → peak bin → walk out to
  half-power (−3dB) region → power-weighted spectral centroid within that region.
- **Bandwidth**: noise-floor (25th percentile) subtracted PSD → continuous span ≥ −10dB of peak.
- **SNR**: in-band power (center ± bw/2) vs noise floor, dB, clipped to [−30, 60].
- **Baud rate**: downconvert to baseband using center_freq → build "transition signal"
  (envelope-derivative + phase-derivative jumps) → FFT of transition signal → strongest
  peak within [Nyquist/bandwidth]-bounded window → autocorrelation fallback.
- **Modulation classifier**: 2nd/4th/6th-order cumulants + envelope/phase/frequency
  statistics + spectral entropy (12 features total) → StandardScaler → Random Forest
  (100 trees, depth 12) trained on **self-generated synthetic data**, not RadioML2016.10a
  as the README claims.

### The 12 classifier features
1. `abs_c20` — 2nd-order cumulant magnitude (AM vs digital)
2. `abs_c40` — 4th-order cumulant magnitude (BPSK/QPSK/8PSK discriminator)
3. `abs_c41` — 4th-order cumulant combination (adds separability, incl. FSK)
4. `c42` — constant-modulus vs QAM discriminator
5. `abs_c60` — 6th-order cumulant (resolves QAM order)
6. `c63` — 6th-order cumulant combination (resolves QAM order)
7. `gamma_max` — peak normalized squared envelope
8. `sigma_aa` — envelope std dev (flat vs varying envelope)
9. `sigma_dp` — instantaneous phase std dev
10. `sigma_af` — instantaneous frequency std dev (primary FSK detector)
11. `spec_entropy` — Shannon entropy of PSD
12. `kurtosis_env` — envelope excess kurtosis

RF hyperparameters: `n_estimators=100`, `max_depth=12`, `random_state=42`, plus a
`StandardScaler` pre-step. 8 output classes.

## 4. Tasks & Solutions

| # | Task (Problem) | Why it hurts accuracy | Solution |
|---|---|---|---|
| 1 | Classifier runs on **raw, non-baseband** signal | Residual carrier offset rotates phase continuously → 4th/6th-order cumulants average toward zero → features lose discriminative power | Downconvert with the already-estimated `center_freq` (same trick used in `estimate_baud_rate`) before calling `classifier.predict()` |
| 2 | Training data is synthetic, not RadioML2016.10a (despite README) | Rectangular pulses (no RRC shaping), no CFO, no timing offset, no multipath, SNR floor only 5dB → real captures look statistically different from training data | Wire up `scripts/download_datasets.py` + `train_classifier.py` to actually train/load a RadioML-based `model.pkl`; if keeping synthetic data, add RRC pulse shaping, random CFO, timing jitter, and SNR range down to ~0dB |
| 3 | DC offset not removed before the **first** spectral pass | LO leakage (common in SDRs) creates a fake 0Hz "carrier" that biases center-frequency estimate | Call `remove_dc_offset()` unconditionally before the first `compute_psd()`, not only inside `--filter`/`--denoise` |
| 4 | `load_iq()` hard-codes float32 interleaved format | Real captures are often int8/uint8/int16 (RTL-SDR, HackRF); wrong dtype = garbage data with no error | Add `--dtype` flag, or support SigMF `.sigmf-meta` sidecar files (also fixes sample-rate metadata gap in Task 6) |
| 5 | Noise floor = 25th percentile of **entire** spectrum (bandwidth + SNR) | Overestimates noise floor when signal occupies a large fraction of the analyzed band → underestimates both BW and SNR | Exclude bins already inside the detected half-power region before computing the percentile |
| 6 | `.iq` sample rate has no metadata, defaults silently to 1MHz | Wrong `fs` silently corrupts every downstream number (BW, baud, cumulant features) | Remove the silent default — require `--fs` explicitly, or adopt SigMF metadata standard |
| 7 | Baud-rate FFT has no windowing | Spectral leakage smears the symbol-clock peak, especially at low SNR | Apply Hann/Hamming window to `transition_signal` before `rfft`; consider smoothing `psd_trans` like the PSD smoothing already used for center frequency |
| 8 | `find_peaks(height=mean(...))` in baud estimator | Weak threshold on noisy spectra → false/missed peaks | Use a percentile-based or SNR-adaptive threshold instead |
| 9 | No tests for `parameter_extractor.py`, `feature_extractor.py`, `classifier.py` | Can't verify accuracy improvements are real — README even claims tests that don't exist in the repo (`test_preprocess.py`, `test_feature_extractor.py`) | Add ground-truth synthetic tests (known fc, known baud, known modulation, known SNR) asserting estimator outputs land within tolerance |
| 10 | No IQ imbalance / image-rejection correction | Mirror image signal around DC (common IQ-imbalance artifact) can be mistaken for real spectral content, corrupting center-freq/BW/SNR | Add an I/Q amplitude-phase balance correction step before spectral analysis |

**Recommended fix order:** #1 → #3 → #2 (biggest, cheapest wins first), then #4/#6 (metadata/format
robustness), then #5/#7/#8 (estimator polish), #9 (so you can measure progress), #10 (hardware-specific).

## 5. Things missing from the current context / worth clarifying next
- **Ground-truth data**: none of the analysis above has been validated against real labeled
  captures — all conclusions are from static code reading. To actually measure "how much did
  accuracy improve," you need a labeled test set (real or well-modeled synthetic with CFO/multipath).
- **Target deployment scenario not specified**: is this for live streaming capture, offline batch
  analysis of stored files, or a hackathon demo? This changes priority — e.g., IQ imbalance
  correction matters a lot more for real SDR hardware capture than for idealized simulation.
- **What SDR/recording hardware produces the `.iq` files** (if any) — determines actual sample
  format (int8/int16/float32) and typical impairments (LO leakage, IQ imbalance) to design fixes for.
- **Acceptable latency/compute budget** — some fixes (spectral denoising, IQ-imbalance correction,
  larger RF model) add compute cost; unclear if this needs to run in real time.
- **Whether RadioML2016.10a is actually available/licensed for your use** — README assumes it, but
  `download_datasets.py` should be checked/run to confirm it actually works end-to-end.
- **Definition of "accuracy" for grading/evaluation** — per-class accuracy, confusion matrix, or a
  single overall number? Matters because the classes are imbalanced in difficulty (PSK orders are
  much easier than QAM orders to separate cleanly).
