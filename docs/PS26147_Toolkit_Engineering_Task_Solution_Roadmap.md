# PS26147 Toolkit — Engineering Improvement Task & Solution Plan

## 1. Purpose

This document converts the repository audit into an actionable engineering roadmap.

The current PS26147 Toolkit is a broad prototype for IQ/signal analysis, modulation classification, demodulation, de-interleaving, FEC decoding, correlation, and a Streamlit/CLI interface.

The main objective of this roadmap is to move the project from:

> **prototype implementations that work on controlled examples**

toward:

> **a robust, measurable, confidence-aware signal-analysis and demodulation toolkit suitable for realistic communications signals.**

---

# 2. Current Overall Assessment

| Area | Current State | Priority |
|---|---|---|
| IQ/file loading | Basic; format handling needs correction | 🔴 Critical |
| PSD/spectrogram | Reasonable prototype | 🟢 Good |
| Center-frequency estimation | Heuristic | 🟠 High |
| Bandwidth estimation | Can be substantially wrong | 🔴 Critical |
| SNR estimation | Not a calibrated estimator | 🔴 Critical |
| Baud-rate estimation | Major weakness | 🔴 Critical |
| Modulation classifier | Prototype-level | 🟠 High |
| Carrier recovery | Needs modulation-specific architecture | 🔴 Critical |
| Timing recovery | Integer decimation rather than true recovery | 🔴 Critical |
| BPSK/QPSK | Controlled-condition prototype | 🟠 High |
| FSK | Oversimplified | 🔴 Critical |
| QAM | Synchronization/mapping improvements needed | 🔴 Critical |
| EVM | Useful but incomplete | 🟠 High |
| De-interleaving | Algorithms exist; autodetection unreliable | 🔴 Critical |
| Viterbi | Basic hard-decision implementation | 🟠 High |
| Reed-Solomon | Needs robust block/status handling | 🟠 High |
| LDPC | Not standards-compatible | 🔴 Critical |
| Bit correlation | Good starting point | 🟡 Medium |
| Preamble discovery | High false-positive risk | 🔴 Critical |
| CLI | Functional but needs robustness | 🟡 Medium |
| Streamlit | Good prototype; caching/UX improvements possible | 🟡 Medium |
| Testing | Too limited for DSP claims | 🔴 Critical |
| Architecture | Good prototype foundation; needs pipeline/result redesign | 🟠 High |

---

# 3. Core Engineering Principle

The current pipeline tends to trust each previous estimate:

```text
Raw IQ
  ↓
PSD
  ↓
Estimated parameters
  ↓
Modulation guess
  ↓
Demodulation
  ↓
Deinterleaving
  ↓
FEC
```

This is risky for unknown signals.

The target architecture should instead be:

```text
Acquire
  ↓
Validate
  ↓
Characterize
  ↓
Detect signal region
  ↓
Estimate parameters + confidence
  ↓
Synchronize
  ↓
Classify
  ↓
Modulation-specific demodulation
  ↓
Soft bits / LLRs
  ↓
Deinterleave
  ↓
FEC
  ↓
Frame synchronization
  ↓
CRC / protocol validation
  ↓
Final payload + confidence + diagnostics
```

Every major stage should provide:

- result
- method
- confidence
- alternatives/candidates where relevant
- warnings
- diagnostics

---

# 4. Task Group A — IQ Input and Metadata

## TASK A1 — Correct `.iq` dtype handling

### Problem

The loader documentation claims support for binary `int16` or `float32`, but the implementation reads the file as `float32`.

An `int16` IQ file can therefore be interpreted incorrectly.

### Solution

Implement explicit format handling:

- `int8`
- `uint8`
- `int16`
- `uint16` where appropriate
- `float32`
- `float64`
- `complex64`
- `complex128`

Also support:

- IQ vs QI ordering
- endianness
- scaling
- DC offset

Example API:

```python
load_iq(
    path,
    dtype=np.int16,
    iq_order="IQ",
    scale=None,
    offset=0,
)
```

### Acceptance criteria

- Known `int16` test file loads correctly.
- Known `float32` test file loads correctly.
- IQ/QI ordering can be selected.
- Invalid/truncated input produces a useful error.
- Loader tests compare loaded samples with known ground truth.

---

## TASK A2 — Introduce signal metadata

### Problem

The application defaults sample rate to approximately 1 MSPS even when raw IQ files may have been recorded at completely different rates.

Incorrect sample rate corrupts:

- bandwidth
- frequency
- baud rate
- filtering
- timing recovery
- synchronization

### Solution

Create:

```python
SignalMetadata(
    sample_rate=1e6,
    center_frequency=433e6,
    iq_dtype="int16",
    iq_order="IQ",
    scale=1/32768,
)
```

Metadata should be explicit rather than guessed.

### UI behavior

If metadata is missing:

> Sample rate is required for accurate frequency/time-domain measurements.

Clearly mark user-supplied metadata.

### Acceptance criteria

- Analysis refuses or warns on missing sample rate.
- Sample-rate range is validated.
- Metadata appears in CLI and Streamlit reports.

---

# 5. Task Group B — Signal Detection and Spectral Analysis

## TASK B1 — Replace peak-only center-frequency estimation

### Problem

Current logic effectively selects the strongest PSD peak.

This fails with:

- multiple signals
- interferers
- FSK
- suppressed-carrier signals
- low SNR
- burst signals
- asymmetric spectra

### Solution

Build a signal mask:

```text
PSD
 │
 │              ███████
 │           ███████████
 │___________████████████________ noise
             ↑       ↑
          f_low    f_high
```

Steps:

1. Compute Welch PSD.
2. Estimate noise floor robustly.
3. Convert to dB.
4. Threshold relative to noise.
5. Find connected spectral regions.
6. Reject insignificant regions.
7. Score candidate regions.
8. Select the signal candidate.
9. Compute weighted centroid.

### Acceptance criteria

Test with:

- one signal
- two signals
- strong adjacent interferer
- low-SNR signal
- FSK
- QAM

Center-frequency error should be reported rather than hidden.

---

## TASK B2 — Implement proper bandwidth measurements

### Problem

The current threshold method treats all above-threshold bins as one signal and does not provide a standards-friendly occupied bandwidth.

### Solution

Provide multiple measurements:

```text
BW -3 dB
BW -10 dB
OBW 95%
OBW 99%
```

For occupied bandwidth, integrate PSD and find the smallest frequency interval containing the requested percentage of power.

### Output

```json
{
  "bw_3db": ...,
  "bw_10db": ...,
  "obw_95": ...,
  "obw_99": ...
}
```

### Acceptance criteria

Compare results against generated signals with known occupied bandwidth.

---

## TASK B3 — Redesign SNR estimation

### Problem

Using a PSD percentile as the noise floor is not a robust SNR measurement.

### Solution

Implement several methods:

1. Noise-band SNR.
2. In-band/out-of-band power method.
3. Pilot/preamble SNR where applicable.
4. Constellation/EVM-derived SNR.
5. Modulation-aware estimates.

Report method and confidence.

Example:

```json
{
  "snr_db": 11.8,
  "method": "noise_band",
  "confidence": 0.76
}
```

### Acceptance criteria

Evaluate SNR estimator on synthetic signals with known SNR from -10 to +30 dB.

---

# 6. Task Group C — Timing and Baud-Rate Estimation

## TASK C1 — Replace simplistic baud-rate estimator

### Problem

The strongest spectral feature in magnitude/phase-difference data is not necessarily the symbol rate.

Harmonics and modulation characteristics can create incorrect estimates.

### Solution

Use multiple candidate estimators:

- spectral nonlinear transforms
- autocorrelation
- cyclostationary features
- eye-opening evaluation
- Gardner timing recovery
- Mueller & Müller after coarse synchronization

Generate candidate symbol rates rather than selecting one blindly.

### Candidate scoring

Score each candidate using:

- timing error
- eye opening
- constellation compactness
- EVM
- carrier lock quality
- preamble correlation

### Acceptance criteria

For generated signals with known baud rates:

- 10 kbaud
- 50 kbaud
- 100 kbaud
- 250 kbaud
- 1 Mbaud

measure estimation error across multiple SNRs.

---

## TASK C2 — Implement fractional timing recovery

### Problem

Current processing effectively uses:

```python
int_sps = round(fs / baud)
```

and integer sample stepping.

For non-integer samples/symbol this accumulates timing error.

### Solution

Implement:

```text
matched filter
      ↓
fractional interpolator
      ↓
Gardner TED
      ↓
timing loop
      ↓
one sample/symbol
```

Optionally support Mueller & Müller.

### Acceptance criteria

Test at:

```text
fs / baud = 2.0
fs / baud = 4.0
fs / baud = 7.37
fs / baud = 10.5
```

with timing offsets from 0 to 1 symbol.

---

# 7. Task Group D — Carrier Synchronization

## TASK D1 — Make carrier recovery modulation-specific

### Problem

The current pipeline applies Costas-style carrier recovery too broadly.

FSK and AM should not be processed as if they were QPSK/BPSK.

### Solution

Use a dispatch architecture:

```text
BPSK  → Costas loop
QPSK  → Costas loop
8PSK  → M-th power / appropriate carrier loop
QAM   → decision-directed PLL
2FSK  → frequency discriminator
4FSK  → frequency discriminator
AM    → envelope or coherent detector
```

### Acceptance criteria

Each modulation has its own synchronization path.

---

## TASK D2 — Add coarse + fine frequency synchronization

### Solution

Use:

```text
FFT/coarse CFO estimate
        ↓
frequency correction
        ↓
PLL/Costas/decision-directed loop
        ↓
lock detector
        ↓
fine tracking
```

Return:

```json
{
  "frequency_offset_hz": ...,
  "phase_offset_rad": ...,
  "lock_metric": ...,
  "locked": true
}
```

---

# 8. Task Group E — Modulation-Specific Demodulation

## TASK E1 — Improve BPSK/QPSK

### Fixes

- phase ambiguity resolution
- coarse CFO
- timing recovery
- carrier lock detection
- soft output
- Gray mapping configuration
- constellation validation

### Phase ambiguity

Try candidate rotations:

```text
0°
90°
180°
270°
```

for QPSK where appropriate, and choose using:

- preamble correlation
- EVM
- known frame structure

---

## TASK E2 — Improve 8PSK

### Problem

Sector slicing alone does not resolve arbitrary constellation rotation.

### Solution

Use phase ambiguity candidates and score them using known symbols/preamble and EVM.

---

## TASK E3 — Correct 16QAM/64QAM mapping

### Problem

The 64QAM implementation uses binary amplitude indexing rather than a Gray-coded mapping.

### Solution

For Gray-coded 64QAM use:

```text
000
001
011
010
110
111
101
100
```

Make mapping configurable because protocols may use different mappings and bit order.

### Acceptance criteria

Round-trip tests must recover exactly the transmitted bit sequence.

---

## TASK E4 — Rebuild FSK demodulation

### Problem

Using phase differences between symbol samples is too simplistic.

### Solution

Estimate instantaneous frequency:

```text
f[n] = fs/(2π) * angle(x[n] * conj(x[n-1]))
```

Then:

```text
frequency smoothing
       ↓
frequency clustering
       ↓
symbol decisions
```

For 4FSK, estimate four clusters rather than using fixed percentiles.

### Acceptance criteria

Test frequency deviations, CFO, noise and unequal tone amplitudes.

---

# 9. Task Group F — EVM, MER and IQ Quality

## TASK F1 — Improve EVM

### Problem

Directly comparing received symbols against reference points without optimal amplitude/phase alignment can inflate EVM.

### Solution

Estimate complex gain:

```text
a = sum(conj(r) * s) / sum(|r|²)
```

Then:

```text
error = s - a*r
```

Calculate:

- RMS EVM %
- EVM dB
- MER

### Acceptance criteria

Known clean constellation should approach very low EVM.

---

## TASK F2 — Add IQ impairment diagnostics

Measure:

- DC offset
- I/Q gain imbalance
- I/Q phase imbalance
- image rejection
- amplitude imbalance
- clipping
- LO leakage

This allows the toolkit to distinguish:

> poor demodulation

from:

> poor source IQ quality.

---

# 10. Task Group G — Modulation Classifier

## TASK G1 — Improve synthetic training data

### Problem

Current synthetic examples are too clean and can teach the model waveform-specific artifacts instead of modulation identity.

### Solution

Generate realistic channels:

```text
random symbols
   ↓
Gray mapping
   ↓
RRC pulse shaping
   ↓
timing offset
   ↓
frequency offset
   ↓
phase offset
   ↓
AWGN
   ↓
Rayleigh/Rician fading
   ↓
multipath
   ↓
IQ imbalance
   ↓
DC offset
   ↓
quantization
```

Train across SNR and channel conditions.

---

## TASK G2 — Evaluate better classifiers

Compare:

- Random Forest
- SVM
- gradient-boosting model
- MLP
- CNN on spectrogram
- CNN on IQ
- hybrid IQ + spectral + engineered features

A strong future architecture:

```text
IQ ──────────────┐
                 ├── feature fusion → classifier
Spectrogram ─────┤
                 │
HOC/features ────┘
```

---

## TASK G3 — Add classifier confidence

Instead of:

```text
QPSK
```

return:

```json
{
  "modulation": "QPSK",
  "confidence": 0.91,
  "alternatives": [
    ["QPSK", 0.91],
    ["BPSK", 0.06],
    ["8PSK", 0.03]
  ]
}
```

The pipeline should refuse automatic demodulation when confidence is too low.

---

# 11. Task Group H — Soft Bits and FEC

## TASK H1 — Preserve soft information

### Problem

The current architecture converts decisions into hard 0/1 bits too early.

### Solution

Preserve:

```text
symbol
confidence
LLR
```

through:

```text
demodulation
→ deinterleaving
→ FEC
```

Example:

```text
bit = 0
LLR = +4.7
```

is much more useful than simply:

```text
0
```

---

## TASK H2 — Implement soft-decision Viterbi

### Problem

Current Viterbi uses hard-decision Hamming metrics.

### Solution

Accept LLRs and calculate likelihood-based branch metrics.

### Acceptance criteria

Compare BER against hard-decision Viterbi at multiple SNRs.

---

## TASK H3 — Fix Viterbi termination assumptions

### Problem

The decoder always removes `K-1` bits, assuming termination.

### Solution

Make termination explicit:

```python
terminated=True
```

or configure an expected final state.

---

## TASK H4 — Fix Reed-Solomon partial blocks

### Problem

An incomplete final block is padded with zeros.

This can invent decoded data.

### Solution

Either:

- reject incomplete blocks, or
- explicitly implement shortened RS codes.

Return clear status:

```text
clean
corrected
uncorrectable
incomplete
invalid
```

---

## TASK H5 — Replace experimental LDPC with standards-based implementations

### Problem

A randomly generated parity-check matrix is not a decoder for a real communications standard.

Also, slicing the first `k` decoded bits assumes a systematic arrangement that may not exist.

### Solution

Support standard-specific matrices/codecs where required:

- DVB-S2
- 5G NR
- Wi-Fi
- CCSDS
- other selected standards

Also use soft LLR input.

### Acceptance criteria

Use official/reference test vectors.

---

# 12. Task Group I — Interleaving and Frame Detection

## TASK I1 — Replace entropy-only deinterleaver selection

### Problem

Low entropy does not imply correct deinterleaving.

Compressed, encrypted or arbitrary payloads may have high entropy.

### Solution

Score candidate deinterleavings using:

```text
CRC success
+ FEC convergence
+ frame synchronization
+ known headers
+ packet structure
+ protocol validity
+ correlation
```

Entropy should be only a weak secondary feature.

---

## TASK I2 — Improve standard sync-word definitions

Represent sync patterns as structured objects:

```python
SyncPattern(
    name="...",
    bits=...,
    protocol="...",
    expected_frame_lengths=[...],
    min_repetitions=2,
    crc="..."
)
```

A sync word should not by itself claim that an entire protocol has been identified.

---

## TASK I3 — Improve preamble discovery

### Problem

Autocorrelation can discover periodicity in random data.

### Solution

Evaluate statistical significance and require stronger evidence.

Use:

- correlation threshold
- false-alarm probability
- repeated occurrence
- expected spacing
- payload/frame validation

---

## TASK I4 — Fix candidate-length selection

Current candidate selection can effectively choose the last valid length instead of evaluating all lengths.

### Solution

For every candidate preamble length:

1. generate candidate
2. measure correlation
3. evaluate repeatability
4. calculate false-alarm likelihood
5. score
6. choose best candidate

---

# 13. Task Group J — Processing Pipeline

## TASK J1 — Introduce a unified analysis result

Create:

```python
SignalAnalysisResult
```

containing:

```text
metadata
raw_stats
spectral_stats
frequency_stats
bandwidth_stats
snr_stats
timing_stats
carrier_stats
modulation
demodulation
interleaving
fec
frame_sync
confidence
warnings
```

This prevents dozens of independent variables being passed around.

---

## TASK J2 — Make preprocessing traceable

Do not silently replace raw measurements with filtered measurements.

Report:

```text
RAW
center frequency = ...
SNR = ...

FILTERED
center frequency = ...
SNR = ...

DENOISED
center frequency = ...
SNR = ...
```

This lets the user understand whether preprocessing helped or damaged the signal.

---

## TASK J3 — Make filtering candidate-based

Bad bandwidth estimation can create a bad filter, which then damages all later stages.

Instead of blindly applying one filter:

```text
raw
 ├── wide filter
 ├── medium filter
 └── narrow filter
```

Evaluate each candidate and choose based on:

- SNR
- classifier confidence
- constellation quality
- synchronization quality
- EVM

---

# 14. Task Group K — CLI and Streamlit

## TASK K1 — Improve CLI file resolution

Support:

- explicit files
- directories
- glob patterns
- recursive globbing

If nothing matches, print:

```text
No input files found.
```

instead of silently continuing.

---

## TASK K2 — Isolate batch failures

One bad file should not stop an entire batch.

Use:

```python
for file in files:
    try:
        analyze(file)
    except Exception:
        record_error(file)
        continue
```

Produce a final batch summary.

---

## TASK K3 — Improve exception handling

Avoid:

```python
except Exception:
    pass
```

and silent fallbacks.

Use structured errors and logging:

```python
logger.exception("Classifier prediction failed")
```

If a fallback is used, report:

```json
{
  "fallback_used": true,
  "reason": "model unavailable"
}
```

---

## TASK K4 — Cache Streamlit resources

Cache expensive objects such as ML models using Streamlit resource caching.

Avoid repeatedly constructing the classifier.

---

# 15. Task Group L — Testing and Validation

## TASK L1 — Build a communications test-signal generator

Create a reusable API:

```python
generate_signal(
    modulation="QPSK",
    symbol_rate=100_000,
    sample_rate=1_000_000,
    snr_db=10,
    carrier_offset=20_000,
    frequency_offset=500,
    phase_offset=0.4,
    rolloff=0.35,
    timing_offset=0.2,
)
```

Every generated signal should retain ground truth.

Example:

```json
{
  "modulation": "QPSK",
  "baud": 100000,
  "sample_rate": 1000000,
  "snr_db": 10
}
```

---

## TASK L2 — Build a realistic channel test matrix

For each modulation:

- BPSK
- QPSK
- 8PSK
- 16QAM
- 64QAM
- 2FSK
- 4FSK
- AM

Test SNR:

```text
-10
-5
0
5
10
15
20
25
30 dB
```

Also vary:

- carrier frequency offset
- phase offset
- timing offset
- pulse-shaping rolloff
- fading
- multipath
- IQ imbalance
- DC offset
- clipping
- sample-rate mismatch

---

## TASK L3 — Add objective metrics

Measure:

```text
center-frequency error
bandwidth error
SNR error
baud-rate error
modulation accuracy
BER
EVM
MER
frame-sync success
FEC recovery rate
false detection rate
```

---

## TASK L4 — Add BER regression tests

For known transmitted bits:

```python
BER = bit_errors / total_bits
```

The CI suite should detect when a DSP modification increases BER.

---

## TASK L5 — Expand existing tests

The current test suite passing is not enough.

Add:

```text
unit tests
integration tests
channel tests
regression tests
golden-vector tests
```

Use official/reference vectors for standards-based FEC where possible.

---

# 16. Task Group M — Accuracy and Confidence Framework

## TASK M1 — Add confidence to every estimator

Example:

```json
{
  "baud_rate": 100000,
  "confidence": 0.83,
  "method": "Gardner",
  "alternatives": [50000, 200000]
}
```

Likewise:

```text
Center frequency: 125 kHz
Confidence: 0.97

Bandwidth: 42.3 kHz
Confidence: 0.91

SNR: 11.8 dB
Confidence: 0.76

Modulation: QPSK
Confidence: 0.88

Baud rate: 100 kbaud
Confidence: 0.83
```

---

## TASK M2 — Add pipeline-level confidence

Combine evidence from:

- spectral detection
- SNR
- timing lock
- carrier lock
- classifier confidence
- EVM
- preamble correlation
- CRC
- FEC convergence

Do not let one weak heuristic dominate the entire result.

---

# 17. Task Group N — Performance and Maintainability

## TASK N1 — Avoid unnecessary recomputation

Cache/reuse:

- FFTs
- PSD
- spectrograms
- filtered signals
- classifier models
- metadata

---

## TASK N2 — Add configurable processing limits

Large IQ recordings can consume significant RAM.

Support:

- chunked processing
- decimation
- analysis windows
- streaming PSD
- maximum sample limits

Avoid loading enormous files entirely into memory when unnecessary.

---

## TASK N3 — Add structured logging

Log:

```text
INFO
DEBUG
WARNING
ERROR
```

Include processing stage and file.

Example:

```text
[baud_estimator] candidate=100000 score=0.87
[carrier_sync] CFO=1247 Hz lock=0.94
[fec] RS corrected=3 blocks
```

---

# 18. Task Group O — Security and Reliability

## TASK O1 — Validate all user inputs

Check:

- file paths
- sample rate
- baud rate
- bandwidth
- modulation selection
- FEC parameters
- interleaver dimensions

Reject impossible combinations.

---

## TASK O2 — Avoid arbitrary execution through inputs

Do not allow file names, configuration fields or protocol fields to become shell commands or executable expressions.

---

## TASK O3 — Dependency and environment validation

Provide:

```text
Python version
NumPy version
SciPy version
scikit-learn version
Streamlit version
```

and fail clearly when incompatible.

---

# 19. Recommended Development Order

## Phase 1 — Correctness

1. Fix IQ dtype handling.
2. Introduce metadata.
3. Validate sample rate.
4. Improve file errors.
5. Fix RS partial-block behavior.
6. Fix 64QAM mapping.
7. Fix Viterbi termination assumptions.
8. Remove silent exception handling.

### Goal

Make current functionality trustworthy.

---

## Phase 2 — DSP Core v2

1. Signal-region detection.
2. Robust center frequency.
3. Occupied bandwidth.
4. SNR estimation.
5. Coarse CFO.
6. Fractional timing.
7. Gardner timing recovery.
8. Proper carrier loops.

### Goal

Make synchronization and measurements reliable.

---

## Phase 3 — Demodulation v2

1. BPSK.
2. QPSK.
3. 8PSK.
4. 16QAM.
5. 64QAM.
6. 2FSK.
7. 4FSK.
8. AM.

### Goal

Use modulation-specific processing rather than generic assumptions.

---

## Phase 4 — Soft Communications Chain

1. LLR generation.
2. Soft Viterbi.
3. Robust RS.
4. Standards-based LDPC.
5. Soft deinterleaving.
6. CRC validation.

### Goal

Improve low-SNR recovery.

---

## Phase 5 — Intelligent Classification

1. Realistic synthetic channels.
2. Better features.
3. Classifier comparison.
4. Confidence scores.
5. Confusion matrices.
6. SNR-vs-accuracy benchmarks.

### Goal

Make automatic modulation recognition defensible.

---

## Phase 6 — Protocol/Frame Intelligence

1. Better sync-word database.
2. Preamble discovery.
3. Frame-length inference.
4. CRC detection.
5. Protocol candidate ranking.
6. FEC-assisted validation.

### Goal

Move from "modulation detector" toward "signal intelligence pipeline."

---

## Phase 7 — Production Hardening

1. Chunked processing.
2. Caching.
3. Structured logging.
4. Batch error isolation.
5. Documentation.
6. CI/CD.
7. Regression corpus.
8. Benchmark suite.

---

# 20. Recommended Final Architecture

```text
                         RAW IQ / WAV
                              │
                              ▼
                     ┌─────────────────┐
                     │ Input Validator │
                     └────────┬────────┘
                              ▼
                       Signal Metadata
                              │
                              ▼
                     DC / IQ Diagnostics
                              │
                              ▼
                      PSD + Spectrogram
                              │
                              ▼
                     Signal Detection
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        Frequency Estimator        Bandwidth Estimator
                 │                         │
                 └────────────┬────────────┘
                              ▼
                         SNR Estimator
                              │
                              ▼
                     Coarse Synchronizer
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
              CFO                        Baud
           Estimator                   Candidates
                 │                         │
                 └────────────┬────────────┘
                              ▼
                      Timing Recovery
                              │
                              ▼
                      Carrier Recovery
                              │
                              ▼
                    Modulation Classifier
                              │
                              ▼
                 Modulation-specific Demod
                              │
                              ▼
                       Soft Symbols/LLR
                              │
                              ▼
                        Deinterleaver
                              │
                              ▼
                             FEC
                              │
                              ▼
                       Frame Detector
                              │
                              ▼
                       CRC / Validation
                              │
                              ▼
                       Payload + Report
```

---

# 21. Final Output Should Look Like This

Instead of:

```text
Modulation: QPSK
Baud: 100000
SNR: 12 dB
```

the final analyzer should produce something like:

```text
SIGNAL ANALYSIS REPORT
======================

Input
-----
Format: int16 IQ
Sample rate: 1.000 MSPS
Center frequency metadata: 433.920 MHz

Signal
------
Detected center: +125.4 kHz
Confidence: 97%

Occupied bandwidth
------------------
95% OBW: 42.1 kHz
99% OBW: 45.3 kHz
Confidence: 91%

SNR
---
11.8 dB
Method: noise-band
Confidence: 76%

Synchronization
---------------
CFO: +1.24 kHz
Carrier lock: 94%
Timing lock: 88%

Modulation
----------
QPSK
Confidence: 91%

Alternatives
------------
BPSK: 6%
8PSK: 3%

Demodulation
------------
Symbol rate: 100 kbaud
Confidence: 83%
EVM: 4.1%
MER: 27.7 dB

Frame
-----
Preamble: candidate detected
Correlation: 0.94
CRC: not available

FEC
---
Unknown

Overall assessment
------------------
HIGH-CONFIDENCE QPSK SIGNAL

Warnings
--------
- FEC type not identified.
- CRC validation unavailable.
- SNR estimate has moderate confidence.
```

This is much more useful than presenting every heuristic result as fact.

---

# 22. Highest-Priority Bug List

The following should be treated as the immediate corrective backlog:

1. **🔴 `.iq` int16 files are incorrectly interpreted as float32.**
2. **🔴 Default sample rate can produce completely wrong measurements.**
3. **🔴 Baud-rate estimator is unreliable.**
4. **🔴 Timing recovery is integer stepping, not fractional recovery.**
5. **🔴 Costas/carrier recovery is incorrectly generalized to FSK/AM.**
6. **🔴 FSK demodulation is oversimplified.**
7. **🔴 64QAM mapping is not Gray-coded.**
8. **🔴 SNR estimator is heuristic and not sufficiently calibrated.**
9. **🔴 Bandwidth estimator can merge separate signals.**
10. **🔴 Deinterleaver autodetection cannot reliably use entropy as the main criterion.**
11. **🔴 LDPC implementation is not a standards decoder and has systematic-bit assumptions.**
12. **🟠 Reed-Solomon silently pads incomplete blocks.**
13. **🔴 Preamble discovery can produce false positives.**
14. **🟠 Candidate preamble-length selection needs actual scoring.**
15. **🟠 EVM needs amplitude/phase alignment.**
16. **🟠 Viterbi should support soft decisions.**
17. **🟠 Viterbi termination should be configurable.**
18. **🟠 Classifier training data is too clean.**
19. **🟠 Classifier needs confidence and alternative predictions.**
20. **🟠 Silent exception handling hides real failures.**
21. **🟡 Streamlit should cache expensive model resources.**
22. **🟡 CLI batch processing should isolate file failures.**
23. **🔴 Testing needs realistic channel simulations and BER/EVM/FEC metrics.**

---

# 23. Definition of Done

The toolkit should not mark a component as "Done" merely because it executes successfully.

Use these statuses:

```text
⚪ Planned
🔵 Prototype
🟡 Implemented
🟠 Needs validation
🟢 Validated
🔴 Known limitation / incorrect for production
```

A DSP component should become `🟢 Validated` only when:

- unit tests pass
- synthetic ground-truth tests pass
- noisy-channel tests pass
- edge cases are tested
- numerical limits are understood
- output confidence is meaningful
- regression tests exist
- documentation explains limitations

---

# 24. Strategic Direction

The project should evolve from:

> **"A collection of signal-processing functions."**

into:

> **"A confidence-aware, modular communications signal-analysis pipeline."**

The key technical philosophy should be:

```text
Don't guess once.
Generate candidates.
Measure evidence.
Validate.
Rank candidates.
Expose uncertainty.
```

That principle should govern frequency estimation, bandwidth, SNR, baud rate, modulation, synchronization, interleaving, FEC and protocol identification.

The most important next milestone is therefore:

# DSP CORE V2

```text
Correct IQ ingestion
        ↓
Robust spectral detection
        ↓
Accurate parameter estimation
        ↓
Real timing recovery
        ↓
Proper carrier synchronization
        ↓
Modulation-specific demodulation
        ↓
Soft bits
        ↓
Validated FEC
        ↓
Frame/CRC validation
```

Once this foundation is reliable, additional protocols, AI classification, automatic protocol discovery, visualization and higher-level intelligence can be added without building on unstable assumptions.
