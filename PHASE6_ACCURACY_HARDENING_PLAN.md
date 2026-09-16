# Phase 6: Accuracy Hardening & Ground-Truth Validation Plan

**Date:** 2026-09-16
**Follows:** `docs/PHASE5_SUMMARY.md` (De-Interleaving & FEC Decoder Remediation)
**Goal:** Take the toolkit from "runs without crashing on happy-path demo files" to
"produces numbers a signal analyst would trust on an unknown, unlabeled `.iq`/`.wav`
capture" — which is literally what PS26147 asks for.

This plan assumes the reader has `docs/ps26147_toolkit_review_and_fixes.md` open
alongside it. That document is still the primary reference for module-level code
diffs; this document (a) confirms which of its findings are still live in the
current code, (b) adds several **new, independently verified** bugs found while
testing against a known-content signal (`hello world`, 2FSK), and (c) — the part
that was actually asked for — lays out the test-signal infrastructure and process
needed to make "accurate" a measurable, checkable claim instead of a hope.

---

## 0. Current State Snapshot

| Area | Status | Evidence |
|---|---|---|
| `auto_discover_preamble` missing-key crash | ✅ Fixed | Defensive `.get()` on caller + early-return dict shape (`1e6ac94`, `d751b52`) — see §1.1 |
| `auto_discover_preamble` overwrites `discovered_preamble` in a loop | ✅ Fixed | Preamble length now selected by frame stability, not longest-fit (`ee4c6d9`) — see §1.2 |
| Soft-decision Viterbi "validation needed" | ✅ Fixed & regression-gated | §1.3 root-caused; `test_soft_decision_viterbi_awgn_ber_curve` exercises a real BER-vs-SNR curve (`1453ea7`, `600a40e`) — see §1.3/§1.3.1 |
| `estimate_snr` clipped at 50 dB ceiling | ✅ Fixed | Ceiling → 80 dB; `snr_clipped` flag + `format_snr()` in UI (`aaccbab`) — §1.4 |
| `estimate_baud_rate` transition detector destroys periodicity via `abs()` | ✅ Fixed | Signed phase-diff FSK baud detection (`2567f4c`) — §1.5 |
| EVM undefined/meaningless for FSK | ✅ Fixed | Deviation-domain `compute_fsk_evm()` + FSK soft-LLR branch (`4d8b502`) — §1.6 |
| Bandwidth (`bw_10db`) underestimates vs. Carson's rule | ✅ Fixed | Per-modulation contour-ladder calibration (`2452e32` + `f8c6f3f`) — §1.7 |
| No ground-truth test-signal corpus | ✅ Partially built | Generator (`00955e3`); `_cal/` now holds the 14-file .wav MPC leg (`0e2581b`) — §2, `.iq` leg + SNR ladder still to generate |
| No automated accuracy scoring / CI regression gate | ✅ Implemented | `scripts/run_accuracy_report.py` built & run on `_cal/` (§3). Gate correctly FAILS at **74% vs ≥95% target**: QAM/PSK misclassified as AM, 4FSK as 2FSK → cascades into baud/bandwidth fails |

---

## 1. Bug Ledger (new findings this session)

Each entry: **defect → evidence → fix → how to verify the fix**.

### 1.1 `auto_discover_preamble` early return missing dict keys (fixed, unpushed)

`correlator.py` line ~257:
```python
if len(bitstream) < min_frame_len * 2:
    return {"discovered": False, "reason": "Bitstream too short for periodicity analysis"}
```
This dict lacks `matched_standard_sync`, `candidate_preamble_bits`, etc., which
`web_demo/app.py` line 314 accesses unconditionally
(`if discovery["matched_standard_sync"]:`). Crashes whenever the bitstream
reaching frame-sync (post FEC/de-interleave) is under `min_frame_len*2` bits —
e.g. any short test file run through a mismatched FEC decoder.

**Status:** you've applied a `.get()`-style fix locally. Before pushing, make sure
the fix covers *both* sides:
- `app.py`: `discovery.get("matched_standard_sync")` — defensive on the caller side.
- `correlator.py`: the early-return dict should still carry the full key set (with
  `None`/`0.0` defaults) so **every other caller** of this function (tests,
  notebooks, future UI code) doesn't hit the same shape mismatch. Fixing only the
  call site in `app.py` leaves a landmine for the next caller.

**Verify:** add a unit test that calls `auto_discover_preamble(np.zeros(4, dtype=np.uint8))`
(a bitstream shorter than `min_frame_len*2`) and asserts the returned dict has
every key the "normal" return path has, not just that it doesn't crash.

### 1.2 `auto_discover_preamble` preamble-length selection picks the *longest* candidate, not the *best* one

Still present, `correlator.py`:
```python
for p_len in preamble_lens:
    if p_len < best_period:
        discovered_preamble = candidate_bits[:p_len]   # overwritten every iteration
```
Already documented in `ps26147_toolkit_review_and_fixes.md` §3.7 with a drop-in
fix (score each candidate length by cross-correlation against subsequent frames,
keep the best-scoring one, not the longest one that fits). Not yet applied.
Low effort, real accuracy impact on multi-frame captures — pull this into Phase 6.

### 1.3 Soft-decision Viterbi: root cause found and reproduced (closes PHASE5 open item)

> **Status: Implemented ✅ (2026-09-16).** Root cause confirmed, no decoder code
> change needed, and a committed test now regression-gates it. See §1.3.1.

PHASE5_SUMMARY.md flagged: *"Both hard-decision and soft-decision decoders
exhibit poor performance with realistic AWGN noise... ~96% BER at Eb/N0=8dB...
root cause unknown, affects both equally."*

**This is now root-caused.** Reproduction (uses the actual `ConvolutionalCodec`
from `fec_decoders.py`, no modifications):

```python
import numpy as np
from ps26147_toolkit.fec_decoders import ConvolutionalCodec

codec = ConvolutionalCodec(k=7, polys=(0o171, 0o133))
rng = np.random.default_rng(456)
msg = rng.integers(0, 2, size=200, dtype=np.uint8)
encoded = codec.encode(msg)

tx = 2 * encoded.astype(np.float32) - 1          # BPSK map: bit1 -> +1, bit0 -> -1
noise_std = np.sqrt(1 / 10**(8/10))               # 8 dB SNR
rx = tx + noise_std * rng.standard_normal(len(tx)).astype(np.float32)

hard_bits = (rx >= 0).astype(np.uint8)
print(np.mean(codec.decode(hard_bits, max_len=len(msg)) != msg))          # -> 0.0   (hard decision is FINE)
print(np.mean(codec.decode_soft( 4*rx, max_len=len(msg)) != msg))         # -> 0.985 (as-tested in repo: BROKEN)
print(np.mean(codec.decode_soft(-4*rx, max_len=len(msg)) != msg))         # -> 0.0   (sign-corrected: FINE)
```

**Root cause:** `decode_soft()`'s documented convention is *"LLR > 0 → bit likely
0, LLR < 0 → bit likely 1"*. But the channel-simulation code used in
`tests/test_phase5_improvements.py::test_soft_decision_viterbi_wrapper`
(`rx_llrs = 4 * rx_signal`, where `rx_signal ≈ tx_signal = 2*encoded_bit - 1`)
produces a **positive** LLR when the transmitted bit is **1** — the exact
opposite of what `decode_soft()` expects. The trellis math inside `decode_soft()`
is correct (confirmed by the sign-corrected run above: 0% BER at 8 dB, matching
theory). The bug is entirely in the LLR generation step, not the decoder.

Two consequences:
1. **The "hard-decision also fails" claim in PHASE5_SUMMARY.md is very likely a
   red herring** from whatever exploratory script produced those numbers — my
   repro shows `decode()` at 0% BER from 4 dB to 20 dB SNR. The most likely
   explanation: that script also fed un-thresholded floats or sign-inverted data
   into the hard path. This needs a **committed, repeatable test** (see §1.3.1)
   so this doesn't get re-litigated from memory next time.
2. `test_soft_decision_viterbi_wrapper` doesn't actually check BER — it only
   asserts `len(bits) > 0` — so this ~98% BER bug passed CI silently. A test that
   exercises a code path but never checks its numerical output isn't catching
   what it's named for.

**Fix (three places, in order of blast radius):**

a) **The test itself** — `tests/test_phase5_improvements.py`:
   ```python
   rx_llrs = -4 * rx_signal   # negate: bit=1 -> tx=+1 -> rx>0 -> must be LLR<0
   ```
   Add a BER assertion (`< 0.01` at 8 dB, say) so this can't silently regress again.

b) **`compute_soft_llr()` in `demodulator.py`** — the BPSK branch already applies
   this negation correctly (`llrs = -symbols.real / noise_variance`, with a
   comment explaining exactly this convention). Good — that one's not broken.
   But this proves the convention is well understood *somewhere* in the codebase;
   the test file just didn't apply it. Add a one-line comment at the top of
   `decode_soft()`'s docstring cross-referencing `compute_soft_llr()`'s BPSK
   branch as the canonical worked example, so the convention doesn't get
   re-inverted a third time by whoever touches this next.

c) **The FSK LLR fallback is a separate, deeper bug — see §1.6.** Fixing (a) and
   (b) does not fix FSK; FSK's LLR is wrong for a completely different reason
   (wrong discriminant, not wrong sign).

#### 1.3.1 Required new test (committed, not exploratory)

> **Status: Committed ✅ (`600a40e`).** `test_soft_decision_viterbi_awgn_ber_curve`
> now exists in `tests/test_fec.py` and produces a real BER-vs-SNR curve.

Committed requirements:
- Sweep Eb/N0 ∈ {2, 4, 6, 8, 10} dB.
- At each point: encode → BPSK map → AWGN → hard-decode AND soft-decode →
  compute BER for both.
- Assert **soft BER ≤ hard BER at every point** (soft decision must never be
  worse than hard — that's the entire point of carrying LLRs through).
- Guard against a vacuous sweep: the lowest point must be in the error regime
  (`hard BER > 0`), so a noise-normalization regression can't silently pass.
- At the highest point, assert soft BER `< 0.02`.
- This single test is exactly what would have caught the original bug in code
  review instead of in a hackathon judging round.

**Two additional defects found while making this test real (both fixed in the
same commit):**

1. **The test's noise normalization was wrong by +3 dB.** The old
   `noise_std = sqrt(1 / (2 * snr_lin))` is not the correct relation for BPSK
   rate-1/2. With ±1 symbols, `Eb = Es/r = 2` and `N0 = 2*sigma²`, so
   `Eb/N0 = Es/(r·2·sigma²) = 1/sigma²`, i.e. `sigma² = 1/snr_lin`. The extra
   factor of 2 in the denominator *halved* the injected noise relative to the
   label, shifting the whole sweep +3 dB — which landed every point in the
   error-free regime (all BERs `0.0000`). The test therefore **passed while
   proving nothing** about soft-vs-hard gain; it wasn't exercising §1.3's bug
   at all because it never entered the waterfall. Fixed, and the sweep now
   actually shows the gain:

   | Eb/N0 (dB) | hard BER | soft BER |
   |---|---|---|
   | 2 | 0.141 | 0.000 |
   | 4 | 0.0045 | 0.000 |
   | 6/8/10 | 0.000 | 0.000 |

2. **A no-op assertion** masked the real one: `assert ber_soft <= ber_soft + 0.001`
   compared a value to itself (always true). Corrected to the intended
   `ber_soft <= ber_hard + 0.001`. (The cross-curve `bs <= bh + 0.005` check at the
   end was already correct.)

**Still outstanding (optional, note in review):** the "within 1 dB of the
textbook K=7 rate-1/2 hard-decision BER curve" comparison — generate a reference
with `commpy`/`sionna` per `ps26147_toolkit_review_and_fixes.md` §1.6's LDPC
suggestion, or tabulate textbook values. The current assertions (soft ≤ hard,
hard > 0 at the low end) already prevent the original class of regression
silently; the absolute reference would additionally pin the *shape* of the curve.

### 1.4 `estimate_snr` clips at a hardcoded 50 dB ceiling

`parameter_extractor.py`, `estimate_snr()`:
```python
return float(np.clip(snr_spectral, -20.0, 50.0))
```
On any clean/near-noiseless synthetic test signal (which is most of what you'll
generate for calibration — see §2), the true SNR is far above 50 dB, so the
reading saturates at the ceiling and stops being a measurement. Confirmed: two
runs of the same file with noise filtering toggled on/off both read exactly
`50.00 dB` with a `+0.00 dB` delta — a real (uncapped) SNR would move at least
slightly between those two runs.

**Fix:** raise the ceiling to something a synthetic clean signal can't hit
(e.g. 80 dB, tied to 16-bit quantization noise floor: ~96 dB theoretical for a
full-scale int16 signal), **or** surface "≥50 dB (clipped)" distinctly from an
actual measured value in the UI so a judge/analyst doesn't read a pegged number
as a precise one. Either way, this can only be verified once you have signals at
*known* SNR (§2) to check the estimator actually tracks reality below the ceiling.

### 1.5 `estimate_baud_rate`'s transition detector discards the signal it needs

`parameter_extractor.py`, inside `estimate_baud_rate()`:
```python
conj_prod = sig_bb[1:] * np.conj(sig_bb[:-1])
phase_diff = np.abs(np.angle(conj_prod))
```
For constant-envelope, continuous-phase FSK, the instantaneous frequency
alternates between `+Δf` and `−Δf` at the baud rate — that alternating **sign**
is the only thing that encodes *when* a bit boundary occurs. Taking `np.abs()`
collapses it to a near-constant series, destroying the periodicity the
subsequent FFT/autocorrelation search depends on. Empirically this showed up as
baud-rate estimates swinging from 1224.3 Hz (close to the true 1200 Baud) to
460.8 Hz just from toggling the bandpass filter on/off — a 2.6× shift that
shouldn't be possible if the estimator were actually tracking bit transitions.

**Fix:** don't `abs()` the signed instantaneous frequency before building the
transition indicator. Use the signed value directly, or better, `np.diff()` of
it — which produces a sharp impulse of the correct sign exactly at each bit
transition and zero elsewhere, giving the FFT/autocorrelation stage a genuinely
periodic signal to lock onto regardless of the data's bit pattern.
`ps26147_toolkit_review_and_fixes.md` §2.1 documents a different (older,
simpler) version of this function with a different but related permissiveness
issue — the current code has evidently been rewritten since that doc was
written, so treat that section as superseded by this finding, not additive to it.

### 1.6 EVM is not a meaningful metric for FSK as currently computed

`demodulator.py`, `slice_symbols_to_bits()`, FSK branch:
```python
elif "FSK" in mod_upper or "2FSK" in mod_upper:
    diff = np.diff(np.unwrap(np.angle(symbols)))
    for d in diff:
        b = 1 if d >= 0 else 0
        bits_list.append(b)
        ref_symbols.append(1.0 if b == 1 else -1.0)   # <- decision-derived, real-valued
```
`compute_evm(symbols_norm, ref_symbols)` then measures
`|symbols_norm - ref_symbols|`. But `symbols_norm` for FSK are full complex
baseband samples whose **phase continuously rotates** (that rotation *is* the
frequency information) — they never actually sit at `±1` the way a PSK symbol
does. Comparing a continuously-rotating point against a fixed `±1` reference
measures phase rotation, not demodulation error — the number is not
meaningless-by-accident, it's measuring the wrong physical quantity by
construction. This explains why a genuinely noiseless synthetic 2FSK file still
reads 2.9–3.6 dB EVM (i.e., ~45–55% RMS error) — there's no actual error to
measure; the metric itself doesn't apply to this modulation as implemented.

Compounding this, `compute_soft_llr()` doesn't have an `FSK` entry in
`CONSTELLATIONS` (only `BPSK/QPSK/8PSK/16QAM/64QAM`), so it silently falls
through to the "unrecognized modulation" branch:
```python
llrs = -symbols.real / noise_variance   # wrong discriminant for FSK
```
— the real part of the raw IQ sample, which has no relationship to the phase-derivative-sign decision the FSK hard-slicer actually uses. Even setting aside
the length mismatch (`llr` is one sample per symbol; the FSK hard-bit array is
`len(symbols)-1` due to `np.diff`), the LLR magnitude and sign here are
decorrelated from the real decision. This is the same class of bug as §1.3 but
in the *demodulator's* LLR generator rather than the *decoder's* — meaning
wiring up soft-decision FEC for FSK specifically needs its own fix, independent
of the sign fix in §1.3.

**Fix, in order of effort:**
- **Cheap/immediate:** for FSK, either suppress the EVM tile in the UI, or
  compute an FSK-appropriate analogue: RMS error between the estimated
  instantaneous-frequency deviation and the ideal `±Δf`, normalized by `Δf`
  (this is the actual FSK equivalent of "how far off the ideal symbol was").
- **For soft-FEC on FSK:** add an `FSK` branch to `compute_soft_llr()` that
  derives LLR from the signed, `np.diff`'d instantaneous-frequency estimate
  (the same signal §1.5 recommends fixing), scaled by an estimate of
  discriminator noise variance — not from `symbols.real`.

### 1.7 Bandwidth (`bw_10db`) likely underestimates true occupied bandwidth

> **Status: Implemented ✅ (`2452e32`, `f8c6f3f`).** Went beyond the plan's
> "surface `obw_99`" cheap fix: added a full **per-modulation contour-ladder**
> calibration in `parameter_extractor.py`. `estimate_bandwidth_all()` already
> exposed `bw_10db`/`obw_95`/`obw_99`; the new estimator measures bandwidth at a
> ladder of contours (−10…−30 dB from the noise-compensated peak,
> `_BW_CONTOURS`) and selects the contour calibrated to each modulation class
> (`_BW_CONTOUR_DB`: BPSK/QPSK/8PSK/16QAM/64QAM → −25 dB, 2FSK → −20 dB, 4FSK →
> −25 dB). `extract_signal_parameters` accepts a `modulation` argument and
> returns both `bandwidth_contour_db` and the full `bandwidth_ladder_hz`.
> Calibrated against the ground-truth corpus: linear 1–10% error, 2FSK <4%,
> 4FSK ~10–16%. The CLI (`process_file`) classifies first, then passes the
> modulation through.

Measured ~1.3 kHz on the reference file vs. a Carson's-rule prediction of
~3.4 kHz and an independent FFT-based measurement of ~2.3 kHz on the same raw
samples. `estimate_bandwidth_all()` already computes `obw_95`/`obw_99` in the
same function call — these are more representative of true occupied bandwidth
for a modulation index this narrow (h≈0.83 for this test signal) but aren't
surfaced on the UI's summary tile (only `bw_10db` is shown). Cheapest fix:
show `obw_99` (or let the user pick which bandwidth definition to display) —
no code changes to the estimator needed, just wire up numbers that already
exist. Calibration against the ground-truth matrix (§2) will tell you whether
`_compute_masked_noise_floor`'s 5%-of-peak signal mask needs adjusting too.

---

### 1.8 New (found by §3 harness): classifier mislabels QAM/PSK as AM, 4FSK as 2FSK

> **Status: Open 🔴 (2026-09-16).** Not a bug in this plan's original ledger —
> surfaced by the first run of the newly built `scripts/run_accuracy_report.py`.
>
> On the clean `.wav` corpus the modulation classifier returns:
> `QPSK → AM`, `8PSK → AM`, `16QAM → AM`, `64QAM → AM`, `4FSK → 2FSK`.
> Because `process_file()` classifies *first* and then branches the whole
> extractor on the result, a wrong modulation cascades: baud (144/83/116/300 vs
> 1200) and bandwidth then fail too, even though the underlying measurements are
> good on the modulations that classify correctly (BPSK/2FSK → exact baud).
>
> **Priority fix for §3's definition-of-done** (≥95% on the minimum corpus):
> the classifier's AM-vs-(Q)PSK/QAM decision boundary and the 2FSK-vs-4FSK
> boundary both need retuning. Fix the classifier, then re-run
> `scripts/run_accuracy_report.py` — the gate will confirm.

---

## 2. The Missing Piece: A Ground-Truth Test-Signal Corpus

This is the actual ask, and it's the right instinct — **you cannot claim a
parameter-extraction pipeline is "accurate" against anything you haven't
independently generated with known values.** Right now every number the UI
shows is unfalsifiable: there's no file in the repo where you know, a priori,
what the correct Center Freq / Bandwidth / SNR / Baud / EVM / decoded payload
*should* be. `scripts/generate_synthetic_iq.py` exists but (per the ROADMAP) is
narrow; `synthetic.iq` is a single committed file. Neither is a corpus.

### 2.1 Design principles

1. **Every generated file ships with a matching ground-truth JSON** — not just
   "this is 2FSK" but the exact numeric values a perfect pipeline should report:
   `{modulation, center_freq_hz, deviation_hz or symbol_map, bandwidth_hz (theoretical, Carson's rule), snr_db (as injected), baud_rate, fec_scheme, fec_params, interleaver, sync_word, payload_text, payload_bits, sample_rate, file_format}`.
2. **Vary one axis at a time first**, then combine. Start with clean, high-SNR,
   no-FEC, no-interleaving files per modulation to validate the demodulator and
   parameter extractor in isolation. Only once those pass should FEC and
   interleaving be layered on — otherwise a failure could be anywhere in the
   chain and you'll spend hours re-deriving what this document already derived
   for FSK.
3. **Known text payload, always.** Use human-readable ASCII (`"hello world"`,
   or better, a fixed pangram like `"the quick brown fox jumps over the lazy
   dog 0123456789"` — covers all letter cases and digits) so a decode failure
   is visually obvious in the "Decoded ASCII / Text Preview" tab without cross-
   referencing a bit table.
4. **Both file formats, always.** `.wav` (real-valued, needs Hilbert transform /
   analytic signal reconstruction internally) and `.iq` (complex64 interleaved,
   as the problem statement explicitly names both formats and `preprocess.py`'s
   `load_iq` has its own documented bug in
   `ps26147_toolkit_review_and_fixes.md` §1.1 that a `.iq`-format test file
   would immediately have caught).
5. **A noise ladder per modulation**, not just clean signals — this is the only
   way to validate §1.4 (SNR) and §1.3.1 (Viterbi BER curve) meaningfully.

### 2.2 The test matrix

| Axis | Values to cover |
|---|---|
| **Modulation** | BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK |
| **FEC** | None, Viterbi (K=7, r=1/2), Reed-Solomon (255,223), Concatenated (RS+Viterbi), LDPC (802.11n, rate 1/2) |
| **Interleaving** | None, Block, Convolutional, Diagonal, Pseudo-Random |
| **Sync word** | Auto-Discover target: one **standard** word actually in `STANDARD_SYNC_WORDS` (e.g. Barker-13, or CCSDS-32 ASM) — not a custom pattern, and *not* the alternating `Preamble-1010` used as a training sequence, since that aliases with the built-in entry of the same name and produces false multi-frame detections (see the earlier `hello_world_plain.wav` "8 Frames" artifact) |
| **SNR ladder** | Clean (no noise), 20 dB, 15 dB, 10 dB, 8 dB, 6 dB, 4 dB, 0 dB |
| **File format** | `.wav` (16-bit PCM, real, with a carrier — needs a real center frequency, not baseband) AND `.iq` (complex64, baseband or with a synthetic LO offset) |
| **Sample rate** | At least two per modulation (e.g. 44.1 kHz and 1 MHz) so `fs`-dependent bugs (like the FFT segment-length assumptions in `parameter_extractor.py`) get exercised |

Full cross product is large — you don't need every cell. **Minimum viable
corpus for Phase 6:**

- 7 modulations × clean, no-FEC, `.wav` + `.iq` = **14 files** (validates
  demodulator + parameter extractor per modulation, in isolation)
- 7 modulations × SNR ladder (8 points) × `.wav` only = **56 files** (validates
  SNR estimator §1.4 and gives real BER-vs-SNR curves to check against theory)
- 5 FEC schemes × clean + 3 SNR points × `.wav` only = **20 files** (validates
  FEC decoders including the Viterbi fix in §1.3, RS from
  `ps26147_toolkit_review_and_fixes.md` §1.3, and LDPC from §1.6)
- 4 interleaving methods × 1 modulation × clean = **4 files** (validates
  `auto_detect_and_deinterleave`)
- 2 standard sync words × 2 modulations × multi-frame (≥4 repeats) = **4 files**
  (validates real frame-sync counting, not preamble-aliasing artifacts)

**~98 files total** for a first pass — scriptable in an afternoon once the
generator (§2.3) exists, and every single one carries its own ground truth so
"is this number right" stops being a judgment call.

### 2.3 Generator architecture

Extend `scripts/generate_synthetic_iq.py` (or add a sibling
`scripts/generate_ground_truth_corpus.py`) with this shape:

```python
def generate_test_case(
    modulation: str,             # "BPSK", "2FSK", ...
    fec_scheme: str = "none",
    interleave_method: str = "none",
    sync_preset: str = "Barker-13",
    snr_db: float | None = None, # None = no noise added
    fs: float = 44100,
    center_freq: float = 10_000,
    baud: float = 1200,
    payload_text: str = "the quick brown fox jumps over the lazy dog 0123456789",
    file_format: str = "wav",    # "wav" | "iq"
) -> tuple[Path, dict]:
    """Returns (file_path, ground_truth_dict). Ground truth is also written
    alongside as <file_path>.json."""
```

Reuse what you already have: `ConvolutionalCodec.encode()`,
`ReedSolomonCodec.encode_block()`, the interleaver module's forward methods,
and `STANDARD_SYNC_WORDS` from `correlator.py` directly — don't reimplement
encoders separately from decoders, or a bug in one will "pass" purely because
the same bug exists in the other. (I can build this generator script with you
directly against your actual encoder classes if useful — same approach as the
`hello_world_plain.wav` file earlier, but driven by this matrix instead of one
ad hoc file.)

---

## 3. Validation Harness & Regression Gate

> **Status: Implemented ✅ (2026-09-16).** `scripts/run_accuracy_report.py`
> is built and works: it feeds every ground-truth-paired `_cal/` file through
> the real `process_file()` pipeline, diffs each parameter against the corpus
> JSON, writes a Markdown report + machine-readable JSON summary, and exits
> non-zero unless ≥95% of applicable checks pass.
>
> **First run (clean `.wav` leg, 7 files): pass rate 74% (31/42) — gate FAILS.**
> The harness is doing its job, and it surfaced two real, un-fixed accuracy
> problems rather than a harness defect:
>
> 1. **Classifier mislabels QAM/PSK as AM** (16QAM, 64QAM, 8PSK, QPSK → AM)
>    and **4FSK as 2FSK**. This is the dominant failure: baud/bandwidth then
>    fail *downstream* because `extract_signal_parameters` branches on the
>    (wrong) modulation. BPSK/2FSK/4FSK-that-classified bauds are exact.
> 2. **8PSK center freq 1.51%** (just over the 1% tol) and **4FSK bandwidth
>    29.6%** (over the 25% tol) — secondary, only where classification held.
>
> Definition-of-done for §3 (≥95% on the minimum corpus, 100% payload on
> FEC+sync files) is therefore **not met yet** — the fix now belongs in the
> classifier, and the report is the regression gate that will confirm it.

Once §2's corpus exists, add `scripts/run_accuracy_report.py`:

1. Feed every corpus file through the actual pipeline (`process_file` /
   the underlying functions directly, not the Streamlit UI).
2. For each file, diff every extracted parameter against its ground-truth JSON:
   - Modulation: exact match required.
   - Center freq / bandwidth / baud: within a tolerance band (e.g. ±5%) —
     define the acceptable tolerance *per parameter*, not one global number.
   - SNR: within ±3 dB below the clip ceiling; flag (not fail) anything at
     or above the ceiling as "unverifiable — clipped."
   - Decoded payload: exact byte match against `payload_text` for any file
     with FEC+sync enabled — this is the sharpest, least ambiguous check you
     have, since it's binary (matches or doesn't).
3. Emit a single Markdown/HTML report: a table of file × parameter × pass/fail,
   plus a BER-vs-SNR curve per modulation/FEC combo (matplotlib, or reuse
   `plotly_visualizations.py`).
4. Wire this into CI (or at minimum, run it before every SIH demo/judging round)
   so a regression in, say, `estimate_baud_rate` doesn't silently ship again the
   way §1.3's LLR sign bug did.

**Definition of done for Phase 6:** the accuracy report shows ≥95% of
Minimum-Viable-Corpus files (§2.2) passing every parameter check, and 100% of
FEC+sync files decoding the exact payload text at every SNR point at or above
the scheme's designed correction threshold (e.g. Viterbi K=7 r=1/2 should
recover the exact payload down to ~4–5 dB SNR; below that, failure is
*expected* and the report should show it failing gracefully, not crashing).

---

## 4. Suggested Order of Work

1. **Push the `.get()` fix** (§1.1) — already done, just ship it. Add the
   defensive early-return dict shape while you're in that function.
2. **Fix the Viterbi LLR sign bug** (§1.3) — ✅ **done.** No decoder change was
   needed (the sign bug was in the test/LLR-generation convention, not the
   trellis); `test_soft_decision_viterbi_awgn_ber_curve` is committed with a
   real BER assertion and a vacuous-sweep guard (§1.3.1).
3. **Wire soft-decision Viterbi into `decode_fec()`/`app.py`** — partially
   verified (note: the plan's original claim here was already stale on the
   decoder side). `decode_fec()` **does** forward `llr=` to
   `viterbi_decode`/`concatenated_decode`, which select the soft path
   automatically when LLRs are present — so soft-decision is reachable at the
   `decode_fec()` boundary. The remaining open check is the **`app.py`** side:
   confirm the UI actually passes `demod_data["llr"]` into `decode_fec()`
   (and that `decode_fec()`'s Viterbi branch receives it — note it does *not*
   currently pass the `soft_decision` flag explicitly, it relies on the `llr`
   auto-detect). Verify with a known-SNR file from the §2 corpus once it exists.
4. **Build the corpus generator** (§2.3) and generate the Minimum Viable
   Corpus (§2.2) — do this before chasing any more individual metric bugs,
   because half of §1's remaining items (SNR ceiling, bandwidth
   underestimate) can't actually be *verified* fixed without known-SNR,
   known-bandwidth ground truth to check against.
5. **Fix §1.5 (baud rate) and §1.4 (SNR ceiling)** against the new corpus,
   checking each fix against the accuracy report (§3) rather than eyeballing
   one file at a time.
6. **Address FSK EVM/LLR** (§1.6) — lowest urgency of the open bugs, since
   EVM is a secondary/diagnostic metric rather than a primary deliverable
   (modulation ID, demodulated payload, and FEC-decoded payload are what
   PS26147's description actually asks the tool to produce).
7. **Pull in `ps26147_toolkit_review_and_fixes.md`'s still-open items**
   (§1.2 [preamble selection], the RS/LDPC/classifier items in that doc's
   §1.2–1.6) once the corpus can verify each one lands correctly, rather than
   trusting the proposed diffs blind.

---

## 5. What This Plan Deliberately Does Not Cover

- UI/UX polish (covered informally in the video-review feedback given
  separately — mostly fine as-is; a couple of truncated metric tiles to widen).
- The classifier/ML model tuning items on `ROADMAP.md`'s pending list
  (waterfall plots, HOC boundary refinement) — those are accuracy-adjacent but
  are a different, larger workstream than "make the numbers this tool already
  tries to report actually correct."
- Anything in `ps26147_toolkit_review_and_fixes.md` §3 (code quality /
  architecture) and §4 (feature suggestions) beyond what's cross-referenced
  above — those are real but lower-priority than closing out known-wrong
  numbers before a judging round.
