# PS26147 Toolkit — Cross-Document Analysis

Condensed comparison of three review documents:
- `ps26147_toolkit_analysis.md`
- `PS26147_Toolkit_Engineering_Task_Solution_Roadmap.md`
- `ps26147_toolkit_review_and_fixes.md`

---

## 🔴 Common to all 3 docs (fix first — everyone agrees these are the top blockers)

| Issue | Analysis doc | Roadmap doc | Fixes doc |
|---|---|---|---|
| **`.iq` loader hard-codes float32**, breaking real SDR captures (int8/16) | Task 4 | Task A1 | Bug 1.1 (has drop-in fix + test) |
| **No/default sample rate** (`fs` defaults silently to 1MHz) | Task 6 | Task A2 | Called out as root cause under A2, ties to loader fix |
| **Classifier runs untrustworthy features / misclassifies** | Task 1 (no baseband downconversion) | Phase 5 "better features" | Bug 1.2 (BPSK/QPSK → misread as FSK) + Bug 1.4 (C63 formula wrong) — most specific root-cause of the three |
| **Testing is a major gap** | Task 9 | "Testing: 🔴 Critical" | Section 5, with actual test code provided |
| **Training data problems (classifier)** | Task 2 (synthetic ≠ RadioML) | Phase 5.1 "realistic synthetic channels" | Issue 3.4 "RadioML loader" blocks retraining |

**Takeaway:** all three independently converge on the same #1/#2 priority: fix IQ dtype handling, then fix the classifier's core correctness — before touching anything else.

---

## 🟠 Similar but described at different depth (same underlying problem, different angle)

| Theme | How each doc frames it |
|---|---|
| **Center frequency / spectral detection** | Analysis: DC offset not removed before first PSD pass (Task 3). Roadmap: peak-only detection fails on multi-signal/FSK/low-SNR — proposes full signal-mask + region-scoring algorithm (Task B1). Fixes doc: doesn't address this directly (focuses on classifier/FEC bugs instead). |
| **Bandwidth / SNR estimation** | Analysis: noise floor = 25th percentile of whole spectrum overestimates it (Task 5). Roadmap: wants multiple standardized measures (BW-3dB, BW-10dB, OBW95/99) plus 5 different SNR methods with confidence scores (Tasks B2/B3). Fixes doc: doesn't cover this. |
| **Baud rate estimation** | Analysis: no windowing on FFT, weak peak threshold (Tasks 7/8). Roadmap: wants to replace the whole approach with multi-candidate scoring (autocorrelation, cyclostationary, Gardner/Mueller-Müller) (Task C1). Fixes doc: flags it only as a missing-test item (2.1). |
| **FEC (RS/LDPC)** | Analysis: doesn't mention FEC at all (scope is signal-analysis only). Roadmap: LDPC "not standards-compatible," RS "needs robust block handling" (high-level). Fixes doc: gives the actual bugs — RS error-magnitude solver is wrong, LDPC H-matrix isn't valid (Bugs 1.3, 1.6) — most concrete of the three. |
| **Silent failure modes** | Analysis: silent `fs` default (#6). Roadmap: "remove silent exception handling" (bug list #20). Fixes doc: shows the exact silent-garbage symptom (`1e-37` magnitudes) for the int16 bug. |

**Takeaway:** the Analysis doc is the narrowest/most surgical (10 tasks, signal-analysis module only). The Roadmap doc is the broadest, reframing the entire pipeline (adds demod/FEC/framing/architecture, confidence-aware redesign). The Fixes doc sits in between — same bug list as Analysis+Roadmap but with actual verified root causes and working code for classifier and FEC bugs that the other two only gesture at.

---

## Unique to one document only

- **Analysis doc only:** IQ imbalance / image-rejection correction (Task 10) — not mentioned elsewhere.
- **Roadmap doc only:** Full pipeline redesign (Acquire → Validate → Characterize → ... → Payload+confidence), 7-phase execution plan, final report format mockup, status taxonomy (⚪🔵🟡🟠🟢🔴). This is a program-management layer the other two don't attempt.
- **Fixes doc only:** Costas loop phase-unwrap/drift bug (1.5), hex-binned constellation plotting, soft-Viterbi LLR upgrade, and the reference cumulant ground-truth table (Appendix B) — genuinely useful for writing the tests all three docs say are missing.

---

## Consolidated action order (merging all three)

1. **Fix `load_iq()` dtype handling** — use the Fixes-doc implementation (has autodetect + round-trip test already written).
2. **Fix classifier root causes** — Bug 1.2 (median-filter phase derivative, add PSK phase-histogram tie-break) + Bug 1.4 (C63 formula) before anything else classifier-related; this subsumes Analysis Task 1.
3. **Require/validate `fs` explicitly** — adopt Roadmap's `SignalMetadata` object instead of a silent default.
4. **Remove DC offset unconditionally** before first PSD pass (Analysis Task 3) — cheap, high-leverage.
5. **Fix RS + LDPC bugs** (Fixes doc 1.3/1.6) — currently silently wrong, not just "needs robustness" as Roadmap frames it.
6. **Add the concrete tests** from Fixes doc §5.2 (they cover Analysis Task 9 and Roadmap's testing gap directly).
7. Only after 1–6: tackle the deeper redesigns — bandwidth/SNR multi-method estimators, baud-rate candidate scoring, IQ-imbalance correction, and the full confidence-aware pipeline architecture from the Roadmap doc.

This order respects what all three documents independently agree on (fix ingestion + classifier correctness first) while sequencing the Roadmap's larger architectural ambitions after the codebase is verifiably correct.
