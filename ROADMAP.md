# PS26147 Project Roadmap & Task Tracker

## 📌 Problem Statement Objectives
- **Input:** `.iq` or `.wav` signal files.
- **Task i:** Identify signal parameters ($f_s$, Modulation, FEC, Interleaving).
- **Task ii:** Demodulate signals (FSK, QAM, PSK).
- **Task iii:** De-interleaving (Block, Convolutional, Diagonal, Pseudo-Random).
- **Task iv:** FEC Decoding (Viterbi for conv codes, RS block codes, Concatenated, LDPC).
- **Task v:** Bit stream correlation & synchronization.

---

## 🚦 Status Summary

| Phase / Feature | Status | Description |
|---|---|---|
| **Scaffold & Loading** | ✅ Done | `.iq` and `.wav` file loaders in `preprocess.py` |
| **GUI & Visualization** | ✅ Done | Streamlit GUI with drag-and-drop & Spectrograms (`web_demo/app.py`) |
| **Negative Bandwidth Fix** | ✅ Done | Fixed unsorted complex frequency bins with `fftshift` in `feature_extractor.py` |
| **IQ Sample Rate Input** | ✅ Done | Added customizable sample rate sidebar in GUI for `.iq` scaling |
| **DSP Noise Filtering** | ✅ Done | Butterworth adaptive bandpass, spectral subtraction denoising, DC offset filter (`filters.py`) |
| **Baud Rate & SNR Calibration** | ✅ Done | Upgraded robust in-band integrated SNR & cyclic envelope transition baud rate estimator (`parameter_extractor.py`) |
| **Modulation Classifier Model** | ✅ Done | Higher-Order Cumulants ($C_{40}, C_{42}, C_{63}$) + automated ML classifier in `classifier.py` |
| **Demodulators (FSK/PSK/QAM)** | ✅ Done | Carrier recovery (Costas PLL), symbol timing, EVM metric, Gray slicing, & Constellation viewer (`demodulator.py`) |
| **De-Interleaving** | ✅ Done | Block, Convolutional, Diagonal, Pseudo-Random de-interleavers + auto-detect (`deinterleaver.py`) |
| **FEC Decoders** | ✅ Done | Viterbi (NASA K=7), Reed-Solomon GF(2^8), Concatenated (Viterbi+RS), LDPC (Min-Sum) (`fec_decoders.py`) |
| **Bit Stream Correlation** | ✅ Done | Barker, CCSDS, DVB-S, auto-preamble discovery, sliding-window bipolar cross-correlation & framing (`correlator.py`) |
| **Waterfall & Time-Domain Plots** | ⏳ Pending | High-resolution interactive 2D/3D waterfall spectral history and In-Phase/Quadrature time-domain waveform viewer |
| **Constellation Accuracy & EVM Tweaks**| ⏳ Pending | Constellation cluster dispersion metrics, decision boundary calibration, and EVM/SNR optimization |
| **Model Tuning & Error Corrections**  | ⏳ Pending | Hyperparameter tuning, dataset expansion, HOC boundary refinement, and overall AMR classifier accuracy boost |
| **Live Floating Step-by-Step Window** | ⏳ Pending | Real-time floating/docked execution log panel displaying each analysis & DSP step as it executes |
| **Aesthetic, Reactive & Simplistic UI** | ⏳ Pending | Redesign frontend layout with a modern, reactive, clean dark/glassmorphic aesthetic based on reference mockup |

---

## 📝 Change Log (Fixes & Features Applied)
1. Fixed `NameError: name 'np' is not defined` in `web_demo/app.py`.
2. Fixed negative bandwidth bug in `ps26147_toolkit/feature_extractor.py` using `np.fft.fftshift` on complex signals.
3. Added `fs_iq` variable in `web_demo/app.py` to allow user-defined IQ sampling frequency instead of hardcoded `1.0`.
4. Added `.gitignore` to protect cache, temporary uploads, and build artifacts from Git tracking.
5. Added `ps26147_toolkit/filters.py` providing adaptive Butterworth bandpass filtering, spectral subtraction denoising, DC offset cancellation, and median filtering.
6. Rewrote `parameter_extractor.py` with in-band integrated SNR and cyclic transition baud rate estimation.
7. Implemented Higher-Order Cumulant ($C_{20}, C_{40}, C_{42}, C_{63}$) features + trained Random Forest AMR in [`classifier.py`](file:///C:/Users/mehja/.gemini/antigravity/scratch/SIH-PS26147/ps26147_toolkit/classifier.py).
8. Created [`demodulator.py`](file:///C:/Users/mehja/.gemini/antigravity/scratch/SIH-PS26147/ps26147_toolkit/demodulator.py) featuring:
   - Costas Loop Decision-Directed Carrier Tracking (order 2, 4, 8)
   - Symbol timing recovery & downsampling
   - Gray-coded bit slicing for BPSK, QPSK, 8PSK, 16QAM, 64QAM, and FSK
   - EVM (Error Vector Magnitude) calculations in dB
   - Bitstream unpacking & Hex preview
9. Upgraded `web_demo/app.py` with multi-tab interface:
   - **Tab 1: 📊 Spectral & Spectrogram Analysis**
   - **Tab 2: 🌌 Constellation Diagram (I-Q Scatter)**
   - **Tab 3: 💾 Demodulated Bitstream (Binary/Hex view & `.bin` download)**
10. Created [`deinterleaver.py`](file:///C:/Users/mehja/.gemini/antigravity/scratch/SIH-PS26147/ps26147_toolkit/deinterleaver.py) implementing:
    - **Block De-interleaver:** Row/column matrix transpose inversion (R×C → C×R readout)
    - **Convolutional De-interleaver:** Forney/Ramsey complementary shift-register delay lines
    - **Diagonal De-interleaver:** Diagonal-fill matrix permutation inverse
    - **Pseudo-Random De-interleaver:** PRBS-seeded permutation inversion with configurable seed
    - **Auto-detect:** Tries all methods with common parameter grids; selects lowest byte-entropy output
    - Integrated into `web_demo/app.py` as **Tab 4: 🔀 De-Interleaved Output** with sidebar controls
11. Created [`fec_decoders.py`](file:///C:/Users/mehja/.gemini/antigravity/scratch/SIH-PS26147/ps26147_toolkit/fec_decoders.py) implementing:
    - **Viterbi Convolutional Decoder:** NASA/ESA standard $K=7, \text{Rate } 1/2$, polynomials $[171_8, 133_8]$ with trellis path metrics and traceback
    - **Reed-Solomon Decoder:** $GF(2^8)$ field arithmetic, syndrome computation, Berlekamp-Massey, Chien search, and Gaussian elimination error-magnitude solver
    - **Concatenated Decoder:** Dual-stage Inner Viterbi + Outer Reed-Solomon DVB/CCSDS receiver pipeline
    - **LDPC Decoder:** Log-Domain normalized Min-Sum belief propagation message passing on Tanner graphs
    - Integrated into `web_demo/app.py` as **Tab 5: 🛡️ FEC Decoded Stream** with sidebar controls, ASCII preview, and clean payload export
12. Created [`correlator.py`](file:///C:/Users/mehja/.gemini/antigravity/scratch/SIH-PS26147/ps26147_toolkit/correlator.py) implementing:
    - **Standard Preamble / Sync Word Library:** Barker-7/11/13, CCSDS-32 ASM, DVB-S, ZigBee-SFD, WiFi-SFD, 1010 Training sequences, and custom Hex patterns
    - **Bipolar Normalized Cross-Correlation:** Sliding window $-1.0 \dots +1.0$ matching with automatic 180° carrier inversion detection
    - **Frame Synchronization & Boundary Slicing:** Multi-frame lock with stride verification and payload stripping
    - **Auto-Preamble Discovery:** Unsupervised frame period estimation and standard library sync identification
    - Integrated into `web_demo/app.py` as **Tab 6: 🎯 Frame Correlation & Sync** with interactive correlation curve plotting, extracted frame tables, and aligned payload export
