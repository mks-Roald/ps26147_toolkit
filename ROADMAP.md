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
| **De-Interleaving** | ⏳ Pending | Implement 4 de-interleaving algorithms |
| **FEC Decoders** | ⏳ Pending | Implement Viterbi, Reed-Solomon, Concatenated, LDPC decoders |
| **Bit Stream Correlation** | ⏳ Pending | Frame sync and preamble pattern cross-correlation |

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
