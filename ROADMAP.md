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
| **Baud Rate & SNR Calibration** | ⏳ Pending | Upgrade symbol rate & SNR algorithms for higher accuracy |
| **Modulation Classifier Model** | ⏳ Pending | Bundle/train pre-trained `model.pkl` for automatic modulation recognition |
| **Demodulators (FSK/PSK/QAM)** | ⏳ Pending | Implement coherent/non-coherent symbol recovery & constellation viewer |
| **De-Interleaving** | ⏳ Pending | Implement 4 de-interleaving algorithms |
| **FEC Decoders** | ⏳ Pending | Implement Viterbi, Reed-Solomon, Concatenated, LDPC decoders |
| **Bit Stream Correlation** | ⏳ Pending | Frame sync and preamble pattern cross-correlation |

---

## 📝 Change Log (Fixes Applied)
1. Fixed `NameError: name 'np' is not defined` in `web_demo/app.py`.
2. Fixed negative bandwidth bug in `ps26147_toolkit/feature_extractor.py` using `np.fft.fftshift` on complex signals.
3. Added `fs_iq` variable in `web_demo/app.py` to allow user-defined IQ sampling frequency instead of hardcoded `1.0`.
4. Added `.gitignore` to protect cache, temporary uploads, and build artifacts from Git tracking.
