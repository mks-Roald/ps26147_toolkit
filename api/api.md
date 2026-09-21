# API Contract

## Endpoints

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/process/file` | POST | multipart/form-data (file) | `{baud_rate?, snr?, modulation, confidence, waveform_data[], constellation_data[], psd_data[], waterfall_data}` |
| `/classify/` | POST | multipart/form-data (file) | `{modulation, confidence, features}` |
| `/decode/` | POST | multipart/form-data (file) | `{modulation, confidence, num_bits, decoded_bits[], demodulated_bits[], deinterleaved_bits[], deinterleaver_method, ...}` |
| `/correlate/` | POST | multipart/form-data (file) | `{status, sync_found, peak_indices[], num_frames, frames[], max_correlation, is_inverted, preamble_discovery, bits[]}` |
| `/correlate/bits` | POST | JSON (`CorrelateRequest`) | `{status, sync_found, peak_indices[], num_frames, frames[], max_correlation, is_inverted, preamble_discovery, bits[]}` |
| `/correlate/sync-words` | GET | — | `{ [sync_word_name]: { length, bit_string, hex } }` |
| `/health` | GET | — | `{status: "ok"}` |

### Notes
- `/decode/` returns the full untruncated bit arrays for `decoded_bits` (post-FEC), pre-FEC `demodulated_bits`, and `deinterleaved_bits` from automatic de-interleaver detection.
- `/correlate/` supports file upload or direct bitstream arrays for preamble auto-discovery, Barker/CCSDS/DVB-S/WiFi sync word correlation, 180° phase inversion handling, and frame slicing.
- All file endpoints accept raw binary files (e.g., .wav, .iq) and convert them to float32 complex/real signals.