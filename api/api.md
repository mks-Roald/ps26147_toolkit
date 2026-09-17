# API Contract

## Endpoints

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/process/file` | POST | multipart/form-data (file) | `{baud_rate?, snr?, modulation, confidence, waveform_data[]}` |
| `/classify/` | POST | multipart/form-data (file) | `{modulation, confidence}` |
| `/decode/` | POST | multipart/form-data (file) | `{modulation, confidence, decoded_bits[]}` |
| `/health` | GET | — | `{status: "ok"}` |

### Notes
- The `baud_rate` and `snr` fields are placeholders; implement extraction from the signal as needed.
- `waveform_data` returns a time-series array (first 1000 samples) for plotting.
- `decoded_bits` is a placeholder for future FEC decoding implementation.
- All file endpoints accept raw binary files (e.g., .wav, .iq) and convert them to a float32 signal in [-1, 1] assuming int16 PCM. Adjust the file parsing based on your actual file format.