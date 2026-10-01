# Ground-Truth Accuracy Report

- **Corpus:** `_cal`
- **Run:** 7 file(s)

## Summary

| metric | value |
|---|---|
| checks passed | 42/42 |
| overall pass rate | 100.00% (target >= 95%) |
| files crashed | 0 |

## Per-File Detail

### `16qam_clean_wav.wav` (GT: 16QAM)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | 16QAM | 16QAM | PASS exact match |
| center_frequency_hz | 10000.0 | 10028.235886678103 | PASS GT 1e+04 vs got 1.003e+04 (0.28%) |
| bandwidth_hz | 1620.0 | 1593.75 | PASS GT 1620 vs got 1594 (1.62%) |
| baud_rate | 1200.0 | 1199.416357370108 | PASS GT 1200 vs got 1199 (0.05%) |
| snr_db | clean | 53.31433383148682 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |

### `2fsk_clean_wav.wav` (GT: 2FSK)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | 2FSK | 2FSK | PASS exact match |
| center_frequency_hz | 10000.0 | 9954.83592384494 | PASS GT 1e+04 vs got 9955 (0.45%) |
| bandwidth_hz | 2196.0 | 2203.125 | PASS GT 2196 vs got 2203 (0.32%) |
| baud_rate | 1200.0 | 1200.1445535367109 | PASS GT 1200 vs got 1200 (0.01%) |
| snr_db | clean | 52.285324886345144 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |

### `4fsk_clean_wav.wav` (GT: 4FSK)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | 4FSK | 4FSK | PASS exact match |
| center_frequency_hz | 10000.0 | 9953.532403042751 | PASS GT 1e+04 vs got 9954 (0.46%) |
| bandwidth_hz | 2196.0 | 1968.75 | PASS GT 2196 vs got 1969 (10.35%) |
| baud_rate | 1200.0 | 1200.1561935255777 | PASS GT 1200 vs got 1200 (0.01%) |
| snr_db | clean | 56.81839873149109 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |

### `64qam_clean_wav.wav` (GT: 64QAM)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | 64QAM | 64QAM | PASS exact match |
| center_frequency_hz | 10000.0 | 9968.503728387992 | PASS GT 1e+04 vs got 9969 (0.31%) |
| bandwidth_hz | 1620.0 | 1640.625 | PASS GT 1620 vs got 1641 (1.27%) |
| baud_rate | 1200.0 | 1199.415629234614 | PASS GT 1200 vs got 1199 (0.05%) |
| snr_db | clean | 52.72798613259405 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |

### `8psk_clean_wav.wav` (GT: 8PSK)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | 8PSK | 8PSK | PASS exact match |
| center_frequency_hz | 10000.0 | 9939.508993349325 | PASS GT 1e+04 vs got 9940 (0.60%) |
| bandwidth_hz | 1620.0 | 1593.75 | PASS GT 1620 vs got 1594 (1.62%) |
| baud_rate | 1200.0 | 1200.1447793172 | PASS GT 1200 vs got 1200 (0.01%) |
| snr_db | clean | 53.25588028887769 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |

### `bpsk_clean_wav.wav` (GT: BPSK)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | BPSK | BPSK | PASS exact match |
| center_frequency_hz | 10000.0 | 9999.978629943425 | PASS GT 1e+04 vs got 1e+04 (0.00%) |
| bandwidth_hz | 1620.0 | 1640.625 | PASS GT 1620 vs got 1641 (1.27%) |
| baud_rate | 1200.0 | 1200.1459178871223 | PASS GT 1200 vs got 1200 (0.01%) |
| snr_db | clean | 52.501185267069914 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |

### `qpsk_clean_wav.wav` (GT: QPSK)

| parameter | GT | extracted | pass |
|---|---|---|---|
| modulation | QPSK | QPSK | PASS exact match |
| center_frequency_hz | 10000.0 | 9983.156415405123 | PASS GT 1e+04 vs got 9983 (0.17%) |
| bandwidth_hz | 1620.0 | 1593.75 | PASS GT 1620 vs got 1594 (1.62%) |
| baud_rate | 1200.0 | 1200.1451985299143 | PASS GT 1200 vs got 1200 (0.01%) |
| snr_db | clean | 52.32687007989561 | PASS N/A (GT clean -> reading pegs at ceiling) |
| payload_text | the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789the quick brown fox jumps over the lazy dog 0123456789 | None | PASS N/A (measure fields only; demod payload path not yet wired) |
