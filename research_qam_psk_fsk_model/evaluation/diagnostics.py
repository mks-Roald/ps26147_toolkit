"""Conditional QAM/FSK diagnostic tables; groups without metadata are omitted."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..labels import CLASS_NAMES


def snr_bin(snr: float) -> str:
    if snr < -5: return "< -5 dB"
    if snr < 0: return "-5 to <0 dB"
    if snr < 5: return "0 to <5 dB"
    if snr < 10: return "5 to <10 dB"
    return ">= 10 dB"


def conditional_error_tables(y_true: list[int], y_pred: list[int], metadata: list[dict[str,Any]]) -> dict[str,Any]:
    if not (len(y_true)==len(y_pred)==len(metadata)):
        raise ValueError("Labels, predictions, metadata must be aligned")
    result={"qam_by_snr":{},"fsk_by_snr":{},"fsk_by_sps":{},"fsk_by_sample_rate":{}}
    dimensions=(
      ("qam_by_snr",lambda y:y in (3,4),lambda m:_safe_snr_bin(m)),
      ("fsk_by_snr",lambda y:y in (5,6),lambda m:_safe_snr_bin(m)),
      ("fsk_by_sps",lambda y:y in (5,6),lambda m:str(m.get("sps",m.get("samples_per_symbol"))) if m.get("sps",m.get("samples_per_symbol")) is not None else None),
      ("fsk_by_sample_rate",lambda y:y in (5,6),lambda m:str(m.get("sample_rate_hz")) if m.get("sample_rate_hz") is not None else None),)
    for name,is_target,key_fn in dimensions:
        groups=defaultdict(list)
        for t,p,m in zip(y_true,y_pred,metadata):
            if is_target(t):
                key=key_fn(m)
                if key is not None: groups[key].append((t,p))
        for key,rows in groups.items():
            true,pred=zip(*rows)
            result[name][key]={"n":len(rows),"accuracy":float(np.mean(np.asarray(true)==np.asarray(pred))),
                "recall_by_class":{CLASS_NAMES[c]:float(np.mean(np.asarray(pred)[np.asarray(true)==c]==c))
                                   for c in sorted(set(true))},
                "4FSK_to_2FSK":int(sum(t==6 and p==5 for t,p in rows)),
                "16QAM_to_64QAM":int(sum(t==3 and p==4 for t,p in rows)),
                "64QAM_to_16QAM":int(sum(t==4 and p==3 for t,p in rows))}
    return result


def _safe_snr_bin(metadata):
    try:
        value=float(metadata["snr_db"])
        return snr_bin(value) if np.isfinite(value) else None
    except (KeyError,TypeError,ValueError):
        return None
