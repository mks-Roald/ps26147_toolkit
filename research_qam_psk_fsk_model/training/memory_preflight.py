"""Measure multimodal model memory without an optimizer step or data access.

The backward pass exists only to retain/measure training activations. It uses
zero-valued shape probes, does not load samples, call optimizer.step, or save a
checkpoint. This is a memory preflight, not a training/evaluation command.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..config import load_config
from ..model.multimodal_model import MultimodalAMR, count_parameters


def activation_stage_elements_per_sample() -> dict[str, int]:
    iq = 32*4096 + 64*4096 + 64*2048 + 128*2048
    temporal = 32*4096 + 32*2048 + 64*2048 + 64*1024 + 96*512
    psd = 16*2048 + 16*512 + 32*512 + 32*128 + 64*128
    return {"iq": iq, "polar_psk_if_combined": 3*temporal, "psd": psd,
            "total": iq + 3*temporal + psd}


def _probe(model: MultimodalAMR, batch_size: int, device: torch.device, amp: bool) -> dict:
    model.to(device).train()
    probe = (
        torch.zeros(batch_size, 2, 4096, device=device),
        torch.zeros(batch_size, 4, 4096, device=device),
        torch.zeros(batch_size, 4, 4096, device=device),
        torch.zeros(batch_size, 1, 4096, device=device),
        torch.zeros(batch_size, 1, 2048, device=device),
        torch.zeros(batch_size, 22, device=device),
    )
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=bool(amp and device.type == "cuda")):
        logits = model(*probe)
        loss = logits.float().square().sum()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        allocated = reserved = None
    elapsed = time.perf_counter() - started
    params = sum(p.numel() for p in model.parameters())
    parameter_bytes = params * 4
    activation_elements = activation_stage_elements_per_sample()["total"] * batch_size
    # Stage-output floor only; backward-saved intermediates and library workspace
    # are captured by the measured probe instead of guessed into this floor.
    stage_activation_fp16 = activation_elements * 2
    stage_activation_fp32 = activation_elements * 4
    input_bytes = batch_size * (2*4096 + 4*4096 + 4*4096 + 4096 + 2048 + 22) * 4
    return {
        "batch_size": batch_size,
        "amp": bool(amp and device.type == "cuda"),
        "probe_seconds": elapsed,
        "probe_peak_allocated_bytes": allocated,
        "probe_peak_reserved_bytes": reserved,
        "input_tensor_bytes_fp32": input_bytes,
        "parameter_bytes_fp32": parameter_bytes,
        "parameter_gradient_adamw_static_bytes_estimate": parameter_bytes * 4,
        "conv_pool_stage_activation_floor_fp16_bytes": stage_activation_fp16,
        "conv_pool_stage_activation_floor_fp32_bytes": stage_activation_fp32,
        "training_peak_lower_estimate_bytes": (allocated + 2*parameter_bytes) if allocated is not None else None,
        "training_peak_with_25pct_margin_bytes": int((allocated + 2*parameter_bytes) * 1.25) if allocated is not None else None,
        "oom": False,
    }


def run(config_path: str | Path, batches: list[int], amp: bool = True) -> dict:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {"device": str(device), "pytorch": torch.__version__,
                   "cuda_version": torch.version.cuda, "amp_requested": amp}
    if device.type == "cuda":
        torch.cuda.init()
        device_info.update({"gpu": torch.cuda.get_device_name(0),
                            "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory)})
    results = []
    for batch in batches:
        model = MultimodalAMR(feature_count=22, num_classes=7,
                              iq_dim=config.architecture.iq_embedding_dim,
                              aux_dim=config.architecture.auxiliary_embedding_dim,
                              fused_dim=config.architecture.fused_embedding_dim,
                              dropout=config.architecture.dropout)
        try:
            result = _probe(model, batch, device, amp)
            results.append(result)
            print(f"batch={batch} allocated={result['probe_peak_allocated_bytes']} "
                  f"reserved={result['probe_peak_reserved_bytes']} OOM=False")
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results.append({"batch_size": batch, "oom": True, "error": str(exc)[:500],
                            "probe_peak_allocated_bytes": None, "probe_peak_reserved_bytes": None})
            print(f"batch={batch} OOM=True")
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {"scope": "memory preflight only; zero probes; no optimizer step; no data access",
               "device_info": device_info, "parameter_counts": count_parameters(
                   MultimodalAMR(feature_count=22, num_classes=7,
                                 iq_dim=config.architecture.iq_embedding_dim,
                                 aux_dim=config.architecture.auxiliary_embedding_dim,
                                 fused_dim=config.architecture.fused_embedding_dim,
                                 dropout=config.architecture.dropout)),
               "activation_stage_elements_per_sample": activation_stage_elements_per_sample(),
               "batches": results}
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research_qam_psk_fsk_model/configs/real_finetune_7class.yaml")
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--output", default="research_qam_psk_fsk_model/results/real_finetune/memory_preflight.json")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.config, args.batches, amp=not args.no_amp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Memory preflight report saved to {output.resolve()}")


if __name__ == "__main__":
    main()
