"""Future training entry point. Requires explicit --execute-training opt-in."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.multimodal_dataset import MultimodalDataset, fit_train_scalers, load_manifest
from ..data.psd_utils import PSDStandardizer
from ..labels import CLASS_NAMES, CLASS_TO_INDEX, FEATURE_NAMES
from ..model.multimodal_model import MultimodalAMR, count_parameters
from ..evaluation.diagnostics import conditional_error_tables
from .losses import FocalLoss, low_snr_qam_sample_weights

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scaler_state(scaler: StandardScaler) -> dict[str, Any]:
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(), "n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist(),
            "n_features_in": int(scaler.n_features_in_), "method": "StandardScaler",
            "fit_split": "train_only"}


def _restore_feature_scaler(state: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var"], dtype=np.float64)
    scaler.n_samples_seen_ = np.asarray(state["n_samples_seen"])
    if scaler.n_samples_seen_.ndim == 0:
        scaler.n_samples_seen_ = int(scaler.n_samples_seen_)
    scaler.n_features_in_ = int(state["n_features_in"])
    return scaler


def _stage_paths(config, stage: str):
    d = config.data
    if stage == "synthetic_pretrain":
        return d.synthetic_pretraining_dataset_root, d.synthetic_pretraining_train_manifest, d.synthetic_pretraining_validation_manifest
    if stage == "real_finetune":
        return d.real_finetuning_dataset_root, d.real_finetuning_train_manifest, d.real_finetuning_validation_manifest
    if stage == "real_from_scratch":
        return d.dataset_root, d.train_manifest, d.validation_manifest
    raise ValueError(f"Unsupported stage {stage!r}")


def _verify_records(train, val):
    tr = {str(Path(x["path"]).resolve()).casefold() for x in train}
    va = {str(Path(x["path"]).resolve()).casefold() for x in val}
    if tr & va:
        raise ValueError("TRAIN/VALIDATION path overlap")
    for name, rows in (("TRAIN", train), ("VALIDATION", val)):
        if not rows:
            raise ValueError(f"{name} manifest is empty")
        unexpected = sorted({r["label"] for r in rows} - set(CLASS_NAMES))
        if unexpected:
            raise ValueError(f"Unexpected classes in {name}: {unexpected}")
    missing_train = sorted(set(CLASS_NAMES) - {r["label"] for r in train})
    missing_val = sorted(set(CLASS_NAMES) - {r["label"] for r in val})
    if missing_train or missing_val:
        raise ValueError(f"All canonical classes are required: missing train={missing_train}, val={missing_val}")


def _device_and_amp(request_amp: bool):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    fallback_error = None
    if cuda_available:
        try:
            torch.cuda.init()
            _ = torch.empty(1, device="cuda")
        except Exception as exc:
            fallback_error = f"CUDA initialization failed: {type(exc).__name__}: {exc}"
            warnings.warn(fallback_error + "; falling back to CPU", RuntimeWarning)
            device = torch.device("cpu")
    amp = bool(request_amp and device.type == "cuda" and hasattr(torch, "autocast") and hasattr(torch, "amp"))
    info = {"pytorch": torch.__version__, "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda, "cudnn_available": torch.backends.cudnn.is_available(),
            "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
            "selected_device": str(device), "amp": amp, "cuda_fallback_error": fallback_error}
    print("DEVICE CHECK\n------------")
    for key, value in info.items(): print(f"{key}: {value}")
    return device, amp, info


def _model_inputs(batch, device):
    names = ("iq", "polar", "psk", "ifreq", "psd", "features")
    return [batch[name].to(device, non_blocking=True) for name in names]


_ENCODER_PREFIXES = ("iq_encoder.", "polar_encoder.", "psk_encoder.",
                     "if_encoder.", "psd_encoder.", "feature_encoder.")


def _set_transfer_trainability(model, transfer, epoch: int) -> dict[str, int]:
    """Freeze all encoders for warm-up, then unfreeze configured module prefixes."""
    start_epoch = transfer.selective_unfreeze_epoch
    if start_epoch is None:
        start_epoch = int(transfer.encoder_freeze_epochs) + 1
    selected = tuple(transfer.selective_unfreeze_modules or ())
    active = epoch >= int(start_epoch)
    counts = {"encoder_trainable": 0, "fusion_head_trainable": 0}
    for name, parameter in model.named_parameters():
        is_encoder = name.startswith(_ENCODER_PREFIXES)
        if not is_encoder:
            parameter.requires_grad_(True)
            counts["fusion_head_trainable"] += parameter.numel()
            continue
        allow_all = not selected
        allow_selected = any(name == prefix or name.startswith(prefix + ".") for prefix in selected)
        trainable = active and (allow_all or allow_selected)
        parameter.requires_grad_(trainable)
        if trainable:
            counts["encoder_trainable"] += parameter.numel()
    return counts


def _set_transfer_module_modes(model, transfer, epoch: int) -> None:
    """Keep frozen encoder BatchNorm in eval mode; train selected blocks only."""
    if int(transfer.encoder_freeze_epochs) <= 0 and not transfer.selective_unfreeze_modules:
        return
    encoder_modules = (model.iq_encoder, model.polar_encoder, model.psk_encoder,
                       model.if_encoder, model.psd_encoder, model.feature_encoder)
    for module in encoder_modules:
        module.eval()
    start_epoch = transfer.selective_unfreeze_epoch
    if start_epoch is None:
        start_epoch = int(transfer.encoder_freeze_epochs) + 1
    if epoch >= int(start_epoch):
        modules = dict(model.named_modules())
        for name in transfer.selective_unfreeze_modules:
            module = modules.get(name)
            if module is None:
                raise ValueError(f"Unknown selective_unfreeze_modules entry: {name!r}")
            module.train()


def _optimizer_param_groups(model, config):
    encoder_lr = config.transfer.encoder_learning_rate or config.training.learning_rate
    head_lr = config.transfer.fusion_head_learning_rate or config.training.learning_rate
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (encoder_params if name.startswith(_ENCODER_PREFIXES) else head_params).append(parameter)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(encoder_lr), "group_name": "encoders"})
    if head_params:
        groups.append({"params": head_params, "lr": float(head_lr), "group_name": "fusion_head"})
    return groups


def _loader_options(data_config, device):
    workers = max(0, int(data_config.num_workers))
    options = {"num_workers": workers,
               "pin_memory": bool(data_config.pin_memory and device.type == "cuda")}
    if workers > 0:
        options["persistent_workers"] = bool(data_config.persistent_workers)
        options["prefetch_factor"] = max(1, int(data_config.prefetch_factor))
    return options


def _gpu_snapshot(enabled: bool) -> dict[str, int | None] | None:
    if not enabled or shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=3, check=True)
        values = [int(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        if len(values) != 3:
            return None
        return {"utilization_pct": values[0], "memory_used_mb": values[1], "memory_total_mb": values[2]}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _checkpoint_scalers(payload):
    feature = payload.get("feature_scaler")
    psd = payload.get("psd_scaler")
    if not isinstance(feature, dict) or not isinstance(psd, dict):
        raise ValueError("Pretrained checkpoint does not contain serialized feature and PSD scalers")
    feature_scaler = _restore_feature_scaler(feature)
    psd_scaler = PSDStandardizer.from_state_dict(psd)
    audit = {"fit_split": "synthetic_train_checkpoint", "reused_from_checkpoint": True,
             "feature_names": list(payload.get("feature_names", FEATURE_NAMES)),
             "feature_scaler": {"mean": feature["mean"], "scale": feature["scale"]},
             "psd_scaler": psd, "scaler_policy": "reuse frozen source-stage transforms"}
    return feature_scaler, psd_scaler, audit


def _metrics(y_true, y_pred, probabilities, records):
    report = classification_report(y_true, y_pred, labels=list(range(7)), target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    result = {"accuracy": float(accuracy_score(y_true, y_pred)),
              "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
              "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
              "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
              "per_class": report, "confusion_matrix": matrix.tolist()}
    result["cross_confusions"] = {"16QAM_to_64QAM": int(matrix[3,4]),
        "64QAM_to_16QAM": int(matrix[4,3]), "2FSK_to_4FSK": int(matrix[5,6]),
        "4FSK_to_2FSK": int(matrix[6,5])}
    factor_rows = []
    for i, record in enumerate(records):
        meta = record.get("metadata", {})
        factor_rows.append({"true": CLASS_NAMES[y_true[i]], "pred": CLASS_NAMES[y_pred[i]],
                            "snr_db": meta.get("snr_db"), "sps": meta.get("sps", meta.get("samples_per_symbol")),
                            "sample_rate_hz": meta.get("sample_rate_hz"), "fading": meta.get("fading_type")})
    result["qam_by_snr"] = _group_summary(factor_rows, lambda r: r["true"] in {"16QAM", "64QAM"}, "snr_db")
    result["fsk_by_snr"] = _group_summary(factor_rows, lambda r: r["true"] in {"2FSK", "4FSK"}, "snr_db")
    result["fsk_by_sps"] = _group_summary(factor_rows, lambda r: r["true"] in {"2FSK", "4FSK"}, "sps")
    result["fsk_by_sample_rate"] = _group_summary(factor_rows, lambda r: r["true"] in {"2FSK", "4FSK"}, "sample_rate_hz")
    result["conditional_diagnostics"] = conditional_error_tables(y_true,y_pred,
        [r.get("metadata",{}) for r in records])
    return result


def _group_summary(rows, include, field):
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if include(row) and row.get(field) is not None:
            groups.setdefault(str(row[field]), []).append(row)
    return {key: {"n": len(group), "accuracy": float(np.mean([r["true"] == r["pred"] for r in group])),
                  "4FSK_recall": (float(np.mean([r["pred"] == "4FSK" for r in group if r["true"] == "4FSK"]))
                                  if any(r["true"] == "4FSK" for r in group) else None),
                  "4FSK_to_2FSK": sum(r["true"] == "4FSK" and r["pred"] == "2FSK" for r in group)}
            for key, group in groups.items()}


@torch.inference_mode()
def evaluate(model, loader, records, device, criterion):
    model.eval(); truth, predictions, probabilities = [], [], []
    loss_total, count = 0.0, 0
    for batch in loader:
        inputs = _model_inputs(batch, device)
        target = batch["target"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
            logits = model(*inputs)
            loss = criterion(logits, target)  # no low-SNR sample weights on validation
        probs = torch.softmax(logits.float(), dim=1)
        loss_total += float(loss) * len(target); count += len(target)
        truth.extend(target.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
    return loss_total / max(count, 1), _metrics(truth, predictions, probabilities, records), truth, predictions, probabilities


def _preflight(model, device, amp, batch_size: int):
    model.to(device); model.eval()
    b = 1  # deliberately lightweight shape/gradient preflight; no training batch is consumed
    args = (torch.zeros(b,2,4096,device=device), torch.zeros(b,4,4096,device=device),
            torch.zeros(b,4,4096,device=device), torch.zeros(b,1,4096,device=device),
            torch.zeros(b,1,2048,device=device), torch.zeros(b,22,device=device))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    # Keep eval mode so the gradient shape check cannot alter BatchNorm buffers.
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(*args)
        loss = logits.float().sum()
    loss.backward()
    if not torch.isfinite(logits).all() or not any(p.grad is not None for p in model.parameters()):
        raise FloatingPointError("Preflight produced invalid output/gradients")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    model.zero_grad(set_to_none=True)
    if device.type == "cuda": torch.cuda.empty_cache()
    print(f"GPU PREFLIGHT: parameters={sum(p.numel() for p in model.parameters())}; device={device}; "
          f"probe_batch=1; configured_physical_batch={batch_size}; AMP={amp}; peak VRAM={peak}; PASS")
    return peak


def run_training(config_path: str | Path, stage: str | None = None, *, output_dir: str | Path | None = None,
                 init_checkpoint: str | Path | None = None):
    config = load_config(config_path)
    if config.data.input_length != 4096 or config.architecture.psd_bins != 2048:
        raise ValueError("This model version requires input_length=4096 and psd_bins=2048")
    stage = stage or config.stage
    root, train_manifest, val_manifest = _stage_paths(config, stage)
    if not train_manifest or not val_manifest:
        raise ValueError(f"Configure TRAIN and VALIDATION manifests for stage={stage}; no data is fabricated")
    train = load_manifest(train_manifest, root); val = load_manifest(val_manifest, root)
    _verify_records(train, val)
    seed_everything(config.training.seed)
    device, amp, device_info = _device_and_amp(config.training.amp)
    checkpoint_in = Path(init_checkpoint or config.init_checkpoint) if (init_checkpoint or config.init_checkpoint) else None
    checkpoint_payload = None
    if checkpoint_in:
        checkpoint_payload = torch.load(checkpoint_in, map_location="cpu", weights_only=False)
        if checkpoint_payload.get("class_names") != list(CLASS_NAMES) or checkpoint_payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Initial checkpoint class/feature ordering is incompatible")
    scaler_policy = config.transfer.scaler_policy
    if scaler_policy == "pretrained_checkpoint":
        if checkpoint_payload is None:
            raise ValueError("scaler_policy=pretrained_checkpoint requires --init-checkpoint")
        feature_scaler, psd_scaler, scaler_audit = _checkpoint_scalers(checkpoint_payload)
    elif scaler_policy == "fit_train":
        feature_scaler, psd_scaler, scaler_audit = fit_train_scalers(
            train, config.data.iq_fallback_sample_rate_hz, config.training.seed)
    else:
        raise ValueError(f"Unsupported scaler_policy={scaler_policy!r}; choose fit_train or pretrained_checkpoint")
    scaler_state = _scaler_state(feature_scaler)
    psd_state = psd_scaler.state_dict()
    train_ds = MultimodalDataset(train, feature_scaler, psd_scaler, "train", seed=config.training.seed,
                                 phase_rotation=config.training.phase_rotation_augmentation,
                                 fallback_fs=config.data.iq_fallback_sample_rate_hz)
    val_ds = MultimodalDataset(val, feature_scaler, psd_scaler, "val", seed=config.training.seed,
                               fallback_fs=config.data.iq_fallback_sample_rate_hz)
    generator = torch.Generator().manual_seed(config.training.seed)
    loader_args = _loader_options(config.data, device)
    print("DATALOADER CONFIG: " + json.dumps(loader_args, sort_keys=True))
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                              generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, **loader_args)
    model = MultimodalAMR(feature_count=22, num_classes=len(CLASS_NAMES),
                          iq_dim=config.architecture.iq_embedding_dim,
                          aux_dim=config.architecture.auxiliary_embedding_dim,
                          fused_dim=config.architecture.fused_embedding_dim,
                          dropout=config.architecture.dropout)
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state_dict"])
    initial_trainability = _set_transfer_trainability(model, config.transfer, epoch=1)
    peak_preflight = _preflight(model, device, amp, config.training.batch_size)
    optimizer = torch.optim.AdamW(_optimizer_param_groups(model, config), lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    criterion = FocalLoss(config.training.focal_gamma, reduction="mean")
    train_criterion = FocalLoss(config.training.focal_gamma, reduction="none")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    scaler_rows = []
    missing_snr = 0
    for record in train:
        value=record.get("metadata", {}).get("snr_db")
        try: valid_snr=value is not None and np.isfinite(float(value))
        except (TypeError,ValueError): valid_snr=False
        if not valid_snr: missing_snr += 1
    if missing_snr:
        warnings.warn(f"SNR metadata missing for {missing_snr}/{len(train)} TRAIN records; those samples use loss weight 1.0.",
                      RuntimeWarning)
    out_root = Path(output_dir or config.training.output_dir).resolve()
    try:
        out_root.relative_to(PACKAGE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Experiment outputs must remain under {PACKAGE_ROOT}") from exc
    out = out_root / stage / f"seed_{config.training.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config_record = config.to_dict()
    config_record.update({"stage": stage, "train_manifest": str(Path(train_manifest).resolve()),
                          "train_manifest_sha256": _sha256(train_manifest),
                          "validation_manifest": str(Path(val_manifest).resolve()),
                          "validation_manifest_sha256": _sha256(val_manifest),
                          "train_count": len(train), "validation_count": len(val),
                          "train_validation_disjoint": True, "test_manifest_opened": False,
                          "test_iq_opened": False, "validation_used_for_scaler_fit": False,
                          "device_info": device_info,
                          "phase_rotation_rebuilds_all_representations": True,
                          "low_snr_qam_missing_snr_weight": 1.0})
    (out / "config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    (out / "scaler_audit.json").write_text(json.dumps(scaler_audit, indent=2), encoding="utf-8")
    history = []; best_state = None; best_key = None; best_epoch = 0; stale = 0
    start_all = time.perf_counter()
    gpu_utilization_log = []
    timing_totals = {"batch_loading_seconds": 0.0, "file_load_sample_seconds": 0.0,
                     "preprocessing_sample_seconds": 0.0,
                     "gpu_compute_seconds": 0.0, "validation_seconds": 0.0, "epoch_seconds": 0.0}
    for epoch in range(1, config.training.max_epochs + 1):
        epoch_started = time.perf_counter()
        trainability = _set_transfer_trainability(model, config.transfer, epoch)
        model.train(); _set_transfer_module_modes(model, config.transfer, epoch)
        train_ds.set_epoch(epoch-1)
        running_loss, seen = 0.0, 0
        batch_loading_seconds, file_load_sample_seconds, preprocessing_sample_seconds = 0.0, 0.0, 0.0
        gpu_events = []
        optimizer.zero_grad(set_to_none=True)
        batches = len(train_loader); accumulation = max(1, config.training.gradient_accumulation_steps)
        iterator = iter(train_loader)
        for bi in range(batches):
            batch_wait_started = time.perf_counter()
            batch = next(iterator)
            batch_loading_seconds += time.perf_counter() - batch_wait_started
            file_load_sample_seconds += float(batch["timing_load_seconds"].sum())
            preprocessing_sample_seconds += float(batch["timing_preprocess_seconds"].sum())
            group_start = (bi // accumulation) * accumulation
            group_end = min(group_start + accumulation, batches)
            group_n = sum(min(config.training.batch_size, len(train) - j*config.training.batch_size)
                          for j in range(group_start, group_end))
            inputs = _model_inputs(batch, device); target = batch["target"].to(device, non_blocking=True)
            snr = batch["snr_db"].to(device, non_blocking=True)
            sample_weights = low_snr_qam_sample_weights(target, snr,
                config.training.low_snr_qam_weight, warn_missing=False)
            event_start = event_end = None
            if config.training.profile_timings and device.type == "cuda":
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(*inputs)
                per_loss = train_criterion(logits, target, sample_weights=sample_weights)
                loss = per_loss.mean()
                backward_loss = loss * (len(target) / max(group_n, 1))
            if not torch.isfinite(backward_loss) or not torch.isfinite(logits).all():
                raise FloatingPointError(f"NaN/Inf in {stage} epoch {epoch}; no silent precision change is made")
            amp_scaler.scale(backward_loss).backward()
            running_loss += float(per_loss.detach().sum()); seen += len(target)
            if bi + 1 == group_end:
                if config.training.gradient_clip_norm:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                amp_scaler.step(optimizer); amp_scaler.update(); optimizer.zero_grad(set_to_none=True)
            if event_end is not None:
                event_end.record()
                gpu_events.append((event_start, event_end))
        if gpu_events:
            torch.cuda.synchronize(device)
            gpu_compute_seconds = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000.0
        else:
            gpu_compute_seconds = 0.0
        validation_started = time.perf_counter()
        val_loss, val_metrics, y, pred, probs = evaluate(model, val_loader, val, device, criterion)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        gpu_snapshot = _gpu_snapshot(config.training.gpu_utilization_logging)
        if gpu_snapshot is not None:
            gpu_utilization_log.append({"epoch": epoch, **gpu_snapshot})
        timing_totals["batch_loading_seconds"] += batch_loading_seconds
        timing_totals["file_load_sample_seconds"] += file_load_sample_seconds
        timing_totals["preprocessing_sample_seconds"] += preprocessing_sample_seconds
        timing_totals["gpu_compute_seconds"] += gpu_compute_seconds
        timing_totals["validation_seconds"] += validation_seconds
        timing_totals["epoch_seconds"] += epoch_seconds
        row = {"epoch": epoch, "train_loss": running_loss/max(seen,1), "validation_loss": val_loss,
               "validation_accuracy": val_metrics["accuracy"], "validation_macro_f1": val_metrics["macro_f1"],
               "validation_macro_precision": val_metrics["macro_precision"],
               "validation_macro_recall": val_metrics["macro_recall"],
               "validation_64qam_recall": val_metrics["per_class"]["64QAM"]["recall"],
               "batch_loading_seconds": batch_loading_seconds,
               "file_load_sample_seconds": file_load_sample_seconds,
               "preprocessing_sample_seconds": preprocessing_sample_seconds,
               "gpu_compute_seconds": gpu_compute_seconds,
               "validation_seconds": validation_seconds,
               "epoch_seconds": epoch_seconds,
               "train_samples_per_second": seen / max(epoch_seconds - validation_seconds, 1e-9),
               "gpu_utilization_pct": gpu_snapshot["utilization_pct"] if gpu_snapshot else None,
               "gpu_memory_used_mb": gpu_snapshot["memory_used_mb"] if gpu_snapshot else None}
        history.append(row)
        selection = (row["validation_macro_f1"], row["validation_64qam_recall"], row["validation_accuracy"])
        if best_key is None or selection > best_key:
            best_key, best_state, best_epoch, stale = selection, copy.deepcopy(model.state_dict()), epoch, 0
            _save_checkpoint(out / "best.pt", model, scaler_state, psd_state, config_record,
                             config.training.seed, stage)
            np.savez_compressed(out / "best_validation_outputs.npz", y_true=y, y_pred=pred, probabilities=probs)
            (out / "best_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        else:
            stale += 1
        print(f"{stage} epoch {epoch}/{config.training.max_epochs}: train_loss={row['train_loss']:.4f} "
              f"val_macroF1={row['validation_macro_f1']:.4f} val_acc={row['validation_accuracy']:.4f} "
              f"epoch_s={epoch_seconds:.1f} load_s={batch_loading_seconds:.1f} "
              f"gpu_s={gpu_compute_seconds:.1f}", flush=True)
        if stale >= config.training.early_stopping_patience:
            break
    elapsed = time.perf_counter() - start_all
    _save_checkpoint(out / "final.pt", model, scaler_state, psd_state, config_record,
                     config.training.seed, stage)
    import csv
    with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    # Re-evaluate the selected best state only on VALIDATION.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model_state_dict"])
    best_loss, metrics, y, pred, probs = evaluate(model, val_loader, val, device, criterion)
    selection_names = tuple(config.training.model_selection_priority)
    selected_values = {
        "macro_f1": metrics["macro_f1"],
        "recall_64QAM": metrics["per_class"]["64QAM"]["recall"],
        "accuracy": metrics["accuracy"],
    }
    learned_weights = {
        name: float(value) for name, value in zip(
            model.fusion.NAMES, torch.softmax(model.fusion.logits.detach().float(), dim=0).cpu().tolist())
    }
    metrics_record = {**metrics, "validation_loss": best_loss, "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "model_selection_metric": selection_names[0] if selection_names else "macro_f1",
        "model_selection_priority": list(selection_names),
        "model_selection_values": {name: selected_values[name] for name in selection_names},
        "learned_fusion_weights": learned_weights,
        "best_checkpoint_path": str((out / "best.pt").resolve()),
        "final_checkpoint_path": str((out / "final.pt").resolve())}
    (out / "metrics.json").write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")
    _write_reports(out, metrics, y, pred, probs, val)
    hardware = {**device_info, "amp_enabled": amp, "physical_batch_size": config.training.batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "effective_batch_size": config.training.batch_size*config.training.gradient_accumulation_steps,
                "peak_preflight_vram_bytes": peak_preflight,
                "peak_training_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "peak_training_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                "training_seconds": elapsed, "best_epoch": best_epoch, "epochs_completed": len(history),
                "parameter_counts": count_parameters(model),
                "data_loader": loader_args,
                "timing_totals": timing_totals,
                "gpu_utilization_by_epoch": gpu_utilization_log,
                "transfer": config.to_dict().get("transfer", {}),
        "trainability_at_epoch_1": initial_trainability,
        "trainability_at_best_epoch": _set_transfer_trainability(model, config.transfer, best_epoch)}
    (out / "training_hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    log_lines = [f"Stage: {stage}", f"Seed: {config.training.seed}",
        f"Best epoch: {best_epoch}", "Model selection: " + " -> ".join(selection_names),
        "Epoch | train_loss | validation_loss | accuracy | macro_precision | macro_recall | macro_f1 | "
        "64QAM_recall | batch_wait_s | file_load_worker_s | prep_worker_s | gpu_compute_s | validation_s | epoch_s | train_samples_s"]
    log_lines.extend("{epoch} | {train_loss:.6f} | {validation_loss:.6f} | {validation_accuracy:.6f} | "
        "{validation_macro_precision:.6f} | {validation_macro_recall:.6f} | {validation_macro_f1:.6f} | "
        "{validation_64qam_recall:.6f} | {batch_loading_seconds:.3f} | {file_load_sample_seconds:.3f} | "
        "{preprocessing_sample_seconds:.3f} | "
        "{gpu_compute_seconds:.3f} | {validation_seconds:.3f} | {epoch_seconds:.3f} | "
        "{train_samples_per_second:.3f}".format(**row) for row in history)
    log_lines.append("Dataloader: " + json.dumps(loader_args, sort_keys=True))
    log_lines.append("GPU utilization snapshots: " + json.dumps(gpu_utilization_log, sort_keys=True))
    log_lines.extend([f"Final selected validation accuracy: {metrics['accuracy']:.6f}",
        f"Final selected validation macro-F1: {metrics['macro_f1']:.6f}",
        "Learned fusion weights: " + json.dumps(learned_weights, sort_keys=True),
        f"Best checkpoint: {(out / 'best.pt').resolve()}",
        f"Final checkpoint: {(out / 'final.pt').resolve()}",
        f"Training seconds: {elapsed:.3f}"])
    (out / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Artifacts saved under {out}; best epoch={best_epoch}; training_seconds={elapsed:.1f}")
    return out


def _save_checkpoint(path, model, feature_scaler, psd_scaler, config, seed, stage):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "research_qam_psk_fsk_multimodal_v1",
        "model_state_dict": model.state_dict(), "class_names": list(CLASS_NAMES),
        "feature_count": 22, "feature_names": list(FEATURE_NAMES),
        "feature_scaler": feature_scaler, "psd_scaler": psd_scaler,
        "input_length": 4096, "architecture_config": config.get("architecture", {}),
        "representation_config": {"iq": "joint IQ RMS", "polar_amplitude_scale": "abs(x)/sqrt(2), clipped to 6",
          "polar_phase": ["sin(phi)", "cos(phi)"], "polar_dphi": "angle(x[n]conj(x[n-1]))/pi",
          "psk": ["cos(2phi)","sin(2phi)","cos(4phi)","sin(4phi)"],
          "if": "angle(x[n]conj(x[n-1]))/(2*pi) = f_inst/Fs", "psd": "two-sided Hann Welch; nperseg=1024; overlap=512; nfft=2048; density; fftshift; log; train-standardized"},
        "focal_gamma": config["training"]["focal_gamma"],
        "low_snr_qam_weight": config["training"]["low_snr_qam_weight"],
        "normalization_config": {"iq": "joint I/Q RMS over valid samples; zero pad to 4096", "feature_scaler_fit": "TRAIN only", "psd_scaler_fit": "TRAIN only"},
        "training_seed": seed, "stage": stage, "repository_version": config.get("repository_version"),
        "experiment_config": config}, path)


def _write_reports(out, metrics, y, pred, probs, records):
    import csv
    (out / "classification_report.json").write_text(json.dumps(metrics["per_class"], indent=2), encoding="utf-8")
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["true/pred", *CLASS_NAMES])
        for name,row in zip(CLASS_NAMES,metrics["confusion_matrix"]): writer.writerow([name,*row])
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields=["index","sample_id","true_label","predicted_label","confidence","snr_db","sps","sample_rate_hz","fading",
                *[f"p_{c}" for c in CLASS_NAMES]]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,(t,p,pr) in enumerate(zip(y,pred,probs)):
            meta=records[i].get("metadata",{})
            writer.writerow({"index":i,"true_label":CLASS_NAMES[t],"predicted_label":CLASS_NAMES[p],
                             "sample_id":records[i].get("sample_id"),"confidence":max(pr),
                             "snr_db":meta.get("snr_db"),"sps":meta.get("sps",meta.get("samples_per_symbol")),
                             "sample_rate_hz":meta.get("sample_rate_hz"),"fading":meta.get("fading_type"),
                             **{f"p_{c}":pr[j] for j,c in enumerate(CLASS_NAMES)}})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="research_qam_psk_fsk_model/configs/multimodal_7class.yaml")
    parser.add_argument("--stage",choices=("real_from_scratch","synthetic_pretrain","real_finetune"))
    parser.add_argument("--output-dir")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--execute-training",action="store_true",help="Required explicit opt-in; without it no training occurs")
    args=parser.parse_args(argv)
    if not args.execute_training:
        print("TRAINING NOT STARTED. Review config/data paths; rerun with --execute-training only when ready.")
        return 0
    run_training(args.config,args.stage,output_dir=args.output_dir,init_checkpoint=args.init_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
