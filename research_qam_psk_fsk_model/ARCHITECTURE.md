# Architecture and preprocessing specification

The experiment implements six independent encoders. IQ preserves the current branch's 2→32→64→128 channel widths and 7/5/3 kernels, one max-pool, and adaptive global-average pooling. Polar is 4→32→64→96; PSK nonlinear phase is 4→32→64→96; IF is 1→32→64→96; PSD is 1→16→32→64. Auxiliary encoders use short temporal convolutions, normalization/activation, pooling, and adaptive global-average pooling. Signal encoders project to 128-D (IQ) or 64-D (auxiliary views). Feature encoder is 22→32→32→128. Polar/PSK/IF/PSD get linear projections to 128-D. Every branch is LayerNormed before scalar modality fusion.

For branch vectors `z_m`, learnable scalar logits `a_m` define `w_m=softmax(a)_m`; `fused=sum_m(w_m * LayerNorm(z_m))`. A compact 128→128→64→7 classifier returns logits. Initial modality weights are uniform because logits initialize to zero.

The instantiated default model has **228,701 trainable parameters**. This is a static count from the configured implementation; it is not a trained checkpoint.

All branches use one 4096-sample crop. IQ joint RMS is `sqrt(mean(I²+Q²)/2)` over valid samples, matching joint I/Q channel RMS. IF/phase use signed adjacent-sample differential phase. The phase channel padding repeats the first valid differential; no crop shift occurs. PSD follows fixed complex two-sided Hann Welch parameters and TRAIN-only per-bin log-power standardization. All feature names and mappings are centralized in `labels.py`.
