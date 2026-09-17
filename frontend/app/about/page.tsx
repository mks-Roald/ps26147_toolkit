import Link from 'next/link';

export default function About() {
  const phases = [
    {
      title: 'Phase 1: Ingestion & Preprocessing',
      description: 'Auto-detection of IQ dtypes (int8, uint8, int16, float32, complex64) and WAV audio signals with DC removal and spectral denoising.',
    },
    {
      title: 'Phase 2: Parametric Estimation',
      description: 'Robust center-frequency centroid, multi-contour occupied bandwidth (3dB, 95% OBW, 99% OBW), M2M4 split-moment SNR, and cyclic transition Baud rate.',
    },
    {
      title: 'Phase 3: Modulation Classification',
      description: 'Hybrid classification engine combining higher-order cumulants (C20, C21, C40, C41, C42) with Random Forest models for 12+ modulation schemes.',
    },
    {
      title: 'Phase 4: Demodulation & Timing Recovery',
      description: 'Gardner Timing Error Detector (TED) fractional interpolation, Costas PLL carrier recovery, and soft Log-Likelihood Ratio (LLR) calculation.',
    },
    {
      title: 'Phase 5: Forward Error Correction (FEC)',
      description: 'Hardware-standard Viterbi decoding (K=7, Rate 1/2), Reed-Solomon (255,223), Concatenated NASA/ESA code, and Log-Domain Min-Sum LDPC.',
    },
    {
      title: 'Phase 6: Frame Synchronization & Correlation',
      description: 'Preamble discovery, Barker/sync-word cross-correlation, bitstream alignment, and packet payload reconstruction.',
    },
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-10">
      <div className="border-b border-slate-800 pb-6">
        <h1 className="text-3xl font-black text-white tracking-tight mb-2">About SIH PS26147 Toolkit</h1>
        <p className="text-slate-400 text-sm">
          Autonomous RF Signal Intelligence, Modulation Recognition, and Multi-Stage Waveform Decoding Suite.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {phases.map((p, idx) => (
          <div key={idx} className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-cyan-500/30 transition-all">
            <h3 className="font-bold text-cyan-400 text-sm mb-2">{p.title}</h3>
            <p className="text-xs text-slate-300 leading-relaxed">{p.description}</p>
          </div>
        ))}
      </div>

      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-base font-bold text-white font-mono">Backend REST API Contract</h2>
        <div className="text-xs font-mono text-slate-300 space-y-2">
          <p><span className="text-cyan-400 font-bold">POST /process/file</span> — Ingests file, extracts physical parameters (SNR, Baud, BW, fc), predicts modulation, and returns time-series waveform & PSD.</p>
          <p><span className="text-fuchsia-400 font-bold">POST /classify/</span> — Fast modulation classifier with confidence and cumulants.</p>
          <p><span className="text-blue-400 font-bold">POST /decode/</span> — Demodulates signal to raw symbols, applies Gardner/Costas sync, and executes optional FEC decoding.</p>
          <p><span className="text-emerald-400 font-bold">GET /health</span> — Service liveness and health probe.</p>
        </div>
      </div>

      <div className="text-center pt-4">
        <Link href="/" className="inline-block px-5 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium text-xs hover:opacity-90 transition-all">
          ← Start Analyzing Signals
        </Link>
      </div>
    </div>
  );
}
