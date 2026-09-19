import Link from 'next/link';
import Layout from '@/components/Layout';
import Card from '@/components/base/Card';

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
    <Layout>
      <div className="max-w-4xl mx-auto space-y-16">
        {/* Header */}
        <section className="space-y-4 text-center">
          <h1 className="text-3xl sm:text-4xl font-geist font-weight-600 tracking-tighter text-ink">
            About SIH PS26147 Toolkit
          </h1>
          <p className="text-ink-muted text-base max-w-2xl mx-auto leading-relaxed">
            Autonomous RF Signal Intelligence, Modulation Recognition, and Multi-Stage Waveform Decoding Suite.
          </p>
        </section>

        {/* Phases Grid */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {phases.map((phase, idx) => (
            <Card key={idx} className="p-6 hover:floating-shadow transition-all duration-300">
              <div className="flex items-center justify-start mb-4">
                <div className="w-10 h-10 flex items-center justify-center bg-cyan-950/50 text-cyan-400 rounded-full text-lg">
                  {idx + 1}
                </div>
                <h3 className="font-geist font-weight-600 text-lg text-ink mb-0 ml-3">{phase.title}</h3>
              </div>
              <p className="text-ink-faint text-sm font-geist-mono leading-relaxed">
                {phase.description}
              </p>
            </Card>
          ))}
        </section>

        {/* API Contract */}
        <section className="bg-canvas-elevated border-hairline rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-4">
            Backend REST API Contract
          </h2>
          <div className="space-y-3 text-xs font-geist-mono text-ink-faint">
            <p>
              <span className="font-geist-mono font-weight-500 text-ink">POST /process/file</span> —
              Ingests file, extracts physical parameters (SNR, Baud, BW, fc), predicts modulation, and returns time-series waveform & PSD.
            </p>
            <p>
              <span className="font-geist-mono font-weight-500 text-ink">POST /classify/</span> —
              Fast modulation classifier with confidence and cumulants.
            </p>
            <p>
              <span className="font-geist-mono font-weight-500 text-ink">POST /decode/</span> —
              Demodulates signal to raw symbols, applies Gardner/Costas sync, and executes optional FEC decoding.
            </p>
            <p>
              <span className="font-geist-mono font-weight-500 text-ink">GET /health</span> —
              Service liveness and health probe.
            </p>
          </div>
        </section>

        {/* CTA */}
        <div className="text-center pt-8">
          <Link href="/" className="inline-flex items-center space-x-2 px-6 py-3 rounded-pill bg-ink text-on-primary font-geist font-weight-500 hover:bg-ink/90 transition-all duration-200">
            <span>⚡ Start Analyzing Signals</span>
          </Link>
        </div>
      </div>
    </Layout>
  );
}