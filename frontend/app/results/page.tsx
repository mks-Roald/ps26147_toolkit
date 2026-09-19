'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
} from 'recharts';
import { ProcessResult } from '@/services/api';
import Card from '@/components/base/Card';

export default function Results() {
  const router = useRouter();
  const [data, setData] = useState<ProcessResult | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'waveform' | 'psd' | 'constellation'>('waveform');

  useEffect(() => {
    const stored = sessionStorage.getItem('lastResult');
    const storedName = sessionStorage.getItem('lastFileName') || 'Signal File';
    if (stored) {
      try {
        setData(JSON.parse(stored));
        setFileName(storedName);
      } catch {
        // Failed to parse
      }
      setLoading(false);
    } else {
      router.push('/');
    }
  }, [router]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <div className="w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="font-geist-mono font-weight-500 text-ink-muted text-base">Loading analysis results…</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-6">
        <p className="text-ink-muted text-base">No analysis results found.</p>
        <Link href="/" className="inline-flex items-center space-x-2 px-6 py-3 rounded-pill bg-ink text-on-primary font-geist font-weight-500 hover:bg-ink/90 transition-all duration-200">
          <span>⬆️ Upload Signal File</span>
        </Link>
      </div>
    );
  }

  const waveformData = data.waveform_data ? data.waveform_data.map((v, i) => ({ x: i, y: v })) : [];
  const psdData = data.psd_data ? data.psd_data.map((p) => ({ freq: Math.round(p.freq), psd: Number(p.psd.toFixed(2)) })) : [];
  const constellationData = data.constellation_data ? data.constellation_data.map((pt) => ({ i: Number(pt.i.toFixed(4)), q: Number(pt.q.toFixed(4)) })) : [];

  return (
    <div className="space-y-16">
      {/* Top Header & Breadcrumb */}
      <header className="border-b border-hairline pb-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div>
            <div className="flex items-center space-x-3 mb-4">
              <Link href="/" className="flex items-center space-x-2 px-4 py-2 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-xs font-geist-mono font-weight-500 hover:bg-cyan-950/70 transition-colors duration-200">
                ← Back to Upload
              </Link>
              <span>/</span>
              <span className="text-xs font-geist-mono font-weight-500 text-ink-faint truncate max-w-xs">{fileName}</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-geist font-weight-600 tracking-tighter text-ink">
              Signal Analysis Report
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            <Link
              href="/"
              className="flex items-center space-x-3 px-6 py-3 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary"
            >
              <span>+ Analyze Another Signal</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Primary Key Metric Cards */}
      <section className="grid gap-6">
        {/* Modulation */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-1 p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-cyan-950/50 text-cyan-400 rounded-full text-lg">
              1
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
              Modulation
            </h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-cyan-400">
            {data.modulation ?? '—'}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            Confidence: <span className="text-emerald-400 font-geist-mono">{((data.confidence ?? 0) * 100).toFixed(1)}%</span>
          </p>
        </Card>

        {/* SNR */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-1 p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-blue-950/50 text-blue-400 rounded-full text-lg">
              2
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
              SNR
            </h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-blue-400">
            {data.snr_db !== null && data.snr_db !== undefined ? `${data.snr_db.toFixed(1)} dB` : '—'}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            Signal-to-Noise Ratio
          </p>
        </Card>

        {/* Baud Rate */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-1 p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-fuchsia-950/50 text-fuchsia-400 rounded-full text-lg">
              3
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
              Baud Rate
            </h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-fuchsia-400">
            {data.baud_rate !== null && data.baud_rate !== undefined && data.baud_rate > 0 ? (
              data.baud_rate >= 1000 ? `${(data.baud_rate / 1000).toFixed(2)} kBd` : `${data.baud_rate.toFixed(1)} Bd`
            ) : (
              '—'
            )}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            Symbol Rate
          </p>
        </Card>

        {/* Center Frequency */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-1 p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-emerald-950/50 text-emerald-400 rounded-full text-lg">
              4
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
              Center Frequency
            </h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-emerald-400">
            {data.center_frequency_hz !== null && data.center_frequency_hz !== undefined ? (
              Math.abs(data.center_frequency_hz) >= 1e6
                ? `${(data.center_frequency_hz / 1e6).toFixed(2)} MHz`
                : `${(data.center_frequency_hz / 1e3).toFixed(1)} kHz`
            ) : (
              '—'
            )}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            RF Carrier
          </p>
        </Card>

        {/* Bandwidth */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-1 p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-violet-950/50 text-violet-400 rounded-full text-lg">
              5
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
              Bandwidth
            </h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-violet-400">
            {data.bandwidth_hz !== null && data.bandwidth_hz !== undefined && data.bandwidth_hz > 0 ? (
              data.bandwidth_hz >= 1e6
                ? `${(data.bandwidth_hz / 1e6).toFixed(2)} MHz`
                : `${(data.bandwidth_hz / 1e3).toFixed(1)} kHz`
            ) : (
              '—'
            )}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            Occupied Bandwidth
          </p>
        </Card>
      </section>

      {/* Visualizations Panel */}
      <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
        {/* Tab Headers */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-hairline pb-6 mb-6">
          <div className="flex space-x-3">
            <button
              onClick={() => setActiveTab('waveform')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'waveform'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              🌊 Time-Domain Waveform
            </button>
            <button
              onClick={() => setActiveTab('psd')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'psd'
                  ? 'bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              📊 Power Spectral Density (PSD)
            </button>
            <button
              onClick={() => setActiveTab('constellation')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'constellation'
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              🌌 I/Q Constellation Diagram
            </button>
          </div>

          <div className="text-xs font-geist-mono text-ink-faint">
            {data.num_samples.toLocaleString()} Samples • {data.duration_sec.toFixed(3)}s @ {(data.sample_rate / 1000).toFixed(0)} kHz
          </div>
        </div>

        {/* Tab 1: Waveform */}
        {activeTab === 'waveform' && (
          <div className="space-y-4">
            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="x" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Sample Index', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[-1.1, 1.1]} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace' }} />
                  <Line type="monotone" dataKey="y" name="Amplitude" stroke="#00f2fe" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-center text-xs font-geist-mono text-ink-faint">Real baseband amplitude representation (first {waveformData.length} downsampled points)</p>
          </div>
        )}

        {/* Tab 2: PSD Spectrum */}
        {activeTab === 'psd' && (
          <div className="space-y-4">
            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={psdData}>
                  <defs>
                    <linearGradient id="psdGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f355da" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#f355da" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="freq" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'dB/Hz', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace' }} />
                  <Area type="monotone" dataKey="psd" name="PSD (dB)" stroke="#f355da" strokeWidth={1.5} fillOpacity={1} fill="url(#psdGrad)" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="text-center text-xs font-geist-mono text-ink-faint">Welch Power Spectral Density over estimated frequency domain</p>
          </div>
        )}

        {/* Tab 3: Constellation */}
        {activeTab === 'constellation' && (
          <div className="space-y-4">
            <div className="h-96 w-full flex items-center justify-center">
              {constellationData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="i" name="In-Phase (I)" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <YAxis type="number" dataKey="q" name="Quadrature (Q)" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace' }} />
                    <Scatter name="I/Q Symbols" data={constellationData} fill="#38bdf8" />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-ink-faint text-xs font-geist-mono">Constellation points not available for this real signal.</p>
              )}
            </div>
            <p className="text-center text-xs font-geist-mono text-ink-faint">Normalized complex baseband constellation scatter diagram</p>
          </div>
        )}
      </section>

      {/* Signal Metadata Details Table */}
      <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
        <h2 className="text-geist font-weight-600 text-lg text-ink mb-6">
          Extracted Signal Parameters
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-geist-mono">
          <div className="p-4 bg-canvas-elevated/90 border border-hairline rounded-md">
            <span className="block text-xs font-geist-mono font-weight-500 text-ink-faint">Sampling Rate</span>
            <span className="block font-geist-mono font-weight-600 text-ink">{data.sample_rate.toLocaleString()} Hz</span>
          </div>
          <div className="p-4 bg-canvas-elevated/90 border border-hairline rounded-md">
            <span className="block text-xs font-geist-mono font-weight-500 text-ink-faint">Total Duration</span>
            <span className="block font-geist-mono font-weight-600 text-ink">{data.duration_sec.toFixed(4)} s</span>
          </div>
          <div className="p-4 bg-canvas-elevated/90 border border-hairline rounded-md">
            <span className="block text-xs font-geist-mono font-weight-500 text-ink-faint">Total Samples</span>
            <span className="block font-geist-mono font-weight-600 text-ink">{data.num_samples.toLocaleString()}</span>
          </div>
          <div className="p-4 bg-canvas-elevated/90 border border-hairline rounded-md">
            <span className="block text-xs font-geist-mono font-weight-500 text-ink-faint">3dB Bandwidth</span>
            <span className="block font-geist-mono font-weight-600 text-ink">
              {data.bandwidth_3db_hz !== undefined ? `${(data.bandwidth_3db_hz / 1000).toFixed(1)} kHz` : 'N/A'}
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}