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
        <div className="w-10 h-10 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="font-mono text-sm text-slate-400">Loading analysis results…</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-20 space-y-4">
        <p className="text-slate-400">No analysis results found.</p>
        <Link href="/" className="inline-block px-4 py-2 bg-cyan-600 rounded-lg text-sm text-white font-medium hover:bg-cyan-500">
          Upload Signal File
        </Link>
      </div>
    );
  }

  const waveformData = data.waveform_data ? data.waveform_data.map((v, i) => ({ x: i, y: v })) : [];
  const psdData = data.psd_data ? data.psd_data.map((p) => ({ freq: Math.round(p.freq), psd: Number(p.psd.toFixed(2)) })) : [];
  const constellationData = data.constellation_data ? data.constellation_data.map((pt) => ({ i: Number(pt.i.toFixed(4)), q: Number(pt.q.toFixed(4)) })) : [];

  return (
    <div className="space-y-8">
      {/* Top Header & Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2 text-xs font-mono text-slate-400 mb-1">
            <Link href="/" className="hover:text-cyan-400 transition-colors">← Back to Upload</Link>
            <span>/</span>
            <span className="text-slate-200 truncate max-w-xs">{fileName}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Signal Analysis Report</h1>
        </div>

        <div className="flex items-center space-x-3">
          <Link
            href="/"
            className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition-all hover:border-cyan-500/40"
          >
            + Analyze Another Signal
          </Link>
        </div>
      </div>

      {/* Primary Key Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Modulation */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden group hover:border-cyan-500/40 transition-all">
          <div className="absolute top-0 left-0 h-1 w-full bg-cyan-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Modulation</p>
          <p className="text-2xl sm:text-3xl font-black text-cyan-400">{data.modulation}</p>
          <p className="text-xs font-mono text-slate-400 mt-2">
            Confidence: <span className="text-emerald-400 font-semibold">{((data.confidence ?? 1) * 100).toFixed(1)}%</span>
          </p>
        </div>

        {/* Baud Rate */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden group hover:border-fuchsia-500/40 transition-all">
          <div className="absolute top-0 left-0 h-1 w-full bg-fuchsia-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Baud Rate</p>
          <p className="text-2xl sm:text-3xl font-black text-fuchsia-400">
            {data.baud_rate !== undefined && data.baud_rate > 0 ? (
              data.baud_rate >= 1000 ? `${(data.baud_rate / 1000).toFixed(2)} kBd` : `${data.baud_rate.toFixed(1)} Bd`
            ) : (
              'N/A'
            )}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-2">Cyclic Transition Est.</p>
        </div>

        {/* SNR */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden group hover:border-blue-500/40 transition-all">
          <div className="absolute top-0 left-0 h-1 w-full bg-blue-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Estimated SNR</p>
          <p className="text-2xl sm:text-3xl font-black text-blue-400">
            {data.snr !== undefined ? `${data.snr.toFixed(1)} dB` : 'N/A'}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-2">M2M4 In-Band Split</p>
        </div>

        {/* Bandwidth */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden group hover:border-emerald-500/40 transition-all">
          <div className="absolute top-0 left-0 h-1 w-full bg-emerald-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Occupied Bandwidth</p>
          <p className="text-2xl sm:text-3xl font-black text-emerald-400">
            {data.bandwidth_hz !== undefined && data.bandwidth_hz > 0 ? (
              data.bandwidth_hz >= 1e6
                ? `${(data.bandwidth_hz / 1e6).toFixed(2)} MHz`
                : `${(data.bandwidth_hz / 1e3).toFixed(1)} kHz`
            ) : (
              'N/A'
            )}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-2">
            Center: {data.center_frequency_hz !== undefined ? `${(data.center_frequency_hz / 1000).toFixed(1)} kHz` : '0 Hz'}
          </p>
        </div>
      </div>

      {/* Visualizations Panel */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 backdrop-blur-xl shadow-xl">
        {/* Tab Headers */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4 mb-6">
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveTab('waveform')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                activeTab === 'waveform'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              🌊 Time-Domain Waveform
            </button>
            <button
              onClick={() => setActiveTab('psd')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                activeTab === 'psd'
                  ? 'bg-fuchsia-500/20 text-fuchsia-300 border border-fuchsia-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              📊 Power Spectral Density (PSD)
            </button>
            <button
              onClick={() => setActiveTab('constellation')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                activeTab === 'constellation'
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              🌌 I/Q Constellation Diagram
            </button>
          </div>

          <div className="text-xs font-mono text-slate-400">
            {data.num_samples.toLocaleString()} Samples • {data.duration_sec.toFixed(3)}s @ {(data.sample_rate / 1000).toFixed(0)} kHz
          </div>
        </div>

        {/* Tab 1: Waveform */}
        {activeTab === 'waveform' && (
          <div className="space-y-2">
            <div className="h-72 w-full">
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
            <p className="text-center text-xs font-mono text-slate-400">Real baseband amplitude representation (first {waveformData.length} downsampled points)</p>
          </div>
        )}

        {/* Tab 2: PSD Spectrum */}
        {activeTab === 'psd' && (
          <div className="space-y-2">
            <div className="h-72 w-full">
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
            <p className="text-center text-xs font-mono text-slate-400">Welch Power Spectral Density over estimated frequency domain</p>
          </div>
        )}

        {/* Tab 3: Constellation */}
        {activeTab === 'constellation' && (
          <div className="space-y-2">
            <div className="h-72 w-full flex items-center justify-center">
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
                <p className="text-slate-400 text-xs font-mono">Constellation points not available for this real signal.</p>
              )}
            </div>
            <p className="text-center text-xs font-mono text-slate-400">Normalized complex baseband constellation scatter diagram</p>
          </div>
        )}
      </div>

      {/* Signal Metadata Details Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
        <h3 className="font-semibold text-sm text-slate-300 font-mono mb-3">Extracted Signal Parameters</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-3 bg-slate-950/60 rounded border border-slate-800">
            <span className="text-slate-400 block">Sampling Rate</span>
            <span className="text-slate-200 font-bold">{data.sample_rate.toLocaleString()} Hz</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded border border-slate-800">
            <span className="text-slate-400 block">Total Duration</span>
            <span className="text-slate-200 font-bold">{data.duration_sec.toFixed(4)} s</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded border border-slate-800">
            <span className="text-slate-400 block">Total Samples</span>
            <span className="text-slate-200 font-bold">{data.num_samples.toLocaleString()}</span>
          </div>
          <div className="p-3 bg-slate-950/60 rounded border border-slate-800">
            <span className="text-slate-400 block">3dB Bandwidth</span>
            <span className="text-slate-200 font-bold">
              {data.bandwidth_3db_hz !== undefined ? `${(data.bandwidth_3db_hz / 1000).toFixed(1)} kHz` : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}