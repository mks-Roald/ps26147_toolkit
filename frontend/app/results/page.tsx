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
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
} from 'recharts';
import { ProcessResult, processFile, DecodeResult, CorrelateResult, decodeSignal, correlateSignal } from '@/services/api';
import Card from '@/components/base/Card';
import WaterfallPlot from '@/components/WaterfallPlot';

export default function Results() {
  const router = useRouter();
  const [data, setData] = useState<ProcessResult | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'waveform' | 'psd' | 'constellation'>('waveform');
  const [editingSampleRate, setEditingSampleRate] = useState<boolean>(false);

  // Fetch full analysis including decode and correlate results
  const fetchFullAnalysis = async (file: File, sampleRate: number): Promise<ProcessResult> => {
    try {
      // Step 1: Process file to get base results
      const processRes = await processFile(file, sampleRate);

      // Step 2: Fetch decode results (using default fecScheme "none")
      const decodeRes = await decodeSignal(file, "none", sampleRate);

      // Step 3: Fetch correlate results
      const correlateRes = await correlateSignal(file, { sampleRate });

      // Merge all results
      return {
        ...processRes,
        demodulated_bits: decodeRes.demodulated_bits,
        demodulated_bits_count: decodeRes.demodulated_bits_count,
        deinterleaved_bits: decodeRes.deinterleaved_bits,
        deinterleaved_bits_count: decodeRes.deinterleaved_bits_count,
        decoded_bits: decodeRes.decoded_bits,
        correlate_result: correlateRes,
      };
    } catch (err) {
      throw err;
    }
  };

  // Helper to decode base64 string back to File object
  const base64ToFile = (base64String: string, fileName: string): File | null => {
    try {
      // Remove data URL prefix if present
      const base64Data = base64String.split(',')[1] || base64String;
      const binaryString = window.atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      // Determine MIME type from file extension
      const ext = fileName.split('.').pop()?.toLowerCase() || '';
      const mimeTypes: Record<string, string> = {
        wav: 'audio/wav',
        iq: 'application/octet-stream',
        bin: 'application/octet-stream',
        raw: 'application/octet-stream',
        'sigmf-data': 'application/octet-stream'
      };
      const mimeType = mimeTypes[ext] || 'application/octet-stream';
      return new File([bytes], fileName, { type: mimeType });
    } catch (error) {
      console.error('Failed to decode base64 to File:', error);
      return null;
    }
  };

  // Handle sample rate changes with file re-processing
  const handleSampleRateChange = async () => {
    if (!data) return;

    const newSampleRate = data.sample_rate;
    if (!newSampleRate || newSampleRate <= 0) return;

    setLoading(true);

    try {
      // Retrieve the original file from sessionStorage
      const base64String = sessionStorage.getItem('lastFileBase64');
      const fileName = sessionStorage.getItem('lastFileName') || 'signal.file';

      if (!base64String) {
        throw new Error('Original file data not found in storage');
      }

      // Decode base64 back to File object
      const originalFile = base64ToFile(base64String, fileName);
      if (!originalFile) {
        throw new Error('Failed to reconstruct file from stored data');
      }

      // Re-process with new sample rate (including decode and correlate)
      const res = await fetchFullAnalysis(originalFile, newSampleRate);

      // Update data and session storage
      setData(res);
      sessionStorage.setItem('lastResult', JSON.stringify(res));
      sessionStorage.setItem('lastFileName', fileName);
      sessionStorage.setItem('lastFileSize', String(originalFile.size));
      sessionStorage.setItem('lastSampleRate', String(newSampleRate));

    } catch (err: any) {
      setError(err.message || 'Failed to re-process signal with new sample rate');
    } finally {
      setLoading(false);
      setEditingSampleRate(false);
    }
  };

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

  // Handle sample rate changes from session storage
  useEffect(() => {
    const storedSampleRate = sessionStorage.getItem('lastSampleRate');
    if (storedSampleRate) {
      const parsed = parseInt(storedSampleRate, 10);
      if (!isNaN(parsed) && parsed > 0) {
        setData(prev => prev ? { ...prev, sample_rate: parsed } : null);
        // Clear the stored value to avoid re-applying on every render
        sessionStorage.removeItem('lastSampleRate');
      }
    }
  }, []);

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
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-pill bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-all duration-200 dark:bg-cyan-400/20 dark:text-cyan-500 dark:hover:bg-cyan-400/30"
            >
              <span>Analyse another Signal</span>
            </Link>
            {!editingSampleRate && data && (
              <>
                <button
                  onClick={() => {
                    const json = JSON.stringify(data, null, 2);
                    const blob = new Blob([json], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'signal-analysis.json';
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
                >
                  <span>💾 Download JSON</span>
                </button>
                <button
                  onClick={() => {
                    if (!data) return;
                    const rows = [
                      ['Parameter', 'Value'],
                      ['Modulation', data.modulation ?? ''],
                      ['Confidence (%)', ((data.confidence ?? 0) * 100).toFixed(1)],
                      ['SNR (dB)', data.snr_db !== null && data.snr_db !== undefined ? data.snr_db.toFixed(1) : ''],
                      ['Baud Rate', data.baud_rate !== null && data.baud_rate !== undefined && data.baud_rate > 0 ? (data.baud_rate >= 1000 ? `${(data.baud_rate / 1000).toFixed(2)} kBd` : `${data.baud_rate.toFixed(1)} Bd`) : ''],
                      ['Center Frequency', data.center_frequency_hz !== null && data.center_frequency_hz !== undefined ? (Math.abs(data.center_frequency_hz) >= 1e6 ? `${(data.center_frequency_hz / 1e6).toFixed(2)} MHz` : `${(data.center_frequency_hz / 1e3).toFixed(1)} kHz`) : ''],
                      ['Bandwidth', data.bandwidth_hz !== null && data.bandwidth_hz !== undefined && data.bandwidth_hz > 0 ? (data.bandwidth_hz >= 1e6 ? `${(data.bandwidth_hz / 1e6).toFixed(2)} MHz` : `${(data.bandwidth_hz / 1e3).toFixed(1)} kHz`) : ''],
                      ['Sample Rate (Hz)', data.sample_rate.toLocaleString()],
                      ['Duration (s)', data.duration_sec.toFixed(4)],
                      ['Total Samples', data.num_samples.toLocaleString()],
                    ];
                    const csvContent = rows.map(e => e.join(',')).join('\n');
                    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'signal-summary.csv';
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
                >
                  <span>📥 Download CSV</span>
                </button>
              </>
            )}
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
            <h3 className="font-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
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
            <h3 className="font-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
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
            <h3 className="font-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
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
            <h3 className="font-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
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
            <h3 className="font-geist font-weight-600 text-lg text-ink-faint mb-0 ml-3 uppercase">
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
            <div className="h-96 w-full bg-canvas-elevated">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="x" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Sample Index', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[-1.1, 1.1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace', color: '#e2e8f0' }}
                    itemStyle={{ color: '#94a3b8' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
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
            <div className="h-96 w-full bg-canvas-elevated">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={psdData} style={{ backgroundColor: 'transparent' }}>
                  <defs>
                    <linearGradient id="psdGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f355da" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#f355da" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="freq" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#64748b' }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'dB/Hz', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#64748b' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace', color: '#e2e8f0' }}
                    itemStyle={{ color: '#94a3b8' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
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
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }} style={{ backgroundColor: 'transparent' }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="i" name="In-Phase (I)" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <YAxis type="number" dataKey="q" name="Quadrature (Q)" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3', stroke: '#334155' }}
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace', color: '#e2e8f0' }}
                      itemStyle={{ color: '#94a3b8' }}
                      labelStyle={{ color: '#e2e8f0' }}
                    />
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

      {/* Spectrogram & 3D Waterfall Display Section */}
      <WaterfallPlot data={data.waterfall_data} />

      {/* NEW SECTIONS START */}
      {data?.demodulated_bits && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
            Demodulated Bitstream
          </h2>
          <div className="space-y-4">
            <div className="h-96 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-xs text-ink">
                {data.demodulated_bits
                  .slice(0, 100)
                  .map(bit => bit.toString())
                  .join('')}
                {data.demodulated_bits.length > 100 ? '...' : ''}
              </pre>
            </div>
            <button
              onClick={() => {
                const bitsString = data.demodulated_bits?.join('') || '';
                const blob = new Blob([bitsString], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'demodulated_bits.txt';
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
            >
              <span>💾 Download Full Data</span>
            </button>
          </div>
        </section>
      )}
      {data?.deinterleaved_bits && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
            Deinterleaved Output
          </h2>
          <div className="space-y-4">
            <div className="h-96 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-xs text-ink">
                {data.deinterleaved_bits
                  .slice(0, 100)
                  .map(bit => bit.toString())
                  .join('')}
                {data.deinterleaved_bits.length > 100 ? '...' : ''}
              </pre>
            </div>
            <button
              onClick={() => {
                const bitsString = data.deinterleaved_bits?.join('') || '';
                const blob = new Blob([bitsString], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'deinterleaved_bits.txt';
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
            >
              <span>💾 Download Full Data</span>
            </button>
          </div>
        </section>
      )}
      {data?.decoded_bits && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
            FEC Decoded Stream
          </h2>
          <div className="space-y-4">
            <div className="h-96 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-xs text-ink">
                {data.decoded_bits
                  .slice(0, 100)
                  .map(bit => bit.toString())
                  .join('')}
                {data.decoded_bits.length > 100 ? '...' : ''}
              </pre>
            </div>
            <button
              onClick={() => {
                const bitsString = data.decoded_bits?.join('') || '';
                const blob = new Blob([bitsString], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'decoded_bits.txt';
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
            >
              <span>💾 Download Full Data</span>
            </button>
          </div>
        </section>
      )}
      {data?.correlate_result && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
            Correlation/Sync Results
          </h2>
          <div className="space-y-4">
            <div className="h-96 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-xs text-ink">
                {data.correlate_result.bits ? (
                  data.correlate_result.bits
                    .slice(0, 100)
                    .map(bit => bit.toString())
                    .join('') + (data.correlate_result.bits.length > 100 ? '...' : '')
                ) : (
                  JSON.stringify(data.correlate_result, null, 2)
                )}
              </pre>
            </div>
            <button
              onClick={() => {
                let content = '';
                if (data.correlate_result.bits) {
                  content = data.correlate_result.bits.join('');
                } else {
                  content = JSON.stringify(data.correlate_result, null, 2);
                }
                const blob = new Blob([content], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'correlate_result.txt';
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
            >
              <span>💾 Download Full Data</span>
            </button>
          </div>
        </section>
      )}
      {/* NEW SECTIONS END */}

      {/* Signal Metadata Details Table */}
      <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
        <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
          Extracted Signal Parameters
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-geist-mono">
          <div className="p-4 bg-canvas-elevated/90 border border-hairline rounded-md">
            <span className="block text-xs font-geist-mono font-weight-500 text-ink-faint">Sampling Rate</span>
            {editingSampleRate ? (
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  value={data.sample_rate}
                  onChange={(e) => setData({ ...data, sample_rate: Number(e.target.value) })}
                  onKeyDown={async (e) => {
                    if (e.key === "Enter") {
                      await handleSampleRateChange();
                    }
                    if (e.key === "Escape") {
                      setEditingSampleRate(false); // cancel edit
                    }
                  }}
                  onBlur={async (e) => {
                    await handleSampleRateChange();
                  }}
                  min="1"
                  step="1"
                  className="w-[150px] bg-canvas-elevated border border-hairline rounded-md px-3 py-1 text-sm font-geist-mono text-ink focus:outline-none focus:ring-2 focus-ring-blue focus:border-blue transition-colors duration-200"
                />
                <span className="text-xs font-geist-mono text-ink-faint">Hz</span>
              </div>
            ) : (
              <span className="block font-geist-mono font-weight-600 text-ink">{data.sample_rate.toLocaleString()} Hz</span>
            )}
            {!editingSampleRate && (
              <button
                onClick={() => setEditingSampleRate(true)}
                className="text-xs font-geist-mono text-cyan-500 hover:text-cyan-400 mt-1"
              >
                Edit
              </button>
            )}
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