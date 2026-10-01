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
import { ProcessResult, createAnalysisSession } from '@/services/api';

interface ExtendedProcessResult extends ProcessResult {
  fec_scheme: string;
}
import Card from '@/components/base/Card';
import WaterfallPlot from '@/components/WaterfallPlot';
import { clearSignalSession, loadSignalFile, loadSignalResult, saveSignalResult, readSignalSession, writeSignalSession } from '@/services/signalStorage';

export default function Results() {
  const router = useRouter();
  const [data, setData] = useState<ProcessResult | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'waveform' | 'psd' | 'constellation' | 'spectrogram' | 'waterfall'>('waveform');
  const [editingSampleRate, setEditingSampleRate] = useState<boolean>(false);
  const [fecScheme, setFecScheme] = useState<string>('none');
  const [editingFecScheme, setEditingFecScheme] = useState<boolean>(false);

  const leaveAnalysis = async () => {
    const session = readSignalSession();
    if (session) await clearSignalSession(session.signalSessionId);
    sessionStorage.removeItem('signalSession');
  };

  const fetchFullAnalysis = async (file: File, sampleRate: number, fecScheme: string = "none"): Promise<ProcessResult> => {
    return createAnalysisSession(file, sampleRate, { fecScheme, syncWord: data?.sync_metadata?.word || 'Barker-13', autoDetectSync: false, autoDeinterleave: fecScheme.toLowerCase() === 'viterbi' });
  };

  // Handle sample rate changes with file re-processing
  const handleSampleRateChange = async () => {
    if (!data) return;

    const newSampleRate = data.sample_rate;
    if (!newSampleRate || newSampleRate <= 0) return;

    setLoading(true);

    try {
      const session = readSignalSession();
      if (!session) throw new Error('Signal session information is unavailable.');
      const originalFile = await loadSignalFile(session.signalSessionId, session.filename);
      if (!originalFile) {
        throw new Error('Original signal file is unavailable in IndexedDB.');
      }

      // Re-process with new sample rate (including decode and correlate)
      const res = await fetchFullAnalysis(originalFile, newSampleRate, fecScheme);

      // Update the cached result in IndexedDB and keep only metadata in sessionStorage.
      setData(res);
      await saveSignalResult(session.signalSessionId, res);
      writeSignalSession({ ...session, sampleRate: newSampleRate, resultStatus: 'ready' });

    } catch (err: any) {
      setError(err.message || 'Failed to re-process signal with new sample rate');
    } finally {
      setLoading(false);
      setEditingSampleRate(false);
    }
  };

  useEffect(() => {
    let active = true;
    (async () => {
      const session = readSignalSession();
      if (!session) { router.push('/'); return; }
      try {
        const storedResult = await loadSignalResult<ProcessResult>(session.signalSessionId);
        if (!active || !storedResult) { if (active) router.push('/'); return; }
        setData(storedResult);
        setFileName(session.filename || 'Signal File');
        setFecScheme((storedResult as ExtendedProcessResult).fec_scheme || '');
      } catch (err: any) {
        if (active) setError(err.message || 'Unable to load analysis from IndexedDB.');
      } finally { if (active) setLoading(false); }
    })();
    return () => { active = false; };
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
  const constellationData = data.recovered_symbols ? data.recovered_symbols.map((pt) => ({ i: Number(pt.i.toFixed(4)), q: Number(pt.q.toFixed(4)) })) : [];
  const isFsk = data.modulation.toUpperCase().includes('FSK');
  const fskPlotData = data.fsk_visualization_data?.instantaneous_frequency.map((frequency, x) => ({ x, frequency })) ?? [];

  // Helper to format decoded bits to Hex and ASCII
  const getFecOutputs = () => {
    if (!data?.decoded_bits || data.decoded_bits.length === 0) {
      return {
        hex: data?.decoded_hex || '',
        ascii: data?.decoded_ascii || '',
        bytes: null as Uint8Array | null,
        byteCount: 0,
        bitCount: 0,
      };
    }

    const bitCount = data.decoded_bits.length;
    const byteCount = Math.ceil(bitCount / 8);
    const bytes = new Uint8Array(byteCount);
    for (let i = 0; i < bitCount; i++) {
      if (data.decoded_bits[i]) {
        bytes[Math.floor(i / 8)] |= 1 << (7 - (i % 8));
      }
    }

    // Space-separated uppercase hex pairs (e.g. 4A 6F 68 6E)
    const hex = data.decoded_hex || Array.from(bytes, (b) => b.toString(16).padStart(2, '0').toUpperCase()).join(' ');

    // ASCII: printable characters with non-printable shown as '.' placeholders
    const ascii = data.decoded_ascii || Array.from(bytes, (b) => (b >= 32 && b <= 126 ? String.fromCharCode(b) : '.')).join('');

    return {
      hex,
      ascii,
      bytes,
      byteCount,
      bitCount,
    };
  };

  const fecOutput = getFecOutputs();

  return (
    <div className="space-y-16">
      {/* Top Header & Breadcrumb */}
      <header className="border-b border-hairline pb-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div>
            <div className="flex items-center space-x-3 mb-4">
              <Link href="/" onClick={(event) => { if (loading) event.preventDefault(); else void leaveAnalysis(); }} className="flex items-center space-x-2 px-4 py-2 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-xs font-geist-mono font-weight-500 hover:bg-cyan-950/70 transition-colors duration-200">
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
              onClick={(event) => { if (loading) event.preventDefault(); else void leaveAnalysis(); }}
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-pill bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-all duration-200 dark:bg-cyan-400/20 dark:text-cyan-500 dark:hover:bg-cyan-400/30"
            >
              <span>Analyse another Signal</span>
            </Link>
            {!editingSampleRate && data && (
              <>
                <button
                  onClick={() => {
                    if (!data) return;
                    // Individual signal information (basic parameters only) for JSON
                    const basicParams = {
                      modulation: data.modulation ?? '',
                      confidence: data.confidence ?? 0,
                      snr_db: data.snr_db ?? null,
                      baud_rate: data.baud_rate ?? null,
                      center_frequency_hz: data.center_frequency_hz ?? null,
                      bandwidth_hz: data.bandwidth_hz ?? null,
                      bandwidth_3db_hz: data.bandwidth_3db_hz ?? null,
                      num_samples: data.num_samples,
                      duration_sec: data.duration_sec,
                      sample_rate: data.sample_rate
                    };
                    const json = JSON.stringify(basicParams, null, 2);
                    const blob = new Blob([json], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    // Use actual file name for download
                    const cleanFileName = fileName.replace(/\.[^/.]+$/, ""); // Remove extension
                    a.download = `${cleanFileName}-individual-info.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
                >
                  <span>Download JSON</span>
                </button>
                <button
                  onClick={() => {
                    if (!data) return;
                    // Overall all parameters including bit streams for CSV
                    const rows = [];

                    // Basic parameters
                    rows.push(['Parameter', 'Value']);
                    rows.push(['Modulation', data.modulation ?? '']);
                    rows.push(['Confidence (%)', ((data.confidence ?? 0) * 100).toFixed(1)]);
                    rows.push(['SNR (dB)', data.snr_db !== null && data.snr_db !== undefined ? data.snr_db.toFixed(1) : '']);
                    rows.push(['Baud Rate', data.baud_rate !== null && data.baud_rate !== undefined && data.baud_rate > 0 ? (data.baud_rate >= 1000 ? `${(data.baud_rate / 1000).toFixed(2)} kBd` : `${data.baud_rate.toFixed(1)} Bd`) : '']);
                    rows.push(['Center Frequency', data.center_frequency_hz !== null && data.center_frequency_hz !== undefined ? (Math.abs(data.center_frequency_hz) >= 1e6 ? `${(data.center_frequency_hz / 1e6).toFixed(2)} MHz` : `${(data.center_frequency_hz / 1e3).toFixed(1)} kHz`) : '']);
                    rows.push(['Bandwidth', data.bandwidth_hz !== null && data.bandwidth_hz !== undefined && data.bandwidth_hz > 0 ? (data.bandwidth_hz >= 1e6 ? `${(data.bandwidth_hz / 1e6).toFixed(2)} MHz` : `${(data.bandwidth_hz / 1e3).toFixed(1)} kHz`) : '']);
                    rows.push(['Bandwidth 3dB (Hz)', data.bandwidth_3db_hz !== null && data.bandwidth_3db_hz !== undefined ? data.bandwidth_3db_hz.toLocaleString() : '']);
                    rows.push(['Sample Rate (Hz)', data.sample_rate.toLocaleString()]);
                    rows.push(['Duration (s)', data.duration_sec.toFixed(4)]);
                    rows.push(['Total Samples', data.num_samples.toLocaleString()]);
                    rows.push([]); // Empty row for separation

                    // Demodulated bits
                    if (data.demodulated_bits) {
                      rows.push(['Demodulated Bits', data.demodulated_bits.join('')]);
                      rows.push(['Demodulated Bits Count', data.demodulated_bits_count ?? 0]);
                      rows.push([]); // Empty row
                    }

                    // Deinterleaved bits
                    if (data.deinterleaved_bits) {
                      rows.push(['Deinterleaved Bits', data.deinterleaved_bits.join('')]);
                      rows.push(['Deinterleaved Bits Count', data.deinterleaved_bits_count ?? 0]);
                      rows.push([]); // Empty row
                    }

                    if (data.synchronized_bits) rows.push(['Synchronized Bits', data.synchronized_bits.join('')]);
                    // Only call the output FEC decoded when a decoder actually ran.
                    if (data.decoded_bits) {
                      rows.push([data.fec_ran ? 'FEC Decoded Bits' : 'Processed Bits (FEC not run)', data.decoded_bits.join('')]);
                      rows.push(['Decoded Bits Count', data.decoded_bits.length]);
                      if (fecOutput.hex) rows.push(['Decoded Hex (FEC Output)', fecOutput.hex]);
                      if (fecOutput.ascii) rows.push(['Decoded ASCII (FEC Output)', fecOutput.ascii]);
                      rows.push([]); // Empty row
                    }

                    // Correlate result bits
                    if (data.correlate_result && data.correlate_result.bits) {
                      rows.push(['Correlate Result Bits', data.correlate_result.bits.join('')]);
                      rows.push(['Correlate Result Bits Count', data.correlate_result.num_bits ?? 0]);
                      rows.push([]); // Empty row
                    }

                    // Add correlate result details if no bits
                    if (data.correlate_result && (!data.correlate_result.bits || data.correlate_result.bits.length === 0)) {
                      rows.push(['Correlate Result Status', data.correlate_result.status ?? '']);
                      rows.push(['Correlate Sync Found', data.correlate_result.sync_found ? 'true' : 'false']);
                      rows.push(['Correlate Peak Indices', data.correlate_result.peak_indices.join(',')]);
                      rows.push(['Correlate Num Frames', data.correlate_result.num_frames ?? 0]);
                      rows.push(['Correlate Max Correlation', data.correlate_result.max_correlation ?? 0]);
                      rows.push(['Correlate Is Inverted', data.correlate_result.is_inverted ? 'true' : 'false']);
                      rows.push([]); // Empty row
                    }

                    const csvContent = rows.map(e => e.join(',')).join('\n');
                    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    // Use actual file name for download
                    const cleanFileName = fileName.replace(/\.[^/.]+$/, ""); // Remove extension
                    a.download = `${cleanFileName}-complete-summary.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink"
                >
                  <span>Download CSV</span>
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Primary Key Metric Cards */}
      <section className="grid gap-6 grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
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
              Time-Domain Waveform
            </button>
            <button
              onClick={() => setActiveTab('psd')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'psd'
                  ? 'bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              Power Spectral Density (PSD)
            </button>
            <button
              onClick={() => setActiveTab('constellation')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'constellation'
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              {isFsk ? 'Frequency States' : 'I/Q Constellation Diagram'}
            </button>
            <button
              onClick={() => setActiveTab('spectrogram')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'spectrogram'
                  ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              Spectrogram
            </button>
            <button
              onClick={() => setActiveTab('waterfall')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-sm font-geist font-weight-500 transition-all duration-200 ${
                activeTab === 'waterfall'
                  ? 'bg-teal-500/20 text-teal-400 border border-teal-500/40'
                  : 'text-ink-muted hover:text-ink hover:bg-canvas/90'
              }`}
            >
              Waterfall
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
              {isFsk && fskPlotData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={fskPlotData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="x" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Sample Index', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Frequency (Hz)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="frequency" name="Instantaneous Frequency" stroke="#38bdf8" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : !isFsk && constellationData.length > 0 ? (
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
                <p className="text-ink-faint text-xs font-geist-mono">{isFsk ? 'Frequency state data not available.' : 'Recovered symbols not available.'}</p>
              )}
            </div>
            <p className="text-center text-xs font-geist-mono text-ink-faint">{isFsk ? 'FSK Symbol Frequencies' : 'Recovered, timing and carrier corrected I/Q symbols'}</p>
          </div>
        )}
      </section>

      {/* Tab 4: Spectrogram */}
      {activeTab === 'spectrogram' && (
        <div className="space-y-4">
          <WaterfallPlot data={data.waterfall_data} viewMode="2d" />
        </div>
      )}

      {/* Tab 5: Waterfall */}
      {activeTab === 'waterfall' && (
        <div className="space-y-4">
          <WaterfallPlot data={data.waterfall_data} viewMode="3d" />
        </div>
      )}

      {/* NEW SECTIONS START */}
      {data?.demodulated_bits && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
            Demodulated Bitstream
          </h2>
          <div className="space-y-4">
            <div className="h-48 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-lg text-ink">
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
              <span>Download Full Data</span>
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
            <div className="h-48 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-lg text-ink">
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
              <span>Download Full Data</span>
            </button>
          </div>
        </section>
      )}
      {/* FEC Decoded Output */}
      {data?.decoded_bits && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div className="flex items-center space-x-3">
              <h2 className="font-geist font-weight-600 text-lg text-ink">
                {data.fec_ran ? 'FEC Decoded Output' : 'Processed Bits (FEC not run)'}
              </h2>
              {fecScheme && fecScheme.toLowerCase() !== 'none' && (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-geist-mono font-weight-500 bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  {fecScheme.toUpperCase()}
                </span>
              )}
            </div>
            <div className="text-xs font-geist-mono text-ink-faint">
              {fecOutput.bitCount.toLocaleString()} bits • {fecOutput.byteCount.toLocaleString()} bytes
            </div>
          </div>

          {fecOutput.bitCount === 0 ? (
            <div className="p-6 rounded-md bg-canvas/60 border border-hairline text-center">
              <p className="text-sm font-geist-mono text-ink-faint">
                No decoded output available.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Bits Preview */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-geist text-sm font-weight-500 text-ink-muted">
                    Decoded Bitstream (first {Math.min(fecOutput.bitCount, 512)} bits):
                  </span>
                  <span className="text-xs font-geist-mono text-ink-faint">
                    Total: {fecOutput.bitCount.toLocaleString()} bits
                  </span>
                </div>
                <div className="h-32 w-full bg-canvas/80 border border-hairline rounded-md overflow-x-auto overflow-y-auto p-4">
                  <pre className="font-geist-mono text-sm text-ink whitespace-pre-wrap break-all leading-relaxed">
                    {data.decoded_bits.slice(0, 512).join('')}
                    {fecOutput.bitCount > 512 ? '...' : ''}
                  </pre>
                </div>
              </div>

              {/* Hex Preview */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-geist text-sm font-weight-500 text-ink-muted">
                    Hex:
                  </span>
                  <span className="text-xs font-geist-mono text-ink-faint">
                    Space-separated uppercase hex pairs
                  </span>
                </div>
                <div className="h-32 w-full bg-canvas/80 border border-hairline rounded-md overflow-x-auto overflow-y-auto p-4">
                  <pre className="font-geist-mono text-sm text-ink whitespace-pre-wrap break-all leading-relaxed tracking-wider">
                    {fecOutput.hex || '—'}
                  </pre>
                </div>
              </div>

              {/* ASCII Preview */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-geist text-sm font-weight-500 text-ink-muted">
                    ASCII:
                  </span>
                  <span className="text-xs font-geist-mono text-ink-faint">
                    Non-printable bytes shown as &quot;.&quot;
                  </span>
                </div>
                <div className="h-32 w-full bg-canvas/80 border border-hairline rounded-md overflow-x-auto overflow-y-auto p-4">
                  <pre className="font-geist-mono text-sm text-ink whitespace-pre-wrap break-all leading-relaxed tracking-wide">
                    {fecOutput.ascii || '—'}
                  </pre>
                </div>
              </div>

              {/* Download buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  onClick={() => {
                    const bitsString = data.decoded_bits?.join('') || '';
                    const blob = new Blob([bitsString], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    const cleanFileName = fileName.replace(/\.[^/.]+$/, '');
                    a.download = `${cleanFileName}_fec_decoded_bits.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 bg-ink text-on-primary dark:hover:bg-ink/20 dark:bg-ink/10 dark:text-ink text-xs"
                >
                  <span>Download Bits (.txt)</span>
                </button>

                {fecOutput.bytes && (
                  <button
                    onClick={() => {
                      if (!fecOutput.bytes) return;
                      const blob = new Blob([fecOutput.bytes.buffer as ArrayBuffer], { type: 'application/octet-stream' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      const cleanFileName = fileName.replace(/\.[^/.]+$/, '');
                      a.download = `${cleanFileName}_fec_decoded.bin`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-cyan-500/20 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs"
                  >
                    <span>Download Decoded (.bin)</span>
                  </button>
                )}
              </div>
            </div>
          )}
        </section>
      )}
      {data && data.correlate_result && (
        <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
          <h2 className="font-geist font-weight-600 text-lg text-ink mb-6">
            Correlation/Sync Results
          </h2>
          <div className="space-y-4">
            <div className="h-48 w-full bg-canvas-elevated overflow-auto p-4">
              <pre className="font-geist-mono text-lg text-ink">
                {data.correlate_result!.bits ? (
                  data.correlate_result!.bits
                    .slice(0, 100)
                    .map(bit => bit.toString())
                    .join('') + (data.correlate_result!.bits.length > 100 ? '...' : '')
                ) : (
                  JSON.stringify(data.correlate_result!, null, 2)
                )}
              </pre>
            </div>
            <button
              onClick={() => {
                let content = '';
                if (data.correlate_result!.bits) {
                  content = data.correlate_result!.bits.join('');
                } else {
                  content = JSON.stringify(data.correlate_result!, null, 2);
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
              <span>Download Full Data</span>
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
                      e.currentTarget.blur();
                      await handleSampleRateChange();
                    }
                    if (e.key === "Escape") {
                      setEditingSampleRate(false); // cancel edit
                    }
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
            <span className="block text-xs font-geist-mono font-weight-500 text-ink-faint">FEC Scheme</span>
            {editingFecScheme ? (
              <div className="flex items-center space-x-2">
                <select
                  value={fecScheme}
                  onChange={(e) => {
                    setFecScheme(e.target.value);
                  }}
                  disabled={loading}
                  className="flex-1 bg-canvas-elevated border border-hairline rounded-md px-4 py-2 text-sm font-geist-mono text-ink focus:outline-none focus:ring-2 focus-ring-blue focus:border-blue transition-colors duration-200"
                >
                  <option value="none">None</option>
                  <option value="viterbi">Viterbi</option>
                  <option value="reed-solomon">Reed-Solomon</option>
                  <option value="concatenated">Concatenated</option>
                  <option value="ldpc">LDPC</option>
                </select>
                <button
                  type="button"
                  disabled={loading || !data}
                  onClick={() => {
                    setEditingFecScheme(false);
                    // Trigger re-processing with current FEC scheme
                    if (data) {
                      handleSampleRateChange();
                    }
                  }}
                  className="shrink-0 px-3 py-2 text-xs font-geist-mono text-ink-faint bg-canvas-elevated border border-hairline rounded-md hover:border-cyan-500/50 hover:text-ink transition-colors duration-200"
                >
                  Apply
                </button>
              </div>
            ) : (
              <span className="block font-geist-mono font-weight-600 text-ink">
                {fecScheme || 'None'}
              </span>
            )}
            {!editingFecScheme && (
              <button
                onClick={() => setEditingFecScheme(true)}
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
