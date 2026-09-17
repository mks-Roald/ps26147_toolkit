'use client';

import { useState, useRef, DragEvent } from 'react';
import { useRouter } from 'next/navigation';
import { processFile } from '@/services/api';
import Link from 'next/link';

export default function Home() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [sampleRate, setSampleRate] = useState<number>(1000000);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressStage, setProgressStage] = useState<string>('');

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setProgressStage('Ingesting & parsing raw signal samples…');

    try {
      setTimeout(() => setProgressStage('Extracting spectral features & cumulants…'), 400);
      setTimeout(() => setProgressStage('Classifying modulation & calculating SNR/Baud…'), 800);

      const res = await processFile(file, sampleRate);
      
      // Store the result and metadata for the Results view
      sessionStorage.setItem('lastResult', JSON.stringify(res));
      sessionStorage.setItem('lastFileName', file.name);
      sessionStorage.setItem('lastFileSize', String(file.size));

      router.push('/results');
    } catch (err: any) {
      setError(err.message || 'Signal processing failed. Please check the backend connection.');
    } finally {
      setLoading(false);
      setProgressStage('');
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-10">
      {/* Hero Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-2">
          <span>🛰️ Parametric Signal Intelligence Engine</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-black tracking-tight text-white">
          Analyze, Classify & Demodulate <br />
          <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-fuchsia-400 bg-clip-text text-transparent">
            Radio Frequency Signals
          </span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto">
          Upload raw IQ streams, SigMF recordings, or WAV files to instantly detect modulation schemes, estimated SNR, Baud rate, occupied bandwidth, and spectral density.
        </p>
      </div>

      {/* Main Upload Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-xl shadow-2xl shadow-cyan-950/20">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Drag & Drop Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all ${
              isDragging
                ? 'border-cyan-400 bg-cyan-950/30 scale-[1.01]'
                : file
                ? 'border-emerald-500/50 bg-emerald-950/10'
                : 'border-slate-700 hover:border-slate-500 bg-slate-950/40 hover:bg-slate-950/70'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.iq,.bin,.raw,.sigmf-data"
              onChange={handleFileChange}
              className="hidden"
              disabled={loading}
            />

            <div className="space-y-3">
              <div className="w-14 h-14 mx-auto rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center text-2xl">
                {file ? '📊' : '📁'}
              </div>
              
              {file ? (
                <div>
                  <p className="font-semibold text-emerald-400 text-lg">{file.name}</p>
                  <p className="text-xs font-mono text-slate-400 mt-1">
                    {(file.size / 1024).toFixed(1)} KB • Click or drag to replace
                  </p>
                </div>
              ) : (
                <div>
                  <p className="font-semibold text-slate-200 text-base">
                    Click to select or drag and drop signal file
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports <span className="text-cyan-400 font-mono">.wav</span>, <span className="text-cyan-400 font-mono">.iq</span>, <span className="text-cyan-400 font-mono">.raw</span>, <span className="text-cyan-400 font-mono">.bin</span>, <span className="text-cyan-400 font-mono">.sigmf-data</span>
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Configuration Inputs */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <div>
              <label className="block text-xs font-mono text-slate-300 mb-1.5">
                Sampling Rate (Hz)
              </label>
              <select
                value={sampleRate}
                onChange={(e) => setSampleRate(Number(e.target.value))}
                disabled={loading}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-cyan-500"
              >
                <option value={1000000}>1,000,000 Hz (1.0 MHz SDR Default)</option>
                <option value={2000000}>2,000,000 Hz (2.0 MHz RTL-SDR)</option>
                <option value={2400000}>2,400,000 Hz (2.4 MHz RTL-SDR)</option>
                <option value={5000000}>5,000,000 Hz (5.0 MHz HackRF)</option>
                <option value={10000000}>10,000,000 Hz (10.0 MHz)</option>
                <option value={44100}>44,100 Hz (Audio WAV)</option>
                <option value={48000}>48,000 Hz (Studio WAV)</option>
              </select>
            </div>

            <div className="flex flex-col justify-end">
              <button
                type="submit"
                disabled={loading || !file}
                className="w-full py-2.5 px-4 rounded-lg font-semibold text-sm transition-all bg-gradient-to-r from-cyan-500 via-blue-600 to-fuchsia-600 text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:opacity-95 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <span>Processing Signal…</span>
                  </>
                ) : (
                  <>
                    <span>⚡ Run Full Pipeline</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Loading status message */}
          {loading && progressStage && (
            <div className="p-3 bg-cyan-950/40 border border-cyan-800/50 rounded-lg text-xs font-mono text-cyan-300 flex items-center space-x-2">
              <span className="animate-pulse">▶</span>
              <span>{progressStage}</span>
            </div>
          )}

          {/* Error display */}
          {error && (
            <div className="p-4 bg-red-950/40 border border-red-800/50 rounded-lg text-sm text-red-300 space-y-1">
              <p className="font-semibold">Processing Failed</p>
              <p className="text-xs font-mono">{error}</p>
            </div>
          )}
        </form>
      </div>

      {/* Feature Pills */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
        <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
          <div className="text-cyan-400 text-lg mb-1">🎯 12+ Modulations</div>
          <div className="text-xs text-slate-400">BPSK, QPSK, 8PSK, 16QAM, 64QAM, FSK2, FSK4, AM, FM, etc.</div>
        </div>
        <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
          <div className="text-fuchsia-400 text-lg mb-1">📈 Parametric Extraction</div>
          <div className="text-xs text-slate-400">Cyclic transition Baud rate, M2M4 split-moment SNR, 3dB & 99% OBW.</div>
        </div>
        <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
          <div className="text-blue-400 text-lg mb-1">🔓 Demodulation & FEC</div>
          <div className="text-xs text-slate-400">Gardner TED timing sync, Costas PLL, Viterbi, RS, Concatenated & LDPC.</div>
        </div>
      </div>
    </div>
  );
}