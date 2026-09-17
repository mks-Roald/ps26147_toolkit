'use client';

import { useState, useRef, DragEvent } from 'react';
import { processFile, startAsyncProcess, pollJobStatus } from '@/services/api';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import UploadZone from '@/components/UploadZone';

export default function Home() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [sampleRate, setSampleRate] = useState<number>(1000000);
  const [useAsync, setUseAsync] = useState<boolean>(true);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressPercent, setProgressPercent] = useState<number>(0);
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
    setProgressPercent(10);
    setProgressStage('Initializing file upload…');

    try {
      let res;
      if (useAsync) {
        setProgressStage('Submitting job to background task queue…');
        const job = await startAsyncProcess(file, sampleRate);
        setProgressStage('Job queued. Polling execution status…');

        res = await pollJobStatus(job.job_id, (status) => {
          setProgressPercent(Math.round(status.progress * 100));
          setProgressStage(status.stage);
        });
      } else {
        setProgressStage('Processing synchronous request…');
        res = await processFile(file, sampleRate);
      }

      // Store results and navigate
      sessionStorage.setItem('lastResult', JSON.stringify(res));
      sessionStorage.setItem('lastFileName', file.name);
      sessionStorage.setItem('lastFileSize', String(file.size));

      router.push('/results');
    } catch (err: any) {
      setError(err.message || 'Signal processing failed. Please check the backend connection.');
    } finally {
      setLoading(false);
      setProgressPercent(0);
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
        <UploadZone />
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