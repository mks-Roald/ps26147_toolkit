'use client';

import { useState, useRef, useEffect, DragEvent } from 'react';
import { processFile, startAsyncProcess, pollJobStatus } from '@/services/api';
import { useRouter } from 'next/navigation';
import Button from '@/components/base/Button';

interface UploadZoneProps {
  onSuccess?: (result: any) => void;
  sampleRate?: number;
  setSampleRate?: (rate: number) => void;
  useAsync?: boolean;
  setUseAsync?: (async: boolean) => void;
}

export default function UploadZone({
  onSuccess,
  sampleRate: sampleRateProp,
  setSampleRate: setSampleRateProp,
  useAsync: useAsyncProp,
  setUseAsync: setUseAsyncProp
}: UploadZoneProps = {}) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  // Use props if provided, otherwise fallback to internal state (for backward compatibility)
  const [sampleRate, setSampleRate] = useState<number>(sampleRateProp ?? 1000000);
  const [useAsync, setUseAsync] = useState<boolean>(useAsyncProp ?? true);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressStage, setProgressStage] = useState<string>('');

  // Sync prop changes to state (if parent updates)
  useEffect(() => {
    if (sampleRateProp !== undefined) setSampleRate(sampleRateProp);
  }, [sampleRateProp]);

  useEffect(() => {
    if (useAsyncProp !== undefined) setUseAsync(useAsyncProp);
  }, [useAsyncProp]);

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

      if (onSuccess) {
        onSuccess(res);
      } else {
        router.push('/results');
      }
    } catch (err: any) {
      setError(err.message || 'Signal processing failed. Please check the backend connection.');
    } finally {
      setLoading(false);
      setProgressPercent(0);
      setProgressStage('');
    }
  };

  return (
    <div className="space-y-8">
      {/* Drag & Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200
          ${isDragging
            ? 'border-accent bg-accent/30 scale-[1.01]'
            : file
            ? 'border-accent bg-accent/10'
            : 'border-hairline hover:border-accent/50 bg-bg/90 hover:bg-bg/95'
          }
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.iq,.bin,.raw,.sigmf-data"
          onChange={handleFileChange}
          className="hidden"
          disabled={loading}
        />

        <div className="space-y-4">
          <div className="w-16 h-16 mx-auto rounded-full bg-panel/80 border border-border flex items-center justify-center text-3xl">
            {file ? '📊' : '📁'}
          </div>

          {file ? (
            <div>
              <p className="font-semibold text-text text-lg">{file.name}</p>
              <p className="text-xs font-geist-mono text-text-faint mt-1">
                {(file.size / 1024).toFixed(1)} KB • Click or drag to replace
              </p>
            </div>
          ) : (
            <div>
              <p className="font-semibold text-text text-base">
                Click to select or drag and drop signal file
              </p>
              <p className="text-xs text-text-faint mt-1">
                Supports <span className="text-accent font-geist-mono">.wav</span>, <span className="text-accent font-geist-mono">.iq</span>, <span className="text-accent font-geist-mono">.raw</span>, <span className="text-accent font-geist-mono">.bin</span>, <span className="text-accent font-geist-mono">.sigmf-data</span>
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Configuration Card */}
      <div className="bg-panel hairline-border rounded-lg p-6 whisper-shadow">
        <h3 className="text-ink font-geist font-weight-600 text-lg mb-4">Configuration</h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="sm:col-span-2">
            <label className="block text-xs font-geist-mono font-weight-500 text-ink-faint mb-2">
              Sampling Rate (Hz)
            </label>
            <select
              value={sampleRate}
              onChange={(e) => setSampleRate(Number(e.target.value))}
              disabled={loading}
              className="w-full bg-canvas-elevated border border-hairline rounded-md px-4 py-2 text-sm font-geist-mono text-ink focus:outline-none focus:ring-2 focus-ring-blue focus:border-blue transition-colors duration-200"
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
            <label className="flex items-center space-x-3 text-xs font-geist-mono font-weight-500 text-ink-faint bg-canvas-elevated border border-hairline rounded-lg px-4 py-3 cursor-pointer hover:border-cyan-500/50 transition-colors duration-200">
              <input
                type="checkbox"
                checked={useAsync}
                onChange={(e) => setUseAsync(e.target.checked)}
                disabled={loading}
                className="rounded border-hairline text-cyan-500 focus:ring-cyan-500 bg-canvas-elevated"
              />
              <span>Async Polling</span>
            </label>
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <div>
        <Button
          variant="primary"
          size="lg"
          className="w-full"
          disabled={loading || !file}
          onClick={handleSubmit as any}
        >
          {loading ? (
            <>
              <svg className="animate-spin h-4 w-4 text-white mr-2" viewBox="0 0 24 24">
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
        </Button>
      </div>

      {/* Live Progress Bar for Async Polling */}
      {loading && (
        <div className="mt-6 p-4 bg-cyan-950/30 border border-cyan-800/40 rounded-lg font-geist-mono text-xs text-cyan-300">
          <div className="flex justify-between items-center mb-2">
            <span className="flex items-center space-x-2">
              <span className="animate-pulse text-cyan-400">▶</span>
              <span>{progressStage || 'Processing…'}</span>
            </span>
            <span className="font-bold">{progressPercent}%</span>
          </div>
          <div className="w-full bg-canvas-elevated rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-cyan-400 via-blue-500 to-fuchsia-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(5, progressPercent)}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="mt-6 p-4 bg-ee0000/40 border border-c50000/50 rounded-lg text-sm text-ee0000 space-y-2">
          <p className="font-semibold text-ee0000">Processing Failed</p>
          <p className="text-xs font-geist-mono">{error}</p>
        </div>
      )}
    </div>
  );
}