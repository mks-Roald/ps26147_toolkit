'use client';

import { useState, useRef, useEffect, DragEvent } from 'react';
import { processFile, startAsyncProcess, pollJobStatus, decodeSignal, correlateSignal } from '@/services/api';
import { useRouter } from 'next/navigation';
import Button from '@/components/base/Button';
import { clearSignalSession, saveSignalFile, saveSignalResult, readSignalSession, writeSignalSession } from '@/services/signalStorage';

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
  const [sampleRate, setSampleRateState] = useState<number>(sampleRateProp ?? 1000000);
  const [useAsync, setUseAsync] = useState<boolean>(useAsyncProp ?? true);
  const [isCustomRate, setIsCustomRate] = useState<boolean>(false);
  const [customRateInput, setCustomRateInput] = useState<string>('');
  const [isAutoDetected, setIsAutoDetected] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressStage, setProgressStage] = useState<string>('');

  // Utility function to extract sample rate from WAV file
  const getSampleRateFromWav = async (file: File): Promise<number | null> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const array = e.target?.result as ArrayBuffer;
        if (!array) { resolve(null); return; }
        // RIFF header: "RIFF", then file size, then "WAVE"
        // Then chunks: we look for "fmt " chunk.
        const view = new DataView(array);
        // Check RIFF and WAVE
        if (
          getString(view, 0, 4) !== "RIFF" ||
          getString(view, 8, 4) !== "WAVE"
        ) {
          resolve(null);
          return;
        }
        // Search for fmt chunk
        let offset = 12;
        while (offset < view.byteLength - 8) {
          const chunkId = getString(view, offset, 4);
          const chunkSize = view.getUint32(offset + 4, true); // little-endian
          if (chunkId === "fmt ") {
            // Audio format (2 bytes) at offset+8, sample rate (4 bytes) at offset+12
            const sampleRate = view.getUint32(offset + 12, true);
            resolve(sampleRate);
            return;
          }
          // Move to next chunk: align to even boundary
          offset += 8 + Math.ceil(chunkSize / 2) * 2;
        }
        resolve(null);
      };
      reader.onerror = () => resolve(null);
      // Read only enough bytes to find fmt chunk; we'll read first 64 KB which is more than enough.
      const slice = file.slice(0, 64 * 1024);
      reader.readAsArrayBuffer(slice);
    });
  };

  function getString(view: DataView, offset: number, length: number): string {
    let str = "";
    for (let i = 0; i < length; i++) {
      str += String.fromCharCode(view.getUint8(offset + i));
    }
    return str;
  }

  // Sync prop changes to state (if parent updates)
  useEffect(() => {
    if (sampleRateProp !== undefined) setSampleRateState(sampleRateProp);
  }, [sampleRateProp]);

  useEffect(() => {
    if (useAsyncProp !== undefined) setUseAsync(useAsyncProp);
  }, [useAsyncProp]);

  // Update prop setter when sample rate changes locally
  useEffect(() => {
    if (setSampleRateProp) {
      setSampleRateProp(sampleRate);
    }
  }, [sampleRate, setSampleRateProp]);

  // Initialize customRateInput when entering custom mode or when sampleRate changes
  useEffect(() => {
    if (isCustomRate) {
      setCustomRateInput(sampleRate.toString());
    }
  }, [isCustomRate, sampleRate]);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0];
      setFile(file);
      // Reset auto-detection flag
      setIsAutoDetected(false);
      // Try to auto-detect sample rate for WAV files
      if (file.name.toLowerCase().endsWith('.wav')) {
        const detectedRate = await getSampleRateFromWav(file);
        if (detectedRate !== null) {
          setSampleRateState(detectedRate);
          setIsCustomRate(false);
          setIsAutoDetected(true);
        }
      }
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      setFile(file);
      // Reset auto-detection flag
      setIsAutoDetected(false);
      // Try to auto-detect sample rate for WAV files
      if (file.name.toLowerCase().endsWith('.wav')) {
        const detectedRate = await getSampleRateFromWav(file);
        if (detectedRate !== null) {
          setSampleRateState(detectedRate);
          setIsCustomRate(false);
          setIsAutoDetected(true);
        }
      }
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
      const previousSession = readSignalSession();
      if (previousSession) await clearSignalSession(previousSession.signalSessionId);
      sessionStorage.removeItem('signalSession');
      const signalSessionId = crypto.randomUUID();
      await saveSignalFile(signalSessionId, file);
      writeSignalSession({ signalSessionId, filename: file.name, fileSize: file.size, sampleRate, resultStatus: 'processing' });

      let processRes;
      if (useAsync) {
        setProgressStage('Submitting job to background task queue…');
        const job = await startAsyncProcess(file, sampleRate);
        setProgressStage('Job queued. Polling execution status…');

        processRes = await pollJobStatus(job.job_id, (status) => {
          setProgressPercent(Math.round(status.progress * 100));
          setProgressStage(status.stage);
        });
      } else {
        setProgressStage('Processing synchronous request…');
        processRes = await processFile(file, sampleRate);
      }

      // Fetch decode results
      setProgressStage('Fetching decode results…');
      const decodeRes = await decodeSignal(file, "", sampleRate);

      // Fetch correlate results
      setProgressStage('Fetching correlation results…');
      const correlateRes = await correlateSignal(file, { sampleRate });

      // Merge all results
      const fullRes = {
        ...processRes,
        demodulated_bits: decodeRes.demodulated_bits,
        demodulated_bits_count: decodeRes.demodulated_bits_count,
        deinterleaved_bits: decodeRes.deinterleaved_bits,
        deinterleaved_bits_count: decodeRes.deinterleaved_bits_count,
        decoded_bits: decodeRes.decoded_bits,
        decoded_hex: decodeRes.decoded_hex,
        decoded_ascii: decodeRes.decoded_ascii,
        correlate_result: correlateRes,
      };

      await saveSignalResult(signalSessionId, fullRes);
      writeSignalSession({ signalSessionId, filename: file.name, fileSize: file.size, sampleRate, resultStatus: 'ready' });

      if (onSuccess) {
        onSuccess(fullRes);
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
      <div className="bg-panel border-hairline rounded-lg p-6 whisper-shadow">
        <h3 className="text-ink font-geist font-weight-600 text-lg mb-4">Configuration</h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="sm:col-span-2">
            <label className="block text-xs font-geist-mono font-weight-500 text-ink-faint mb-2">
              Sampling Rate (Hz)
            </label>
            {isCustomRate ? (
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={customRateInput}
                  onChange={(e) => {
                    const value = e.target.value;
                    setCustomRateInput(value);
                    const numValue = Number(value);
                    if (!isNaN(numValue) && numValue > 0) {
                      setSampleRateState(numValue);
                      setIsAutoDetected(false); // Manual entry overrides auto-detection
                    }
                  }}
                  min="1"
                  step="1"
                  disabled={loading}
                  className="flex-1 bg-canvas-elevated border border-hairline rounded-md px-4 py-2 text-sm font-geist-mono text-ink focus:outline-none focus:ring-2 focus-ring-blue focus:border-blue transition-colors duration-200"
                  placeholder="Enter custom rate (Hz)"
                />
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => {
                    setIsCustomRate(false);
                    // Snap back to nearest preset, defaulting to 1 MHz
                    const presets = [1000000, 2000000, 2400000, 5000000, 10000000, 44100, 48000];
                    if (!presets.includes(sampleRate)) {
                      setSampleRateState(1000000);
                    }
                  }}
                  className="shrink-0 px-3 py-2 text-xs font-geist-mono text-ink-faint bg-canvas-elevated border border-hairline rounded-md hover:border-cyan-500/50 hover:text-ink transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Switch back to preset rates"
                >
                  ← Presets
                </button>
              </div>
            ) : (
              <select
                value={sampleRate}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (value === -1) { // Custom option selected
                    setIsCustomRate(true);
                  } else {
                    setSampleRateState(value);
                    setIsCustomRate(false);
                    setIsAutoDetected(false); // Manual selection overrides auto-detection
                  }
                }}
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
                <option value={-1}>Custom...</option>
              </select>
            )}
            {isAutoDetected && (
              <p className="mt-1 text-xs font-geist-mono text-accent">
                Auto-detected: {sampleRate.toLocaleString()} Hz
              </p>
            )}
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
          className={`
            font-geist-mono font-weight-500 transition-all duration-150 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100 w-full px-8 py-3 rounded-md
            ${loading
              ? 'bg-accent/30 border border-accent/40 text-accent'
              : 'bg-white text-black border-hairline hover:bg-white/80 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700'
            }
          `}
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
              <span>Run Full Pipeline</span>
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
        <div className="mt-6 p-4 bg-danger/40 border border-danger/50 rounded-lg text-sm text-danger space-y-2">
          <p className="font-semibold text-danger">Processing Failed</p>
          <p className="text-xs font-geist-mono">{error}</p>
        </div>
      )}
    </div>
  );
}
