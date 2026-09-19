'use client';

import { useEffect, useState, useRef } from 'react';
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
import { createSDRWebSocket, SDRFrame, SDRStreamControls } from '@/services/stream';
import Card from '@/components/base/Button';

export default function LiveStreamPage() {
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<SDRFrame | null>(null);
  const [fpsCount, setFpsCount] = useState<number>(0);
  const [controls, setControls] = useState<SDRStreamControls>({
    modulation: 'QPSK',
    snr_db: 20,
    baud_rate: 50000,
    cfo_hz: 0,
    fps: 15,
  });

  const wsRef = useRef<ReturnType<typeof createSDRWebSocket> | null>(null);
  const frameCounterRef = useRef<number>(0);
  const lastFpsCalcTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    const ws = createSDRWebSocket(
      (frame) => {
        setCurrentFrame(frame);
        frameCounterRef.current += 1;

        const now = Date.now();
        if (now - lastFpsCalcTimeRef.current >= 1000) {
          setFpsCount(frameCounterRef.current);
          frameCounterRef.current = 0;
          lastFpsCalcTimeRef.current = now;
        }
      },
      (err) => {
        console.error('WebSocket error:', err);
        setIsConnected(false);
      },
      () => {
        setIsConnected(true);
      },
      () => {
        setIsConnected(false);
      }
    );

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const handleControlChange = (field: keyof SDRStreamControls, value: any) => {
    const updated = { ...controls, [field]: value };
    setControls(updated);
    if (wsRef.current) {
      wsRef.current.updateConfig(updated);
    }
  };

  const togglePause = () => {
    if (!wsRef.current) return;
    if (isPaused) {
      wsRef.current.resume();
      setIsPaused(false);
    } else {
      wsRef.current.pause();
      setIsPaused(true);
    }
  };

  const waveformData = currentFrame?.waveform
    ? currentFrame.waveform.map((y, x) => ({ x, y }))
    : [];

  const constellationData = currentFrame?.constellation || [];
  const psdData = currentFrame?.psd || [];

  return (
    <div className="space-y-12 max-w-7xl mx-auto">
      {/* Top Title & Status Header */}
      <header className="border-b border-hairline pb-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-3 px-4 py-2 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-xs font-geist-mono font-weight-500 mb-4">
              <span>🛰️ REAL-TIME SDR STREAMING PIPELINE</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-geist font-weight-600 tracking-tighter text-ink">
              Live RF Signal Oscilloscope & Constellation
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3 px-4 py-2 rounded-full bg-emerald-950/60 border border-emerald-500/50 text-emerald-400 font-geist-mono font-weight-500">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>{isConnected ? 'LIVE WS STREAM' : 'DISCONNECTED'}</span>
            </div>

            <div className="flex items-center space-x-3 px-4 py-2 rounded-full bg-slate-950 border border-hairline text-slate-300 font-geist-mono font-weight-500">
              {fpsCount} FPS
            </div>

            <button
              onClick={togglePause}
              disabled={!isConnected}
              className={`flex items-center space-x-2 px-5 py-2.5 rounded-pill font-geist font-weight-500 transition-all duration-200 hover:bg-ink/90 ${isPaused ? 'bg-ink text-on-primary' : 'bg-canvas-elevated text-ink hairline-border'}`}
            >
              <span>{isPaused ? '▶ Resume' : '⏸ Pause'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Live Telemetry HUD */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Detected Modulation */}
        <Card className="p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-cyan-950/50 text-cyan-400 rounded-full text-lg">
              1
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink mb-0 ml-3">Live Modulation</h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-cyan-400">
            {currentFrame?.detected_modulation || '—'}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            Confidence: <span className="text-emerald-400 font-geist-mono">{currentFrame ? `${(currentFrame.confidence * 100).toFixed(0)}%` : '—'}</span>
          </p>
        </Card>

        {/* Instantaneous SNR */}
        <Card className="p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-blue-950/50 text-blue-400 rounded-full text-lg">
              2
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink mb-0 ml-3">Channel SNR</h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-blue-400">
            {currentFrame ? `${currentFrame.snr_db.toFixed(1)} dB` : '—'}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">Simulated AWGN</p>
        </Card>

        {/* Baud Rate */}
        <Card className="p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-fuchsia-950/50 text-fuchsia-400 rounded-full text-lg">
              3
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink mb-0 ml-3">Symbol Rate</h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-fuchsia-400">
            {currentFrame ? `${(currentFrame.baud_rate / 1000).toFixed(1)} kBd` : '—'}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">
            Fs: {currentFrame ? `${(currentFrame.sample_rate / 1e6).toFixed(1)} MS/s` : '—'}
          </p>
        </Card>

        {/* Frame Sequence */}
        <Card className="p-6 hover:floating-shadow transition-all duration-300">
          <div className="flex items-center justify-start mb-4">
            <div className="w-10 h-10 flex items-center justify-center bg-emerald-950/50 text-emerald-400 rounded-full text-lg">
              4
            </div>
            <h3 className="text-geist font-weight-600 text-lg text-ink mb-0 ml-3">Frame Counter</h3>
          </div>
          <p className="text-2xl font-geist font-weight-600 text-emerald-400">
            #{currentFrame?.frame_idx ?? 0}
          </p>
          <p className="text-xs font-geist-mono text-ink-faint mt-2">Streaming over WebSocket</p>
        </Card>
      </section>

      {/* Main Real-Time Visualizations */}
      <section className="grid gap-8">
        {/* Constellation Cloud */}
        <Card className="col-span-1 lg:col-span-1 p-6">
          <div className="space-y-4">
            <h3 className="flex items-center justify-between">
              <span className="text-geist font-weight-600 text-lg text-ink">🌌 I/Q Constellation Diagram</span>
              <span className="text-xs font-geist-mono text-ink-faint">{constellationData.length} live symbols</span>
            </h3>
            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" dataKey="i" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis type="number" dataKey="q" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }} />
                  <Scatter data={constellationData} fill="#00f2fe" isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>

        {/* Time-Domain Waveform Oscilloscope */}
        <Card className="col-span-1 lg:col-span-1 p-6">
          <div className="space-y-4">
            <h3 className="flex items-center justify-between">
              <span className="text-geist font-weight-600 text-lg text-ink">🌊 Live Oscilloscope Waveform</span>
              <span className="text-xs font-geist-mono text-ink-faint">Real I-Channel</span>
            </h3>
            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="x" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis domain={[-1.5, 1.5]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }} />
                  <Line type="monotone" dataKey="y" stroke="#f355da" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>

        {/* Full-Width PSD Spectrum Analyzer */}
        <Card className="lg:col-span-2 p-6">
          <div className="space-y-4">
            <h3 className="flex items-center justify-between">
              <span className="text-geist font-weight-600 text-lg text-ink">📊 Live Power Spectral Density (PSD)</span>
              <span className="text-xs font-geist-mono text-ink-faint">Welch FFT Spectrum (dB/Hz)</span>
            </h3>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={psdData}>
                  <defs>
                    <linearGradient id="livePsdGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="freq" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis domain={[-80, 0]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }} />
                  <Area type="monotone" dataKey="psd" stroke="#38bdf8" strokeWidth={1.5} fill="url(#livePsdGrad)" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>
      </section>

      {/* Interactive Stream Transmitter Controls */}
      <section className="bg-canvas-elevated hairline-border rounded-lg p-6 whisper-shadow">
        <h2 className="text-geist font-weight-600 text-lg text-ink mb-6">
          SDR Transmitter & Channel Impairment Simulator
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-xs font-geist-mono">
          {/* Modulation Scheme */}
          <div className="space-y-4">
            <label className="block text-xs font-geist-mono font-weight-500 text-ink-faint mb-2">
              Modulation Scheme
            </label>
            <select
              value={controls.modulation}
              onChange={(e) => handleControlChange('modulation', e.target.value)}
              disabled={isConnected}
              className="w-full bg-canvas-elevated border border-hairline rounded-md px-4 py-2 text-sm font-geist-mono text-ink focus:outline-none focus:ring-2 focus-ring-blue focus:border-blue transition-colors duration-200"
            >
              <option value="BPSK">BPSK</option>
              <option value="QPSK">QPSK</option>
              <option value="8PSK">8PSK</option>
              <option value="16QAM">16-QAM</option>
              <option value="64QAM">64-QAM</option>
              <option value="FSK2">2-FSK</option>
              <option value="FSK4">4-FSK</option>
            </select>
          </div>

          {/* SNR Slider */}
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="flex justify-between text-xs font-geist-mono font-weight-500 text-ink-faint">
                <span>Channel SNR</span>
                <span className="font-mono text-cyan-400">{controls.snr_db} dB</span>
              </label>
            </div>
            <input
              type="range"
              min="0"
              max="40"
              step="1"
              value={controls.snr_db}
              onChange={(e) => handleControlChange('snr_db', Number(e.target.value))}
              className="w-full bg-canvas-elevated h-2 rounded-full cursor-pointer bg-gradient-to-r from-cyan-400 via-blue-500 to-fuchsia-500"
            />
          </div>

          {/* Carrier Frequency Offset Slider */}
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="flex justify-between text-xs font-geist-mono font-weight-500 text-ink-faint">
                <span>Carrier Offset (CFO)</span>
                <span className="font-mono text-fuchsia-400">{controls.cfo_hz} Hz</span>
              </label>
            </div>
            <input
              type="range"
              min="-2000"
              max="2000"
              step="50"
              value={controls.cfo_hz}
              onChange={(e) => handleControlChange('cfo_hz', Number(e.target.value))}
              className="w-full bg-canvas-elevated h-2 rounded-full cursor-pointer bg-gradient-to-r from-fuchsia-400 via-magenta-500 to-pink-500"
            />
          </div>
        </div>
      </section>
    </div>
  );
}