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
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Title & Status Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
            <span>REAL-TIME SDR STREAMING PIPELINE</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">Live RF Signal Oscilloscope & Constellation</h1>
        </div>

        <div className="flex items-center space-x-3 font-mono text-xs">
          <div className={`px-3 py-1.5 rounded-full border flex items-center space-x-2 ${
            isConnected
              ? 'bg-emerald-950/60 border-emerald-500/50 text-emerald-400'
              : 'bg-red-950/60 border-red-500/50 text-red-400'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`}></span>
            <span>{isConnected ? 'LIVE WS STREAM' : 'DISCONNECTED'}</span>
          </div>

          <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            {fpsCount} FPS
          </div>

          <button
            onClick={togglePause}
            disabled={!isConnected}
            className={`px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              isPaused
                ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
            }`}
          >
            {isPaused ? '▶ Resume' : '⏸ Pause'}
          </button>
        </div>
      </div>

      {/* Live Telemetry HUD */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Detected Modulation */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
          <div className="absolute top-0 left-0 h-1 w-full bg-cyan-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase">Live Modulation</p>
          <p className="text-2xl font-black text-cyan-400">
            {currentFrame?.detected_modulation || '—'}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-1">
            Confidence: <span className="text-emerald-400">{currentFrame ? `${(currentFrame.confidence * 100).toFixed(0)}%` : '—'}</span>
          </p>
        </div>

        {/* Instantaneous SNR */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
          <div className="absolute top-0 left-0 h-1 w-full bg-blue-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase">Channel SNR</p>
          <p className="text-2xl font-black text-blue-400">
            {currentFrame ? `${currentFrame.snr_db.toFixed(1)} dB` : '—'}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-1">Simulated AWGN</p>
        </div>

        {/* Baud Rate */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
          <div className="absolute top-0 left-0 h-1 w-full bg-fuchsia-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase">Symbol Rate</p>
          <p className="text-2xl font-black text-fuchsia-400">
            {currentFrame ? `${(currentFrame.baud_rate / 1000).toFixed(1)} kBd` : '—'}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-1">
            Fs: {currentFrame ? `${(currentFrame.sample_rate / 1e6).toFixed(1)} MS/s` : '—'}
          </p>
        </div>

        {/* Frame Sequence */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
          <div className="absolute top-0 left-0 h-1 w-full bg-emerald-400"></div>
          <p className="text-xs font-mono text-slate-400 uppercase">Frame Counter</p>
          <p className="text-2xl font-black text-emerald-400">
            #{currentFrame?.frame_idx ?? 0}
          </p>
          <p className="text-xs font-mono text-slate-400 mt-1">Streaming over WebSocket</p>
        </div>
      </div>

      {/* Main Real-Time Visualizations (2-Column Grid) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Constellation Cloud */}
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 backdrop-blur-xl space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-cyan-300 font-mono">🌌 I/Q Constellation Diagram</h3>
            <span className="text-xs font-mono text-slate-400">{constellationData.length} live symbols</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="i" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <YAxis type="number" dataKey="q" domain={[-2, 2]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }} />
                <Scatter data={constellationData} fill="#00f2fe" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Time-Domain Waveform Oscilloscope */}
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 backdrop-blur-xl space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-fuchsia-300 font-mono">🌊 Live Oscilloscope Waveform</h3>
            <span className="text-xs font-mono text-slate-400">Real I-Channel</span>
          </div>
          <div className="h-64 w-full">
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

        {/* Full-Width PSD Spectrum Analyzer */}
        <div className="lg:col-span-2 bg-slate-900/70 border border-slate-800 rounded-2xl p-5 backdrop-blur-xl space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-blue-300 font-mono">📊 Live Power Spectral Density (PSD)</h3>
            <span className="text-xs font-mono text-slate-400">Welch FFT Spectrum (dB/Hz)</span>
          </div>
          <div className="h-56 w-full">
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
      </div>

      {/* Interactive Stream Transmitter Controls */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-5">
        <h3 className="text-sm font-bold text-slate-200 font-mono flex items-center space-x-2">
          <span>🎛️ SDR Transmitter & Channel Impairment Simulator</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-xs font-mono">
          {/* Modulation Scheme */}
          <div className="space-y-2">
            <label className="text-slate-300 block">Modulation Scheme</label>
            <select
              value={controls.modulation}
              onChange={(e) => handleControlChange('modulation', e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
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
          <div className="space-y-2">
            <div className="flex justify-between text-slate-300">
              <span>Channel SNR</span>
              <span className="text-cyan-400 font-bold">{controls.snr_db} dB</span>
            </div>
            <input
              type="range"
              min="0"
              max="40"
              step="1"
              value={controls.snr_db}
              onChange={(e) => handleControlChange('snr_db', Number(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
            />
          </div>

          {/* Carrier Frequency Offset Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-slate-300">
              <span>Carrier Offset (CFO)</span>
              <span className="text-fuchsia-400 font-bold">{controls.cfo_hz} Hz</span>
            </div>
            <input
              type="range"
              min="-2000"
              max="2000"
              step="50"
              value={controls.cfo_hz}
              onChange={(e) => handleControlChange('cfo_hz', Number(e.target.value))}
              className="w-full accent-fuchsia-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
