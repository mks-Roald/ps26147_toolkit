export interface SDRFrame {
  type: "sdr_frame";
  frame_idx: number;
  timestamp: number;
  configured_modulation: string;
  detected_modulation: string;
  confidence: number;
  snr_db: number;
  baud_rate: number;
  sample_rate: number;
  constellation: Array<{ i: number; q: number }>;
  waveform: number[];
  psd: Array<{ freq: number; psd: number }>;
}

export interface SDRStreamControls {
  modulation?: string;
  snr_db?: number;
  baud_rate?: number;
  cfo_hz?: number;
  fps?: number;
}

export function createSDRWebSocket(
  onFrame: (frame: SDRFrame) => void,
  onError?: (err: Event) => void,
  onOpen?: () => void,
  onClose?: () => void
): {
  socket: WebSocket;
  updateConfig: (controls: SDRStreamControls) => void;
  pause: () => void;
  resume: () => void;
  close: () => void;
} {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const wsUrl = baseUrl.replace(/^http/, "ws") + "/stream/live";

  const socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    if (onOpen) onOpen();
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "sdr_frame") {
        onFrame(data);
      }
    } catch {
      // Failed to parse frame
    }
  };

  socket.onerror = (err) => {
    if (onError) onError(err);
  };

  socket.onclose = () => {
    if (onClose) onClose();
  };

  const updateConfig = (controls: SDRStreamControls) => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: "configure", ...controls }));
    }
  };

  const pause = () => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: "pause" }));
    }
  };

  const resume = () => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: "resume" }));
    }
  };

  const close = () => {
    socket.close();
  };

  return { socket, updateConfig, pause, resume, close };
}
