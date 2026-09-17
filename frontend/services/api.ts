const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ConstellationPoint {
  i: number;
  q: number;
}

export interface PsdPoint {
  freq: number;
  psd: number;
}

export interface ProcessResult {
  modulation: string;
  confidence: number;
  baud_rate?: number;
  snr?: number;
  snr_db?: number;
  center_frequency_hz?: number;
  bandwidth_hz?: number;
  bandwidth_3db_hz?: number;
  num_samples: number;
  duration_sec: number;
  sample_rate: number;
  waveform_data: number[];
  constellation_data?: ConstellationPoint[];
  psd_data?: PsdPoint[];
}

export interface ClassifyResult {
  modulation: string;
  confidence: number;
  features?: Record<string, number>;
}

export interface DecodeResult {
  modulation: string;
  confidence: number;
  num_bits: number;
  bit_string_preview: string;
  hex_preview: string;
  evm_db?: number;
  evm_percent?: number;
  fec_scheme?: string;
  decoded_bits_count: number;
  decoded_bits: number[];
}

export async function processFile(file: File, sampleRate: number = 1000000): Promise<ProcessResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/process/file?fs=${sampleRate}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

export async function classifySignal(file: File, sampleRate: number = 1000000): Promise<ClassifyResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/classify/?fs=${sampleRate}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

export async function decodeSignal(
  file: File,
  fecScheme: string = "none",
  sampleRate: number = 1000000
): Promise<DecodeResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/decode/?fec_scheme=${fecScheme}&fs=${sampleRate}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}