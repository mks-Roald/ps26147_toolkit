const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ProcessResult {
  baud_rate?: number;
  snr?: number;
  modulation: string;
  confidence: number;
  waveform_data: number[]; // time-domain samples (first N samples)
}

export async function processFile(file: File): Promise<ProcessResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/process/file`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Backend error: ${res.status} ${err}`);
  }
  return res.json();
}