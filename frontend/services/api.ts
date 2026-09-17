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

export interface AsyncJobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  stage: string;
  created_at: number;
  result?: ProcessResult;
  error?: string;
}

/** Synchronous file processing */
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

/** Queue file for asynchronous background processing */
export async function startAsyncProcess(file: File, sampleRate: number = 1000000): Promise<AsyncJobResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/process/async?fs=${sampleRate}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

/** Fetch current status and live progress of an asynchronous job */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/process/status/${jobId}`);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

/** Poll asynchronous job until completion with progress callback */
export async function pollJobStatus(
  jobId: string,
  onProgress?: (status: JobStatusResponse) => void,
  intervalMs: number = 400,
  maxAttempts: number = 150
): Promise<ProcessResult> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const status = await getJobStatus(jobId);
    if (onProgress) {
      onProgress(status);
    }
    if (status.status === "completed" && status.result) {
      return status.result;
    }
    if (status.status === "failed") {
      throw new Error(status.error || "Background processing failed.");
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error("Job polling timed out.");
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