const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ConstellationPoint {
  i: number;
  q: number;
}

export interface PsdPoint {
  freq: number;
  psd: number;
}

export interface WaterfallData {
  time: number[];
  frequency: number[];
  power_db: number[][];
}

export interface FskVisualizationData {
  instantaneous_frequency: number[];
  recovered_frequency_states: number[];
  symbol_frequency_values: number[];
  frequency_state_count: number;
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
  recovered_symbols?: ConstellationPoint[];
  constellation_data?: ConstellationPoint[];
  constellation_metadata?: { representation: string; symbol_rate?: number; timing_recovery_used: boolean; carrier_recovery_used: boolean };
  fsk_visualization_data?: FskVisualizationData;
  psd_data?: PsdPoint[];
  waterfall_data?: WaterfallData;
  demodulated_bits?: number[];
  demodulated_bits_count?: number;
  deinterleaved_bits?: number[];
  deinterleaved_bits_count?: number;
  decoded_bits?: number[];
  decoded_hex?: string;
  decoded_ascii?: string;
  synchronized_bits?: number[];
  fec_ran?: boolean;
  fec_decoder_result?: Record<string, any>;
  errors_corrected?: number;
  sync_method?: string;
  sync_confidence?: number;
  deinterleaver_method?: string;
  demodulation_quality?: Record<string, number>;
  correlate_result?: CorrelateResult;
  session_id?: string;
  pipeline_stages?: Record<string, string>;
  classifier_probabilities?: Record<string, number>;
  parameter_confidence?: Record<string, number>;
  timing_quality?: number;
  carrier_quality?: number;
  evm_db?: number;
  synchronized_bits?: number[];
  fec_scheme?: string;
}

export async function createAnalysisSession(file: File, sampleRate = 1000000, options: { fecScheme?: string; syncWord?: string; autoDetectSync?: boolean; autoDeinterleave?: boolean } = {}): Promise<ProcessResult> {
  const form = new FormData(); form.append("file", file);
  const params = new URLSearchParams({ fs: String(sampleRate), fec_scheme: options.fecScheme || "none", auto_detect_sync: String(options.autoDetectSync ?? true), auto_deinterleave: String(options.autoDeinterleave ?? false) });
  if (options.syncWord) params.set("sync_word", options.syncWord);
  const res = await fetch(`${API_BASE}/process/session?${params}`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API error (${res.status}): ${await res.text()}`);
  return res.json();
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
  decoded_hex?: string;
  decoded_ascii?: string;
  demodulated_bits?: number[];
  demodulated_bits_count?: number;
  deinterleaved_bits?: number[];
  deinterleaved_bits_count?: number;
  deinterleaver_method?: string;
  deinterleaver_params?: Record<string, any>;
  deinterleaver_entropy?: number;
  deinterleaver_baseline_entropy?: number;
  synchronized_bits?: number[];
  fec_ran?: boolean;
  fec_decoder_result?: Record<string, any>;
  errors_corrected?: number;
  demodulation_quality?: Record<string, number>;
  sync_method?: string;
  sync_confidence?: number;
  center_frequency_hz?: number;
  baud_rate?: number;
}

export interface CorrelatedFrame {
  start_bit: number;
  end_bit: number;
  frame_bits: number[];
  payload_bits: number[];
  correlation: number;
}

export interface PreambleDiscoveryResult {
  discovered: boolean;
  estimated_frame_period?: number;
  periodicity_strength: number;
  matched_standard_sync?: string;
  standard_sync_confidence?: number;
  candidate_preamble_bits?: number[];
  candidate_preamble_hex?: string;
}

export interface CorrelateResult {
  status: string;
  sync_found: boolean;
  peak_indices: number[];
  num_frames: number;
  frames: CorrelatedFrame[];
  max_correlation: number;
  is_inverted: boolean;
  detected_frame_length?: number;
  sync_word_len?: number;
  correlation_curve?: number[];
  preamble_discovery?: PreambleDiscoveryResult;
  num_bits?: number;
  bits?: number[];
}

export interface CorrelateRequest {
  bits?: number[];
  hex_string?: string;
  bit_string?: string;
  sync_word?: string;
  sync_word_hex?: string;
  sync_word_bits?: number[];
  frame_length?: number;
  threshold?: number;
  tolerate_inverted?: boolean;
  auto_discover?: boolean;
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
  sampleRate: number = 1000000,
  options: { syncWord?: string; autoDetectSync?: boolean; autoDeinterleave?: boolean } = {}
): Promise<DecodeResult> {
  const params = new URLSearchParams({ fec_scheme: fecScheme || "none", fs: sampleRate.toString() });
  if (options.syncWord) params.set("sync_word", options.syncWord);
  params.set("auto_detect_sync", String(options.autoDetectSync ?? false));
  params.set("auto_deinterleave", String(options.autoDeinterleave ?? false));
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/decode/?${params.toString()}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

export async function correlateSignal(
  file: File,
  options: {
    syncWord?: string;
    frameLength?: number;
    threshold?: number;
    tolerateInverted?: boolean;
    sampleRate?: number;
    fecScheme?: string;
    autoDeinterleave?: boolean;
    autoDiscover?: boolean;
  } = {}
): Promise<CorrelateResult> {
  const form = new FormData();
  form.append("file", file);

  const params = new URLSearchParams();
  if (options.syncWord) params.append("sync_word", options.syncWord);
  if (options.frameLength !== undefined) params.append("frame_length", options.frameLength.toString());
  if (options.threshold !== undefined) params.append("threshold", options.threshold.toString());
  if (options.tolerateInverted !== undefined) params.append("tolerate_inverted", options.tolerateInverted.toString());
  if (options.sampleRate !== undefined) params.append("fs", options.sampleRate.toString());
  if (options.fecScheme !== undefined) params.append("fec_scheme", options.fecScheme);
  if (options.autoDeinterleave !== undefined) params.append("auto_deinterleave", options.autoDeinterleave.toString());
  if (options.autoDiscover !== undefined) params.append("auto_discover", options.autoDiscover.toString());

  const queryString = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`${API_BASE}/correlate/file${queryString}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

export async function correlateBits(request: CorrelateRequest): Promise<CorrelateResult> {
  const res = await fetch(`${API_BASE}/correlate/bits`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

export async function getSyncWords(): Promise<Record<string, { length: number; bit_string: string; hex: string }>> {
  const res = await fetch(`${API_BASE}/correlate/sync-words`);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}
