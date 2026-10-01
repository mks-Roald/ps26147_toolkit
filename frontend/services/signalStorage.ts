const DB_NAME = 'signal-analysis';
const DB_VERSION = 1;
const FILES_STORE = 'files';
const RESULTS_STORE = 'results';

export interface SignalSessionRecord {
  signalSessionId: string;
  filename: string;
  fileSize: number;
  sampleRate: number;
  resultStatus: 'ready' | 'processing';
}

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB is unavailable in this browser. Signal persistence is required to continue.'));
  }
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(FILES_STORE)) db.createObjectStore(FILES_STORE);
      if (!db.objectStoreNames.contains(RESULTS_STORE)) db.createObjectStore(RESULTS_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Unable to open IndexedDB.'));
    request.onblocked = () => reject(new Error('IndexedDB is blocked by another open connection.'));
  });
}

function requestResult<T>(db: IDBDatabase, store: string, mode: IDBTransactionMode, action: (objectStore: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(store, mode);
    const request = action(tx.objectStore(store));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB operation failed.'));
    tx.onabort = () => reject(tx.error || new Error('IndexedDB transaction aborted.'));
  }).finally(() => db.close());
}

export async function saveSignalFile(id: string, file: File | Blob): Promise<void> {
  const db = await openDatabase();
  await requestResult(db, FILES_STORE, 'readwrite', store => store.put(file, id));
}

export async function loadSignalFile(id: string, filename?: string): Promise<File | null> {
  const db = await openDatabase();
  const blob = await requestResult(db, FILES_STORE, 'readonly', store => store.get(id)) as Blob | undefined;
  if (!blob) return null;
  if (typeof File !== 'undefined' && blob instanceof File) return blob;
  return new File([blob], filename || 'signal.file', { type: blob.type || 'application/octet-stream', lastModified: Date.now() });
}

export async function deleteSignalFile(id: string): Promise<void> {
  const db = await openDatabase();
  await requestResult(db, FILES_STORE, 'readwrite', store => store.delete(id));
}

export async function saveSignalResult(id: string, result: unknown): Promise<void> {
  const db = await openDatabase();
  await requestResult(db, RESULTS_STORE, 'readwrite', store => store.put(result, id));
}

export async function loadSignalResult<T>(id: string): Promise<T | null> {
  const db = await openDatabase();
  return (await requestResult(db, RESULTS_STORE, 'readonly', store => store.get(id)) as T | undefined) ?? null;
}

export async function clearSignalSession(id: string): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([FILES_STORE, RESULTS_STORE], 'readwrite');
    tx.objectStore(FILES_STORE).delete(id);
    tx.objectStore(RESULTS_STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('Unable to clear signal session.'));
    tx.onabort = () => reject(tx.error || new Error('Unable to clear signal session.'));
  }).finally(() => db.close());
}

export const SIGNAL_SESSION_KEY = 'signalSession';

export function readSignalSession(): SignalSessionRecord | null {
  try {
    const value = sessionStorage.getItem(SIGNAL_SESSION_KEY);
    return value ? JSON.parse(value) as SignalSessionRecord : null;
  } catch {
    return null;
  }
}

export function writeSignalSession(record: SignalSessionRecord): void {
  sessionStorage.setItem(SIGNAL_SESSION_KEY, JSON.stringify(record));
}
