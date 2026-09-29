import test from 'node:test';
import assert from 'node:assert/strict';
import { readSignalSession, writeSignalSession } from '../services/signalStorage.ts';

test('sessionStorage persists only the signal pointer and metadata, never file bytes', () => {
  const values = new Map();
  globalThis.sessionStorage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  const record = { signalSessionId: 'session-1', filename: 'large.iq', fileSize: 10_485_760,
    sampleRate: 1_000_000, resultStatus: 'ready' };
  writeSignalSession(record);
  const stored = values.get('signalSession');
  assert.deepEqual(readSignalSession(), record);
  assert.deepEqual(Object.keys(JSON.parse(stored)).sort(), Object.keys(record).sort());
  assert.equal(stored.includes('base64'), false);
});
