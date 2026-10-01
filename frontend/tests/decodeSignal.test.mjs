import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeSignal } from '../services/api.ts';

test('decode request forwards configured pipeline parameters', async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl;
  globalThis.fetch = async (url) => {
    requestedUrl = String(url);
    return { ok: true, json: async () => ({ fec_scheme: 'viterbi' }) };
  };
  try {
    await decodeSignal(new File(['iq'], 'signal.iq'), 'viterbi', 1_000_000, {
      syncWord: 'Barker-13', autoDeinterleave: true, autoDetectSync: true,
    });
    const params = new URL(requestedUrl).searchParams;
    assert.equal(params.get('fec_scheme'), 'viterbi');
    assert.equal(params.get('sync_word'), 'Barker-13');
    assert.equal(params.get('auto_deinterleave'), 'true');
    assert.equal(params.get('auto_detect_sync'), 'true');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('decode defaults to explicit no FEC and no automatic transforms', async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl;
  globalThis.fetch = async (url) => {
    requestedUrl = String(url);
    return { ok: true, json: async () => ({}) };
  };
  try {
    await decodeSignal(new File(['iq'], 'signal.iq'));
    const params = new URL(requestedUrl).searchParams;
    assert.equal(params.get('fec_scheme'), 'none');
    assert.equal(params.get('auto_deinterleave'), 'false');
    assert.equal(params.get('auto_detect_sync'), 'false');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
