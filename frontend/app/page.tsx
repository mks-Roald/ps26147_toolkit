'use client';

import { useState } from 'react';
import { processFile } from '@/services/api';
import Link from 'next/link';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await processFile(file);
      // Store the result in sessionStorage for the Results page to read
      sessionStorage.setItem('lastResult', JSON.stringify(res));
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Link href="/results">
        <a className="text-sm text-blue-600 hover:underline mb-4 inline-block">View Results →</a>
      </Link>

      <form onSubmit={handleSubmit} className="space-y-4">
        <label className="flex flex-col space-y-2">
          <span className="font-medium">Upload Signal File (.wav, .iq, .bin, …)</span>
          <input
            type="file"
            accept=".wav,.iq,.bin,.raw"
            onChange={handleChange}
            className="border rounded px-3 py-2"
            disabled={loading}
          />
          {file && (
            <p className="text-sm text-gray-600">
              Selected: {file.name} ({Math.round(file.size / 1024)} KB)
            </p>
          )}
        </label>

        <button
          type="submit"
          disabled={loading || !file}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Processing…' : 'Analyze'}
        </button>
      </form>

      {error && (
        <p className="text-red-600 bg-red-50 p-3 rounded border border-red-200">
          {error}
        </p>
      )}
    </>
  );
}