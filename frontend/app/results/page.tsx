'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Results() {
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem('lastResult');
    if (stored) {
      try {
        setData(JSON.parse(stored));
      } catch {
        setError('Failed to parse result data');
      }
      setLoading(false);
    } else {
      // No result available, go back home
      router.push('/');
    }
  }, [router]);

  if (loading) return <p className="text-center py-8">Loading…</p>;
  if (!data) return <p className="text-center py-8">No result available.</p>;
  if (error) return <p className="text-red-600 text-center py-8">{error}</p>;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-center">Analysis Results</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
          <h3 className="font-semibold text-gray-600 dark:text-gray-300">Modulation</h3>
          <p className="text-2xl font-bold">{data.modulation}</p>
        </div>
        <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
          <h3 className="font-semibold text-gray-600 dark:text-gray-300">Confidence</h3>
          <p className="text-2xl font-bold">{`${(data.confidence * 100).toFixed(1)}%`}</p>
        </div>
        {data.baud_rate !== undefined && (
          <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
            <h3 className="font-semibold text-gray-600 dark:text-gray-300">Baud Rate</h3>
            <p className="text-2xl font-bold">{data.baud_rate.toFixed(1)} Baud</p>
          </div>
        )}
        {data.snr !== undefined && (
          <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
            <h3 className="font-semibold text-gray-600 dark:text-gray-300">SNR</h3>
            <p className="text-2xl font-bold">{data.snr.toFixed(1)} dB</p>
          </div>
        )}
      </div>

      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        <h3 className="font-semibold text-gray-600 dark:text-gray-300 mb-2">
          Waveform (first {data.waveform_data.length} samples)
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data.waveform_data.map((v, i) => ({ x: i, y: v }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" dark:stroke="#4b5563" />
            <XAxis dataKey="x" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Legend verticalAlign="top" height={36} />
            <Line type="monotone" dataKey="y" stroke="#00ffff" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 text-center">
        <Link href="/" className="text-blue-600 hover:underline">
          ← Analyze another file
        </Link>
      </div>
    </div>
  );
}