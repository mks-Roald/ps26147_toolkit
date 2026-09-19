'use client';

import { useRouter } from 'next/navigation';
import UploadZone from '@/components/UploadZone';
import  Card  from '@/components/base/Card';

export default function HomePage() {
  const router = useRouter();

  const modulationOptions = [
    { label: 'BPSK', value: 'BPSK' },
    { label: 'QPSK', value: 'QPSK' },
    { label: '8PSK', value: '8PSK' },
    { label: '16-QAM', value: '16QAM' },
    { label: '64-QAM', value: '64QAM' },
    { label: '2-FSK', value: 'FSK2' },
    { label: '4-FSK', value: 'FSK4' },
  ];

  return (
    <div className="space-y-12">
      {/* Header */}
      <header className="border-b border-hairline pb-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-3 px-4 py-2 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-xs font-geist-mono font-weight-500 mb-4">
              <span>📤 Upload & Process</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-geist font-weight-600 tracking-tighter text-ink">
              Satellite Signal Processor
            </h1>
          </div>
        </div>
      </header>

      {/* Upload Zone */}
      <UploadZone
        onSuccess={(result) => {
          router.push('/results');
        }}
      />

      {/* Supported Modulations */}
      <Card className="p-6">
        <div className="space-y-4">
          <h3 className="font-geist font-weight-600 text-lg text-ink">
            Supported Modulations
          </h3>
          <div className="flex flex-wrap gap-2">
            {modulationOptions.map((mod) => (
              <span
                key={mod.value}
                className={`px-3 py-1.5 rounded-full text-xs font-geist-mono font-weight-500 border border-hairline bg-canvas/80 hover:bg-canvas/90 transition-colors duration-200`}
              >
                {mod.label}
              </span>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}