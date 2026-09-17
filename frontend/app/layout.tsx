import type { Metadata } from 'next';
import './globals.css';
import Layout from '@/components/Layout';

export const metadata: Metadata = {
  title: 'SIH PS26147 - Signal Analysis Dashboard',
  description: 'Parametric RF Signal Processing, Modulation Classification, Demodulation & FEC Decoding',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased selection:bg-cyan-500 selection:text-black">
        <Layout>{children}</Layout>
      </body>
    </html>
  );
}
