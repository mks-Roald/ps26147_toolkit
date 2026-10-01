import type { Metadata } from 'next';
import './globals.css';
import Layout from '@/components/Layout';
import Head from 'next/head';

export const metadata: Metadata = {
  title: 'SIGextract - Signal Analysis Dashboard',
  description: 'Parametric RF Signal Processing, Modulation Classification, Demodulation & FEC Decoding',
  icons: {
    icon: '/icons/icon.svg',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-canvas text-ink min-h-screen antialiased selection:bg-cyan-500 selection:text-black">
        <Layout>{children}</Layout>
      </body>
    </html>
  );
}