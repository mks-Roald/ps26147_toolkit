'use client';

import { ThemeToggle } from './ThemeToggle';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';

export interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [scrolled, setScrolled] = useState(false);

  // Header scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 4);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial check

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { href: '/', label: 'Upload & Process' },
    { href: '/live', label: '🔴 Live SDR Stream' },
    { href: '/results', label: 'Dashboard Results' },
    { href: '/about', label: 'About & Toolkit' },
  ];

  return (
    <div className="min-h-screen bg-canvas text-ink flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation Bar */}
      <header
        className={`sticky top-0 z-50 backdrop-blur-md bg-canvas/80 border-b border-hairline/80 px-6 py-3.5 transition-all duration-200 ${
          scrolled ? 'bg-canvas/95 shadow-whisper' : ''
        }`}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 via-blue-600 to-fuchsia-500 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
              ⚡
            </div>
            <div>
              <Link href="/" className="font-extrabold text-lg tracking-tighter bg-gradient-to-r from-cyan-400 via-blue-300 to-fuchsia-400 bg-clip-text text-transparent">
                SIH PS26147
              </Link>
              <span className="hidden sm:inline-block ml-2 text-xs font-geist-mono font-weight-500 text-ink-faint border border-hairline/60 rounded px-2 py-0.5">
                v0.2.0
              </span>
            </div>
          </div>

          <nav className="hidden md:flex items-center space-x-2">
            {navLinks.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`${active
                    ? 'text-cyan-400 font-weight-500 border-b-2 border-cyan-400 px-2 py-1.5'
                    : 'text-ink-muted hover:text-ink hover:border-hairline/50 border-b-2 border-transparent px-2 py-1.5'
                  } transition-all duration-200`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center space-x-3">
            <div className="hidden md:flex items-center text-xs font-geist-mono font-weight-500 text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2.5 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5"></span>
              API Online
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-hairline bg-canvas/90 py-8 text-center text-xs font-geist-mono font-weight-400 text-ink-faint">
        <p>SIH PS26147 Signal Processing & Demodulation Engine • Automated RF Signal Pipeline</p>
      </footer>
    </div>
  );
}