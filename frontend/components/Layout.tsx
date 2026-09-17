'use client';

import { ThemeToggle } from './ThemeToggle';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

export interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const pathname = usePathname();

  const navLinks = [
    { href: '/', label: 'Upload & Process' },
    { href: '/results', label: 'Dashboard Results' },
    { href: '/about', label: 'About & Toolkit' },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-900/80 border-b border-slate-800/80 px-6 py-3.5 transition-all">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 via-blue-600 to-fuchsia-500 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
              ⚡
            </div>
            <div>
              <Link href="/" className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-cyan-400 via-blue-300 to-fuchsia-400 bg-clip-text text-transparent">
                SIH PS26147
              </Link>
              <span className="hidden sm:inline-block ml-2 text-xs font-mono text-slate-400 border border-slate-700/60 rounded px-1.5 py-0.5">
                v0.2.0
              </span>
            </div>
          </div>

          <nav className="flex items-center space-x-1 sm:space-x-2">
            {navLinks.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    active
                      ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center space-x-3">
            <div className="hidden md:flex items-center text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2.5 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5"></span>
              API Online
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-400 font-mono">
        <p>SIH PS26147 Signal Processing & Demodulation Engine • Automated RF Signal Pipeline</p>
      </footer>
    </div>
  );
}