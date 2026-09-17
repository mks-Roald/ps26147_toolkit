'use client';

import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [theme, setTheme] = useState<'system' | 'light' | 'dark'>('dark');

  useEffect(() => {
    const saved = localStorage.getItem('theme') as 'system' | 'light' | 'dark' | null;
    const initial = saved || 'dark';
    setTheme(initial);
    applyTheme(initial);
  }, []);

  const applyTheme = (next: 'system' | 'light' | 'dark') => {
    const isDark =
      next === 'dark' ||
      (next === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('theme', next);
  };

  const cycleTheme = () => {
    const next = theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark';
    setTheme(next);
    applyTheme(next);
  };

  return (
    <button
      onClick={cycleTheme}
      title={`Current theme: ${theme} (Click to switch)`}
      className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-xs font-mono text-slate-300 transition-all hover:border-cyan-500/50"
    >
      <span>{theme === 'dark' ? '🌙' : theme === 'light' ? '☀️' : '💻'}</span>
      <span className="capitalize">{theme}</span>
    </button>
  );
}