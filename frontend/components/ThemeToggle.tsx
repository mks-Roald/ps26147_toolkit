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
      className="flex items-center space-x-2 px-3 py-1.5 rounded-sm bg-panel/80 backdrop-blur-sm hairline-border hover:bg-panel/90 transition-all duration-200 font-geist-mono font-weight-500 text-text hover:text-text/80"
    >
      <span className="flex h-4 w-4 items-center justify-center">
        {theme === 'dark' ? (
          <span className="text-cyan-400">🌙</span>
        ) : theme === 'light' ? (
          <span className="text-yellow-400">☀️</span>
        ) : (
          <span className="text-blue-400">💻</span>
        )}
      </span>
      <span className="hidden md:inline">{theme === 'dark' ? 'Dark' : theme === 'light' ? 'Light' : 'System'}</span>
    </button>
  );
}