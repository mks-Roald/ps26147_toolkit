'use client';

import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  useEffect(() => {
    const saved = localStorage.getItem('theme') as 'light' | 'dark' | null;
    const initial = saved || 'dark';
    setTheme(initial);
    applyTheme(initial);
  }, []);

  const applyTheme = (next: 'light' | 'dark') => {
    const isDark = next === 'dark';
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('theme', next);
  };

  const cycleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    applyTheme(next);
  };

  return (
    <button
      onClick={cycleTheme}
      title={`Current theme: ${theme} (Click to switch)`}
      className="flex items-center space-x-2 px-3 py-1.5 rounded-sm bg-panel/80 backdrop-blur-sm hairline-border hover:bg-panel/90 transition-all duration-200 font-geist-mono font-weight-500 text-ink hover:text-ink/80"
    >
      <span className="flex h-4 w-4 items-center justify-center">
        {theme === 'dark' ? (
          <span className="text-cyan-400">🌙</span>
        ) : (
          <span className="text-yellow-400">☀️</span>
        )}
      </span>
      <span className="hidden md:inline">{theme === 'dark' ? 'Dark (Recommended)' : 'Light'}</span>
    </button>
  );
}