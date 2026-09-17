'use client';

import { useEffect } from 'react';

export function ThemeToggle() {
  useEffect(() => {
    const apply = (theme: 'system' | 'light' | 'dark') => {
      const isDark =
        theme === 'dark' ||
        (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
      document.documentElement.classList.toggle('dark', isDark);
      localStorage.setItem('theme', theme);
    };

    const saved = localStorage.getItem('theme') as 'system' | 'light' | 'dark' | null;
    if (saved) apply(saved);
    else apply('system'); // default
  }, []);

  return (
    <button
      onClick={() => {
        const cur = localStorage.getItem('theme') || 'system';
        const next =
          cur === 'system' ? 'light' : cur === 'light' ? 'dark' : 'system';
        localStorage.setItem('theme', next);
        const isDark =
          next === 'dark' ||
          (next === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
        document.documentElement.classList.toggle('dark', isDark);
      }}
      className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
    >
      🌓
    </button>
  );
}