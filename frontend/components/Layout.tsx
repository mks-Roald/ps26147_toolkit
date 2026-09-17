import { ThemeToggle } from './ThemeToggle';
import type { ReactNode } from 'react';

export interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors">
      <header className="flex items-center justify-between px-6 py-4 bg-white dark:bg-gray-800 shadow">
        <h1 className="text-xl font-bold">Signal Analysis Dashboard</h1>
        <ThemeToggle />
      </header>
      <main className="container mx-auto px-4 py-8">{children}</main>
    </div>
  );
}