'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { WaterfallData } from '@/services/api';
import Card from '@/components/base/Card';

interface WaterfallPlotProps {
  data?: WaterfallData;
  className?: string;
}

type PlotViewMode = '2d' | '3d';
type ColorScheme = 'Viridis' | 'Jet' | 'Plasma' | 'Turbo' | 'Inferno';

export default function WaterfallPlot({ data, className = '' }: WaterfallPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<PlotViewMode>('2d');
  const [colorScale, setColorScale] = useState<ColorScheme>('Viridis');
  const [isDark, setIsDark] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const plotlyInstanceRef = useRef<any>(null);

  // Monitor theme changes on document.documentElement
  useEffect(() => {
    const checkTheme = () => {
      const dark = document.documentElement.classList.contains('dark');
      setIsDark(dark);
    };

    checkTheme();

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          checkTheme();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  // Dynamically load Plotly and render chart
  const renderPlot = useCallback(async () => {
    if (!containerRef.current || !data || !data.power_db || data.power_db.length === 0) {
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      const Plotly = (await import('plotly.js-dist-min')).default;
      plotlyInstanceRef.current = Plotly;

      // Prepare coordinate axes
      // x: Time in milliseconds
      const timeMs = data.time.map((t) => Number((t * 1000).toFixed(3)));
      // y: Frequency in kHz
      const freqKhz = data.frequency.map((f) => Number((f / 1000).toFixed(3)));
      // z: 2D power array (freq rows x time cols)
      const powerDb = data.power_db;

      // Theme-specific colors
      const textColor = isDark ? '#e6edf3' : '#0f172a';
      const textMuted = isDark ? '#8b98a5' : '#64748b';
      const gridColor = isDark ? '#1e2530' : '#e2e8f0';
      const bgColor = 'rgba(0, 0, 0, 0)';

      let plotData: any[] = [];
      let layout: any = {};

      if (viewMode === '2d') {
        plotData = [
          {
            type: 'heatmap',
            x: timeMs,
            y: freqKhz,
            z: powerDb,
            colorscale: colorScale,
            colorbar: {
              title: {
                text: 'Power (dB)',
                font: { color: textColor, family: 'JetBrains Mono, monospace', size: 12 },
              },
              tickfont: { color: textMuted, family: 'JetBrains Mono, monospace', size: 10 },
              len: 0.9,
              thickness: 16,
              outlinewidth: 0,
            },
            hovertemplate:
              '<b>Time:</b> %{x:.2f} ms<br><b>Freq:</b> %{y:.2f} kHz<br><b>Power:</b> %{z:.1f} dB<extra></extra>',
          },
        ];

        layout = {
          paper_bgcolor: bgColor,
          plot_bgcolor: bgColor,
          font: {
            color: textColor,
            family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
          },
          margin: { l: 65, r: 30, t: 30, b: 60 },
          xaxis: {
            title: {
              text: 'Time (ms)',
              font: { color: textColor, size: 12, family: 'JetBrains Mono, monospace' },
            },
            tickfont: { color: textMuted, size: 10, family: 'JetBrains Mono, monospace' },
            gridcolor: gridColor,
            zerolinecolor: gridColor,
          },
          yaxis: {
            title: {
              text: 'Frequency (kHz)',
              font: { color: textColor, size: 12, family: 'JetBrains Mono, monospace' },
            },
            tickfont: { color: textMuted, size: 10, family: 'JetBrains Mono, monospace' },
            gridcolor: gridColor,
            zerolinecolor: gridColor,
          },
          autosize: true,
          height: 480,
          hovermode: 'closest',
        };
      } else {
        // 3D Surface Waterfall
        plotData = [
          {
            type: 'surface',
            x: timeMs,
            y: freqKhz,
            z: powerDb,
            colorscale: colorScale,
            colorbar: {
              title: {
                text: 'Power (dB)',
                font: { color: textColor, family: 'JetBrains Mono, monospace', size: 12 },
              },
              tickfont: { color: textMuted, family: 'JetBrains Mono, monospace', size: 10 },
              len: 0.85,
              thickness: 16,
              outlinewidth: 0,
            },
            hovertemplate:
              '<b>Time:</b> %{x:.2f} ms<br><b>Freq:</b> %{y:.2f} kHz<br><b>Power:</b> %{z:.1f} dB<extra></extra>',
          },
        ];

        layout = {
          paper_bgcolor: bgColor,
          plot_bgcolor: bgColor,
          font: {
            color: textColor,
            family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
          },
          margin: { l: 20, r: 20, t: 20, b: 20 },
          scene: {
            xaxis: {
              title: {
                text: 'Time (ms)',
                font: { color: textColor, size: 11, family: 'JetBrains Mono, monospace' },
              },
              tickfont: { color: textMuted, size: 9, family: 'JetBrains Mono, monospace' },
              gridcolor: gridColor,
              backgroundcolor: bgColor,
              showbackground: false,
            },
            yaxis: {
              title: {
                text: 'Frequency (kHz)',
                font: { color: textColor, size: 11, family: 'JetBrains Mono, monospace' },
              },
              tickfont: { color: textMuted, size: 9, family: 'JetBrains Mono, monospace' },
              gridcolor: gridColor,
              backgroundcolor: bgColor,
              showbackground: false,
            },
            zaxis: {
              title: {
                text: 'Power (dB)',
                font: { color: textColor, size: 11, family: 'JetBrains Mono, monospace' },
              },
              tickfont: { color: textMuted, size: 9, family: 'JetBrains Mono, monospace' },
              gridcolor: gridColor,
              backgroundcolor: bgColor,
              showbackground: false,
            },
            camera: {
              eye: { x: 1.5, y: -1.5, z: 1.3 },
            },
          },
          autosize: true,
          height: 520,
        };
      }

      const config: any = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'hoverClosestCartesian', 'hoverCompareCartesian'],
      };

      await Plotly.react(containerRef.current, plotData, layout, config);
      setIsLoading(false);
    } catch (err) {
      console.error('Error rendering Plotly spectrogram/waterfall:', err);
      setIsLoading(false);
    }
  }, [data, viewMode, colorScale, isDark]);

  useEffect(() => {
    renderPlot();
  }, [renderPlot]);

  // Handle window resizing
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && plotlyInstanceRef.current) {
        plotlyInstanceRef.current.Plots.resize(containerRef.current);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current && plotlyInstanceRef.current) {
        plotlyInstanceRef.current.purge(containerRef.current);
      }
    };
  }, []);

  if (!data || !data.power_db || data.power_db.length === 0) {
    return (
      <Card className={`p-6 ${className}`}>
        <h3 className="font-geist font-weight-600 text-lg text-ink mb-2">
          Spectrogram & Waterfall Display
        </h3>
        <p className="text-xs font-geist-mono text-ink-faint">
          Waterfall data is not available for this signal.
        </p>
      </Card>
    );
  }

  const numTimeBins = data.time.length;
  const numFreqBins = data.frequency.length;

  return (
    <Card className={`p-6 hover:floating-shadow transition-all duration-300 ${className}`}>
      {/* Header & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-hairline pb-4 mb-6">
        <div>
          <div className="flex items-center space-x-3 mb-1">
            <h2 className="font-geist font-weight-600 text-lg text-ink">
              Spectrogram & 3D Waterfall Display
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-geist-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Interactive
            </span>
          </div>
          <p className="text-xs font-geist-mono text-ink-faint">
            Time-frequency-power spectral distribution ({numTimeBins} time bins × {numFreqBins} frequency bins)
          </p>
        </div>

        {/* View mode and Colormap toggles */}
        <div className="flex flex-wrap items-center gap-3">
          {/* 2D / 3D Switch */}
          <div className="inline-flex rounded-sm p-1 bg-canvas border border-hairline">
            <button
              onClick={() => setViewMode('2d')}
              className={`px-3 py-1.5 rounded-sm text-xs font-geist font-weight-500 transition-all duration-200 ${
                viewMode === '2d'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 shadow-sm'
                  : 'text-ink-muted hover:text-ink'
              }`}
            >
              🗺️ 2D Heatmap
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-3 py-1.5 rounded-sm text-xs font-geist font-weight-500 transition-all duration-200 ${
                viewMode === '3d'
                  ? 'bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/40 shadow-sm'
                  : 'text-ink-muted hover:text-ink'
              }`}
            >
              🏔️ 3D Surface
            </button>
          </div>

          {/* Colormap selection */}
          <div className="flex items-center space-x-2">
            <label className="text-xs font-geist-mono text-ink-faint hidden sm:inline">
              Palette:
            </label>
            <select
              value={colorScale}
              onChange={(e) => setColorScale(e.target.value as ColorScheme)}
              className="bg-canvas border border-hairline rounded-sm px-2.5 py-1.5 text-xs font-geist-mono text-ink focus:outline-none focus:ring-2 focus-ring-blue transition-colors duration-200"
            >
              <option value="Viridis">Viridis (Default)</option>
              <option value="Jet">Jet (Classic Waterfall)</option>
              <option value="Plasma">Plasma</option>
              <option value="Turbo">Turbo</option>
              <option value="Inferno">Inferno</option>
            </select>
          </div>
        </div>
      </div>

      {/* Plot container */}
      <div className="relative w-full rounded-md overflow-hidden bg-canvas-elevated">
        {isLoading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-canvas-elevated/70 backdrop-blur-xs">
            <div className="w-8 h-8 border-3 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-xs font-geist-mono text-ink-muted mt-2">
              Rendering {viewMode === '2d' ? '2D Spectrogram' : '3D Waterfall'}…
            </span>
          </div>
        )}
        <div
          ref={containerRef}
          className="w-full min-h-[480px]"
          style={{ width: '100%' }}
        />
      </div>

      {/* Footer Info */}
      <div className="flex flex-wrap items-center justify-between gap-2 mt-4 pt-3 border-t border-hairline/60 text-xs font-geist-mono text-ink-faint">
        <span>
          {viewMode === '2d'
            ? 'Hover over spectrogram bins to inspect precise time, frequency, and power (dB).'
            : 'Click and drag to rotate the 3D surface. Scroll to zoom, and right-click to pan.'}
        </span>
        <span className="text-cyan-400 font-geist-mono">
          STFT Spectral Waterfall
        </span>
      </div>
    </Card>
  );
}
