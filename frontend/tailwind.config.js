/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors');

module.exports = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark "instrument panel" palette
        bg: '#0A0E14',
        panel: '#12161F',
        'panel-2': '#171C26',
        border: '#1E2530',
        text: '#E6EDF3',
        'text-dim': '#8B98A5',
        'text-faint': '#545F6D',
        accent: '#3DD9C4',
        blue: '#4FC3F7',
        amber: '#F5A623',
        danger: '#F0546B',

        // For backwards compatibility with existing classes
        ink: '#E6EDF3',
        'ink-muted': '#8B98A5',
        'ink-faint': '#545F6D',
        'ink-subtle': '#8B98A5',
        canvas: '#0A0E14',
        'canvas-elevated': '#12161F',
        'hairline': '#1E2530',
        'hairline-soft': '#1E2530',

        // Brand colors (keeping some original accents)
        primary: '#0A0E14',
        'on-primary': '#E6EDF3',

        // Accent colors from Geist (replaced with new palette)
        cyan: '#3DD9C4',
        'cyan-soft': '#66EAD8',
        blue: '#4FC3F7',
        'blue-deep': '#3AB8E8',
        'blue-soft': '#D0F4FF',
        violet: '#7928ca',
        'violet-soft': '#d8ccf1',
        pink: '#ff0080',
        magenta: '#eb367f',

        // Gradient stops
        'gradient-develop-start': '#007cf0',
        'gradient-develop-end': '#00dfd8',
        'gradient-preview-start': '#7928ca',
        'gradient-preview-end': '#ff0080',
        'gradient-ship-start': '#ff4d4d',
        'gradient-ship-end': '#f9cb28',
      },
      borderRadius: {
        // Geist-inspired radius scale
        'none': '0px',
        'sm': '6px',
        'md': '12px',
        'lg': '16px',
        'pill-category': '64px',
        'pill': '100px',
        'full': '9999px',
      },
      spacing: {
        // 4px base unit spacing system
        'xxs': '4px',
        'xs': '8px',
        'sm': '12px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
        '2xl': '40px',
        '3xl': '64px',
        '4xl': '96px',
        'section': '128px',
        // Added numeric keys for the classes we are using in @apply
        '1.5': '6px',
        '2': '8px',
        '3': '12px',
        '4': '16px',
        '6': '24px',
        '8': '32px',
        '16': '64px',
      },
      fontSize: {
        '3xl': '1.875rem', // 30px
        'lg': '1.125rem',  // 18px
      },
      maxWidth: {
        '7xl': '80rem', // 1280px
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            color: theme('colors.text-dim'),
            '[class~="lead"]:not(:where([class~="not-prose"]*))': {
              color: theme('colors.text'),
            },
            a: {
              color: theme('colors.accent'),
              '&:hover': {
                color: theme('colors.blue'),
              },
            },
            strong: {
              color: theme('colors.text'),
            },
            'ol > li::before': {
              backgroundColor: theme('colors.accent'),
            },
            'ul > li::before': {
              backgroundColor: theme('colors.accent'),
            },
            hr: {
              borderColor: theme('colors.border'),
            },
            blockquote: {
              color: theme('colors.text'),
              borderLeftColor: theme('colors.border'),
            },
            h1: {
              color: theme('colors.text'),
              letterSpacing: '-2.4px',
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            h2: {
              color: theme('colors.text'),
              letterSpacing: '-1.28px',
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            h3: {
              color: theme('colors.text'),
              letterSpacing: '-0.4px',
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            'h4, h5, h6': {
              color: theme('colors.text'),
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            'p, li': {
              fontWeight: '400',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            code: {
              color: theme('colors.text'),
              backgroundColor: theme('colors.panel'),
              padding: '0.2em 0.4em',
              borderRadius: theme('rounded.md'),
              border: `1px solid ${theme('colors.border')}`,
              fontFamily: 'JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace',
            },
          },
        },
        dark: {
          css: {
            color: theme('colors.text-dim'),
            '[class~="lead"]:not(:where([class~="not-prose"]*))': {
              color: theme('colors.text'),
            },
            a: {
              color: theme('colors.accent'),
              '&:hover': {
                color: theme('colors.blue'),
              },
            },
            strong: {
              color: theme('colors.text'),
            },
            'ol > li::before': {
              backgroundColor: theme('colors.accent'),
            },
            'ul > li::before': {
              backgroundColor: theme('colors.accent'),
            },
            hr: {
              borderColor: theme('colors.border'),
            },
            blockquote: {
              color: theme('colors.text'),
              borderLeftColor: theme('colors.border'),
            },
            h1: {
              color: theme('colors.text'),
              letterSpacing: '-2.4px',
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            h2: {
              color: theme('colors.text'),
              letterSpacing: '-1.28px',
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            h3: {
              color: theme('colors.text'),
              letterSpacing: '-0.4px',
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            'h4, h5, h6': {
              color: theme('colors.text'),
              fontWeight: '600',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            'p, li': {
              fontWeight: '400',
              fontFamily: 'Inter, system-ui, sans-serif',
            },
            code: {
              color: theme('colors.text'),
              backgroundColor: theme('colors.panel'),
              padding: '0.2em 0.4em',
              borderRadius: theme('rounded.md'),
              border: `1px solid ${theme('colors.border')}`,
              fontFamily: 'JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace',
            },
          },
        },
      }),
      boxShadow: {
        // Whisper and floating shadows from Geist
        whisper: '0px 1px 1px rgba(0,0,0,0.04)',
        floating: '0px 2px 2px rgba(0,0,0,0.04), 0px 8px 16px -4px rgba(0,0,0,0.04)',
      },
      keyframes: {
        // Subtle animations
        pulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-4px)' },
        },
        gradientShift: {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        },
      },
      animation: {
        pulse: 'pulse 3s ease-in-out infinite',
        float: 'float 6s ease-in-out infinite',
        gradientShift: 'gradientShift 8s ease infinite',
      },
    },
  },
  plugins: [],
};