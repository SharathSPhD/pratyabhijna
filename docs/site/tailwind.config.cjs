/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        serif: ['"Source Serif 4"', '"Source Serif Pro"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        ink: {
          DEFAULT: '#1a1a1a',
          soft: '#2d2d2d',
        },
        paper: {
          DEFAULT: '#fafaf7',
          soft: '#f3f1ea',
        },
        accent: {
          DEFAULT: '#7c3aed',
          ink: '#a78bfa',
        },
      },
      maxWidth: {
        prose: '72ch',
        wide: '88ch',
      },
    },
  },
  plugins: [],
};
