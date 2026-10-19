import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // GitHub Pages serves the site under /rele/; local dev and preview stay at /
  base: process.env.PAGES_BASE ?? '/',
  plugins: [react(), tailwindcss()],
  worker: { format: 'es' },
});
