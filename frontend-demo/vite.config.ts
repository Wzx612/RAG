import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backend = 'http://127.0.0.1:8000';
const proxy = {
  '/healthz': { target: backend, changeOrigin: true },
  '/readyz': { target: backend, changeOrigin: true },
  '/answer': { target: backend, changeOrigin: true },
  '/api': { target: backend, changeOrigin: true },
  '/acceptance': { target: backend, changeOrigin: true },
  '/docs': { target: backend, changeOrigin: true },
  '/openapi.json': { target: backend, changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: false,
    proxy,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: false,
    proxy,
  },
});
