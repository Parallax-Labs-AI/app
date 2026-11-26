import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/katex/') || id.includes('/node_modules/rehype-katex/')) return 'math'
          if (id.includes('/node_modules/highlight.js/') || id.includes('/node_modules/lowlight/')) return 'syntax'
        },
      },
    },
  },
  server: { host: '127.0.0.1', port: 1420, strictPort: true },
})
