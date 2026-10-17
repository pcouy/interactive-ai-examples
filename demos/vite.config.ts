import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    open: '/playground/index.html',
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    lib: {
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: 'drl-demos',
    },
    rollupOptions: {
      output: {
        chunkFileNames: (chunkInfo) =>
          chunkInfo.name === 'three.module' ? 'drl-three.js' : `drl-${chunkInfo.name}.js`,
      },
    },
  },
});
