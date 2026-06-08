import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 开发默认直连 Django；若已启动 Spring BFF，可改为 http://127.0.0.1:8080
      "/api": {
        target: process.env.VITE_BFF_URL || "http://127.0.0.1:8001",
        changeOrigin: true,
      },
    },
  },
});
