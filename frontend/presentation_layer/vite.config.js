import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendOrigin = env.VITE_BACKEND_ORIGIN || "http://127.0.0.1:8000";

  return {
    plugins: [vue()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/api": {
          target: backendOrigin,
          changeOrigin: true
        },
        "/outputs": {
          target: backendOrigin,
          changeOrigin: true
        },
        "/source-assets": {
          target: backendOrigin,
          changeOrigin: true
        }
      }
    }
  };
});
