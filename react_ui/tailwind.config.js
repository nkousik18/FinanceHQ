/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        amber: {
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
        charcoal: {
          50:  "#f5f0e8",
          100: "#e8e0d0",
          200: "#c8bcaa",
          300: "#a89880",
          400: "#7a6a58",
          500: "#4a3c2c",
          800: "#1c1714",
          900: "#161210",
          950: "#0f0d0b",
        },
      },
      fontFamily: {
        sans: ["Geist Variable", "Geist", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
