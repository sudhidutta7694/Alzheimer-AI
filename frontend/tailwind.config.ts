import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        medical: {
          bg: '#0b1f3a',
          card: '#112a4d',
          accent: '#00bcd4',
          text: '#e6f0f7',
        },
      },
    },
  },
  plugins: [],
}
export default config
