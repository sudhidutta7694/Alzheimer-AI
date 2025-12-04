Frontend (Next.js + Tailwind)

Environment
- Set `NEXT_PUBLIC_BACKEND_URL` to your Flask server (default `http://localhost:5000`).

Setup
```zsh
cd frontend
pnpm install  # or npm install / yarn
pnpm dev      # or npm run dev / yarn dev
```

Use
- Open http://localhost:3000, upload an MRI image, and click Analyze.
- The UI displays prediction, confidence, probability chart, and Grad-CAM heatmap.
