import './globals.css'
import type { Metadata } from 'next'
import Image from 'next/image'
import logo from '../assets/logo.png'

export const metadata: Metadata = {
  title: 'Alzheimer MRI Analysis',
  description: 'Clinical-grade Alzheimer MRI classification with explainability and ROI-based insights',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet" />
      </head>
      <body className="font-ui bg-surface text-slate-900 antialiased">
        <header className="bg-white/60 backdrop-blur sticky top-0 z-40 border-b border-slate-200">
          <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
                <Image src={logo} alt="Alzheimer AI" className="h-9 w-9 rounded-lg object-contain" />
              <div>
                <div className="text-sm font-semibold">Alzheimer AI</div>
                <div className="text-xs text-slate-500 -mt-0.5">Clinical Imaging · Explainable AI</div>
              </div>
            </div>
            <nav className="flex items-center gap-4 text-sm text-slate-600">
              <a className="hover:text-slate-800" href="/docs">Docs</a>
            </nav>
          </div>
        </header>

        <main className='mt-10Su'>{children}</main>

        <footer className="mt-12 border-t border-slate-200">
          <div className="max-w-7xl mx-auto px-6 py-6 text-xs text-slate-500">
            For clinical decision support only. Not a diagnostic device. See model card for dataset & limitations.
          </div>
        </footer>
      </body>
    </html>
  )
}
