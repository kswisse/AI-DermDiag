import React, { useState } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import AnalysisResult from './components/AnalysisResult'
import Disclaimer from './components/Disclaimer'
import ProjectInfo from './components/ProjectInfo'
import { analyzeImage } from './utils/api'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [view, setView] = useState('upload')

  const handleAnalyze = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeImage(file)
      setResult(data)
      setView('result')
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setView('upload')
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <main className="max-w-5xl mx-auto px-4 py-8">
        {view === 'upload' && (
          <>
            <ImageUpload onAnalyze={handleAnalyze} loading={loading} />
            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}
          </>
        )}
        {view === 'result' && result && (
          <AnalysisResult result={result} onReset={handleReset} />
        )}
        <Disclaimer />
      </main>
      <ProjectInfo />
    </div>
  )
}
