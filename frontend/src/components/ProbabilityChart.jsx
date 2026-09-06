import React from 'react'

export default function ProbabilityChart({ predictions }) {
  const maxProb = Math.max(...predictions.map(p => p.probability))
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-4">Top Predictions</h3>
      <div className="space-y-3">
        {predictions.map((pred, i) => (
          <div key={pred.class} className="flex items-center gap-3">
            <span className="text-sm font-medium text-slate-500 w-5">{i + 1}.</span>
            <span className="text-sm font-medium text-slate-700 w-36 truncate">{pred.name}</span>
            <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
              <div
                className="h-full bg-medical-500 rounded-full transition-all duration-500"
                style={{ width: `${(pred.probability / maxProb) * 100}%` }}
              />
            </div>
            <span className="text-sm font-mono text-slate-600 w-16 text-right">
              {(pred.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
