import React from 'react'
import GradCAMViewer from './GradCAMViewer'
import ProbabilityChart from './ProbabilityChart'

export default function AnalysisResult({ result, onReset }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">AI Screening Result</h2>
        <button onClick={onReset} className="px-4 py-2 text-sm text-medical-600 hover:text-medical-700 font-medium">
          New Analysis
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="text-center mb-4">
          <p className="text-sm text-slate-500 uppercase tracking-wide">Predicted Class</p>
          <p className="text-3xl font-bold text-slate-800 mt-1">{result.class_name}</p>
          <p className="text-lg text-medical-600 font-semibold mt-1">
            Confidence {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>

        {result.class_info && (
          <div className="bg-slate-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-slate-600">{result.class_info.description}</p>
            {result.class_info.risk && (
              <p className="text-sm text-amber-700 mt-2 font-medium">{result.class_info.risk}</p>
            )}
          </div>
        )}
      </div>

      <ProbabilityChart predictions={result.top_predictions} />

      <GradCAMViewer
        original={result.original_image}
        heatmap={result.heatmap_image}
        overlay={result.overlay_image}
      />

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-5">
        <p className="text-sm text-amber-800">
          <strong>Important:</strong> {result.disclaimer}
        </p>
      </div>
    </div>
  )
}
