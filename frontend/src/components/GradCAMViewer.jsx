import React, { useState } from 'react'

const VIEWS = [
  { key: 'original', label: 'Original' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'overlay', label: 'Overlay' },
]

export default function GradCAMViewer({ original, heatmap, overlay }) {
  const [active, setActive] = useState('overlay')
  const images = { original, heatmap, overlay }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-2">Explainability</h3>
      <p className="text-sm text-slate-500 mb-4">
        The Grad-CAM visualization highlights image regions that contributed most strongly to the model's prediction.
      </p>
      <div className="flex gap-2 mb-4">
        {VIEWS.map(v => (
          <button
            key={v.key}
            onClick={() => setActive(v.key)}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors
              ${active === v.key ? 'bg-medical-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
          >
            {v.label}
          </button>
        ))}
      </div>
      <div className="flex justify-center">
        <img
          src={images[active]}
          alt={active}
          className="max-h-96 rounded-lg shadow-sm"
        />
      </div>
    </div>
  )
}
