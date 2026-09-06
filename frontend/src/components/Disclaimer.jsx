import React from 'react'

export default function Disclaimer() {
  return (
    <div className="mt-8 bg-slate-100 border border-slate-200 rounded-xl p-5">
      <p className="text-sm text-slate-600 leading-relaxed">
        <strong>Important:</strong> AI-DermDiag is a research and screening prototype. Its output is not a
        definitive medical diagnosis and should not replace evaluation by a qualified healthcare professional.
        HAM10000 consists of dermatoscopic images and does not automatically represent smartphone-camera conditions.
        Dataset performance does not equal clinical performance. Grad-CAM is an interpretability aid, not proof
        of clinical reasoning. Professional evaluation is required for suspicious lesions.
      </p>
    </div>
  )
}
