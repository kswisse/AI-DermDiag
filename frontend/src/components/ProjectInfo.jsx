import React from 'react'

export default function ProjectInfo() {
  return (
    <footer className="bg-white border-t border-slate-200 mt-12">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <h2 className="text-xl font-bold text-slate-800 mb-4">About AI-DermDiag</h2>
        <div className="grid md:grid-cols-2 gap-6 text-sm text-slate-600">
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Problem</h3>
            <p>Skin cancer is one of the most common cancers worldwide. Early detection significantly improves outcomes, but access to dermatological expertise is limited in many regions.</p>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Solution</h3>
            <p>AI-DermDiag provides AI-assisted screening of skin lesion images using deep learning, offering preliminary classifications with explainable visualizations.</p>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Dataset</h3>
            <p>Trained on HAM10000 — Human Against Machine with 10,000 training images. Contains 10,015 dermatoscopic images across 7 diagnostic categories.</p>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Technology</h3>
            <p>EfficientNet-B0 for classification, Grad-CAM for explainability. Built with PyTorch, FastAPI, and React.</p>
          </div>
        </div>
        <div className="mt-6 pt-4 border-t border-slate-200 text-sm text-slate-500">
          <p><strong>Classes:</strong> Actinic Keratoses (akiec), Basal Cell Carcinoma (bcc), Benign Keratosis (bkl), Dermatofibroma (df), Melanoma (mel), Melanocytic Nevi (nv), Vascular Lesions (vasc)</p>
          <p className="mt-2"><strong>Limitations:</strong> This prototype is not clinically validated. Dermatoscopic images differ from smartphone photos. Performance may vary across demographics. Not a standalone diagnostic tool.</p>
        </div>
        <p className="mt-4 text-xs text-slate-400">Developed by Develop4Life — Nguyễn Đinh Trọng Khang, Nguyễn Đinh Bích Khuê</p>
      </div>
    </footer>
  )
}
