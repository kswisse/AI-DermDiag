import React, { useState, useRef } from 'react'

export default function ImageUpload({ onAnalyze, loading }) {
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    if (!f.type.match(/^image\/(jpeg|jpg|png|webp)$/)) {
      alert('Please upload a JPG, PNG, or WEBP image.')
      return
    }
    if (f.size > 10 * 1024 * 1024) {
      alert('File too large. Maximum size is 10MB.')
      return
    }
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8">
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer
          ${dragActive ? 'border-medical-500 bg-medical-50' : 'border-slate-300 hover:border-medical-400'}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/webp"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-sm" />
        ) : (
          <div>
            <svg className="w-16 h-16 mx-auto text-slate-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-slate-600 font-medium">Upload a skin lesion image</p>
            <p className="text-slate-400 text-sm mt-1">JPG, PNG — max 10MB</p>
          </div>
        )}
      </div>
      {file && (
        <button
          onClick={() => onAnalyze(file)}
          disabled={loading}
          className="mt-6 w-full py-3 bg-medical-600 hover:bg-medical-700 disabled:bg-slate-400 text-white font-medium rounded-lg transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Analyzing image...
            </span>
          ) : (
            'Analyze image'
          )}
        </button>
      )}
      {loading && (
        <div className="mt-4 text-center text-sm text-slate-500 space-y-1">
          <p>Running EfficientNet-B0 inference</p>
          <p>Generating explainability map</p>
        </div>
      )}
    </div>
  )
}
