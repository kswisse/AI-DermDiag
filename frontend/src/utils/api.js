const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api'

export async function analyzeImage(file) {
  const formData = new FormData()
  formData.append('image', file)
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    throw new Error(data?.detail || `Server error: ${response.status}`)
  }
  return response.json()
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`)
  return response.json()
}

export async function fetchClasses() {
  const response = await fetch(`${API_BASE}/classes`)
  return response.json()
}
