// frontend/src/api.js

const API_URL =
  process.env.REACT_APP_API_URL ||
  "http://127.0.0.1:8000";

export function getApiUrl() {
  return API_URL;
}

export async function preguntar({ texto, mode = "local", top_k = 5, signal }) {
  const res = await fetch(`${API_URL}/preguntar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texto, mode, top_k }),
    signal,
  });

  let data;
  try {
    data = await res.json();
  } catch {
    const raw = await res.text();
    throw new Error(`Respuesta no JSON (HTTP ${res.status}): ${raw}`);
  }

  if (!res.ok) {
    const msg = data?.detail || data?.error || JSON.stringify(data);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }

  return data;
}