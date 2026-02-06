import React, { useMemo, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [pregunta, setPregunta] = useState("");
  const [respuesta, setRespuesta] = useState("");
  const [fuentes, setFuentes] = useState([]);
  const [mode, setMode] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [error, setError] = useState("");

  const badge = useMemo(() => {
    if (mode === "baseline") return { text: "Baseline (sin tokens)", tone: "ok" };
    if (mode === "llm") return { text: "LLM (Gemini + Chroma)", tone: "warn" };
    if (!mode) return null;
    return { text: `Modo: ${mode}`, tone: "neutral" };
  }, [mode]);

  const ask = async () => {
    const q = pregunta.trim();
    if (!q) return;

    setLoading(true);
    setError("");
    setRespuesta("");
    setFuentes([]);
    setShowSources(false);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);

    try {
      const r = await fetch(`${API_URL}/preguntar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texto: q }),
        signal: controller.signal,
      });

      if (!r.ok) {
        const txt = await r.text();
        throw new Error(`HTTP ${r.status}: ${txt}`);
      }

      const data = await r.json();
      setRespuesta(data.respuesta || "");
      setFuentes(data.fuentes || []);
      setMode(data.mode || "");
    } catch (e) {
      const msg =
        e.name === "AbortError"
          ? "⏳ El backend tardó demasiado. Intenta de nuevo."
          : `❌ ${e.message || "Error al conectar con el backend"}`;
      setError(msg);
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask();
  };

  return (
    <div className="page">
      <div className="shell">
        <header className="header">
          <div className="brand">
            <div className="logo">🛡️</div>
            <div>
              <h1>Asistente de Riesgos</h1>
              <p>RAG reproducible: retrieval + respuesta con fuentes</p>
            </div>
          </div>

          {badge && (
            <span className={`badge badge--${badge.tone}`}>
              {badge.text}
            </span>
          )}
        </header>

        <main className="grid">
          <section className="card">
            <div className="card__title">
              <h2>Consulta</h2>
              <span className="hint">Ctrl/⌘ + Enter para enviar</span>
            </div>

            <textarea
              className="input"
              value={pregunta}
              onChange={(e) => setPregunta(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ej: ¿Cuáles son las exclusiones? ¿Qué riesgos cubre la póliza?"
              rows={6}
            />

            <div className="actions">
              <button className="btn btn--primary" onClick={ask} disabled={loading}>
                {loading ? "Consultando…" : "Consultar"}
              </button>
              <button
                className="btn"
                onClick={() => {
                  setPregunta("");
                  setRespuesta("");
                  setFuentes([]);
                  setShowSources(false);
                  setError("");
                }}
                disabled={loading}
              >
                Limpiar
              </button>
            </div>

            <div className="meta">
              <span className="dot" />
              <span>Backend: {API_URL}</span>
            </div>
          </section>

          <section className="card">
            <div className="card__title">
              <h2>Respuesta</h2>
              {fuentes.length > 0 && (
                <button
                  className="link"
                  onClick={() => setShowSources((s) => !s)}
                >
                  {showSources ? "Ocultar fuentes" : `Ver fuentes (${fuentes.length})`}
                </button>
              )}
            </div>

            {error && <div className="alert">{error}</div>}

            {!error && !respuesta && (
              <div className="empty">
                <p>Haz una pregunta para ver la respuesta aquí.</p>
                <div className="chips">
                  {["exclusiones", "riesgos cubiertos", "procedimiento siniestro"].map((x) => (
                    <button key={x} className="chip" onClick={() => setPregunta(`¿${x}?`)}>
                      {x}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {respuesta && (
              <div className="answer">
                <pre>{respuesta}</pre>
              </div>
            )}

            {showSources && fuentes.length > 0 && (
              <div className="sources">
                <ol>
                  {fuentes.map((f, i) => (
                    <li key={i}>
                      <details>
                        <summary>Fuente {i + 1}</summary>
                        <pre>{f}</pre>
                      </details>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </section>
        </main>

        <footer className="footer">
          <span>Modo reproducible sin tokens + modo LLM opcional</span>
        </footer>
      </div>
    </div>
  );
}
