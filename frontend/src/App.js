import React, { useMemo, useState } from "react";
import "./App.css";
import { preguntar, getApiUrl } from "./api";

export default function App() {
  const API_URL = getApiUrl();

  const [pregunta, setPregunta] = useState("");

  // Output principal
  const [respuesta, setRespuesta] = useState("");
  const [fuentes, setFuentes] = useState([]);

  // Debug / contrato
  const [mode, setMode] = useState("local"); // ✅ default local
  const [topK, setTopK] = useState(5);
  const [retrieved, setRetrieved] = useState([]);
  const [noEvidence, setNoEvidence] = useState(false);
  const [gateReason, setGateReason] = useState("");
  const [usedFallback, setUsedFallback] = useState(false);

  // UX
  const [loading, setLoading] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [error, setError] = useState("");

  const badge = useMemo(() => {
    if (mode === "baseline") return { text: "Baseline (sin tokens)", tone: "ok" };
    if (mode === "llm") return { text: "LLM (Gemini + Chroma)", tone: "warn" };
    if (!mode) return null;
    return { text: `Modo: ${mode}`, tone: "neutral" };
  }, [mode]);

  const resetOutput = () => {
    setRespuesta("");
    setFuentes([]);
    setRetrieved([]);
    setNoEvidence(false);
    setGateReason("");
    setUsedFallback(false);
    setShowSources(false);
  };

  const ask = async () => {
    const q = pregunta.trim();
    if (!q) return;

    setLoading(true);
    setError("");
    resetOutput();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    try {
      const data = await preguntar({
        texto: q,
        mode,
        top_k: Number(topK) || 5,
        signal: controller.signal,
      });

      // ✅ defaults defensivos (contrato estable en UI)
      setRespuesta(data?.respuesta || "");
      setFuentes(Array.isArray(data?.fuentes) ? data.fuentes : []);
      setRetrieved(Array.isArray(data?.retrieved) ? data.retrieved : []);
      setNoEvidence(Boolean(data?.no_evidence));
      setGateReason(data?.gate_reason || "");
      setUsedFallback(Boolean(data?.used_fallback));

      // si backend devuelve mode, úsalo
      if (data?.mode) setMode(data.mode);
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

          {badge && <span className={`badge badge--${badge.tone}`}>{badge.text}</span>}
        </header>

        <main className="grid">
          {/* ---------------- CARD 1: Consulta ---------------- */}
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
              {/* ✅ selector de modo */}
              <select
                className="btn"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                disabled={loading}
                title="Modo"
              >
                <option value="baseline">baseline</option>
                <option value="local">local</option>
                <option value="llm">llm</option>
              </select>

              {/* ✅ top_k */}
              <input
                className="btn"
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                disabled={loading}
                title="top_k"
                style={{ width: 90 }}
              />

              <button className="btn btn--primary" onClick={ask} disabled={loading}>
                {loading ? "Consultando…" : "Consultar"}
              </button>

              <button
                className="btn"
                onClick={() => {
                  setPregunta("");
                  resetOutput();
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

          {/* ---------------- CARD 2: Respuesta + Debug ---------------- */}
          <section className="card">
            <div className="card__title">
              <h2>Respuesta</h2>
              {fuentes.length > 0 && (
                <button className="link" onClick={() => setShowSources((s) => !s)}>
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

            {/* ✅ Panel Debug */}
            {(respuesta || retrieved.length > 0 || noEvidence || gateReason || usedFallback) && (
              <div className="sources" style={{ marginTop: 12 }}>
                <details>
                  <summary>Debug</summary>

                  <div className="meta" style={{ marginTop: 10, flexWrap: "wrap" }}>
                    <span><b>mode</b>: {String(mode)}</span>
                    <span><b>no_evidence</b>: {String(noEvidence)}</span>
                    <span><b>gate_reason</b>: {gateReason || "-"}</span>
                    <span><b>used_fallback</b>: {String(usedFallback)}</span>
                    <span><b>top_k</b>: {String(topK)}</span>
                  </div>

                  <div className="sources" style={{ marginTop: 10 }}>
                    <details>
                      <summary>Retrieved ({retrieved.length})</summary>
                      <pre>{JSON.stringify(retrieved || [], null, 2)}</pre>
                    </details>
                  </div>

                  <div className="sources" style={{ marginTop: 10 }}>
                    <details>
                      <summary>Fuentes ({fuentes.length})</summary>
                      <pre>{JSON.stringify(fuentes || [], null, 2)}</pre>
                    </details>
                  </div>
                </details>
              </div>
            )}

            {/* Fuentes “bonitas” colapsables (tu UI original) */}
            {showSources && fuentes.length > 0 && (
              <div className="sources">
                <ol>
                  {fuentes.map((f, i) => (
                    <li key={i}>
                      <details>
                        <summary>Fuente {i + 1}</summary>
                        <pre>{typeof f === "string" ? f : JSON.stringify(f, null, 2)}</pre>
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