import os
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))

        # Defaults “para producción” (rápidos, deterministas y cortos)
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))  # <- clave para latencia
        self.top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))
        self.repeat_penalty = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.05"))

        # stop tokens (separados por | en env)
        stop_env = os.getenv("OLLAMA_STOP", "").strip()
        if stop_env:
            self.stop = [s for s in stop_env.split("|") if s]
        else:
            # default razonable: cortar cuando el modelo empieza a “explicar”
            self.stop = ["\n\n", "\n- ", "\n• "]

        # Mantener el modelo caliente (reduce latencia entre llamadas)
        # Ej: "5m", "10m", "30m" o "-1" para siempre
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Genera una respuesta usando Ollama.
        Puedes sobre-escribir opciones pasando `options`.
        """
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                    "top_p": self.top_p,
                    "repeat_penalty": self.repeat_penalty,
                    # stop dentro de options funciona bien en Ollama
                    "stop": self.stop,
                },
            }

            # override opcional por llamada (experimentos)
            if options:
                payload["options"].update(options)

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            resp.raise_for_status()
            data = resp.json()
            return (data.get("response", "") or "").strip()

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Error: el modelo local tardó demasiado en responder."

        except requests.exceptions.ConnectionError:
            logger.error("No se pudo conectar con Ollama")
            return "Error: no se pudo conectar con Ollama en local."

        except Exception as e:
            logger.exception("Error inesperado en Ollama")
            return f"Error inesperado: {str(e)}"