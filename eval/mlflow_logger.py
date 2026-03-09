import hashlib
import os
import subprocess
from typing import Dict, Any, Optional

import mlflow


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _build_run_name(mode: str, extra_params: Optional[Dict[str, Any]] = None) -> str:
    extra_params = extra_params or {}

    if mode == "baseline":
        baseline_version = extra_params.get("baseline_version")
        if baseline_version:
            return f"baseline_{baseline_version}"
        return "baseline_default"

    if mode == "local":
        model = extra_params.get("OLLAMA_MODEL") or "ollama"
        top_n = extra_params.get("TOP_N_FOR_LLM") or "na"
        ctx = extra_params.get("MAX_CONTEXT_CHARS") or "na"
        return f"local_{model}_top{top_n}_ctx{ctx}"

    if mode == "gemini":
        model = extra_params.get("GEMINI_MODEL") or "gemini"
        return f"gemini_{model}"

    if mode == "hybrid":
        retriever = extra_params.get("RETRIEVER") or "hybrid"
        model = extra_params.get("OLLAMA_MODEL") or extra_params.get("GEMINI_MODEL") or "llm"
        return f"hybrid_{retriever}_{model}"

    if mode == "batch_index":
        modes = extra_params.get("modes") or "batch"
        return f"batch_{modes}"

    return f"{mode}_eval"


def log_run_to_mlflow(
    *,
    experiment_name: str,
    mode: str,
    summary: Dict[str, Any],
    dataset_path: str,
    run_json_path: str,
    index_json_path: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    mlflow.set_experiment(experiment_name)

    run_name = _build_run_name(mode, extra_params)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("mode", mode)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_hash", sha256_file(dataset_path))

        commit = git_commit_sha()
        if commit:
            mlflow.log_param("git_commit", commit)

        if extra_params:
            for k, v in extra_params.items():
                if v is not None:
                    mlflow.log_param(k, str(v))

        for k, v in summary.items():
            if isinstance(v, (int, float)) and v is not None:
                mlflow.log_metric(k, float(v))

        if os.path.exists(run_json_path):
            mlflow.log_artifact(run_json_path, artifact_path="eval_runs")

        if index_json_path and os.path.exists(index_json_path):
            mlflow.log_artifact(index_json_path, artifact_path="eval_runs")