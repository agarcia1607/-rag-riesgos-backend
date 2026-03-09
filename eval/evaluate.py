import argparse
import json
import os
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

try:
    from mlflow_logger import log_run_to_mlflow
except Exception:
    log_run_to_mlflow = None


# =========================
# Utils
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def safe_lower(s: str) -> str:
    return (s or "").lower()


def normalize_semantic_label(x: str) -> str:
    """
    'datos_generales' -> 'datos generales'
    """
    s = (x or "").strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def must_include_hits(answer: str, must_include: List[str]) -> Tuple[int, int, List[str]]:
    ans = safe_lower(answer)
    missing = []
    hits = 0
    for token in must_include or []:
        if safe_lower(token) in ans:
            hits += 1
        else:
            missing.append(token)
    total = len(must_include or [])
    return hits, total, missing


def compute_retrieval_semantic_at_k(
    retrieved: List[Dict[str, Any]],
    gold_evidence: List[Any],
    k: int,
) -> Tuple[Optional[float], Optional[float], List[Any], List[Any]]:
    """
    Soporta dos tipos de gold:
    1) numéricos -> compara contra chunk_id
    2) strings semánticos -> compara contra texto recuperado

    Devuelve:
    precision_at_k, recall_at_k, retrieved_markers, hit_markers
    """
    if not gold_evidence:
        return None, None, [], []

    top = retrieved[:k] if retrieved else []
    if not top:
        return 0.0, 0.0, [], []

    # caso 1: gold numérico
    if all(isinstance(g, int) or (isinstance(g, str) and g.isdigit()) for g in gold_evidence):
        gold_ids = set(int(g) for g in gold_evidence)
        retrieved_ids: List[int] = []

        for r in top:
            cid = r.get("chunk_id")
            if isinstance(cid, int):
                retrieved_ids.append(cid)
            else:
                try:
                    retrieved_ids.append(int(cid))
                except Exception:
                    pass

        hit_ids = [cid for cid in retrieved_ids if cid in gold_ids]
        precision = len(hit_ids) / max(len(retrieved_ids), 1)
        recall = len(hit_ids) / max(len(gold_ids), 1) if gold_ids else 0.0
        return precision, recall, retrieved_ids, hit_ids

    # caso 2: gold semántico textual
    retrieved_texts = [safe_lower(r.get("text", "")) for r in top]
    gold_labels = [normalize_semantic_label(str(g)) for g in gold_evidence]

    hit_labels: List[str] = []
    for gold in gold_labels:
        if any(gold in txt for txt in retrieved_texts):
            hit_labels.append(gold)

    precision = len(hit_labels) / max(len(top), 1)
    recall = len(hit_labels) / max(len(gold_labels), 1)
    return precision, recall, gold_labels, hit_labels


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    xs = sorted(values)
    idx = int(p * (len(xs) - 1))
    return int(xs[idx])


def infer_retriever_type(retrieved: List[Dict[str, Any]]) -> Optional[str]:
    if not retrieved:
        return None
    for r in retrieved:
        t = r.get("retriever")
        if t:
            return str(t)
    return None


# =========================
# API call
# =========================

def post_preguntar(base_url: str, question: str, mode: str, k: int, timeout_s: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/preguntar"
    payload = {
        "texto": question,
        "mode": mode,
        "top_k": k,
        "k": k,  # compatibilidad futura si el backend empieza a usar `k`
    }

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_s)
    latency_ms = int((time.time() - t0) * 1000)

    try:
        data = r.json()
    except Exception:
        data = {"respuesta": r.text}

    data["_http_status"] = r.status_code
    data["_latency_ms"] = latency_ms
    return data


# =========================
# Summaries
# =========================

def summarize(results: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    pr_list: List[float] = []
    rc_list: List[float] = []
    retrieval_eligible_n = 0

    abst_total = 0
    abst_correct = 0
    abst_wrong = 0

    ans_total = 0
    false_no_evidence = 0

    must_total_tokens = 0
    must_hits_tokens = 0
    must_questions = 0
    must_questions_all_hit = 0

    answered_but_no_retrieved = 0
    http_fail = 0
    bad_answer = 0

    latencies: List[int] = []
    retriever_types_seen: set[str] = set()

    for row in results:
        latencies.append(int(row.get("_latency_ms", 0)))
        if int(row.get("_http_status", 200)) >= 400:
            http_fail += 1

        retr_type = row.get("_retriever_type")
        if retr_type:
            retriever_types_seen.add(str(retr_type))

        label = row.get("_label") or row.get("label")
        no_evidence = bool(row.get("no_evidence", False))
        retrieved = row.get("retrieved") or []
        answered = not no_evidence

        respuesta = (row.get("respuesta") or "").strip().lower()
        if answered and (
            respuesta.startswith("error:")
            or "traceback" in respuesta
            or "exception" in respuesta
        ):
            bad_answer += 1

        if answered and len(retrieved) == 0:
            answered_but_no_retrieved += 1

        if label == "UNANSWERABLE":
            abst_total += 1
            if no_evidence:
                abst_correct += 1
            else:
                abst_wrong += 1

        if label == "ANSWERABLE":
            ans_total += 1
            if no_evidence:
                false_no_evidence += 1

            ev = row.get("_evidence_chunk_ids") or []
            p_at_k = row.get("_precision_at_k")
            r_at_k = row.get("_recall_at_k")
            if ev and p_at_k is not None and r_at_k is not None:
                retrieval_eligible_n += 1
                pr_list.append(float(p_at_k))
                rc_list.append(float(r_at_k))

        mi = row.get("_must_include") or []
        if mi:
            must_questions += 1
            must_total_tokens += len(mi)
            must_hits_tokens += int(row.get("_must_hits", 0))
            if int(row.get("_must_hits", 0)) == len(mi):
                must_questions_all_hit += 1

    def avg(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "n": n,
        "k": k,
        "retrieval_eligible_n": retrieval_eligible_n,
        "avg_precision_at_k": avg(pr_list),
        "avg_recall_at_k": avg(rc_list),
        "answerable_total": ans_total,
        "false_no_evidence_rate": (false_no_evidence / max(1, ans_total)) if ans_total else None,
        "unanswerable_total": abst_total,
        "abstention_accuracy": (abst_correct / abst_total) if abst_total else None,
        "abstention_wrong_rate": (abst_wrong / abst_total) if abst_total else None,
        "must_include_token_hit_rate": (must_hits_tokens / must_total_tokens) if must_total_tokens else None,
        "must_include_all_hit_rate": (must_questions_all_hit / must_questions) if must_questions else None,
        "answered_but_no_retrieved_rate": answered_but_no_retrieved / max(1, n),
        "bad_answer_rate": bad_answer / max(1, n),
        "http_fail_rate": http_fail / max(1, n),
        "latency_ms_avg": int(sum(latencies) / max(1, len(latencies))),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
        "retriever_types_seen": sorted(retriever_types_seen),
    }


# =========================
# MLflow helpers
# =========================

def build_extra_params(args: argparse.Namespace, results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    baseline_version = None
    retriever_type = None

    if results:
        baseline_version = results[0].get("baseline_version")
        retriever_type = results[0].get("_retriever_type")

    return {
        "mode": mode,
        "baseline_version": baseline_version or os.getenv("BASELINE_VERSION"),
        "retriever_type": retriever_type or os.getenv("RETRIEVER_TYPE"),
        "hybrid_alpha": os.getenv("HYBRID_ALPHA"),
        "TOP_N_FOR_LLM": os.getenv("TOP_N_FOR_LLM"),
        "MAX_CONTEXT_CHARS": os.getenv("MAX_CONTEXT_CHARS"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        "OLLAMA_NUM_PREDICT": os.getenv("OLLAMA_NUM_PREDICT"),
        "OLLAMA_TEMPERATURE": os.getenv("OLLAMA_TEMPERATURE"),
        "OLLAMA_TOP_P": os.getenv("OLLAMA_TOP_P"),
        "OLLAMA_TIMEOUT": os.getenv("OLLAMA_TIMEOUT"),
        "timeout_s": args.timeout_s,
        "k": args.k,
        "limit": args.limit,
        "base_url": args.base_url,
    }


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--modes", default="baseline,local")

    # nuevo: k como parámetro principal
    ap.add_argument("--k", type=int, default=None, help="Top-k retrieval")
    # compatibilidad con scripts viejos
    ap.add_argument("--top_k", type=int, default=5, help="Top-k retrieval (legacy)")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--out_dir", default="eval/runs")

    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow_experiment", default="rag_riesgos_eval")

    args = ap.parse_args()

    # prioriza --k si fue pasado; si no, usa --top_k
    if args.k is None:
        args.k = args.top_k

    items = load_jsonl(args.dataset)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    for mode in modes:
        results: List[Dict[str, Any]] = []

        for it in items:
            qid = it.get("id")
            question = it.get("question", "")
            label = it.get("label", "ANSWERABLE")
            evidence_gold = it.get("evidence_chunk_ids", []) or []
            must_include = it.get("must_include", []) or []

            resp = post_preguntar(
                args.base_url,
                question,
                mode,
                args.k,
                args.timeout_s,
            )

            retrieved = resp.get("retrieved") or []
            retriever_type = infer_retriever_type(retrieved)

            if label == "ANSWERABLE" and evidence_gold:
                p_at_k, r_at_k, retrieved_markers, hit_markers = compute_retrieval_semantic_at_k(
                    retrieved=retrieved,
                    gold_evidence=evidence_gold,
                    k=args.k,
                )
            else:
                p_at_k, r_at_k, retrieved_markers, hit_markers = None, None, [], []

            mh, mt, missing = must_include_hits(resp.get("respuesta", ""), must_include)

            row = {
                "id": qid,
                "question": question,
                "label": label,
                "mode_sent": mode,
                "mode_returned": resp.get("mode"),
                "requested_mode": resp.get("requested_mode"),
                "baseline_version": resp.get("baseline_version"),
                "no_evidence": resp.get("no_evidence"),
                "used_fallback": resp.get("used_fallback"),
                "gate_reason": resp.get("gate_reason"),
                "respuesta": resp.get("respuesta"),
                "fuentes": resp.get("fuentes") or [],
                "retrieved": retrieved,

                "_http_status": resp.get("_http_status"),
                "_latency_ms": resp.get("_latency_ms"),
                "_retriever_type": retriever_type,

                "_precision_at_k": p_at_k,
                "_recall_at_k": r_at_k,
                "_retrieved_markers": retrieved_markers,
                "_hit_markers": hit_markers,
                "_must_hits": mh,
                "_must_total": mt,
                "_must_missing": missing,
                "_label": label,
                "_evidence_chunk_ids": evidence_gold,
                "_must_include": must_include,
            }
            results.append(row)

        summary = summarize(results, args.k)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"run_{stamp}_{mode}.json"

        payload = {
            "meta": {
                "base_url": args.base_url,
                "dataset": args.dataset,
                "mode": mode,
                "k": args.k,
                "retriever_type_env": os.getenv("RETRIEVER_TYPE"),
                "hybrid_alpha_env": os.getenv("HYBRID_ALPHA"),
                "n": len(results),
                "timestamp_utc": stamp,
            },
            "summary": summary,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n=== RUN: {mode} ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("saved:", str(out_path))

        all_runs.append({"mode": mode, "summary": summary, "file": str(out_path)})

        if args.mlflow:
            if log_run_to_mlflow is None:
                print("warning: MLflow logger no disponible. Revisa eval/mlflow_logger.py")
            else:
                extra_params = build_extra_params(args, results, mode)
                try:
                    log_run_to_mlflow(
                        experiment_name=args.mlflow_experiment,
                        mode=mode,
                        summary=summary,
                        dataset_path=args.dataset,
                        run_json_path=str(out_path),
                        index_json_path=None,
                        extra_params=extra_params,
                    )
                except Exception as e:
                    print(f"warning: no se pudo loggear en MLflow para mode={mode}: {e}")

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    index_path = out_dir / f"index_{stamp}.json"
    index_path.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nindex:", str(index_path))

    if args.mlflow and log_run_to_mlflow is not None:
        try:
            batch_summary = {
                "batch_num_modes": len(modes),
                "batch_num_items": len(items),
            }
            extra_params = {
                "modes": ",".join(modes),
                "k": args.k,
                "timeout_s": args.timeout_s,
                "limit": args.limit,
                "base_url": args.base_url,
                "retriever_type": os.getenv("RETRIEVER_TYPE"),
                "hybrid_alpha": os.getenv("HYBRID_ALPHA"),
            }
            log_run_to_mlflow(
                experiment_name=args.mlflow_experiment,
                mode="batch_index",
                summary=batch_summary,
                dataset_path=args.dataset,
                run_json_path=str(index_path),
                index_json_path=str(index_path),
                extra_params=extra_params,
            )
        except Exception as e:
            print(f"warning: no se pudo loggear batch index en MLflow: {e}")


if __name__ == "__main__":
    main()