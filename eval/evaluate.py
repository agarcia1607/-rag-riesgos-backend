import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

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


def must_include_hits(answer: str, must_include: List[str]) -> Tuple[int, int, List[str]]:
    """
    Returns: (hits, total, missing_list)
    Match is case-insensitive substring.
    """
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


def compute_precision_recall_at_k(
    retrieved: List[Dict[str, Any]],
    evidence_chunk_ids: List[int],
    k: int,
) -> Tuple[float, float, List[int], List[int]]:
    """
    retrieved: list of {chunk_id, score, ...}
    evidence_chunk_ids: gold ids
    returns: precision@k, recall@k, retrieved_ids@k, hit_ids
    """
    gold = set(evidence_chunk_ids or [])
    top = retrieved[:k] if retrieved else []
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

    if not retrieved_ids:
        return 0.0, 0.0, [], []

    hit_ids = [cid for cid in retrieved_ids if cid in gold]
    precision = len(hit_ids) / max(len(retrieved_ids), 1)
    recall = (len(hit_ids) / max(len(gold), 1)) if gold else 0.0
    return precision, recall, retrieved_ids, hit_ids


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    xs = sorted(values)
    idx = int(p * (len(xs) - 1))
    return int(xs[idx])


# =========================
# API call
# =========================

def post_preguntar(base_url: str, question: str, mode: str, top_k: int, timeout_s: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/preguntar"
    payload = {"texto": question, "mode": mode, "top_k": top_k}

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_s)
    latency_ms = int((time.time() - t0) * 1000)

    # robust JSON parsing (even if content-type is wrong or response is HTML)
    ct = (r.headers.get("content-type") or "").lower()
    data: Dict[str, Any]
    if "application/json" in ct:
        try:
            data = r.json()
        except Exception:
            data = {"respuesta": r.text}
    else:
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

    # Retrieval metrics (only for ANSWERABLE with evidence)
    pr_list: List[float] = []
    rc_list: List[float] = []

    # Abstention (UNANSWERABLE)
    abst_total = 0
    abst_correct = 0
    abst_wrong = 0  # UNANSWERABLE but answered anyway

    # False no_evidence on ANSWERABLE
    ans_total = 0
    false_no_evidence = 0  # ANSWERABLE but system abstained

    # Must-include
    must_total_tokens = 0
    must_hits_tokens = 0
    must_questions = 0
    must_questions_all_hit = 0

    # Stability / safety checks
    answered_but_no_retrieved = 0  # answered but retrieved empty
    http_fail = 0

    latencies: List[int] = []

    for row in results:
        latencies.append(int(row.get("_latency_ms", 0)))
        if int(row.get("_http_status", 200)) >= 400:
            http_fail += 1

        label = row.get("_label") or row.get("label")
        no_evidence = bool(row.get("no_evidence", False))
        retrieved = row.get("retrieved") or []
        answered = not no_evidence

        if answered and len(retrieved) == 0:
            answered_but_no_retrieved += 1

        # UNANSWERABLE abstention accuracy
        if label == "UNANSWERABLE":
            abst_total += 1
            if no_evidence:
                abst_correct += 1
            else:
                abst_wrong += 1

        # ANSWERABLE false abstention rate
        if label == "ANSWERABLE":
            ans_total += 1
            if no_evidence:
                false_no_evidence += 1

            ev = row.get("_evidence_chunk_ids") or []
            # only measure retrieval when we have gold evidence
            if ev:
                pr_list.append(float(row.get("_precision_at_k", 0.0)))
                rc_list.append(float(row.get("_recall_at_k", 0.0)))

        mi = row.get("_must_include") or []
        if mi:
            must_questions += 1
            must_total_tokens += len(mi)
            must_hits_tokens += int(row.get("_must_hits", 0))
            if int(row.get("_must_hits", 0)) == len(mi):
                must_questions_all_hit += 1

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    out: Dict[str, Any] = {
        "n": n,
        "k": k,
        "avg_precision_at_k": (avg(pr_list) if pr_list else None),
        "avg_recall_at_k": (avg(rc_list) if rc_list else None),

        "answerable_total": ans_total,
        "false_no_evidence_rate": (false_no_evidence / max(1, ans_total)) if ans_total else None,

        "unanswerable_total": abst_total,
        "abstention_accuracy": (abst_correct / abst_total) if abst_total else None,
        "abstention_wrong_rate": (abst_wrong / abst_total) if abst_total else None,

        "must_include_token_hit_rate": (must_hits_tokens / must_total_tokens) if must_total_tokens else None,
        "must_include_all_hit_rate": (must_questions_all_hit / must_questions) if must_questions else None,

        "answered_but_no_retrieved_rate": answered_but_no_retrieved / max(1, n),
        "http_fail_rate": http_fail / max(1, n),

        "latency_ms_avg": int(sum(latencies) / max(1, len(latencies))),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
    }
    return out


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True, help="e.g. https://rag-riesgos-backend.onrender.com")
    ap.add_argument("--dataset", required=True, help="path to jsonl dataset")
    ap.add_argument("--modes", default="baseline,local", help="comma-separated modes to test")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--out_dir", default="eval/runs")
    args = ap.parse_args()

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
            label = it.get("label", "ANSWERABLE")  # "ANSWERABLE" | "UNANSWERABLE"
            evidence_chunk_ids = it.get("evidence_chunk_ids", [])
            must_include = it.get("must_include", [])

            resp = post_preguntar(
                args.base_url,
                question,
                mode,
                args.top_k,
                args.timeout_s,
            )

            retrieved = resp.get("retrieved") or []

            # Only compute retrieval if ANSWERABLE and we have gold evidence
            if label == "ANSWERABLE" and evidence_chunk_ids:
                p_at_k, r_at_k, retrieved_ids, hit_ids = compute_precision_recall_at_k(
                    retrieved=retrieved,
                    evidence_chunk_ids=evidence_chunk_ids,
                    k=args.top_k,
                )
            else:
                p_at_k, r_at_k, retrieved_ids, hit_ids = 0.0, 0.0, [], []

            mh, mt, missing = must_include_hits(resp.get("respuesta", ""), must_include)

            row = {
                "id": qid,
                "question": question,
                "label": label,
                "mode_sent": mode,
                "mode_returned": resp.get("mode"),
                "requested_mode": resp.get("requested_mode"),
                "no_evidence": resp.get("no_evidence"),
                "used_fallback": resp.get("used_fallback"),
                "gate_reason": resp.get("gate_reason"),
                "respuesta": resp.get("respuesta"),
                "fuentes": resp.get("fuentes") or [],
                "retrieved": retrieved,

                # raw transport
                "_http_status": resp.get("_http_status"),
                "_latency_ms": resp.get("_latency_ms"),

                # eval extras
                "_precision_at_k": p_at_k,
                "_recall_at_k": r_at_k,
                "_retrieved_ids": retrieved_ids,
                "_hit_ids": hit_ids,
                "_must_hits": mh,
                "_must_total": mt,
                "_must_missing": missing,
                "_label": label,
                "_evidence_chunk_ids": evidence_chunk_ids,
                "_must_include": must_include,
            }
            results.append(row)

        summary = summarize(results, args.top_k)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"run_{stamp}_{mode}.json"

        payload = {
            "meta": {
                "base_url": args.base_url,
                "dataset": args.dataset,
                "mode": mode,
                "top_k": args.top_k,
                "n": len(results),
                "timestamp_utc": stamp,
            },
            "summary": summary,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== RUN:", mode, "===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("saved:", str(out_path))

        all_runs.append({"mode": mode, "summary": summary, "file": str(out_path)})

    # index file for this batch
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    index_path = out_dir / f"index_{stamp}.json"
    index_path.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nindex:", str(index_path))


if __name__ == "__main__":
    main()