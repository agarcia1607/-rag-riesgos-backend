import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000/preguntar"
NO_EVIDENCE_TEXT = "No se encontró evidencia suficiente en los documentos."


@dataclass
class Example:
    id: str
    question: str
    label: str  # "ANSWERABLE" | "NO_EVIDENCE"
    evidence_chunk_ids: List[int]
    must_include: List[str]


def load_jsonl(path: Path) -> List[Example]:
    exs: List[Example] = []
    text = path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        exs.append(
            Example(
                id=obj["id"],
                question=obj["question"],
                label=obj["label"],
                evidence_chunk_ids=list(obj.get("evidence_chunk_ids", [])),
                must_include=list(obj.get("must_include", [])),
            )
        )
    return exs


def call_api(api_url: str, question: str, mode: str, timeout_s: int = 120) -> Dict[str, Any]:
    r = requests.post(api_url, json={"texto": question, "mode": mode}, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def precision_recall_at_k(
    retrieved_chunk_ids: List[int], gold_chunk_ids: List[int], k: int
) -> Tuple[float, float, int]:
    topk = retrieved_chunk_ids[:k]
    gold = set(gold_chunk_ids)
    if not gold:
        return 0.0, 0.0, 0
    hits = sum(1 for cid in topk if cid in gold)
    precision = hits / max(1, len(topk))
    recall = hits / max(1, len(gold))
    hit_at_k = 1 if hits > 0 else 0
    return precision, recall, hit_at_k


def must_include_ok(answer: str, must_include: List[str]) -> bool:
    a = (answer or "").lower()
    return all(term.lower() in a for term in must_include)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(p * (len(xs) - 1))
    return xs[idx]


def evaluate(
    dataset_path: str,
    mode: str,
    api_url: str = DEFAULT_API_URL,
    k_list: Tuple[int, ...] = (1, 3, 5),
    save_dir: str = "eval/runs",
    timeout_s: int = 120,
) -> Dict[str, Any]:
    dataset = load_jsonl(Path(dataset_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(save_dir) / f"run_{run_id}_{mode}.json"

    results = []
    agg = {
        "n": 0,
        "answerable_n": 0,
        "no_evidence_n": 0,
        "no_evidence_correct": 0,
        "false_answers": 0,
        "false_no_evidence": 0,
        "fallback_count": 0,
        "fallback_reasons": {},
        "must_include_total": 0,
        "must_include_pass": 0,
        "p_at_k": {str(k): 0.0 for k in k_list},
        "r_at_k": {str(k): 0.0 for k in k_list},
        "hit_at_k": {str(k): 0 for k in k_list},
        "latencies": [],
    }

    for ex in dataset:
        t0 = time.time()
        out = call_api(api_url, ex.question, mode=mode, timeout_s=timeout_s)
        dt = time.time() - t0

        answer = out.get("respuesta", "") or ""
        no_evidence_flag = bool(out.get("no_evidence", False))
        no_evidence = no_evidence_flag or (answer.strip() == NO_EVIDENCE_TEXT)

        used_fallback = bool(out.get("used_fallback", False))
        gate_reason = out.get("gate_reason")

        retrieved = out.get("retrieved", []) or []
        retrieved_chunk_ids: List[int] = []
        for r in retrieved:
            cid = safe_int(r.get("chunk_id"))
            if cid is not None:
                retrieved_chunk_ids.append(cid)

        agg["n"] += 1
        agg["latencies"].append(dt)

        if used_fallback:
            agg["fallback_count"] += 1
            key = str(gate_reason)
            agg["fallback_reasons"][key] = agg["fallback_reasons"].get(key, 0) + 1

        if ex.label == "NO_EVIDENCE":
            agg["no_evidence_n"] += 1
            if no_evidence:
                agg["no_evidence_correct"] += 1
            else:
                agg["false_answers"] += 1

        elif ex.label == "ANSWERABLE":
            agg["answerable_n"] += 1
            if no_evidence:
                agg["false_no_evidence"] += 1

            if ex.evidence_chunk_ids:
                for k in k_list:
                    p, r, hit = precision_recall_at_k(retrieved_chunk_ids, ex.evidence_chunk_ids, k)
                    agg["p_at_k"][str(k)] += p
                    agg["r_at_k"][str(k)] += r
                    agg["hit_at_k"][str(k)] += hit

            if ex.must_include:
                agg["must_include_total"] += 1
                if must_include_ok(answer, ex.must_include):
                    agg["must_include_pass"] += 1
        else:
            raise ValueError(f"Label inválida en ejemplo {ex.id}: {ex.label}")

        results.append(
            {
                "id": ex.id,
                "question": ex.question,
                "label": ex.label,
                "mode_requested": mode,
                "mode_returned": out.get("mode", "unknown"),
                "latency_s": dt,
                "no_evidence": no_evidence,
                "no_evidence_flag": no_evidence_flag,
                "used_fallback": used_fallback,
                "gate_reason": gate_reason,
                "retrieved_chunk_ids_top10": retrieved_chunk_ids[:10],
                "gold_chunk_ids": ex.evidence_chunk_ids,
                "must_include_ok": None if not ex.must_include else must_include_ok(answer, ex.must_include),
            }
        )

    answerable = max(1, agg["answerable_n"])
    noev = max(1, agg["no_evidence_n"])
    mi_total = max(1, agg["must_include_total"])

    p50 = percentile(agg["latencies"], 0.50)
    p95 = percentile(agg["latencies"], 0.95)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "api_url": api_url,
        "mode_requested": mode,
        "n": agg["n"],
        "answerable_n": agg["answerable_n"],
        "no_evidence_n": agg["no_evidence_n"],
        "no_evidence_accuracy": agg["no_evidence_correct"] / noev,
        "false_answer_rate": agg["false_answers"] / noev,
        "false_no_evidence_rate": agg["false_no_evidence"] / answerable,
        "fallback_rate": agg["fallback_count"] / max(1, agg["n"]),
        "fallback_reasons": agg["fallback_reasons"],
        "must_include_pass_rate": agg["must_include_pass"] / mi_total,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "retrieval": {},
    }

    for k in k_list:
        summary["retrieval"][f"precision@{k}"] = agg["p_at_k"][str(k)] / answerable
        summary["retrieval"][f"recall@{k}"] = agg["r_at_k"][str(k)] / answerable
        summary["retrieval"][f"hit@{k}"] = agg["hit_at_k"][str(k)] / answerable

    payload = {"summary": summary, "results": results}
    run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Ej: eval/v1.jsonl")
    parser.add_argument("--mode", required=True, choices=["baseline", "local", "llm"])
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--save_dir", default="eval/runs")
    parser.add_argument("--k", default="1,3,5")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    k_list = tuple(int(x) for x in args.k.split(",") if x.strip())
    out = evaluate(
        args.dataset,
        mode=args.mode,
        api_url=args.api_url,
        k_list=k_list,
        save_dir=args.save_dir,
        timeout_s=args.timeout,
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved run to: {Path(args.save_dir) / ('run_' + out['summary']['run_id'] + '_' + args.mode + '.json')}")
