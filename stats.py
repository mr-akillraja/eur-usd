import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BREAK_EVEN_RATE = 0.549


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + (z ** 2) / total
    center = (p + (z ** 2) / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) / total) + (z ** 2) / (4 * total ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _bucket_terciles(rows):
    values = []
    for row in rows:
        if row.get("atr_1m"):
            try:
                values.append(float(row["atr_1m"]))
            except (TypeError, ValueError):
                pass
    if not values:
        return defaultdict(lambda: "unknown")
    ordered = sorted(values)
    if len(ordered) < 3:
        return {r["row_id"]: "single" for r in rows}
    lower = ordered[len(ordered) // 3]
    upper = ordered[(2 * len(ordered)) // 3]
    buckets = {}
    for row in rows:
        try:
            atr = float(row.get("atr_1m") or 0)
        except (TypeError, ValueError):
            atr = 0
        if atr <= lower:
            buckets[row.get("row_id", id(row))] = "low"
        elif atr <= upper:
            buckets[row.get("row_id", id(row))] = "mid"
        else:
            buckets[row.get("row_id", id(row))] = "high"
    return buckets


def compute_stats(rows, key_name, key_value, bucket_name=None, bucket_value=None):
    outcomes = [int(row.get("result") == "win") if row.get("result") in {"win", "loss"} else None for row in rows]
    valid = [x for x in outcomes if x is not None]
    wins = sum(valid)
    n = len(valid)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ev_per_trade": 0.0, "too_few_samples": True, "ci_lower_exceeds_breakeven": False}
    ci_low, ci_high = wilson_interval(wins, n)
    ev = (wins * 0.82) - ((n - wins) * 1.0)
    ev_per_trade = ev / n
    return {
        "group": key_name,
        "value": key_value,
        "n": n,
        "win_rate": wins / n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ev_per_trade": ev_per_trade,
        "too_few_samples": n < 100,
        "ci_lower_exceeds_breakeven": ci_low > BREAK_EVEN_RATE,
    }


def summarize(rows):
    groupers = [
        ("gate_name", lambda r: r.get("gate_name", "unknown")),
        ("session", lambda r: r.get("session", "unknown")),
        ("atr_bucket", lambda r: "unknown"),
        ("range_ratio_bucket", lambda r: "unknown"),
        ("rule_combo", lambda r: r.get("passed_rules", "") + "|" + r.get("failed_rules", "")),
    ]
    results = []
    for name, fn in groupers:
        groups = defaultdict(list)
        for row in rows:
            key = fn(row)
            groups[key].append(row)
        for key, subset in sorted(groups.items()):
            results.append(compute_stats(subset, name, key))
    return results


def main():
    path = Path("data/decisions.csv")
    rows = _load_rows(path)
    if not rows:
        print("No data/decisions.csv entries found.")
        return
    summaries = summarize(rows)
    for stats in summaries:
        status = "too few samples" if stats["too_few_samples"] else "ok"
        print(json.dumps({
            "group": stats["group"],
            "value": stats["value"],
            "n": stats["n"],
            "win_rate": stats["win_rate"],
            "ci_95": [stats["ci_low"], stats["ci_high"]],
            "ev_per_trade": stats["ev_per_trade"],
            "status": status,
            "ci_lower_exceeds_break_even_54.9": stats["ci_lower_exceeds_breakeven"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
