"""
Estimate the ratio of TOP (viral) trending posts that are 0-1h old.

Samples across many trending topics (broad, shallow) so the ratio isn't biased
by one topic: top N trends x ~1 page of Top posts each, deduped by tweet id.
"""
import json
import time
import pathlib
from datetime import datetime, timezone, timedelta

from crawl_trending import BASE, load_key, get_json, trend_fields

N_TOPICS = 12          # how many trending topics to sample
PAGES_PER_TOPIC = 1    # ~20 Top posts per page (the most viral ones)
WOEID = 1              # worldwide
ICT = timezone(timedelta(hours=7))
ROOT = pathlib.Path(__file__).parent


def parse_created(s):
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None


def main():
    headers = {"x-api-key": load_key()}
    now = datetime.now(timezone.utc)
    print(f"[now] {now:%a %b %d %H:%M} UTC = {now.astimezone(ICT):%H:%M} ICT\n")

    tj = get_json(f"{BASE}/twitter/trends", headers, params={"woeid": WOEID, "count": N_TOPICS})
    trends = tj.get("trends") or tj.get("data", {}).get("trends") or tj.get("data") or []
    topics = [(n, q) for n, q in (trend_fields(t) for t in trends) if n][:N_TOPICS]

    seen = set()
    all_posts = []          # (topic, tweet)
    per_topic = []          # (topic, n, n_fresh)

    for topic, query in topics:
        collected, cursor = [], ""
        for _ in range(PAGES_PER_TOPIC):
            params = {"query": query, "queryType": "Top"}
            if cursor:
                params["cursor"] = cursor
            sj = get_json(f"{BASE}/twitter/tweet/advanced_search", headers, params=params)
            batch = sj.get("tweets", [])
            collected.extend(batch)
            if not sj.get("has_next_page") or not sj.get("next_cursor"):
                break
            cursor = sj["next_cursor"]
            time.sleep(0.2)

        n_fresh = 0
        kept = 0
        for p in collected:
            tid = p.get("id")
            if tid in seen:
                continue
            seen.add(tid)
            kept += 1
            all_posts.append((topic, p))
            dt = parse_created(p.get("createdAt", ""))
            if dt is not None and (now - dt).total_seconds() / 3600 < 1:
                n_fresh += 1
        per_topic.append((topic, kept, n_fresh))
        print(f"  {topic[:24]:<24} {kept:>3} post | 0-1h: {n_fresh}")

    # aggregate
    total = len(all_posts)
    buckets = {"0-1h": 0, "1-3h": 0, "3-6h": 0, "6-24h": 0, ">24h": 0, "unknown": 0}
    for _, p in all_posts:
        dt = parse_created(p.get("createdAt", ""))
        if dt is None:
            buckets["unknown"] += 1
            continue
        h = (now - dt).total_seconds() / 3600
        if h < 1:   buckets["0-1h"] += 1
        elif h < 3: buckets["1-3h"] += 1
        elif h < 6: buckets["3-6h"] += 1
        elif h < 24: buckets["6-24h"] += 1
        else:        buckets[">24h"] += 1

    (ROOT / "data" / "freshness_sample.json").write_text(
        json.dumps({"now_utc": now.isoformat(), "n_topics": len(topics),
                    "total_posts": total, "buckets": buckets,
                    "per_topic": [{"topic": t, "n": n, "fresh_0_1h": f} for t, n, f in per_topic]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    fresh = buckets["0-1h"]
    pct = 100 * fresh / total if total else 0
    print(f"\n{'='*48}")
    print(f"  MAU: {total} top-viral post tu {len(topics)} topic trending")
    print(f"  Tao trong 0-1h: {fresh}  ->  TY LE = {pct:.1f}%  ({fresh}/{total})")
    print(f"{'='*48}")
    print("\n  Phan bo tuoi:")
    for k in ["0-1h", "1-3h", "3-6h", "6-24h", ">24h", "unknown"]:
        n = buckets[k]
        p = 100 * n / total if total else 0
        print(f"    {k:>7}: {n:>3} ({p:4.1f}%)  {'#' * n}")


if __name__ == "__main__":
    main()
