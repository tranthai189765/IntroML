"""
Crawl 50 trending (Top) posts NOW and bucket them by age relative to current time.
Answers: how many of the 50 were created in the last 0-1h?

Reuses helpers from crawl_trending.py. Saves a fresh snapshot to
data/trending_fresh.json (separate file so it won't clash with open CSVs).
"""
import json
import time
import pathlib
from datetime import datetime, timezone, timedelta

from crawl_trending import BASE, load_key, get_json, trend_fields

TARGET = 50
WOEID = 1  # worldwide
ICT = timezone(timedelta(hours=7))
ROOT = pathlib.Path(__file__).parent


def parse_created(s):
    # Twitter format: "Mon Jun 08 11:19:00 +0000 2026"
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None


def main():
    key = load_key()
    headers = {"x-api-key": key}
    now = datetime.now(timezone.utc)
    print(f"[now] {now:%a %b %d %H:%M:%S} UTC  =  "
          f"{now.astimezone(ICT):%H:%M} ICT (UTC+7)\n")

    # trends -> top topic
    tj = get_json(f"{BASE}/twitter/trends", headers, params={"woeid": WOEID, "count": 30})
    trends = tj.get("trends") or tj.get("data", {}).get("trends") or tj.get("data") or []
    parsed = [trend_fields(t) for t in trends]
    parsed = [(n, q) for n, q in parsed if n]
    topic, query = parsed[0]
    print(f"[topic] #1 worldwide trend: {topic!r}  -> collecting {TARGET} Top posts\n")

    # paginate advanced_search Top
    posts, cursor, page = [], "", 0
    while len(posts) < TARGET:
        params = {"query": query, "queryType": "Top"}
        if cursor:
            params["cursor"] = cursor
        sj = get_json(f"{BASE}/twitter/tweet/advanced_search", headers, params=params)
        batch = sj.get("tweets", [])
        page += 1
        print(f"  page {page}: +{len(batch)} (total {len(posts) + len(batch)})")
        if not batch:
            break
        posts.extend(batch)
        if not sj.get("has_next_page") or not sj.get("next_cursor"):
            break
        cursor = sj["next_cursor"]
        time.sleep(0.3)
    posts = posts[:TARGET]

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "trending_fresh.json").write_text(
        json.dumps({"topic": topic, "now_utc": now.isoformat(), "posts": posts},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # --- age analysis ---
    buckets = {"0-1h": 0, "1-3h": 0, "3-6h": 0, "6-24h": 0, ">24h": 0, "unknown": 0}
    fresh = []  # the 0-1h ones
    ages = []
    for p in posts:
        dt = parse_created(p.get("createdAt", ""))
        if dt is None:
            buckets["unknown"] += 1
            continue
        age_h = (now - dt).total_seconds() / 3600
        ages.append((age_h, p))
        if age_h < 1:
            buckets["0-1h"] += 1
            fresh.append((age_h, p))
        elif age_h < 3:
            buckets["1-3h"] += 1
        elif age_h < 6:
            buckets["3-6h"] += 1
        elif age_h < 24:
            buckets["6-24h"] += 1
        else:
            buckets[">24h"] += 1

    print(f"\n[result] collected {len(posts)} Top posts about {topic!r}")
    print(f"\n  ===> Tao trong 0-1h qua: {buckets['0-1h']} / {len(posts)} post\n")
    print("  Phan bo tuoi (so voi now):")
    for k in ["0-1h", "1-3h", "3-6h", "6-24h", ">24h", "unknown"]:
        bar = "#" * buckets[k]
        print(f"    {k:>7}: {buckets[k]:>2}  {bar}")

    if ages:
        ages.sort(key=lambda x: x[0])
        youngest, oldest = ages[0][0], ages[-1][0]
        print(f"\n  Newest post: {youngest*60:.0f} phut truoc | "
              f"Oldest post: {oldest:.1f}h truoc")

    if fresh:
        print(f"\n  Chi tiet {len(fresh)} post moi (0-1h):")
        for age_h, p in sorted(fresh):
            a = (p.get("author") or {}).get("userName", "")
            txt = (p.get("text", "") or "").replace("\n", " ")[:60]
            print(f"    {age_h*60:>4.0f} phut  @{a:<16} ❤{p.get('likeCount',0):>6}  {txt}")
    else:
        print("\n  (Khong co post nao 0-1h — dung voi dac diem 'Top': post hot da co tuoi)")


if __name__ == "__main__":
    main()
