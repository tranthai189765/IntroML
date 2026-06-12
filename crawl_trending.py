"""
Crawl ~50 trending posts from Twitter/X via twitterapi.io.

Flow:
  1. GET /twitter/trends (worldwide) -> list of trending topics
  2. Pick the hottest topic -> /twitter/tweet/advanced_search (queryType=Top)
     paginate until we collect TARGET posts
  3. Save results to data/trending_posts.json and data/trending_posts.csv

API key is read from .env (key: API_KEY). Never hardcoded.
"""
import os
import re
import sys
import csv
import json
import time
import pathlib
import requests

# Windows console is cp1252 by default -> emoji/CJK in tweets crash print()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = "https://api.twitterapi.io"
TARGET = 50
WOEID = 1            # 1 = worldwide, 23424977 = US
OUTDIR = pathlib.Path(__file__).parent / "data"


def load_key() -> str:
    env = pathlib.Path(__file__).parent / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("API_KEY not found in .env")


def get_json(url, headers, params=None, tries=5):
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(tries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code in (429, 500, 502, 503):
            wait = 1.5 * (2 ** attempt)
            print(f"    [{resp.status_code}] backoff {wait:.1f}s "
                  f"(attempt {attempt + 1}/{tries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def trend_fields(entry):
    """Trends come back as {"trend": {name, target:{query}, rank}} (sometimes flat)."""
    d = entry.get("trend", entry) if isinstance(entry, dict) else {}
    if not isinstance(d, dict):
        d = entry if isinstance(entry, dict) else {}
    name = d.get("name") or d.get("trend") or ""
    query = (d.get("target") or {}).get("query") or name
    return str(name), str(query)


def main():
    key = load_key()
    headers = {"x-api-key": key}
    OUTDIR.mkdir(exist_ok=True)

    # --- 0) balance ---
    bal = get_json(f"{BASE}/oapi/my/info", headers)
    print(f"[balance] recharge={bal.get('recharge_credits')} "
          f"bonus={bal.get('total_bonus_credits')}")

    # --- 1) trends ---
    tj = get_json(f"{BASE}/twitter/trends", headers,
                  params={"woeid": WOEID, "count": 30})
    trends = tj.get("trends") or tj.get("data", {}).get("trends") or tj.get("data") or []
    parsed = [trend_fields(t) for t in trends]
    parsed = [(n, q) for (n, q) in parsed if n]
    names = [n for n, _ in parsed]
    print(f"\n[trends] worldwide top {len(names)}:")
    for i, n in enumerate(names[:15], 1):
        print(f"  {i:>2}. {n}")
    if not parsed:
        print("RAW trends response:", json.dumps(tj, ensure_ascii=False)[:800])
        raise SystemExit("No trends parsed.")

    topic, query = parsed[0]
    print(f"\n[search] collecting {TARGET} Top posts for: {topic!r}  (query={query!r})")

    # --- 2) advanced search, paginate ---
    posts, cursor, page = [], "", 0
    while len(posts) < TARGET:
        params = {"query": query, "queryType": "Top"}
        if cursor:
            params["cursor"] = cursor
        sj = get_json(f"{BASE}/twitter/tweet/advanced_search", headers, params=params)
        batch = sj.get("tweets", [])
        page += 1
        print(f"  page {page}: +{len(batch)} (total {len(posts)+len(batch)})")
        if not batch:
            break
        posts.extend(batch)
        if not sj.get("has_next_page"):
            break
        cursor = sj.get("next_cursor") or ""
        if not cursor:
            break
        time.sleep(0.3)

    posts = posts[:TARGET]
    print(f"\n[done] collected {len(posts)} posts")

    # --- 3) save ---
    out_json = OUTDIR / "trending_posts.json"
    out_csv = OUTDIR / "trending_posts.csv"
    out_json.write_text(json.dumps(
        {"topic": topic, "trends": names, "count": len(posts), "posts": posts},
        ensure_ascii=False, indent=2), encoding="utf-8")

    def author(p):
        a = p.get("author") or {}
        return a.get("userName") or a.get("screen_name") or ""

    def extract_media(p):
        """Return (image_urls, video_urls, media_types) from extendedEntities."""
        media = (p.get("extendedEntities") or {}).get("media") or []
        images, videos, types = [], [], []
        for m in media:
            t = m.get("type")
            types.append(t or "")
            if t == "photo":
                if m.get("media_url_https"):
                    images.append(m["media_url_https"])
            elif t in ("video", "animated_gif"):
                variants = [v for v in (m.get("video_info") or {}).get("variants", [])
                            if v.get("content_type") == "video/mp4" and v.get("url")]
                if variants:
                    videos.append(max(variants, key=lambda v: v.get("bitrate") or 0)["url"])
                elif m.get("media_url_https"):
                    videos.append(m["media_url_https"])
        return images, videos, types

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "author", "createdAt", "lang",
                    "likeCount", "retweetCount", "replyCount", "quoteCount",
                    "viewCount", "bookmarkCount",
                    "media_count", "media_types", "image_count", "video_count",
                    "image_urls", "video_urls", "url", "text"])
        for p in posts:
            images, videos, types = extract_media(p)
            w.writerow([
                p.get("id", ""), author(p), p.get("createdAt", ""), p.get("lang", ""),
                p.get("likeCount", ""), p.get("retweetCount", ""),
                p.get("replyCount", ""), p.get("quoteCount", ""),
                p.get("viewCount", ""), p.get("bookmarkCount", ""),
                len(types), "|".join(types), len(images), len(videos),
                " | ".join(images), " | ".join(videos),
                p.get("url", ""),
                (p.get("text", "") or "").replace("\n", " ").strip(),
            ])

    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")

    # preview top 5 by likes
    def likes(p):
        try:
            return int(p.get("likeCount") or 0)
        except (TypeError, ValueError):
            return 0
    print("\n[preview] top 5 by likes:")
    for p in sorted(posts, key=likes, reverse=True)[:5]:
        txt = (p.get("text", "") or "").replace("\n", " ")[:80]
        print(f"  @{author(p):<16} ❤{likes(p):>7}  {txt}")


if __name__ == "__main__":
    main()
