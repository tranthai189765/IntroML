"""
Enrich data_v1/dataset_aligned.csv with AUTHOR metadata (strong virality signal).

The DB only stores the author *username*, so the real author signals
(followers, verified, account age, total tweets) must be fetched once from
twitterapi.io  /twitter/user/info?userName=<name>.

  - fetches each unique author (threaded, retry on 429/network), caches to
    data_v1/authors.json so re-runs are FREE
  - merges per-author stats + derived features into the CSV (new columns)

New columns added to dataset_aligned.csv:
  author_followers          followers count (raw)
  author_following          following count
  author_statuses           total tweets ever
  author_favourites         likes the author has given
  author_blue_verified      1/0  (Twitter Blue / paid checkmark)
  author_verified           1/0  (legacy verified)
  author_age_days           account age in days (now - createdAt)
  author_log_followers      ln(1+followers)            <- strongest single feature
  author_followers_per_day  followers / max(age_days,1)  (growth-rate proxy)
  author_ff_ratio           followers / (following+1)   (influence vs. follow-back)
  author_found              1 if profile fetched ok, 0 if deleted/suspended/error

Run:  python add_author_meta.py
SECURITY: API key read from .env (env var) — never hard-coded/printed.
"""
import os
import sys
import csv
import json
import time
import math
import pathlib
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

CSV = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data_v1/dataset_aligned.csv")
CACHE = pathlib.Path("data_v1/authors.json")
BASE = "https://api.twitterapi.io"
WORKERS = 8
NOW = dt.datetime.now(dt.timezone.utc)


def api_key():
    k = os.environ.get("API_KEY")
    if not k:
        for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("API_KEY not found (.env or env var)")
    return k


def fetch_one(sess, key, name):
    """Return user dict or None (deleted/suspended). Retries on 429/network."""
    url = f"{BASE}/twitter/user/info"
    for attempt in range(5):
        try:
            r = sess.get(url, params={"userName": name},
                         headers={"x-api-key": key}, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            d = r.json()
            u = d.get("data") or (d if "userName" in d else None)
            return u
        except requests.exceptions.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def age_days(created):
    if not created:
        return 0
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%SZ"):
        try:
            d = dt.datetime.strptime(created, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return max(0, (NOW - d).days)
        except ValueError:
            continue
    return 0


def features(u):
    """Map a raw user dict -> the columns we store."""
    if not u:
        return dict(author_followers=0, author_following=0, author_statuses=0,
                    author_favourites=0, author_blue_verified=0, author_verified=0,
                    author_age_days=0, author_log_followers=0.0,
                    author_followers_per_day=0.0, author_ff_ratio=0.0,
                    author_found=0)
    fo = int(u.get("followers") or 0)
    fg = int(u.get("following") or 0)
    ad = age_days(u.get("createdAt"))
    return dict(
        author_followers=fo,
        author_following=fg,
        author_statuses=int(u.get("statusesCount") or 0),
        author_favourites=int(u.get("favouritesCount") or 0),
        author_blue_verified=int(bool(u.get("isBlueVerified"))),
        author_verified=int(bool(u.get("isVerified"))),
        author_age_days=ad,
        author_log_followers=round(math.log1p(fo), 4),
        author_followers_per_day=round(fo / max(ad, 1), 3),
        author_ff_ratio=round(fo / (fg + 1), 3),
        author_found=1,
    )


COLS = ["author_followers", "author_following", "author_statuses",
        "author_favourites", "author_blue_verified", "author_verified",
        "author_age_days", "author_log_followers", "author_followers_per_day",
        "author_ff_ratio", "author_found"]


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    authors = sorted({r["author"] for r in rows})
    print(f"[csv] {len(rows)} posts | {len(authors)} unique authors")

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [a for a in authors if a not in cache]
    print(f"[fetch] cached={len(cache)} | to fetch={len(todo)}")

    if todo:
        key = api_key()
        sess = requests.Session()
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(fetch_one, sess, key, a): a for a in todo}
            for fu in as_completed(futs):
                a = futs[fu]
                cache[a] = features(fu.result())
                done += 1
                if done % 100 == 0 or done == len(todo):
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False),
                                     encoding="utf-8")
                    print(f"  {done}/{len(todo)} fetched")
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    found = sum(1 for a in authors if cache.get(a, {}).get("author_found"))
    print(f"[fetch] profiles found: {found}/{len(authors)} "
          f"({len(authors)-found} deleted/suspended/private)")

    # ---- merge into CSV (append author columns) ----
    in_header = rows[0].keys() if rows else []
    base_cols = [c for c in in_header if c not in COLS]
    header = base_cols + COLS
    with open(CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            feat = cache.get(r["author"]) or features(None)
            out = {c: r[c] for c in base_cols}
            out.update(feat)
            w.writerow(out)
    print(f"[saved] {CSV}  (+{len(COLS)} author columns, total {len(header)} cols)")

    # quick sanity: do followers separate the virality classes?
    import statistics as st
    by = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        fo = cache.get(r["author"], {}).get("author_followers", 0)
        by[int(r["label"])].append(fo)
    print("[check] median followers by final label:")
    for lab in (0, 1, 2, 3):
        v = by[lab]
        print(f"  label {lab}: n={len(v):>4}  median_followers="
              f"{int(st.median(v)) if v else 0:,}")


if __name__ == "__main__":
    main()
