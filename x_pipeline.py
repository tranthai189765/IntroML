"""
X (Twitter) online-learning data pipeline — twitterapi.io.

Collects fresh (0-1h) viral candidate posts from trending topics, spam-filters &
dedups them, downloads their images, and tracks each post's engagement over its
first 24h on a log-spaced snapshot schedule. Built for online-learning datasets
(text + image features -> future engagement labels).

Design (see memory: x-online-learning-intake-window):
  - INTAKE: trends (multi-region) -> advanced_search `<topic> since_time:<now-1h>`
            queryType=Top -> only posts aged 0-1h. NO min_faves filter (we can't
            know virality at age 0-1h; that's the label to predict).
  - SPAM:   drop near-duplicate text clusters (campaigns) + per-author flooders.
  - STORE:  SQLite (posts + snapshots). Images downloaded to data/media/.
  - SNAP:   re-fetch via batch /twitter/tweets at post-ages [1,2,4,8,12,24]h,
            then RETIRE (>24h is converged -> no point paying to re-fetch).
  - BUDGET: hard cost cap (USD). Stops calling the API before exceeding it.

CLI:
  python x_pipeline.py init
  python x_pipeline.py intake   [--target 5000] [--budget 2.0]
  python x_pipeline.py snapshot [--force] [--budget 2.0]
  python x_pipeline.py run      [--target 5000] [--interval 3600] [--budget 8.0]
  python x_pipeline.py stats
"""
import os
import re
import sys
import json
import time
import sqlite3
import argparse
import pathlib
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import requests

try:
    # line_buffering => logs flush per line (real-time logs when piped on a server)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# ----------------------------- config ---------------------------------------
BASE = "https://api.twitterapi.io"
ROOT = pathlib.Path(__file__).parent
DB_PATH = ROOT / "data" / "x_pipeline.db"
MEDIA_DIR = ROOT / "data" / "media"

# Regions to pull trends from (worldwide + high-volume active locales).
WOEIDS = [1, 23424977, 23424856, 23424768, 23424848]  # world, US, JP, BR, IN
TRENDS_PER_WOEID = 10
PAGES_PER_TOPIC = 2          # ~20 fresh posts/page
FRESH_WINDOW_H = 1.0         # intake posts aged 0..FRESH_WINDOW_H hours

SNAPSHOT_AGES_H = [1, 2, 4, 8, 12, 24]   # log-spaced re-snapshot schedule
MAX_TRACK_H = 24             # retire after this age
SNAPSHOT_BATCH = 50         # tweet_ids per /twitter/tweets call (100 => HTTP 400)

# spam heuristics
DUP_TEXT_THRESHOLD = 3        # >=N identical normalized texts => campaign spam
TEMPLATE_THRESHOLD = 4        # >=N posts sharing a skeleton (text minus tags/nums)
AUTHOR_FLOOD_THRESHOLD = 5    # an author appearing >=N times in a pass => keep best
AUTHOR_DB_MAX = 8            # an author already this many posts in DB => drop new
CASHTAG_MAX = 3              # >=N distinct $cashtags => stock-pump promo

CREDITS_PER_USD = 100_000    # pricing: 100k credits = $1
CREDITS_PER_TWEET = 15       # $0.15 / 1k tweets
MIN_CREDITS_PER_REQ = 15     # $0.00015 minimum per request


def load_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("API_KEY not found in .env")


HEADERS = {"x-api-key": load_key()}


# ----------------------------- budget ---------------------------------------
class Budget:
    """Local cost guard so we never blow past the cap between balance checks."""
    def __init__(self, usd):
        self.cap = int(usd * CREDITS_PER_USD)
        self.used = 0

    def ok(self):
        return self.used < self.cap

    def charge_request(self, n_tweets):
        self.used += max(n_tweets * CREDITS_PER_TWEET, MIN_CREDITS_PER_REQ)

    def __str__(self):
        return f"${self.used / CREDITS_PER_USD:.4f} / ${self.cap / CREDITS_PER_USD:.2f}"


def balance():
    try:
        j = requests.get(f"{BASE}/oapi/my/info", headers=HEADERS, timeout=30).json()
        return j.get("recharge_credits", 0) + j.get("total_bonus_credits", 0)
    except Exception:
        return None


# ----------------------------- http -----------------------------------------
def get_json(url, params=None, tries=5):
    last_exc = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=40)
        except requests.exceptions.RequestException as e:
            # network blip (timeout / connection reset) -> back off and retry
            last_exc = e
            time.sleep(1.5 * (2 ** attempt))
            continue
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()
    if last_exc is not None:
        raise last_exc
    r.raise_for_status()
    return r.json()


def trend_fields(entry):
    d = entry.get("trend", entry) if isinstance(entry, dict) else {}
    if not isinstance(d, dict):
        d = entry if isinstance(entry, dict) else {}
    name = d.get("name") or d.get("trend") or ""
    query = (d.get("target") or {}).get("query") or name
    return str(name), str(query)


# ----------------------------- parsing --------------------------------------
def created_epoch(s):
    try:
        return int(datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp())
    except Exception:
        return None


def gi(p, k):
    v = p.get(k)
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return 0


def extract_media(p):
    """images = photo urls + video/gif thumbnails; videos = best-bitrate mp4."""
    media = (p.get("extendedEntities") or {}).get("media") or []
    images, videos = [], []
    for m in media:
        t = m.get("type")
        if t == "photo" and m.get("media_url_https"):
            images.append(m["media_url_https"])
        elif t in ("video", "animated_gif"):
            if m.get("media_url_https"):
                images.append(m["media_url_https"])  # thumbnail as image feature
            variants = [v for v in (m.get("video_info") or {}).get("variants", [])
                        if v.get("content_type") == "video/mp4" and v.get("url")]
            if variants:
                videos.append(max(variants, key=lambda v: v.get("bitrate") or 0)["url"])
    return images, videos


_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")
_TAG = re.compile(r"[#$]\w+")
_NUM = re.compile(r"\d+")
_CASHTAG = re.compile(r"\$[A-Za-z]{2,6}\b")
_HASHTAG = re.compile(r"#\w+")


def norm_text(s):
    s = (s or "").lower()
    s = _URL.sub("", s)
    s = _MENTION.sub("", s)
    s = _WS.sub(" ", s).strip()
    return s


def skeleton(s):
    """Text stripped of tags/numbers/urls -> catches templated campaigns that
    differ only by hashtag/cashtag/number."""
    s = _TAG.sub("", norm_text(s))
    s = _NUM.sub("", s)
    return _WS.sub(" ", s).strip()


def is_promo(text):
    """Stock-pump / tag-stuffed promo: many cashtags, or tag-dominated short text."""
    cash = len(set(m.lower() for m in _CASHTAG.findall(text or "")))
    if cash >= CASHTAG_MAX:
        return True
    words = norm_text(text).split()
    tags = len(_HASHTAG.findall(text or "")) + cash
    return tags >= 5 and len(words) <= 25 and tags / max(len(words), 1) >= 0.4


# ----------------------------- spam filter ----------------------------------
def spam_filter(tweets, conn=None):
    """Return (kept, reasons) where reasons counts why posts were dropped.
    Heuristics: identical text, templated skeleton, cashtag promo, author
    flooding within the pass, and author flooding across passes (DB-aware)."""
    by_text, by_skel, by_author = {}, {}, {}
    for t in tweets:
        by_text.setdefault(norm_text(t.get("text", "")), []).append(t)
        sk = skeleton(t.get("text", ""))
        if len(sk) >= 8:                       # ignore trivial skeletons
            by_skel.setdefault(sk, []).append(t)
        by_author.setdefault((t.get("author") or {}).get("userName", ""), []).append(t)

    reasons = {"dup_text": 0, "template": 0, "promo": 0, "author_flood": 0, "author_db": 0}
    spam = set()

    def drop(t, why):
        if t["id"] not in spam:
            spam.add(t["id"]); reasons[why] += 1

    for txt, grp in by_text.items():
        if txt and len(grp) >= DUP_TEXT_THRESHOLD:
            for t in grp:
                drop(t, "dup_text")
    for sk, grp in by_skel.items():
        if len(grp) >= TEMPLATE_THRESHOLD:
            for t in grp:
                drop(t, "template")
    for t in tweets:
        if t["id"] not in spam and is_promo(t.get("text", "")):
            drop(t, "promo")
    for author, grp in by_author.items():
        if author and len(grp) >= AUTHOR_FLOOD_THRESHOLD:
            keep = max(grp, key=lambda t: gi(t, "likeCount"))
            for t in grp:
                if t["id"] != keep["id"]:
                    drop(t, "author_flood")
    # cross-pass: author already heavily represented in the DB
    if conn is not None:
        for author, grp in by_author.items():
            if not author:
                continue
            prior = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE author=?", (author,)).fetchone()[0]
            if prior >= AUTHOR_DB_MAX:
                for t in grp:
                    drop(t, "author_db")

    kept = [t for t in tweets if t["id"] not in spam]
    return kept, reasons


# ----------------------------- storage --------------------------------------
def db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS posts(
        id TEXT PRIMARY KEY, author TEXT, text TEXT, lang TEXT,
        created_epoch INTEGER, intake_epoch INTEGER, intake_age_h REAL,
        has_image INTEGER, has_video INTEGER,
        image_urls TEXT, video_urls TEXT, media_paths TEXT, url TEXT,
        next_snap_idx INTEGER DEFAULT 0, retired INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS snapshots(
        post_id TEXT, snap_epoch INTEGER, age_h REAL,
        likes INTEGER, views INTEGER, retweets INTEGER,
        replies INTEGER, quotes INTEGER, bookmarks INTEGER,
        PRIMARY KEY(post_id, snap_epoch)
    );
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    CREATE INDEX IF NOT EXISTS ix_active ON posts(retired);
    """)
    conn.commit()
    conn.close()
    print(f"[init] db ready at {DB_PATH}")


def n_posts(conn):
    return conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]


# ----------------------------- images ---------------------------------------
def download_images(posts):
    """posts: list of dicts with id + image_urls. Returns {id: [local_path,...]}."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for p in posts:
        for i, u in enumerate(p["image_urls"][:4]):  # cap 4 imgs/post
            tasks.append((p["id"], i, u))

    def dl(task):
        pid, idx, url = task
        path = MEDIA_DIR / f"{pid}_{idx}.jpg"
        if path.exists():
            return pid, str(path)
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            path.write_bytes(r.content)
            return pid, str(path)
        except Exception:
            return pid, None

    out = {}
    if not tasks:
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pid, path in ex.map(dl, tasks):
            if path:
                out.setdefault(pid, []).append(path)
    return out


# ----------------------------- intake ---------------------------------------
def intake(target, budget):
    conn = db()
    if n_posts(conn) >= target:
        print(f"[intake] target {target} already reached ({n_posts(conn)} posts).")
        conn.close()
        return
    now = int(time.time())
    since = now - int(FRESH_WINDOW_H * 3600)
    existing = {r[0] for r in conn.execute("SELECT id FROM posts").fetchall()}

    seen, candidates = set(), []
    for woeid in WOEIDS:
        if not budget.ok():
            break
        try:
            tj = get_json(f"{BASE}/twitter/trends",
                          params={"woeid": woeid, "count": TRENDS_PER_WOEID})
        except Exception as e:
            print(f"  [woeid {woeid}] trends error: {e}")
            continue
        budget.charge_request(0)
        trends = tj.get("trends") or tj.get("data", {}).get("trends") or tj.get("data") or []
        topics = [(n, q) for n, q in (trend_fields(t) for t in trends) if n][:TRENDS_PER_WOEID]
        for topic, query in topics:
            cursor = ""
            for _ in range(PAGES_PER_TOPIC):
                if not budget.ok():
                    break
                params = {"query": f"{query} since_time:{since}", "queryType": "Top"}
                if cursor:
                    params["cursor"] = cursor
                try:
                    sj = get_json(f"{BASE}/twitter/tweet/advanced_search", params=params)
                except Exception:
                    break
                batch = sj.get("tweets", [])
                budget.charge_request(len(batch))
                for t in batch:
                    tid = t.get("id")
                    if not tid or tid in seen or tid in existing:
                        continue
                    seen.add(tid)
                    candidates.append(t)
                if not sj.get("has_next_page") or not sj.get("next_cursor"):
                    break
                cursor = sj["next_cursor"]
        print(f"  [woeid {woeid}] candidates so far: {len(candidates)}  (budget {budget})")

    # spam filter + cap to remaining target
    kept, reasons = spam_filter(candidates, conn)
    n_spam = sum(reasons.values())
    room = max(0, target - n_posts(conn))
    kept = kept[:room]
    print(f"[intake] raw={len(candidates)} spam_dropped={n_spam} {reasons} kept={len(kept)}")

    # prepare + download images
    prepped = []
    for t in kept:
        imgs, vids = extract_media(t)
        prepped.append({
            "id": t["id"],
            "author": (t.get("author") or {}).get("userName", ""),
            "text": t.get("text", ""),
            "lang": t.get("lang", ""),
            "created_epoch": created_epoch(t.get("createdAt", "")),
            "image_urls": imgs, "video_urls": vids,
            "url": t.get("url", ""),
            "metrics": (gi(t, "likeCount"), gi(t, "viewCount"), gi(t, "retweetCount"),
                        gi(t, "replyCount"), gi(t, "quoteCount"), gi(t, "bookmarkCount")),
        })
    print(f"[intake] downloading images for {len(prepped)} posts ...")
    media_map = download_images(prepped)

    # store
    nowi = int(time.time())
    ins_p = ins_s = 0
    for p in prepped:
        ce = p["created_epoch"] or nowi
        age_h = (nowi - ce) / 3600
        paths = media_map.get(p["id"], [])
        conn.execute("""INSERT OR IGNORE INTO posts
            (id,author,text,lang,created_epoch,intake_epoch,intake_age_h,
             has_image,has_video,image_urls,video_urls,media_paths,url,next_snap_idx,retired)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)""",
            (p["id"], p["author"], p["text"], p["lang"], ce, nowi, age_h,
             int(bool(p["image_urls"])), int(bool(p["video_urls"])),
             json.dumps(p["image_urls"]), json.dumps(p["video_urls"]),
             json.dumps(paths), p["url"]))
        ins_p += conn.total_changes and 1 or 0
        lk, vw, rt, rp, qt, bm = p["metrics"]
        conn.execute("""INSERT OR IGNORE INTO snapshots
            (post_id,snap_epoch,age_h,likes,views,retweets,replies,quotes,bookmarks)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (p["id"], nowi, age_h, lk, vw, rt, rp, qt, bm))
        ins_s += 1
    conn.commit()
    total = n_posts(conn)
    imgs_dl = sum(len(v) for v in media_map.values())
    print(f"[intake] stored {len(prepped)} posts (+{ins_s} t0 snapshots), "
          f"{imgs_dl} images. DB total = {total}/{target}. budget {budget}")
    conn.close()


# ----------------------------- snapshot -------------------------------------
def snapshot(budget, force=False):
    conn = db()
    now = int(time.time())
    active = conn.execute("SELECT * FROM posts WHERE retired=0").fetchall()
    due = []
    for p in active:
        age_h = (now - p["created_epoch"]) / 3600
        idx = p["next_snap_idx"]
        if force or (idx < len(SNAPSHOT_AGES_H) and age_h >= SNAPSHOT_AGES_H[idx]):
            due.append((p["id"], age_h))
    if not due:
        print("[snapshot] nothing due.")
        conn.close()
        return
    print(f"[snapshot] {len(due)} posts due (force={force})")

    by_id = {pid: age for pid, age in due}
    ids = list(by_id.keys())
    fetched = recorded = retired = 0
    for i in range(0, len(ids), SNAPSHOT_BATCH):
        if not budget.ok():
            print(f"[snapshot] budget reached, stopping. {budget}")
            break
        chunk = ids[i:i + SNAPSHOT_BATCH]
        try:
            j = get_json(f"{BASE}/twitter/tweets", params={"tweet_ids": ",".join(chunk)})
        except Exception as e:
            print(f"  batch error: {e}")
            continue
        tweets = j.get("tweets", [])
        budget.charge_request(len(tweets))
        fetched += len(tweets)
        got = {t["id"]: t for t in tweets}
        for pid in chunk:
            row = conn.execute("SELECT created_epoch,next_snap_idx FROM posts WHERE id=?",
                               (pid,)).fetchone()
            age_h = (now - row["created_epoch"]) / 3600
            t = got.get(pid)
            if t is not None:
                conn.execute("""INSERT OR IGNORE INTO snapshots
                    (post_id,snap_epoch,age_h,likes,views,retweets,replies,quotes,bookmarks)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (pid, now, age_h, gi(t, "likeCount"), gi(t, "viewCount"),
                     gi(t, "retweetCount"), gi(t, "replyCount"),
                     gi(t, "quoteCount"), gi(t, "bookmarkCount")))
                recorded += 1
            # advance schedule past every milestone already reached
            idx = row["next_snap_idx"]
            while idx < len(SNAPSHOT_AGES_H) and age_h >= SNAPSHOT_AGES_H[idx]:
                idx += 1
            ret = 1 if (idx >= len(SNAPSHOT_AGES_H) or age_h >= MAX_TRACK_H or t is None) else 0
            conn.execute("UPDATE posts SET next_snap_idx=?, retired=? WHERE id=?",
                         (idx, ret, pid))
            retired += ret
    conn.commit()
    print(f"[snapshot] fetched={fetched} recorded={recorded} retired={retired}. budget {budget}")
    conn.close()


# ----------------------------- run loop -------------------------------------
def run(target, interval, budget):
    print(f"[run] target={target} interval={interval}s budget=${budget.cap/CREDITS_PER_USD:.2f}")
    while budget.ok():
        snapshot(budget)                 # update due posts first
        intake(target, budget)           # then top up with fresh posts
        conn = db(); total = n_posts(conn); conn.close()
        if total >= target:
            print(f"[run] reached target {total}/{target}. Continuing snapshots only "
                  f"until active posts retire is recommended; stopping intake.")
            # keep snapshotting active posts to finish their trajectories
            active = True
            while active and budget.ok():
                time.sleep(interval)
                snapshot(budget)
                conn = db()
                active = conn.execute("SELECT COUNT(*) FROM posts WHERE retired=0").fetchone()[0] > 0
                conn.close()
            break
        print(f"[run] sleeping {interval}s ...")
        time.sleep(interval)
    print(f"[run] done. budget {budget}")


# ----------------------------- stats ----------------------------------------
def stats():
    conn = db()
    total = n_posts(conn)
    if total == 0:
        print("[stats] empty db.")
        conn.close()
        return
    active = conn.execute("SELECT COUNT(*) FROM posts WHERE retired=0").fetchone()[0]
    imgs = conn.execute("SELECT COUNT(*) FROM posts WHERE has_image=1").fetchone()[0]
    vids = conn.execute("SELECT COUNT(*) FROM posts WHERE has_video=1").fetchone()[0]
    media_files = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE media_paths NOT IN ('[]','')").fetchone()[0]
    snaps = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    per_post = conn.execute(
        "SELECT post_id, COUNT(*) c FROM snapshots GROUP BY post_id").fetchall()
    avg_snaps = sum(r["c"] for r in per_post) / len(per_post) if per_post else 0
    print(f"[stats] posts={total} (active={active}, retired={total-active})")
    print(f"        with image={imgs} ({100*imgs/total:.0f}%) | with video={vids} "
          f"({100*vids/total:.0f}%) | posts w/ downloaded media={media_files}")
    print(f"        snapshots={snaps} | avg {avg_snaps:.2f} snapshots/post")
    # sample trajectories with >=2 snapshots
    multi = [r["post_id"] for r in per_post if r["c"] >= 2][:3]
    for pid in multi:
        rows = conn.execute(
            "SELECT age_h,likes,views FROM snapshots WHERE post_id=? ORDER BY age_h", (pid,)
        ).fetchall()
        traj = " -> ".join(f"{r['age_h']:.1f}h:♥{r['likes']}/👁{r['views']}" for r in rows)
        print(f"        traj {pid}: {traj}")
    conn.close()


# ----------------------------- cli ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    pi = sub.add_parser("intake"); pi.add_argument("--target", type=int, default=5000); pi.add_argument("--budget", type=float, default=2.0)
    ps = sub.add_parser("snapshot"); ps.add_argument("--force", action="store_true"); ps.add_argument("--budget", type=float, default=2.0)
    pr = sub.add_parser("run"); pr.add_argument("--target", type=int, default=5000); pr.add_argument("--interval", type=int, default=3600); pr.add_argument("--budget", type=float, default=8.0)
    sub.add_parser("stats")
    a = ap.parse_args()

    b0 = balance()
    if a.cmd == "init":
        init_db()
    elif a.cmd == "intake":
        init_db(); intake(a.target, Budget(a.budget))
    elif a.cmd == "snapshot":
        init_db(); snapshot(Budget(a.budget), force=a.force)
    elif a.cmd == "run":
        init_db(); run(a.target, a.interval, Budget(a.budget))
    elif a.cmd == "stats":
        stats()
    b1 = balance()
    if b0 is not None and b1 is not None and a.cmd != "stats":
        spent = b0 - b1
        print(f"[cost] this run spent {spent} credits = ${spent/CREDITS_PER_USD:.4f}  "
              f"| balance left {b1} credits = ${b1/CREDITS_PER_USD:.2f}")


if __name__ == "__main__":
    main()
