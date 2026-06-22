"""
crawl_mature.py — crawl SINGLE-SHOT các post viral CÓ ẢNH đã chín (tạo > 3 ngày).
Giá trị hiện tại của post = mốc "72h" (đã bão hòa). 12 mốc sớm sẽ được SUY NGƯỢC
ở bước local (donor-shape từ 10k post đầy đủ) — KHÔNG crawl lại.

Tái dùng helper của x_pipeline (advanced_search, get_json, extract_media, spam_filter,
download_images, Budget, balance). Lưu data/mature_raw.jsonl (+ ảnh về data/media/).

  python crawl_mature.py --probe                 # test query, ~vài cent
  python crawl_mature.py --budget 6.5 --minf 1000 # crawl thật (nền)
"""
import sys, os, json, time, argparse, sqlite3, pathlib
from datetime import datetime, timezone, timedelta

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
import x_pipeline as xp   # import-safe (có __main__ guard)

OUT = ROOT / "data" / "mature_raw.jsonl"
NOW = datetime.now(timezone.utc)
# cửa sổ: post tạo trong [now-WINDOW_D, now-3d]  → đã qua mốc 3 ngày
UNTIL = (NOW - timedelta(days=3)).date().isoformat()
WINDOW_D = 60
LANG_TARGET = {"en": 18000, "ja": 8000, "es": 6000, "pt": 4000, "fr": 4000}
FLUSH_EVERY = 300


def weekly_windows():
    """(since, until) dạng YYYY-MM-DD, bước 7 ngày, mới → cũ."""
    until = NOW - timedelta(days=3)
    out = []
    for _ in range(WINDOW_D // 7 + 1):
        since = until - timedelta(days=7)
        out.append((since.date().isoformat(), until.date().isoformat()))
        until = since
    return out


def existing_ids():
    ids = set()
    try:
        conn = sqlite3.connect(str(xp.DB_PATH))
        ids = {str(r[0]) for r in conn.execute("SELECT id FROM posts")}
        conn.close()
    except Exception as e:
        print(f"  [warn] no DB ids: {e}")
    if OUT.exists():                       # resume-safe
        for line in OUT.open(encoding="utf-8"):
            try: ids.add(str(json.loads(line)["id"]))
            except Exception: pass
    return ids


def to_record(t, paths):
    return {
        "id": t["id"],
        "author": (t.get("author") or {}).get("userName", ""),
        "text": t.get("text", ""),
        "lang": t.get("lang", ""),
        "created_epoch": xp.created_epoch(t.get("createdAt", "")),
        "likes":    xp.gi(t, "likeCount"),
        "views":    xp.gi(t, "viewCount"),
        "comments": xp.gi(t, "replyCount"),
        "reposts":  xp.gi(t, "retweetCount"),
        "url": t.get("url", ""),
        "img_paths": paths,
    }


def flush(buf, fout):
    """spam filter → bắt buộc có ảnh → tải ảnh → ghi jsonl. Trả về số đã ghi."""
    if not buf:
        return 0
    kept, _ = xp.spam_filter(buf)
    prepped = []
    for t in kept:
        imgs, _vids = xp.extract_media(t)
        if not imgs:
            continue
        prepped.append({"id": t["id"], "image_urls": imgs})
    media = xp.download_images(prepped)
    raw_by_id = {t["id"]: t for t in kept}
    n = 0
    for p in prepped:
        paths = media.get(p["id"], [])
        if not paths:                      # bắt buộc tải được ảnh
            continue
        fout.write(json.dumps(to_record(raw_by_id[p["id"]], paths), ensure_ascii=False) + "\n")
        n += 1
    fout.flush()
    return n


def probe(minf):
    s, u = weekly_windows()[0]
    q = f"filter:images min_faves:{minf} lang:en since:{s} until:{u}"
    print(f"[probe] query = {q!r}  queryType=Latest")
    sj = xp.get_json(f"{xp.BASE}/twitter/tweet/advanced_search",
                     params={"query": q, "queryType": "Latest"})
    batch = sj.get("tweets", [])
    print(f"[probe] returned {len(batch)} tweets | has_next={sj.get('has_next_page')}")
    for t in batch[:3]:
        imgs, _ = xp.extract_media(t)
        print(f"   id={t.get('id')} lang={t.get('lang')} likes={xp.gi(t,'likeCount')} "
              f"views={xp.gi(t,'viewCount')} imgs={len(imgs)} created={t.get('createdAt')}")
        print(f"      text={ (t.get('text') or '')[:80]!r}")
    b = xp.balance()
    print(f"[probe] balance now: {b} credits = ${None if b is None else round(b/xp.CREDITS_PER_USD,2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--budget", type=float, default=6.5)
    ap.add_argument("--minf", type=int, default=1000)
    a = ap.parse_args()

    if a.probe:
        probe(a.minf)
        return

    b0 = xp.balance()
    print(f"[start] balance {b0} credits (${None if b0 is None else round(b0/xp.CREDITS_PER_USD,2)}) "
          f"| budget cap ${a.budget} | until={UNTIL} window={WINDOW_D}d | minf={a.minf}")
    budget = xp.Budget(a.budget)
    seen = existing_ids()
    print(f"[start] {len(seen)} ids already known (DB + jsonl) — sẽ bỏ qua")

    total = 0
    with OUT.open("a", encoding="utf-8") as fout:
        for lang, tgt in LANG_TARGET.items():
            got, buf = 0, []
            for (s, u) in weekly_windows():
                if got >= tgt or not budget.ok():
                    break
                cursor = ""
                while budget.ok() and got < tgt:
                    q = f"filter:images min_faves:{a.minf} lang:{lang} since:{s} until:{u}"
                    params = {"query": q, "queryType": "Latest"}
                    if cursor:
                        params["cursor"] = cursor
                    try:
                        sj = xp.get_json(f"{xp.BASE}/twitter/tweet/advanced_search", params=params)
                    except Exception as e:
                        print(f"  [{lang} {s}] error: {e}"); break
                    batch = sj.get("tweets", [])
                    budget.charge_request(len(batch))
                    new = [t for t in batch if t.get("id") and str(t["id"]) not in seen]
                    for t in new:
                        seen.add(str(t["id"]))
                    buf.extend(new)
                    if len(buf) >= FLUSH_EVERY:
                        w = flush(buf, fout); got += w; total += w; buf = []
                        print(f"  [{lang}] +{w} (lang {got}/{tgt}, total {total}, budget {budget})")
                    if not sj.get("has_next_page") or not sj.get("next_cursor"):
                        break
                    cursor = sj["next_cursor"]
            if buf:
                w = flush(buf, fout); got += w; total += w
                print(f"  [{lang}] flush tail +{w} (lang {got}/{tgt}, total {total}, budget {budget})")
            print(f"[lang done] {lang}: {got} posts | total {total} | budget {budget}")
            if not budget.ok():
                print("[stop] budget cap reached."); break

    b1 = xp.balance()
    print(f"[done] wrote {total} mature posts -> {OUT}")
    print(f"[cost] balance now {b1} credits = ${None if b1 is None else round(b1/xp.CREDITS_PER_USD,2)} "
          f"| spent this run {None if (b0 is None or b1 is None) else b0-b1} credits")


if __name__ == "__main__":
    main()
