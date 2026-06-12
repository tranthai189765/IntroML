"""
Rebuild a complete CSV from data/trending_posts.json — now WITH image & video URLs.
No API calls (free): just re-parses the JSON already crawled.

Media lives in post["extendedEntities"]["media"][], each item:
  - type == "photo"            -> image at media_url_https
  - type in (video, animated_gif) -> pick highest-bitrate mp4 from video_info.variants
"""
import json
import csv
import pathlib

ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "data" / "trending_posts.json"
OUT = ROOT / "data" / "trending_posts_full.csv"


def author(p):
    a = p.get("author") or {}
    return a.get("userName") or a.get("screen_name") or ""


def extract_media(p):
    """Return (image_urls, video_urls, media_types)."""
    media = (p.get("extendedEntities") or {}).get("media") or []
    images, videos, types = [], [], []
    for m in media:
        t = m.get("type")
        types.append(t or "")
        if t == "photo":
            if m.get("media_url_https"):
                images.append(m["media_url_https"])
        elif t in ("video", "animated_gif"):
            variants = [
                v for v in (m.get("video_info") or {}).get("variants", [])
                if v.get("content_type") == "video/mp4" and v.get("url")
            ]
            if variants:
                best = max(variants, key=lambda v: v.get("bitrate") or 0)
                videos.append(best["url"])
            elif m.get("media_url_https"):  # fallback: thumbnail
                videos.append(m["media_url_https"])
    return images, videos, types


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    posts = data.get("posts", [])

    cols = [
        "id", "author", "createdAt", "lang",
        "likeCount", "retweetCount", "replyCount", "quoteCount",
        "viewCount", "bookmarkCount",
        "media_count", "media_types", "image_count", "video_count",
        "image_urls", "video_urls", "url", "text",
    ]

    n_img = n_vid = n_media_posts = 0
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in posts:
            images, videos, types = extract_media(p)
            if images or videos:
                n_media_posts += 1
            n_img += len(images)
            n_vid += len(videos)
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

    print(f"[saved] {OUT}")
    print(f"  posts: {len(posts)} | with media: {n_media_posts} "
          f"| images: {n_img} | videos: {n_vid}")
    print(f"  columns ({len(cols)}): {', '.join(cols)}")


if __name__ == "__main__":
    main()
