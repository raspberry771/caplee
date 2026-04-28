"""
extract_logos.py — Phase 2: Logo Extraction for Swim Teams

For each team in teams_enriched.csv, collects logo candidates from:
  1. Team website — OG image, <img> tags with logo-like names, favicon
  2. Facebook profile picture (square, ~320px)
  3. Instagram profile picture (square, ~150px)

Scores each candidate by:
  score = (min(w,h) / max(w,h)) * sqrt(w * h)
  → rewards square aspect ratio AND higher resolution

Saves the best candidate to logos/{team_name}_{state}.png

Features:
  - Resumable: skips teams that already have a logo file
  - Concurrent: asyncio + semaphore
  - Logged: logo_extraction_log.jsonl per team

Setup:
  pip install anthropic pandas python-dotenv requests beautifulsoup4 Pillow aiohttp aiofiles
"""

import asyncio
import io
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONCURRENT_REQUESTS = 8     # Max parallel teams being processed
REQUEST_TIMEOUT = 15        # Seconds per HTTP request
MIN_SIZE_PX = 1             # No size minimum — accept any image including small favicons
MAX_ASPECT_RATIO = 4.0      # Ignore images wider/taller than 4:1
LOGOS_DIR = BASE / "logos"
INPUT_CSV = BASE / "teams_enriched.csv"
LOG_FILE = BASE / "logo_extraction_log.jsonl"

# Headers that look like a real browser to avoid 403s
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------
def team_slug(name: str, state: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    slug = slug.strip("_")
    return f"{slug}_{state.lower()}"


# ---------------------------------------------------------------------------
# Image scoring
# ---------------------------------------------------------------------------
def score_image(img: Image.Image, tier: int = 5) -> float:
    """
    Higher score = better logo candidate.

    Scoring: aspect_ratio × resolution × tier_bonus
      - aspect_ratio: 1.0 = perfect square
      - resolution:   sqrt(w × h), capped at 512 to prevent huge images dominating
      - tier_bonus:   tier 1 (favicon) gets 100×, tier 7 (OG) gets 1×
                      This ensures a favicon always beats a page image.

    Tier map: 1=apple-touch-icon, 2=declared icon, 3=/favicon.ico,
              4=nav logo img, 5=nav first img, 6=page logo img, 7=OG/twitter
    """
    w, h = img.size
    if w == 0 or h == 0:
        return 0.0
    aspect_ratio = min(w, h) / max(w, h)
    resolution = min(math.sqrt(w * h), 512)  # cap so a 4000px banner can't beat a 200px icon
    tier_bonus = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20, 6: 8, 7: 1}.get(tier, 1)
    return aspect_ratio * resolution * tier_bonus


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT), allow_redirects=True, ssl=False) as resp:
            if resp.status == 200:
                return await resp.read()
    except Exception:
        pass
    return None


async def fetch_html(session: aiohttp.ClientSession, url: str) -> str | None:
    data = await fetch_bytes(session, url)
    if data:
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            pass
    return None


def bytes_to_image(data: bytes) -> Image.Image | None:
    # Try standard Pillow decode first (PNG, JPG, GIF, WebP, ICO, etc.)
    try:
        img = Image.open(io.BytesIO(data))
        # For ICO files, select the largest available frame instead of the default
        if getattr(img, "format", None) == "ICO":
            sizes = img.info.get("sizes", [img.size])
            if sizes:
                largest = max(sizes, key=lambda s: s[0] * s[1])
                img = img.resize(largest, Image.LANCZOS) if largest != img.size else img
        img.load()
        return img.convert("RGBA")
    except Exception:
        pass
    # Try SVG → PNG via cairosvg if available.
    # Match any common SVG opening: <svg, <?xml, <!DOCTYPE, case-insensitive.
    stripped = data.lstrip()
    is_svg = (
        stripped[:4].lower() == b"<svg"
        or stripped[:5] == b"<?xml"
        or b"<svg" in stripped[:256].lower()
    )
    if is_svg:
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(bytestring=data, output_width=512)
            img = Image.open(io.BytesIO(png_bytes))
            img.load()
            return img.convert("RGBA")
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Candidate collectors
# ---------------------------------------------------------------------------
LOGO_KEYWORDS = re.compile(
    r"\b(logo|badge|crest|emblem|seal|shield|mascot)\b",
    re.IGNORECASE,
)
# "icon" in class/id signals a logo, unless it's a social/share icon
ICON_KEYWORD = re.compile(r"\bicon\b", re.IGNORECASE)
SOCIAL_ICON = re.compile(r"\b(social|share|facebook|instagram|twitter|youtube|linkedin)\b", re.IGNORECASE)
# Paths/attributes that signal non-team-logo content
SPONSOR_KEYWORDS = re.compile(
    r"sponsor|partner|advertis|vendor|supporter|affiliate|banner"
    r"|plumbing|speedo|arena|tyr\b"
    r"|instagram|facebook|twitter|youtube|social|share"
    r"|photoslider|slider|carousel|hero|banner|background",
    re.IGNORECASE,
)


def _img_src(tag) -> str:
    return (tag.get("src") or tag.get("data-src") or
            tag.get("data-lazy-src") or tag.get("data-original") or "")


def _name_words(team_name: str) -> list[str]:
    """Return meaningful words from team name for URL/alt matching."""
    stop = {"of", "the", "and", "swim", "swimming", "aquatic", "aquatics", "club", "team", "inc"}
    return [w.lower() for w in re.split(r"\W+", team_name) if len(w) > 2 and w.lower() not in stop]


HEADER_CLASS_RE = re.compile(
    r"\b(header|navbar|nav-bar|nav-brand|site-header|page-header|top-bar|masthead|topbar)\b",
    re.IGNORECASE,
)


def _nav_containers(soup) -> list:
    """Return header/nav semantic tags + divs with header-like class/id names."""
    containers = list(soup.find_all(["header", "nav"], limit=6))
    for div in soup.find_all("div", limit=300):
        classes = " ".join(div.get("class") or [])
        div_id = div.get("id", "")
        if HEADER_CLASS_RE.search(classes + " " + div_id):
            containers.append(div)
    return containers


def extract_website_candidates(html: str, base_url: str, team_name: str = "") -> list[tuple[str, int]]:
    """
    Pull logo candidate URLs from a team website.
    Returns list of (url, tier) — lower tier number = higher confidence.

    Tier 1 — apple-touch-icon (180×180 square, always the team's own brand mark)
    Tier 2 — <link rel="icon"> with the largest declared size
    Tier 3 — /favicon.ico default path
    Tier 4 — nav/header <img> with "logo" in id/class/alt/src (top-left logo)
    Tier 5 — first <img> in any nav/header container (positional fallback)
    Tier 6 — logo-keyword <img> anywhere on the page (excluding sponsor paths)
    Tier 7 — OG / twitter card (last resort)
    """
    soup = BeautifulSoup(html, "html.parser")
    parsed_base = urlparse(base_url)
    base_origin = f"{parsed_base.scheme}://{parsed_base.netloc}"

    results: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(url: str, tier: int):
        if url and url not in seen:
            seen.add(url)
            results.append((url, tier))

    # ── Tier 1: apple-touch-icon ──
    # Sites set this as their brand mark, it's always square and clean.
    # Prefer the precomposed version (no transparency overlay).
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel") or []).lower()
        if "apple-touch-icon" in rel:
            href = link.get("href")
            if href:
                add(urljoin(base_url, href), 1)

    # ── Tier 2: declared favicon with the largest size ──
    # e.g. <link rel="icon" sizes="192x192" href="...">
    icon_links: list[tuple[int, str]] = []
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel") or []).lower()
        if "icon" in rel and "apple" not in rel and "mask" not in rel:
            href = link.get("href", "")
            sizes = link.get("sizes", "")
            # Parse the largest dimension from "32x32" or "any"
            try:
                dim = max(int(d) for d in re.findall(r"\d+", sizes)) if sizes and sizes != "any" else 0
            except ValueError:
                dim = 0
            if href:
                icon_links.append((dim, href))
    # Add largest first
    for _, href in sorted(icon_links, reverse=True):
        add(urljoin(base_url, href), 2)

    # ── Tier 3: /favicon.ico default path ──
    add(f"{base_origin}/favicon.ico", 3)

    # ── Tier 4: nav/header <img> with logo-like id/class/alt/src ──
    nav_containers = _nav_containers(soup)
    for container in nav_containers:
        for img_tag in container.find_all("img"):
            src = _img_src(img_tag)
            alt = img_tag.get("alt", "")
            classes = " ".join(img_tag.get("class") or [])
            img_id = img_tag.get("id", "")
            combined = src + " " + alt + " " + classes + " " + img_id
            icon_signal = ICON_KEYWORD.search(classes + " " + img_id) and not SOCIAL_ICON.search(classes + " " + img_id)
            is_logo = LOGO_KEYWORDS.search(combined) or icon_signal
            if is_logo and not SPONSOR_KEYWORDS.search(combined):
                if src:
                    add(urljoin(base_url, src), 4)

    # ── Tier 5: first <img> in each nav/header container (positional) ──
    for container in nav_containers:
        for img_tag in container.find_all("img"):
            src = _img_src(img_tag)
            if src and not SPONSOR_KEYWORDS.search(src):
                add(urljoin(base_url, src), 5)
                break

    # ── Tier 6: logo-keyword <img> anywhere on page ──
    for img_tag in soup.find_all("img"):
        src = _img_src(img_tag)
        alt = img_tag.get("alt", "")
        classes = " ".join(img_tag.get("class") or [])
        img_id = img_tag.get("id", "")
        combined = src + " " + alt + " " + classes + " " + img_id
        if LOGO_KEYWORDS.search(combined) and not SPONSOR_KEYWORDS.search(combined):
            if src:
                add(urljoin(base_url, src), 6)

    # ── Tier 7: OG / twitter card ──
    og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if og and og.get("content"):
        add(urljoin(base_url, og["content"]), 7)
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        add(urljoin(base_url, tw["content"]), 7)

    return results


async def get_facebook_profile_pic(session: aiohttp.ClientSession, fb_url: str) -> str | None:
    """
    Facebook profile photos are reliably square and ~320px.
    We scrape the og:image from the page (no API key needed).
    """
    html = await fetch_html(session, fb_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    return None


async def get_instagram_profile_pic(session: aiohttp.ClientSession, ig_url: str) -> str | None:
    """
    Instagram profile photos are square ~150px.
    Scrape og:image from the profile page.
    """
    html = await fetch_html(session, ig_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    return None


# ---------------------------------------------------------------------------
# Per-team logo extraction
# ---------------------------------------------------------------------------
async def extract_logo_for_team(
    session: aiohttp.ClientSession,
    team_name: str,
    state: str,
    website: str | None,
    facebook: str | None,
    instagram: str | None,
) -> dict:
    slug = team_slug(team_name, state)
    out_path = LOGOS_DIR / f"{slug}.png"

    result = {
        "team": team_name,
        "state": state,
        "slug": slug,
        "status": "no_candidates",
        "source": None,
        "score": None,
        "size": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Only use favicon sources — website page images pick wrong logos too often.
    # Priority: apple-touch-icon (tier 1) → declared icon (tier 2) → /favicon.ico (tier 3)
    candidate_urls: list[tuple[str, str, int]] = []

    if website and website.startswith("http"):
        html = await fetch_html(session, website)
        if html:
            for url, tier in extract_website_candidates(html, website, team_name):
                if tier <= 3:  # favicon tiers only
                    candidate_urls.append((url, "website", tier))

    if not candidate_urls:
        result["status"] = "no_candidates"
        return result

    best_img: Image.Image | None = None
    best_url = None
    best_source = None
    best_score = -1.0

    for url, source, tier in candidate_urls:
        data = await fetch_bytes(session, url)
        if not data:
            continue
        img = bytes_to_image(data)
        if not img:
            continue
        w, h = img.size
        if w < MIN_SIZE_PX or h < MIN_SIZE_PX:
            continue
        if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
            continue
        s = score_image(img, tier=tier)
        if s > best_score:
            best_score, best_img, best_url, best_source = s, img, url, source

    if best_img is None:
        result["status"] = "all_candidates_failed"
        return result

    # Save as PNG
    LOGOS_DIR.mkdir(exist_ok=True)
    best_img.save(out_path, "PNG")
    result["status"] = "ok"
    result["source"] = best_source
    result["url"] = best_url
    result["score"] = round(best_score, 1)
    result["size"] = list(best_img.size)
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main():
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    LOGOS_DIR.mkdir(exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def process_team(row):
        team_name = row["Team Name"]
        state = row["State"]
        slug = team_slug(team_name, state)
        out_path = LOGOS_DIR / f"{slug}.png"

        # Skip if already extracted
        if out_path.exists():
            log.info(f"SKIP {team_name} — already have {out_path.name}")
            return

        # Skip unenriched teams — nothing to scrape yet
        if not any([row.get("Website", ""), row.get("Facebook", ""), row.get("Instagram", "")]):
            return

        async with semaphore:
            result = await extract_logo_for_team(
                session=session,
                team_name=team_name,
                state=state,
                website=row.get("Website", ""),
                facebook=row.get("Facebook", ""),
                instagram=row.get("Instagram", ""),
            )
            status_emoji = "✓" if result["status"] == "ok" else "✗"
            log.info(
                f"{status_emoji} {team_name} ({state}) — "
                f"status={result['status']} source={result.get('source')} "
                f"size={result.get('size')} score={result.get('score')}"
            )
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS * 2, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_team(row) for _, row in df.iterrows()]
        await asyncio.gather(*tasks)

    # Summary
    total = len(df)
    ok = sum(1 for _ in LOGOS_DIR.glob("*.png"))
    log.info(f"\nDone. {ok}/{total} logos saved to {LOGOS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
