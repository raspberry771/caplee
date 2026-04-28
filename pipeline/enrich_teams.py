"""
enrich_teams.py — Swim Team Data Enrichment

Reads teams_master_list.csv (Team Name, City, State), uses the Anthropic API
with built-in web search to find each team's Website, Email, Facebook, Instagram,
and writes results to teams_enriched.csv.

Features:
  - Resumable: checkpoint.json tracks processed rows; restart picks up where it stopped
  - Safe: never overwrites cells that already have data
  - Concurrent: asyncio + semaphore (tune CONCURRENT_REQUESTS to your API tier)
  - Logged: enrichment_log.jsonl has one entry per team

Setup:
  pip install anthropic pandas python-dotenv requests beautifulsoup4
  echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import anthropic
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

# ---------------------------------------------------------------------------
# Config — tune these to match your API tier and preferences
# ---------------------------------------------------------------------------
CONCURRENT_REQUESTS = 1  # 1 team at a time — TPM is the bottleneck, not RPM
CHUNK_SIZE = 1           # Save progress after every single team
REQUEST_DELAY_S = 5      # 5s buffer + ~10-12s processing = ~4 teams/min, well under 40k TPM
MAX_RETRIES = 4             # Exponential backoff: 1s, 2s, 4s, 8s
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

INPUT_CSV = BASE / "teams_master_list.csv"
OUTPUT_CSV = BASE / "teams_enriched.csv"
CHECKPOINT_FILE = BASE / "checkpoint.json"
LOG_FILE = BASE / "enrichment_log.jsonl"

ENRICH_COLS = ["Website", "Email", "Facebook", "Instagram", "Phone", "Head Coach", "Admin Contact", "Address", "Mascot"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    # Read from the enriched output if it exists so previous runs' data is preserved.
    # Fall back to the master list for a fresh start.
    source = OUTPUT_CSV if os.path.exists(OUTPUT_CSV) else INPUT_CSV
    df = pd.read_csv(source, dtype=str).fillna("")
    for col in ENRICH_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def load_checkpoint() -> set[int]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(df: pd.DataFrame, checkpoint: set[int]) -> None:
    df.to_csv(OUTPUT_CSV, index=False)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(checkpoint), f)


def append_log(entry: dict) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Email scraper — runs after each checkpoint to catch embedded/hidden emails
# ---------------------------------------------------------------------------

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
SKIP_EMAIL_PREFIXES = {"noreply", "no-reply", "donotreply", "wordpress", "example",
                       "test", "webmaster", "admin", "support", "info@spamarrest"}
CONTACT_PATHS = ["", "/contact", "/contact-us", "/about", "/about-us", "/staff", "/coaches"]


def _scrape_email_sync(website: str) -> str:
    """Synchronously scrape a website for email addresses across common pages."""
    if not website.startswith("http"):
        website = "https://" + website
    base = website.rstrip("/")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; SwimTeamBot/1.0)"}

    for path in CONTACT_PATHS:
        try:
            resp = requests.get(base + path, headers=headers, timeout=8, allow_redirects=True)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # 1. mailto: links — most reliable, works even when email is invisible on page
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().startswith("mailto:"):
                    email = href[7:].split("?")[0].strip().lower()
                    if email and not any(email.startswith(s) for s in SKIP_EMAIL_PREFIXES):
                        return email

            # 2. Regex search across all visible text and raw HTML
            for email in EMAIL_REGEX.findall(resp.text):
                email = email.lower()
                if not any(email.startswith(s) for s in SKIP_EMAIL_PREFIXES):
                    # Skip image/font/asset file extensions accidentally matched
                    if not re.search(r'\.(png|jpg|gif|svg|woff|ttf|css|js)$', email):
                        return email

        except Exception:
            continue

    return ""


async def scrape_emails_for_chunk(df: pd.DataFrame, indices: list[int]) -> int:
    """
    For each row in indices that has a website but no email,
    scrape the website to find a hidden/embedded email.
    Returns the count of emails newly found.
    """
    found = 0
    for idx in indices:
        website = str(df.at[idx, "Website"]).strip()
        email = str(df.at[idx, "Email"]).strip()
        if not website or email:
            continue  # skip: no website to scrape, or already has email

        team_name = df.at[idx, "Team Name"]
        scraped = await asyncio.to_thread(_scrape_email_sync, website)
        if scraped:
            df.at[idx, "Email"] = scraped
            log.info(f"  [email scraper] Found email for {team_name}: {scraped}")
            found += 1

    return found


# ---------------------------------------------------------------------------
# Enrichment logic
# ---------------------------------------------------------------------------

def needs_enrichment(row: pd.Series) -> bool:
    """True if at least one enrichment column is empty."""
    return any(not str(row.get(col, "")).strip() for col in ENRICH_COLS)


PROMPT_TEMPLATE = """\
Search for information about the swim team "{team_name}" located in {city}, {state}.

Find:
- Official website URL (prefer the club's own domain, not USA Swimming or team-finder directories)
- Contact or club email address
- Facebook page URL
- Instagram handle or URL
- Club phone number
- Head coach full name
- Admin contact: the name of the Club President, Aquatics Director, or Owner only — return null if none of these specific titles are found
- Physical address or mailing address of the club
- Team mascot name (return null if the team has no official mascot)

Respond with ONLY a raw JSON object — no markdown fences, no explanation:
{{"website": "...", "email": "...", "facebook": "...", "instagram": "...", "phone": "...", "head_coach": "...", "admin_contact": "...", "address": "...", "mascot": "..."}}

Use null for any field you cannot find with high confidence. Do not guess."""


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a response string."""
    # Strip markdown fences if Claude added them anyway
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group())


async def enrich_team(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    idx: int,
    row: pd.Series,
) -> dict:
    """
    Call the Anthropic API to find online presence for one team.
    Returns a dict with keys: website, email, facebook, instagram, status.
    """
    team_name = row["Team Name"]
    city = row.get("City", "")
    state = row.get("State", "")

    prompt = PROMPT_TEMPLATE.format(team_name=team_name, city=city, state=state)

    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    tools=tools,
                    messages=[{"role": "user", "content": prompt}],
                )
                await asyncio.sleep(REQUEST_DELAY_S)

            # Extract text from the final response block
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            if not text_blocks:
                raise ValueError("No text block in response")

            result = _extract_json(text_blocks[-1])
            # Normalise keys to lowercase, coerce None → ""
            cleaned = {k: (v or "") for k, v in result.items()}
            cleaned["status"] = "ok"
            return cleaned

        except anthropic.RateLimitError as e:
            wait = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s — enough to clear the rolling window
            log.warning(f"[{idx}] Rate limited (attempt {attempt+1}/{MAX_RETRIES}). Waiting {wait}s…")
            await asyncio.sleep(wait)
            if attempt == MAX_RETRIES:
                log.error(f"[{idx}] {team_name}: gave up after rate limit retries")
                return {"status": "error", "error": str(e)}

        except anthropic.APIError as e:
            log.error(f"[{idx}] {team_name}: API error — {e}")
            return {"status": "error", "error": str(e)}

        except (ValueError, json.JSONDecodeError) as e:
            log.warning(f"[{idx}] {team_name}: parse error — {e}. Retrying…")
            if attempt == MAX_RETRIES:
                log.error(f"[{idx}] {team_name}: gave up after parse failures")
                return {"status": "error", "error": str(e)}
            await asyncio.sleep(1)

    return {"status": "error", "error": "unknown"}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Add it to .env or your environment.")

    df = load_data()
    checkpoint = load_checkpoint()

    todo = [
        (idx, row)
        for idx, row in df.iterrows()
        if idx not in checkpoint and needs_enrichment(row)
    ]

    total = len(df)
    already_done = len(checkpoint)
    log.info(f"Total teams: {total} | Already processed: {already_done} | To enrich: {len(todo)}")

    if not todo:
        log.info("Nothing to do. All rows are complete or already processed.")
        save_progress(df, checkpoint)
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    processed = 0
    try:
        for chunk_start in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[chunk_start : chunk_start + CHUNK_SIZE]
            tasks = [enrich_team(client, semaphore, idx, row) for idx, row in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (idx, row), result in zip(chunk, results):
                team_name = row["Team Name"]

                if isinstance(result, Exception):
                    log.error(f"[{idx}] {team_name}: unhandled exception — {result}")
                    result = {"status": "error", "error": str(result)}
                else:
                    # Merge into DataFrame — only fill empty cells
                    col_to_key = {
                        "Website": "website",
                        "Email": "email",
                        "Facebook": "facebook",
                        "Instagram": "instagram",
                        "Phone": "phone",
                        "Head Coach": "head_coach",
                        "Admin Contact": "admin_contact",
                        "Address": "address",
                        "Mascot": "mascot",
                    }
                    for col, key in col_to_key.items():
                        if not str(df.at[idx, col]).strip() and result.get(key):
                            df.at[idx, col] = result[key]

                checkpoint.add(idx)
                processed += 1

                # Log entry
                append_log({
                    "idx": idx,
                    "team_name": team_name,
                    "city": row.get("City", ""),
                    "state": row.get("State", ""),
                    "website": result.get("website", ""),
                    "email": result.get("email", ""),
                    "facebook": result.get("facebook", ""),
                    "instagram": result.get("instagram", ""),
                    "phone": result.get("phone", ""),
                    "head_coach": result.get("head_coach", ""),
                    "admin_contact": result.get("admin_contact", ""),
                    "address": result.get("address", ""),
                    "mascot": result.get("mascot", ""),
                    "status": result.get("status", "error"),
                    "error": result.get("error", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            save_progress(df, checkpoint)

            # After each checkpoint: scrape websites for emails Claude couldn't find
            chunk_indices = [idx for idx, _ in chunk]
            newly_found = await scrape_emails_for_chunk(df, chunk_indices)
            if newly_found:
                save_progress(df, checkpoint)  # save again with the newly scraped emails

            log.info(f"Progress: {already_done + processed}/{total} rows complete "
                     f"({100 * (already_done + processed) / total:.1f}%) "
                     f"| +{newly_found} emails scraped from websites")

    except KeyboardInterrupt:
        log.info(f"Stopped by user. Saving progress ({already_done + processed} rows done)…")
        # Still run email scraper on whatever was in the last partial chunk
        if chunk:
            await scrape_emails_for_chunk(df, [idx for idx, _ in chunk])
        save_progress(df, checkpoint)
        log.info(f"Progress saved. Re-run the script to continue from where you left off.")
        return

    log.info(f"Done. Output: {OUTPUT_CSV} | Log: {LOG_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
