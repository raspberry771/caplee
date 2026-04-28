# Caplee — Swim Team Parka Generator

## Project Goal

Build a 3-phase pipeline to produce custom AI-generated parka designs for 2,000+ U.S. swim clubs:

1. **Scrape** — collect team data (name, city, state, website, contact info, mascot) for every team
2. **Extract Logos** — download each team's logo as a PNG for use in the AI image generator
3. **Generate Parka Designs** — feed each team's logo + a custom AI prompt + a blank parka image into the Nano Banana AI image generator to produce a finished parka design per team

---

## Current State

| File                    | Purpose                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `teamsheets.py`         | Phase 0: Parses `U.S. Swimming Team Club Ranking 2022.pdf` → `teams_master_list.csv`                              |
| `enrich_teams.py`       | Phase 1: Uses Claude API (with web search) to enrich each team with website, email, social, coach, mascot, etc.   |
| `teams_master_list.csv` | Raw team list: Team Name, City, State (~2,000+ rows)                                                              |
| `teams_enriched.csv`    | Output of enrichment: adds Website, Email, Facebook, Instagram, Phone, Head Coach, Admin Contact, Address, Mascot |
| `checkpoint.json`       | Tracks which rows have been processed so enrichment is resumable                                                  |
| `enrichment_log.jsonl`  | Per-team enrichment log with timestamps and status                                                                |

---

## Phase 1 — Team Data Enrichment (`enrich_teams.py`)

- Uses `claude-sonnet-4-6` with the `web_search_20250305` tool
- Concurrent async processing with semaphore (tune `CONCURRENT_REQUESTS` for API tier)
- Resumable via `checkpoint.json` — safe to kill and restart
- After each chunk, scrapes team websites directly for emails Claude couldn't find
- Requires `ANTHROPIC_API_KEY` in `.env`

```bash
pip install anthropic pandas python-dotenv requests beautifulsoup4 pypdf
python enrich_teams.py
```

---

## Phase 2 — Logo Extraction (next to build)

- Input: `teams_enriched.csv` (specifically the `Website` column)
- Goal: download each team's logo as a clean PNG
- Strategy: scrape `<img>` tags, favicon, or OG image from team websites; fall back to searching for the logo via web
- Output: `logos/` directory with one PNG per team, named by a slug of the team name
- Logo quality matters — these go directly into an AI image generator (Nano Banana)

---

## Phase 3 — Parka Design Generation (next to build)

- Tool: **Nano Banana** (powered by **Google AI Studio API** — Gemini image generation)
- Inputs per team:
  1. Team logo PNG — clean, background-removed (from Phase 2)
  2. Blank parka image (one shared template, e.g. `parka_blank.png`)
  3. Auto-generated prompt via Claude vision — Claude analyzes the logo PNG and writes a detailed Nano Banana-style prompt describing logo colors, text, shapes, and embroidery placement
- Output: one rendered parka design PNG per team
- Fully automatable via Google AI Studio API (Gemini) in a batch loop

### Prompt Template (based on validated manual test)

```
A detailed, photorealistic studio photograph of the black hooded swim parka from image_0.png
against a plain white background. The parka is shown in the same 3/4 front-facing pose.
Positioned clearly on the left chest area is the "[TEAM NAME]" logo from image_1.png, rendered
as high-quality embroidery. The background of the original logo has been removed, so
only [LOGO ELEMENTS — text, icon, colors] are visible. The embroidery texture is detailed,
blending seamlessly with the folds and wrinkles of the parka's technical fabric. Soft studio
lighting highlights the texture of both the nylon parka and the threads of the logo.
The perspective is maintained from image_0.png.
```

Claude vision fills in the bracketed fields automatically by analyzing each team's logo PNG.

### Automation Flow

```
for each team:
    1. load logo PNG (Phase 2 output)
    2. Claude vision → analyze logo → generate custom prompt
    3. Gemini API (Google AI Studio) → parka_blank.png + logo PNG + prompt → parka design PNG
    4. save to designs/{team_slug}.png
```

---

## Key Rules

- Enrichment is **never destructive** — only fill empty cells, never overwrite existing data
- Always check `checkpoint.json` before re-running enrichment to avoid duplicate API calls
- Logo files should be named using a consistent slug: `{team_name}_{state}.png` (lowercase, spaces → underscores)
- Do not hardcode API keys — always use `.env` / environment variables
