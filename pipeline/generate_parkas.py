"""
generate_parkas.py — Phase 3: AI Parka Design Generation

For each swim team that has a logo, uses:
  1. Claude vision → analyzes the logo, fills in the prompt template
  2. Gemini image generation → blank parka + logo + prompt → parka design PNG

Outputs to: designs/{team_slug}.png
Resumable: skips teams whose output file already exists.

Setup:
  pip install anthropic google-genai python-dotenv pillow pandas
  Add to .env:
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=...
"""

import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY    = os.environ["GOOGLE_API_KEY"]

GEMINI_MODEL   = "gemini-3-pro-image-preview"
CLAUDE_MODEL   = "claude-sonnet-4-6"

LOGOS_DIR      = BASE / "logos"
DESIGNS_DIR    = BASE / "designs"
PARKA_TEMPLATE = BASE / "sample_blank_parka.jpeg"
CSV_PATH       = BASE / "teams_enriched.csv"
LOG_PATH       = BASE / "parka_generation_log.jsonl"

DESIGNS_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Clients ────────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
gemini_client    = genai.Client(api_key=GOOGLE_API_KEY)


# ── Helpers ────────────────────────────────────────────────────────────────────
def team_slug(name: str, state: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    slug = slug.strip("_")
    return f"{slug}_{state.lower()}"


def encode_image_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def analyze_logo_with_claude(team_name: str, logo_path: Path) -> str:
    """
    Uses Claude vision to analyze the logo and return a filled-in Gemini prompt
    describing the logo's colors, text, shapes, and the parka color.
    """
    logo_b64  = encode_image_b64(logo_path)
    media_type = "image/png"

    system = (
        "You are an expert at writing AI image generation prompts for custom embroidered apparel "
        "and have a strong eye for color theory and aesthetics. "
        "Given a swim team logo, you analyze its visual elements and write a precise, detailed prompt "
        "for a Gemini image generation model that will place the logo as embroidery on a parka. "
        "You choose parka colors that look visually stunning with the logo — not just matching, but complementary and professional ensuring that the logo is fully visible."
    )

    user_text = f"""Analyze this swim team logo for "{team_name}" and fill in the prompt template below.

Look at the logo carefully and identify:
- The best parka color to complement the logo aesthetically. Choose a color that makes the logo pop and looks professional.
- The exact text visible in the logo
- All visual elements: icons, shapes, animals, waves, etc.
- All colors present in the logo elements themselves

Fill in this template, replacing the bracketed sections with your analysis:

---
A professional product photography composite image with a plain white background, split into two panels \
side by side. Use the parka image as the reference for the parka's shape, style, cut, and pose only — decide the visually stunning color of the parka that would pair well \
LEFT PANEL: A parka with the exact same shape, hood, length, and 3/4 front-facing pose as the parka image. ensure full parka is visible, \
with the "{team_name}" logo from the team logo image added as high-quality embroidery on the left chest area. \
RIGHT PANEL: A close-up macro shot of just the embroidered logo on the parka fabric from the left panel, \
showing the thread texture in fine detail. The logo consists of \
[LOGO ELEMENTS — describe all visible text, icons, shapes, and their colors in detail]. \
Both panels are exactly equal in size, each taking up exactly half the total image width. \
Both panels use soft studio lighting on a pure white background. The overall image is wide/landscape format.
---

Return ONLY the filled-in prompt text, nothing else."""

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": logo_b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    return response.content[0].text.strip()


def generate_parka_with_gemini(parka_path: Path, logo_path: Path, prompt: str) -> bytes:
    """
    Sends blank parka + team logo + prompt to Gemini and returns the generated image bytes.
    """
    parka_bytes = parka_path.read_bytes()
    logo_bytes  = logo_path.read_bytes()

    contents = [
        types.Part.from_bytes(data=parka_bytes, mime_type="image/jpeg"),
        types.Part.from_bytes(data=logo_bytes,  mime_type="image/png"),
        types.Part.from_text(text=prompt),
    ]

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini returned no image in response")


def write_log(entry: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH)
    log.info(f"Loaded {len(df)} teams from {CSV_PATH}")

    # Find teams that have logos
    teams_with_logos = []
    for _, row in df.iterrows():
        name  = str(row["Team Name"])
        state = str(row["State"])
        slug  = team_slug(name, state)
        logo  = LOGOS_DIR / f"{slug}.png"
        if logo.exists():
            teams_with_logos.append((name, state, slug, logo))

    log.info(f"Found {len(teams_with_logos)} teams with logos")

    for i, (team_name, state, slug, logo_path) in enumerate(teams_with_logos, 1):
        out_path = DESIGNS_DIR / f"{slug}.png"

        if out_path.exists():
            log.info(f"[{i}/{len(teams_with_logos)}] SKIP {team_name} — already generated")
            continue

        log.info(f"[{i}/{len(teams_with_logos)}] Processing: {team_name} ({state})")
        entry = {
            "team": team_name,
            "state": state,
            "slug": slug,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Step 1: Claude analyzes logo → fills in prompt
            log.info(f"  → Analyzing logo with Claude...")
            prompt = analyze_logo_with_claude(team_name, logo_path)
            log.info(f"  → Prompt: {prompt[:120]}...")
            entry["prompt"] = prompt

            # Step 2: Gemini generates parka design
            log.info(f"  → Generating parka with Gemini...")
            image_bytes = generate_parka_with_gemini(PARKA_TEMPLATE, logo_path, prompt)

            # Save output
            out_path.write_bytes(image_bytes)
            log.info(f"  → Saved: {out_path}")
            entry["status"] = "success"
            entry["output"] = str(out_path)

        except Exception as e:
            log.error(f"  → FAILED: {e}")
            entry["status"] = "error"
            entry["error"]  = str(e)

        write_log(entry)

        # Small pause between teams to be kind to APIs
        if i < len(teams_with_logos):
            time.sleep(2)

    log.info("Done.")


if __name__ == "__main__":
    main()
