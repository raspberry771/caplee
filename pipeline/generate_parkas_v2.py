"""
generate_parkas_v2.py — Phase 3 v2: 3-Panel Parka Design Card Generation

For each swim team that has a logo, produces a 3-panel design card:
  LEFT   — front parka (color-matched to team) with logo embroidered on left chest
  TOP-RIGHT  — team logo upscaled on solid background
  BOTTOM-RIGHT — back parka with team name printed on upper back panel

Pipeline per team:
  1. Upscale logo in memory (LANCZOS 4x + UnsharpMask)
  2. Claude vision → selects best parka color combo + describes logo
  3. Gemini image gen → frame + front parka + back parka + logo → design card

Outputs to: designs_v2/{team_slug}.png
Resumable: skips teams whose output file already exists (checkpoint_v2.json).

Setup:
  pip install anthropic google-genai python-dotenv pillow pandas
  .env must contain: ANTHROPIC_API_KEY and GOOGLE_API_KEY
"""

import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageFilter

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY    = os.environ["GOOGLE_API_KEY"]

CLAUDE_MODEL  = "claude-sonnet-4-6"
GEMINI_MODEL  = "gemini-3-pro-image-preview"

LOGOS_DIR      = BASE / "logos"
DESIGNS_DIR    = BASE / "designs_v2"
PARKA_FRONT    = BASE / "Parka Front"
PARKA_BACK     = BASE / "Parka Back"
FRAME_PATH     = BASE / "templates" / "frame_v2.png"
CSV_PATH       = BASE / "teams_enriched.csv"
LOG_PATH       = BASE / "parka_v2_log.jsonl"

# Upscale logo before sending to Gemini (recommended — logos can be very small)
USE_UPSCALER      = True
UPSCALE_TARGET    = 600   # px on the longest side after upscaling
SAVE_UPSCALED     = True  # Save upscaled logos to logos_upscaled/ for inspection

# Debug: save all 4 Gemini inputs + full prompt to debug/{slug}/ for inspection
DEBUG_SAVE_INPUTS = True

LOGOS_UPSCALED_DIR = BASE / "logos_upscaled"
DEBUG_DIR          = BASE / "debug"

DESIGNS_DIR.mkdir(exist_ok=True)
if SAVE_UPSCALED:
    LOGOS_UPSCALED_DIR.mkdir(exist_ok=True)
if DEBUG_SAVE_INPUTS:
    DEBUG_DIR.mkdir(exist_ok=True)

# All 12 available parka color combos (Primary-Secondary)
# Primary  = outer shell color
# Secondary = arm stripe + hood lining
PARKA_COMBOS = [
    "Black-Royal",
    "Black-Orange",
    "Black-Purple",
    "Black-Red",
    "Black-White",
    "Black-Yellow",
    "Black-Green",
    "Navy-Red",
    "Navy-White",
    "Navy-Yellow",
    "Royal-White",
    "Royal-Yellow",
]

# Suffixes to strip from long team names on the back panel
# Only applied when name exceeds NAME_LENGTH_THRESHOLD characters
NAME_LENGTH_THRESHOLD = 20
NAME_SUFFIXES_TO_STRIP = [
    "Swimming Club",
    "Swim Club",
    "Aquatics Club",
    "Aquatic Club",
    "Swim Team",
    "Swimming",
    "Aquatics",
    "Aquatic",
    ", Inc.",
    " Inc.",
    ", LLC",
]

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
    return f"{slug.strip('_')}_{state.lower()}"


def shorten_name(name: str) -> str:
    """Strip common swim-club suffixes from long team names for the back panel."""
    if len(name) <= NAME_LENGTH_THRESHOLD:
        return name.upper()
    shortened = name
    for suffix in NAME_SUFFIXES_TO_STRIP:
        if shortened.lower().endswith(suffix.lower()):
            shortened = shortened[: -len(suffix)].strip().rstrip(",").strip()
            break
    return shortened.upper()


def get_parka_paths(combo: str) -> tuple[Path, Path]:
    """
    Locate front and back parka images for a given combo (e.g. 'Navy-Red').
    Handles mixed .jpg / .JPG / .jpeg extensions.
    """
    def find_file(directory: Path, stem: str) -> Path:
        for ext in (".jpg", ".JPG", ".jpeg", ".JPEG"):
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No parka image found for stem '{stem}' in {directory}")

    front = find_file(PARKA_FRONT, f"CS1600-{combo}-F")
    back  = find_file(PARKA_BACK,  f"CS1600-{combo}-Back")
    return front, back


def upscale_logo(logo_path: Path) -> bytes:
    """
    Upscale the logo in memory using LANCZOS + UnsharpMask sharpening.
    Returns PNG bytes ready to send to Gemini.
    If USE_UPSCALER is False, returns the original file bytes unchanged.

    Semi-transparent logos (e.g. logos extracted as RGBA with partial alpha)
    are composited onto a white background before upscaling so Gemini receives
    a fully opaque image. The source file in logos/ is never modified.
    """
    if not USE_UPSCALER:
        return logo_path.read_bytes()

    img = Image.open(logo_path).convert("RGBA")

    # Flatten onto white if any pixel is not fully opaque.
    # This prevents Gemini from receiving a ghost/semi-transparent logo.
    alpha = img.getchannel("A")
    if alpha.getextrema()[0] < 255:
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        white.paste(img, mask=alpha)
        img = white.convert("RGB").convert("RGBA")

    w, h = img.size
    longest = max(w, h)

    if longest >= UPSCALE_TARGET:
        # Already large enough — just sharpen slightly and return
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    else:
        scale = UPSCALE_TARGET / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    if SAVE_UPSCALED:
        save_path = LOGOS_UPSCALED_DIR / logo_path.name
        save_path.write_bytes(data)

    return data


def encode_b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("utf-8")


def analyze_logo_with_claude(team_name: str, logo_path: Path) -> dict:
    """
    Send the team logo to Claude. Returns a dict with:
      parka_color    — one of PARKA_COMBOS (e.g. "Navy-Red")
      logo_description — detailed description of logo elements for the Gemini prompt
      back_text_color  — "white" or "black" for best contrast on the chosen parka
    """
    logo_bytes = upscale_logo(logo_path) if USE_UPSCALER else logo_path.read_bytes()
    logo_b64   = encode_b64(logo_bytes)

    combos_str = "\n".join(f"  - {c}" for c in PARKA_COMBOS)

    system = (
        "You are an expert apparel designer specializing in swim team merchandise. "
        "You have a strong eye for color theory and know how to match team logos to "
        "parka colorways that look professional and make the logo pop."
    )

    user_text = f"""Analyze this swim team logo for "{team_name}" and return a JSON object with exactly these four fields:

1. "parka_color": Choose the single best parka colorway from this list:
{combos_str}
   CRITICAL RULE: The logo must be clearly visible against the parka's PRIMARY color (the outer shell).
   If the logo is predominantly blue, do NOT pick Royal or Navy — pick Black instead.
   If the logo is predominantly black, do NOT pick Black — pick Navy or Royal instead.
   Prioritize contrast and visibility above all else. The logo should pop off the parka, not blend in.

2. "back_text_color": Either "white" or "black" — whichever will be most readable
   printed on the PRIMARY color of your chosen parka_color.

3. "display_name": The short name to print on the back of the parka.
   Look at the text visible IN the logo itself and use that as the display name if it is short and clean.
   For example, if the logo says "NOVA SWIMMING", use "NOVA SWIMMING".
   If the logo says "TAC TITANS", use "TAC TITANS".
   If the logo has no readable short name, shorten the team name "{team_name}" by removing
   generic suffixes (Swim Club, Aquatics, Inc, Swimming Club, Swim Team) and return what remains.
   Keep it to 3 words or fewer if possible. Return in ALL CAPS.

Return ONLY valid JSON, no other text."""

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
                            "media_type": "image/png",
                            "data": logo_b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if Claude wraps the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def build_gemini_prompt(
    team_name: str,
    back_name: str,
    back_text_color: str,
    parka_color: str,
) -> str:
    primary_color = parka_color.split("-")[0].lower()  # e.g. "navy"

    return f"""Create a product design card with the exact 3-panel layout shown in image_0. Plain white background.
image_0 = layout reference | image_1 = front parka photo | image_2 = back parka photo | image_3 = team logo

CRITICAL: image_1 and image_2 are real product photos. You must reproduce these exact garments —
do not change the parka style, silhouette, cut, length, hood shape, zipper, pockets, sleeve stripes,
or any physical feature. The only modifications allowed are adding the logo to the front and text to the back.

LEFT PANEL (full height, ~57% of total width):
Reproduce the exact parka from image_1 — same style, same color, same pose, same every detail.
Add the "{team_name}" logo from image_3 as embroidery on the upper left chest area.
The embroidery threads blend naturally with the fabric.

TOP RIGHT PANEL (upper portion of right side):
Show the logo from image_3 sharp and crisp, centered on a solid {primary_color} background square.
Single logo only — do not repeat or tile it.

BOTTOM RIGHT PANEL (lower portion of right side):
Reproduce the exact parka from image_2 — same style, same color, same pose, same every detail.
Add "{back_name}" as bold, centered {back_text_color} capital letters on the upper back panel only.
Clean athletic sans-serif font. No other changes to the parka.

Soft even studio lighting. Pure white background on all panels."""


def save_debug_inputs(slug: str, frame_path: Path, front_path: Path, back_path: Path, logo_bytes: bytes, prompt: str):
    d = DEBUG_DIR / slug
    d.mkdir(exist_ok=True)
    (d / "image_0_frame.png").write_bytes(frame_path.read_bytes())
    (d / "image_1_front_parka.jpg").write_bytes(front_path.read_bytes())
    (d / "image_2_back_parka.jpg").write_bytes(back_path.read_bytes())
    (d / "image_3_logo.png").write_bytes(logo_bytes)
    (d / "prompt.txt").write_text(prompt, encoding="utf-8")


def generate_with_gemini(
    frame_path: Path,
    front_path: Path,
    back_path: Path,
    logo_bytes: bytes,
    prompt: str,
    slug: str = "",
) -> bytes:
    """Send 4 images + prompt to Gemini, return generated image bytes."""
    if DEBUG_SAVE_INPUTS and slug:
        save_debug_inputs(slug, frame_path, front_path, back_path, logo_bytes, prompt)

    contents = [
        types.Part.from_bytes(data=frame_path.read_bytes(), mime_type="image/png"),
        types.Part.from_bytes(data=front_path.read_bytes(), mime_type="image/jpeg"),
        types.Part.from_bytes(data=back_path.read_bytes(),  mime_type="image/jpeg"),
        types.Part.from_bytes(data=logo_bytes,              mime_type="image/png"),
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


def load_checkpoint() -> set:
    """Derive completed set from files that actually exist on disk."""
    if not DESIGNS_DIR.exists():
        return set()
    return {p.stem for p in DESIGNS_DIR.glob("*.png")}


def write_log(entry: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Verify frame template exists
    if not FRAME_PATH.exists():
        log.error(f"Frame template not found at {FRAME_PATH}. Re-run setup.")
        return

    df = pd.read_csv(CSV_PATH)
    log.info(f"Loaded {len(df)} teams from CSV")

    # Find teams that have logos
    teams = []
    for _, row in df.iterrows():
        name  = str(row["Team Name"])
        state = str(row["State"])
        slug  = team_slug(name, state)
        logo  = LOGOS_DIR / f"{slug}.png"
        if logo.exists():
            teams.append((name, state, slug, logo))

    log.info(f"Found {len(teams)} teams with logos")

    done = load_checkpoint()
    remaining = [(n, s, sl, l) for n, s, sl, l in teams if sl not in done]
    log.info(f"{len(done)} already completed, {len(remaining)} remaining")

    for i, (team_name, state, slug, logo_path) in enumerate(remaining, 1):
        out_path = DESIGNS_DIR / f"{slug}.png"
        log.info(f"[{i}/{len(remaining)}] {team_name} ({state})")

        entry = {
            "team": team_name,
            "state": state,
            "slug": slug,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Step 1: Upscale logo in memory
            log.info("  → Upscaling logo...")
            logo_bytes = upscale_logo(logo_path)

            # Step 2: Claude selects color + describes logo
            log.info("  → Analyzing logo with Claude...")
            analysis = analyze_logo_with_claude(team_name, logo_path)
            parka_color      = analysis["parka_color"]
            logo_description = analysis["logo_description"]
            back_text_color  = analysis.get("back_text_color", "white")
            display_name     = analysis.get("display_name") or shorten_name(team_name)

            # Validate Claude's color choice
            if parka_color not in PARKA_COMBOS:
                log.warning(f"  Claude returned unknown combo '{parka_color}', defaulting to Black-Royal")
                parka_color = "Black-Royal"

            log.info(f"  → Parka color: {parka_color}")
            entry["parka_color"] = parka_color

            # Step 3: Load parka images
            front_path, back_path = get_parka_paths(parka_color)

            # Step 4: Build prompt
            prompt = build_gemini_prompt(
                team_name=team_name,
                back_name=display_name,
                logo_description=logo_description,
                back_text_color=back_text_color,
                parka_color=parka_color,
            )
            log.info(f"  → Back name: {display_name}")
            entry["prompt"] = prompt

            # Step 5: Gemini generates the design card
            log.info("  → Generating with Gemini...")
            image_bytes = generate_with_gemini(
                frame_path=FRAME_PATH,
                front_path=front_path,
                back_path=back_path,
                logo_bytes=logo_bytes,
                prompt=prompt,
                slug=slug,
            )

            out_path.write_bytes(image_bytes)
            log.info(f"  → Saved: {out_path.name}")
            entry["status"] = "success"
            entry["output"] = str(out_path)

            done.add(slug)

        except Exception as e:
            log.error(f"  → FAILED: {e}")
            entry["status"] = "error"
            entry["error"]  = str(e)

        write_log(entry)

        if i < len(remaining):
            time.sleep(2)

    log.info(f"Done. {len(done)} total completed.")


if __name__ == "__main__":
    main()
