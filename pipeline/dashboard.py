import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from flask import Flask, jsonify, render_template

BASE = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE / "templates"))
ENRICHED_CSV = BASE / "teams_enriched.csv"
MASTER_CSV = BASE / "teams_master_list.csv"
LOG_FILE = BASE / "enrichment_log.jsonl"
LOGOS_DIR = BASE / "logos"
DESIGNS_DIR = BASE / "designs"
DESIGNS_V2_DIR = BASE / "designs_v2"

# --- Pipeline process management ---

PIPELINE_SCRIPTS = {
    "enrich":    BASE / "pipeline" / "enrich_teams.py",
    "logos":     BASE / "pipeline" / "extract_logos.py",
    "parkas_v1": BASE / "pipeline" / "generate_parkas.py",
    "parkas_v2": BASE / "pipeline" / "generate_parkas_v2_backup.py",
}
_pipeline = {name: {"status": "idle", "lines": [], "proc": None} for name in PIPELINE_SCRIPTS}
_lock = threading.Lock()
MAX_LOG_LINES = 500


def load_teams():
    import pandas as pd
    if ENRICHED_CSV.exists():
        df = pd.read_csv(ENRICHED_CSV, dtype=str).fillna("")
    else:
        df = pd.read_csv(MASTER_CSV, dtype=str).fillna("")
    return df


def get_logos():
    if not LOGOS_DIR.exists():
        return set()
    return {p.stem for p in LOGOS_DIR.glob("*.png")}


def get_designs():
    if not DESIGNS_DIR.exists():
        return set()
    return {p.stem for p in DESIGNS_DIR.glob("*.png")}


def get_designs_v2():
    if not DESIGNS_V2_DIR.exists():
        return set()
    return {p.stem for p in DESIGNS_V2_DIR.glob("*.png")}


def team_slug(name, state):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    slug = slug.strip("_")
    return f"{slug}_{state.lower()}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    df = load_teams()
    total = len(df)
    logos = get_logos()
    designs = get_designs()
    designs_v2 = get_designs_v2()

    fields = ["Website", "Email", "Facebook", "Instagram", "Phone", "Head Coach", "Admin Contact", "Mascot"]
    has_data_mask = df[fields].apply(lambda col: col.str.strip() != "").any(axis=1)
    enrichment_count = int(has_data_mask.sum())

    field_coverage = {}
    for f in fields:
        if f in df.columns:
            filled = (df[f].str.strip() != "").sum()
            field_coverage[f] = {"filled": int(filled), "total": total, "pct": round(filled / total * 100, 1)}
        else:
            field_coverage[f] = {"filled": 0, "total": total, "pct": 0.0}

    state_counts = df["State"].value_counts().to_dict()
    state_enriched = {}
    if "State" in df.columns:
        for state, group in df.groupby("State"):
            state_enriched[state] = int(has_data_mask[group.index].sum())

    return jsonify({
        "total": total,
        "enriched": enrichment_count,
        "logos": len(logos),
        "designs": len(designs),
        "designs_v2": len(designs_v2),
        "field_coverage": field_coverage,
        "state_counts": state_counts,
        "state_enriched": state_enriched,
    })


@app.route("/api/teams")
def teams():
    df = load_teams()
    logos = get_logos()
    designs = get_designs()
    designs_v2 = get_designs_v2()

    rows = []
    for i, row in df.iterrows():
        slug = team_slug(row.get("Team Name", ""), row.get("State", ""))
        has_logo = slug in logos
        has_design = slug in designs
        has_design_v2 = slug in designs_v2
        enrich_fields = ["Website", "Email", "Facebook", "Instagram", "Phone", "Head Coach", "Admin Contact", "Mascot"]
        has_any_data = any(str(row.get(f, "")).strip() for f in enrich_fields)
        status = "enriched" if has_any_data else "not enriched"

        rows.append({
            "idx": i,
            "name": row.get("Team Name", ""),
            "city": row.get("City", ""),
            "state": row.get("State", ""),
            "website": row.get("Website", ""),
            "email": row.get("Email", ""),
            "facebook": row.get("Facebook", ""),
            "instagram": row.get("Instagram", ""),
            "phone": row.get("Phone", ""),
            "coach": row.get("Head Coach", ""),
            "mascot": row.get("Mascot", ""),
            "address": row.get("Address", ""),
            "slug": slug,
            "has_logo": has_logo,
            "has_design": has_design,
            "has_design_v2": has_design_v2,
            "status": status,
        })

    return jsonify(rows)


@app.route("/api/log")
def log():
    if not LOG_FILE.exists():
        return jsonify([])
    lines = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    ok_lines = [e for e in lines if e.get("status") == "ok"]
    return jsonify(ok_lines[-50:])


@app.route("/logos/<path:filename>")
def serve_logo(filename):
    from flask import send_from_directory
    return send_from_directory(LOGOS_DIR, filename)


@app.route("/designs/<path:filename>")
def serve_design(filename):
    from flask import send_from_directory
    return send_from_directory(DESIGNS_DIR, filename)


@app.route("/designs_v2/<path:filename>")
def serve_design_v2(filename):
    from flask import send_from_directory
    return send_from_directory(DESIGNS_V2_DIR, filename)


@app.route("/api/designs")
def designs_list():
    if not DESIGNS_DIR.exists():
        return jsonify([])
    items = []
    for p in sorted(DESIGNS_DIR.glob("*.png")):
        items.append({"slug": p.stem, "filename": p.name})
    return jsonify(items)


@app.route("/api/designs_v2")
def designs_v2_list():
    if not DESIGNS_V2_DIR.exists():
        return jsonify([])
    items = []
    for p in sorted(DESIGNS_V2_DIR.glob("*.png")):
        items.append({"slug": p.stem, "filename": p.name})
    return jsonify(items)


# --- Pipeline control routes ---

@app.route("/api/run/<name>", methods=["POST"])
def run_process(name):
    if name not in PIPELINE_SCRIPTS:
        return jsonify({"error": "unknown process"}), 400
    with _lock:
        state = _pipeline[name]
        if state["status"] == "running":
            return jsonify({"error": "already running"}), 409
        script = PIPELINE_SCRIPTS[name]
        if not script.exists():
            return jsonify({"error": f"script not found: {script}"}), 404
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE),
        )
        state["proc"] = proc
        state["status"] = "running"
        state["lines"] = []

    def _reader():
        for line in proc.stdout:
            with _lock:
                _pipeline[name]["lines"].append(line.rstrip())
                if len(_pipeline[name]["lines"]) > MAX_LOG_LINES:
                    _pipeline[name]["lines"] = _pipeline[name]["lines"][-MAX_LOG_LINES:]
        proc.wait()
        with _lock:
            rc = proc.returncode
            _pipeline[name]["status"] = "done" if rc == 0 else f"error (exit {rc})"
            _pipeline[name]["proc"] = None

    threading.Thread(target=_reader, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/stop/<name>", methods=["POST"])
def stop_process(name):
    if name not in PIPELINE_SCRIPTS:
        return jsonify({"error": "unknown process"}), 400
    with _lock:
        state = _pipeline[name]
        proc = state.get("proc")
        if not proc:
            return jsonify({"error": "not running"}), 409
        proc.terminate()
        state["status"] = "stopped"
        state["proc"] = None
    return jsonify({"ok": True})


@app.route("/api/pipeline-status")
def pipeline_status():
    with _lock:
        result = {
            name: {
                "status": state["status"],
                "lines": list(state["lines"][-150:]),
            }
            for name, state in _pipeline.items()
        }
    return jsonify(result)


# --- Single-team logo extraction ---

@app.route("/api/extract-logo/<slug>", methods=["POST"])
def extract_logo_single(slug):
    if not re.match(r"^[a-z0-9_]+$", slug):
        return jsonify({"error": "invalid slug"}), 400

    df = load_teams()
    matching = next(
        (row for _, row in df.iterrows() if team_slug(row.get("Team Name", ""), row.get("State", "")) == slug),
        None,
    )
    if matching is None:
        return jsonify({"error": "team not found"}), 404

    # Import extract_logos as a module so we reuse all its logic
    import importlib.util, asyncio, aiohttp
    spec = importlib.util.spec_from_file_location("extract_logos", BASE / "pipeline" / "extract_logos.py")
    elo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(elo)

    async def _run():
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            return await elo.extract_logo_for_team(
                session=session,
                team_name=matching["Team Name"],
                state=matching["State"],
                website=matching.get("Website", ""),
                facebook=matching.get("Facebook", ""),
                instagram=matching.get("Instagram", ""),
            )

    result = asyncio.run(_run())
    return jsonify(result)


# --- Per-design deletion ---

@app.route("/api/designs/<version>/<slug>", methods=["DELETE"])
def delete_design(version, slug):
    if not re.match(r"^[a-z0-9_]+$", slug):
        return jsonify({"error": "invalid slug"}), 400
    if version == "v1":
        path = DESIGNS_DIR / f"{slug}.png"
    elif version == "v2":
        path = DESIGNS_V2_DIR / f"{slug}.png"
    else:
        return jsonify({"error": "invalid version"}), 400
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    path.unlink()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5001")
    app.run(debug=True, port=5001)
