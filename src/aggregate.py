"""
Gauntlet — aggregate scores into leaderboard.

Reads scores/{model}/*.json, computes per-category, per-difficulty, and
per-dimension breakdowns, then generates leaderboard JSON and markdown.

Designed by SK. Built by Claude.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from .schema import ABResult, CATEGORY_LABELS, WEIGHTS, JudgeResult, LeaderboardEntry, Question

# ── config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gauntlet.aggregate")

DIFF_LABELS = {1: "easy", 2: "medium", 3: "hard"}


# ── helpers ─────────────────────────────────────────────────────────────
def _load_questions() -> dict[str, Question]:
    """Load questions indexed by ID."""
    questions = {}
    q_dir = ROOT / "questions" / "current"
    if not q_dir.exists():
        dirs = sorted((ROOT / "questions").glob("20*"))
        q_dir = dirs[-1] if dirs else None
    if not q_dir:
        return questions

    for path in sorted(q_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        for item in raw:
            q = Question.model_validate(item)
            questions[q.id] = q
    return questions


def _load_scores(model: str) -> list[JudgeResult]:
    """Load all scores for a model."""
    score_dir = ROOT / "scores" / model
    if not score_dir.exists():
        return []
    results = []
    for path in sorted(score_dir.glob("*.json")):
        data = json.loads(path.read_text())
        results.append(JudgeResult.model_validate(data))
    return results


def _resolve_display_name(dir_name: str) -> str:
    """Get the real model ID from response files, fall back to directory name."""
    resp_dir = ROOT / "responses" / dir_name
    if not resp_dir.exists():
        return dir_name
    for path in resp_dir.glob("*.json"):
        data = json.loads(path.read_text())
        model = data.get("model")
        if model:
            return model
    return dir_name


def _find_baselines() -> tuple[str | None, set[str]]:
    """Find baseline directories. Returns (primary, all_baselines)."""
    resp_dir = ROOT / "responses"
    if not resp_dir.exists():
        return None, set()
    primary = None
    all_bl = set()
    for d in resp_dir.iterdir():
        marker = d / ".baseline"
        if d.is_dir() and marker.exists():
            all_bl.add(d.name)
            if (d / ".primary").exists():
                primary = d.name
    return primary, all_bl


def _load_ab_results(all_baselines: set[str]) -> dict[str, dict]:
    """Load A/B results, return win rates keyed by directory name.

    Each model gets: {"wins": N, "losses": N, "ties": N, "total": N, "win_rate": float}
    """
    ab_dir = ROOT / "ab_results"
    if not ab_dir.exists():
        return {}

    # collect all results across sessions
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})

    for path in sorted(ab_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = ABResult.model_validate(json.loads(line))

            # figure out which is the local model (not a baseline)
            if r.model_a in all_baselines:
                local = r.model_b
            else:
                local = r.model_a

            if r.winner == "tie":
                counts[local]["ties"] += 1
            elif r.winner == local:
                counts[local]["wins"] += 1
            else:
                counts[local]["losses"] += 1

    result = {}
    for model, c in counts.items():
        total = c["wins"] + c["losses"] + c["ties"]
        result[model] = {**c, "total": total, "win_rate": round(c["wins"] / total * 100, 1) if total else 0}
    return result


def _build_entry(rank: int, model: str, results: list[JudgeResult], questions: dict[str, Question]) -> LeaderboardEntry:
    """Build a leaderboard entry from judge results."""
    # overall
    overall = mean(r.composite for r in results)

    # by category
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        q = questions.get(r.question_id)
        if q:
            by_cat[q.category.value].append(r.composite)
    by_category = {k: round(mean(v), 1) for k, v in by_cat.items()}

    # by difficulty
    by_diff: dict[str, list[float]] = defaultdict(list)
    for r in results:
        q = questions.get(r.question_id)
        if q:
            label = DIFF_LABELS.get(q.difficulty.value, str(q.difficulty.value))
            by_diff[label].append(r.composite)
    by_difficulty = {k: round(mean(v), 1) for k, v in by_diff.items()}

    # by dimension
    by_dim: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for dim in WEIGHTS:
            score = getattr(r.scores, dim).score
            by_dim[dim].append(score)
    by_dimension = {k: round(mean(v), 2) for k, v in by_dim.items()}

    return LeaderboardEntry(
        rank=rank,
        model=model,
        overall=round(overall, 1),
        by_category=by_category,
        by_difficulty=by_difficulty,
        by_dimension=by_dimension,
    )


# ── output ──────────────────────────────────────────────────────────────
def _write_leaderboard_md(entries: list[LeaderboardEntry], out_dir: Path, n_questions: int, primary_display: str | None = None, baseline_displays: set[str] | None = None) -> Path:
    """Generate markdown leaderboard."""
    lines = [
        f"# Gauntlet Leaderboard — {date.today().strftime('%B %Y')}",
        "",
        "> Can your model run the Gauntlet?",
        "",
        f"| Rank | Model | Overall | Investment | Biotech | General | Easy | Medium | Hard | A/B vs Baseline |",
        f"|------|-------|---------|------------|---------|---------|------|--------|------|-----------------|",
    ]

    for e in entries:
        inv = e.by_category.get("investment", 0)
        bio = e.by_category.get("biotech", 0)
        tech = e.by_category.get("general", 0)
        easy = e.by_difficulty.get("easy", 0)
        med = e.by_difficulty.get("medium", 0)
        hard = e.by_difficulty.get("hard", 0)
        ab = f"{e.ab_win_rate}%" if e.ab_win_rate is not None else "—"
        bl = baseline_displays or set()
        if e.model == primary_display:
            name = f"**{e.model} (baseline)**"
        elif e.model in bl:
            name = f"{e.model} (baseline)"
        else:
            name = e.model
        lines.append(f"| {e.rank} | {name} | {e.overall} | {inv} | {bio} | {tech} | {easy} | {med} | {hard} | {ab} |")

    lines.extend([
        "",
        f"*{n_questions} questions · Judged by latest Claude Opus (max effort via Claude Code)*",
    ])

    path = out_dir / "leaderboard.md"
    path.write_text("\n".join(lines))
    return path


def _write_leaderboard_json(entries: list[LeaderboardEntry], out_dir: Path) -> Path:
    """Save leaderboard as JSON."""
    path = out_dir / "leaderboard.json"
    data = [e.model_dump() for e in entries]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


# ── main ────────────────────────────────────────────────────────────────
def main():
    questions = _load_questions()
    if not questions:
        log.error("No questions found")
        sys.exit(1)

    scores_root = ROOT / "scores"
    if not scores_root.exists() or not any(scores_root.iterdir()):
        print("\n  No results yet. Run the benchmark first.\n")
        return

    models = sorted(d.name for d in scores_root.iterdir() if d.is_dir())
    if not models:
        print("\n  No results yet. Run the benchmark first.\n")
        return

    # baselines and A/B results
    primary, all_baselines = _find_baselines()
    ab = _load_ab_results(all_baselines)

    # build entries
    raw_entries = []
    primary_display = None
    baseline_displays = set()
    for model in models:
        results = _load_scores(model)
        if not results:
            continue
        display_name = _resolve_display_name(model)
        if model == primary:
            primary_display = display_name
        if model in all_baselines:
            baseline_displays.add(display_name)
        entry = _build_entry(0, display_name, results, questions)
        # attach A/B win rate if available (keyed by directory name)
        if model in ab:
            entry = entry.model_copy(update={"ab_win_rate": ab[model]["win_rate"]})
        raw_entries.append(entry)

    # rank by overall score
    raw_entries.sort(key=lambda e: e.overall, reverse=True)
    entries = []
    for i, e in enumerate(raw_entries, 1):
        entries.append(e.model_copy(update={"rank": i}))

    # output
    today = date.today().strftime("%Y-%m")
    out_dir = ROOT / "results" / today
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = _write_leaderboard_md(entries, out_dir, len(questions), primary_display, baseline_displays)
    json_path = _write_leaderboard_json(entries, out_dir)

    # print to terminal
    console = Console()
    console.print()
    console.rule(f"[bold]Gauntlet Leaderboard — {date.today().strftime('%B %Y')}[/bold]")
    console.print(f"  [dim]{len(questions)} questions · Judged by latest Opus (max effort)[/dim]\n")

    # main table
    t = Table(show_edge=False, pad_edge=False, box=None)
    t.add_column("#", style="bold", justify="right")
    t.add_column("Model")
    t.add_column("Overall", justify="right", style="bold")
    t.add_column("Investment", justify="right")
    t.add_column("Biotech", justify="right")
    t.add_column("General", justify="right")
    t.add_column("Easy", justify="right", style="dim")
    t.add_column("Medium", justify="right", style="dim")
    t.add_column("Hard", justify="right", style="dim")
    t.add_column("A/B", justify="right")

    for e in entries:
        is_primary = e.model == primary_display
        is_bl = e.model in baseline_displays
        inv = str(e.by_category.get("investment", 0))
        bio = str(e.by_category.get("biotech", 0))
        tech = str(e.by_category.get("general", 0))
        easy = str(e.by_difficulty.get("easy", 0))
        med = str(e.by_difficulty.get("medium", 0))
        hard = str(e.by_difficulty.get("hard", 0))
        ab_str = f"{e.ab_win_rate}%" if e.ab_win_rate is not None else "—"
        if is_primary:
            name = f"[cyan]{e.model} (baseline)[/cyan]"
            style = "on grey11"
        elif is_bl:
            name = f"[dim cyan]{e.model} (baseline)[/dim cyan]"
            style = ""
        else:
            name = e.model
            style = ""
        t.add_row(str(e.rank), name, str(e.overall), inv, bio, tech, easy, med, hard, ab_str, style=style)

    console.print(t)

    # dimension breakdown for each model
    for e in entries:
        is_bl = e.model in baseline_displays
        tag = " [cyan](baseline)[/cyan]" if is_bl else ""
        console.print(f"\n  [bold]{e.model}[/bold]{tag}")
        dims = "  ".join(f"{d.replace('_', ' ').title()}: [bold]{s:.1f}[/bold]/5" for d, s in e.by_dimension.items())
        console.print(f"  {dims}")

    console.print(f"\n  [dim]Saved: {md_path}[/dim]")
    console.print(f"  [dim]Saved: {json_path}[/dim]\n")


if __name__ == "__main__":
    main()
