"""
Gauntlet — A/B blind comparison in the terminal.

Shows Opus baseline vs local model responses side-by-side (randomized order).
Human judges pick a winner. Results saved to ab_results/.

Designed by SK. Built by Claude.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .schema import ABResult, Question

# ── config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
console = Console()


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


def _load_response(model: str, qid: str) -> str | None:
    """Load a model response."""
    path = ROOT / "responses" / model / f"{qid}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("content") or data.get("reasoning") or ""


def _find_primary_baseline() -> str | None:
    """Find the primary baseline (has .primary marker)."""
    resp_dir = ROOT / "responses"
    if not resp_dir.exists():
        return None
    for d in resp_dir.iterdir():
        if d.is_dir() and (d / ".primary").exists():
            return d.name
    return None


def _find_baselines() -> set[str]:
    """Find all baseline directories."""
    resp_dir = ROOT / "responses"
    if not resp_dir.exists():
        return set()
    return {d.name for d in resp_dir.iterdir() if d.is_dir() and (d / ".baseline").exists()}


def _find_models(baselines: set[str]) -> list[str]:
    """Find all non-baseline models with responses."""
    resp_dir = ROOT / "responses"
    if not resp_dir.exists():
        return []
    return [d.name for d in sorted(resp_dir.iterdir()) if d.is_dir() and d.name not in baselines]


def _save_result(result: ABResult, session_id: str) -> None:
    """Append result to session JSONL."""
    out_dir = ROOT / "ab_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session_id}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(result.model_dump()) + "\n")


# ── display ─────────────────────────────────────────────────────────────
def _show_question(q: Question, idx: int, total: int) -> None:
    """Display the question."""
    console.print()
    console.rule(f"[bold]Question {idx}/{total}[/bold]  {q.id}  ({q.category.value}, d={q.difficulty.value})")
    console.print()
    console.print(Panel(q.prompt.system, title="System", border_style="dim"))
    for msg in q.prompt.messages:
        console.print(Panel(Markdown(msg.content), title=msg.role, border_style="blue"))


def _show_responses(resp_a: str, resp_b: str) -> None:
    """Display two responses side by side."""
    console.print()
    console.print(Panel(Markdown(resp_a), title="[bold green]Response A[/bold green]", border_style="green"))
    console.print()
    console.print(Panel(Markdown(resp_b), title="[bold yellow]Response B[/bold yellow]", border_style="yellow"))
    console.print()


DEFAULT_COUNT = 10


# ── main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gauntlet — A/B blind comparison")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"Number of questions (default {DEFAULT_COUNT}, 0=all)")
    parser.add_argument("--model", help="Local model to compare (directory name under responses/)")
    args = parser.parse_args()

    questions = _load_questions()
    if not questions:
        console.print("[red]No questions found[/red]")
        sys.exit(1)

    baseline = _find_primary_baseline()
    if not baseline:
        console.print("[red]No baseline found. Run: ./gauntlet.sh baseline[/red]")
        sys.exit(1)

    all_baselines = _find_baselines()
    local_models = _find_models(all_baselines)
    if not local_models:
        console.print("[red]No local model responses found. Run: ./gauntlet.sh local[/red]")
        sys.exit(1)

    # pick model to compare
    if args.model:
        model = args.model
    elif len(local_models) == 1:
        model = local_models[0]
    else:
        console.print("\nAvailable models:")
        for i, m in enumerate(local_models, 1):
            console.print(f"  {i}. {m}")
        choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(local_models) + 1)])
        model = local_models[int(choice) - 1]

    # find questions with responses from both baseline and the local model
    eligible = []
    for qid, q in questions.items():
        bl_resp = _load_response(baseline, qid)
        local_resp = _load_response(model, qid)
        if bl_resp and local_resp:
            eligible.append((q, bl_resp, local_resp))

    if not eligible:
        console.print(f"[red]No questions with both baseline and {model} responses[/red]")
        sys.exit(1)

    random.shuffle(eligible)
    if args.count > 0:
        eligible = eligible[:args.count]
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    console.print(f"\n[bold]Gauntlet A/B Test[/bold]")
    console.print(f"  Baseline ({baseline}) vs {model}")
    console.print(f"  {len(eligible)} questions  session={session_id}")
    console.print(f"  Keys: [green]1[/green]=A wins  [yellow]2[/yellow]=B wins  [blue]3[/blue]=tie  [red]q[/red]=quit\n")

    wins = {"baseline": 0, "local": 0, "tie": 0}

    for i, (q, bl_resp, local_resp) in enumerate(eligible, 1):
        # randomize order
        bl_is_a = random.random() < 0.5
        if bl_is_a:
            resp_a, resp_b = bl_resp, local_resp
        else:
            resp_a, resp_b = local_resp, bl_resp

        _show_question(q, i, len(eligible))
        _show_responses(resp_a, resp_b)

        t0 = time.time()
        choice = Prompt.ask("Winner", choices=["1", "2", "3", "q"])
        elapsed = time.time() - t0

        if choice == "q":
            break

        # map choice to winner
        if choice == "3":
            winner = "tie"
            wins["tie"] += 1
        elif (choice == "1" and bl_is_a) or (choice == "2" and not bl_is_a):
            winner = baseline
            wins["baseline"] += 1
        else:
            winner = model
            wins["local"] += 1

        result = ABResult(
            question_id=q.id,
            model_a=baseline if bl_is_a else model,
            model_b=model if bl_is_a else baseline,
            winner=winner,
            response_time_sec=round(elapsed, 1),
        )
        _save_result(result, session_id)

        # running tally
        total = wins["baseline"] + wins["local"] + wins["tie"]
        console.print(
            f"  [dim]Score: Baseline {wins['baseline']} — {model} {wins['local']} — Tie {wins['tie']}  "
            f"({total} judged)[/dim]"
        )

    # final summary
    total = wins["baseline"] + wins["local"] + wins["tie"]
    if total > 0:
        console.print(f"\n{'='*60}")
        console.print(f"  [bold]Session Complete[/bold]  ({total} questions judged)\n")
        console.print(f"  Baseline:  {wins['baseline']:3d}  ({wins['baseline']/total*100:.0f}%)")
        console.print(f"  {model}: {wins['local']:3d}  ({wins['local']/total*100:.0f}%)")
        console.print(f"  Tie:       {wins['tie']:3d}  ({wins['tie']/total*100:.0f}%)")
        console.print(f"\n  Results saved: ab_results/{session_id}.jsonl")

        # update leaderboard with latest A/B results
        console.print(f"  Updating leaderboard...\n")
        from .aggregate import main as aggregate_main
        aggregate_main()


if __name__ == "__main__":
    main()
