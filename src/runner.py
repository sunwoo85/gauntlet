"""
Gauntlet — run benchmark questions against local models and Opus baseline.

Local models: httpx to OpenAI-compatible endpoints.
Opus baseline: Claude Code CLI subprocess (max effort).

Designed by SK. Built by Claude.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tomllib
from datetime import date
from pathlib import Path

from .clients import _resolve_model, call_claude_code, call_local
from .schema import CATEGORY_LABELS, Question

# ── config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gauntlet")


# ── helpers ─────────────────────────────────────────────────────────────
def _load_questions() -> list[Question]:
    """Load all questions from the current question set."""
    questions = []
    q_dir = ROOT / "questions" / "current"
    if not q_dir.exists():
        dirs = sorted((ROOT / "questions").glob("20*"))
        q_dir = dirs[-1] if dirs else None
    if not q_dir:
        log.error("No question directories found")
        sys.exit(1)

    for path in sorted(q_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        questions.extend(Question.model_validate(q) for q in raw)
    log.info("Loaded %d questions from %s", len(questions), q_dir)
    return questions


def _load_endpoint() -> dict:
    """Load endpoint config from TOML."""
    with open(ROOT / "config" / "models.toml", "rb") as f:
        return tomllib.load(f).get("endpoint", {})


def _load_judge_config() -> dict:
    """Load judge/baseline config from TOML."""
    with open(ROOT / "config" / "judge.toml", "rb") as f:
        return tomllib.load(f)


def _build_prompt(q: Question) -> str:
    """Build a flat prompt string from a question for Claude Code CLI."""
    parts = [q.prompt.system, ""]
    for doc in q.prompt.context_documents:
        parts.append(f"[{doc.type}]\n{doc.content}\n")
    for msg in q.prompt.messages:
        parts.append(msg.content)
    return "\n".join(parts)


def _safe_dirname(model_id: str) -> str:
    """Convert model ID to a safe directory name."""
    return model_id.replace("/", "--")


def _save_response(dir_name: str, question_id: str, response: dict) -> Path:
    """Save a model response to disk."""
    out_dir = ROOT / "responses" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{question_id}.json"
    path.write_text(json.dumps(response, ensure_ascii=False, indent=2))
    return path


# ── runners ─────────────────────────────────────────────────────────────
async def run_local_model(endpoint: dict, questions: list[Question]) -> int:
    """Run all questions against the local model at the endpoint."""
    url = endpoint["url"]

    model_id, max_context = await _resolve_model(url)
    max_tokens = max(32768, max_context // 4) if max_context else 32768
    default_concurrency = (max_context // max_tokens) if max_context else 4
    concurrency = endpoint.get("concurrency", default_concurrency)
    sampling = endpoint.get("sampling")
    dir_name = _safe_dirname(model_id)

    print(f"\n{'='*60}")
    print(f"  {model_id} ({len(questions)} questions, concurrency={concurrency})")
    print(f"  max_tokens={max_tokens} (quarter of {max_context})")
    print(f"{'='*60}\n")

    # filter to uncached questions
    todo = []
    cached = 0
    for q in questions:
        out_path = ROOT / "responses" / dir_name / f"{q.id}.json"
        if out_path.exists():
            log.info("  skip %s (cached)", q.id)
            cached += 1
        else:
            todo.append(q)

    count = cached
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(q: Question) -> bool:
        async with sem:
            messages = [{"role": m.role, "content": m.content} for m in q.prompt.messages]
            try:
                resp = await call_local(
                    endpoint=url,
                    model_id=model_id,
                    system=q.prompt.system,
                    messages=messages,
                    max_tokens=max_tokens,
                    sampling=sampling,
                )
                resp["question_id"] = q.id
                _save_response(dir_name, q.id, resp)
                log.info("  %s  %.1fs  %s", q.id, resp["duration_ms"] / 1000, model_id)
                return True
            except Exception as e:
                log.error("  %s  FAILED  %s: %s", q.id, model_id, e)
                return False

    results = await asyncio.gather(*[_run_one(q) for q in todo])
    count += sum(1 for r in results if r)

    print(f"\n  ✓ {model_id}: {count}/{len(questions)} done\n")
    return count


def run_baseline(questions: list[Question], cfg: dict, is_primary: bool = False) -> int:
    """Generate baseline responses via Claude Code CLI."""
    model = cfg.get("model", "opus")
    effort = cfg.get("effort", "high")
    timeout = cfg.get("timeout_sec", 300)
    month = date.today().strftime("%Y-%m")

    # use model alias as directory name (e.g. "opus", "sonnet", "haiku")
    # the actual model ID gets stored in each response JSON
    dir_name = model

    # check if baseline is from a different month — if so, clear and re-run
    out_dir = ROOT / "responses" / dir_name
    marker = out_dir / ".baseline"
    if marker.exists() and marker.read_text().strip() != month:
        log.info("Baseline %s is from %s, refreshing for %s", model, marker.read_text().strip(), month)
        shutil.rmtree(out_dir)
        scores_dir = ROOT / "scores" / dir_name
        if scores_dir.exists():
            shutil.rmtree(scores_dir)

    # mark as baseline with current month
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(month)
    if is_primary:
        (out_dir / ".primary").touch()

    print(f"\n{'='*60}")
    tag = " (primary)" if is_primary else ""
    print(f"  Baseline{tag}: {model}  effort={effort}  month={month}")
    print(f"  {len(questions)} questions")
    print(f"{'='*60}\n")

    count = 0
    for q in questions:
        out_path = out_dir / f"{q.id}.json"
        if out_path.exists():
            log.info("  skip %s (cached)", q.id)
            count += 1
            continue

        prompt = _build_prompt(q)
        try:
            resp = call_claude_code(prompt, model=model, effort=effort, timeout=timeout)
            resp["question_id"] = q.id
            _save_response(dir_name, q.id, resp)
            count += 1
            log.info("  %s  %.1fs  %s", q.id, resp["duration_ms"] / 1000, model)
        except Exception as e:
            log.error("  %s  FAILED  %s: %s", q.id, model, e)
    return count


# ── clean ───────────────────────────────────────────────────────────────
def _list_models() -> list[str]:
    """List all model directories with responses or scores."""
    dirs = set()
    for parent in [ROOT / "responses", ROOT / "scores"]:
        if parent.exists():
            dirs.update(d.name for d in parent.iterdir() if d.is_dir())
    return sorted(dirs)


def _is_baseline(name: str) -> bool:
    return (ROOT / "responses" / name / ".baseline").exists()


def _is_primary(name: str) -> bool:
    return (ROOT / "responses" / name / ".primary").exists()


def _baseline_month(name: str) -> str | None:
    marker = ROOT / "responses" / name / ".baseline"
    if marker.exists():
        return marker.read_text().strip() or None
    return None


def _clean(name: str) -> None:
    """Remove responses, scores, and ab_results for a model."""
    removed = []
    for parent in [ROOT / "responses", ROOT / "scores"]:
        target = parent / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(ROOT)))

    # remove A/B results referencing this model
    ab_dir = ROOT / "ab_results"
    if ab_dir.exists():
        for path in ab_dir.glob("*.jsonl"):
            lines = path.read_text().splitlines()
            kept = [l for l in lines if name not in l]
            if len(kept) < len(lines):
                path.write_text("\n".join(kept) + "\n" if kept else "")
                removed.append(f"ab_results/{path.name} (filtered)")

    if removed:
        for r in removed:
            print(f"  removed {r}")
    else:
        print(f"  nothing found for '{name}'")


def _reset() -> None:
    """Remove all responses, scores, ab_results, and results."""
    removed = []
    for d in [ROOT / "responses", ROOT / "scores", ROOT / "ab_results", ROOT / "results"]:
        if d.exists():
            for child in list(d.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed.append(str(child.relative_to(ROOT)))

    if removed:
        print(f"\n  Cleaned {len(removed)} items. Fresh state.\n")
    else:
        print("\n  Already clean.\n")


# ── generate references ─────────────────────────────────────────────────
def _generate_references(questions: list[Question]) -> None:
    """Generate reference answers via Opus max effort and update question files."""
    print(f"\n{'='*60}")
    print(f"  Generating reference answers (Opus, max effort)")
    print(f"  {len(questions)} questions")
    print(f"{'='*60}\n")

    # group questions by their source file
    q_files: dict[Path, list] = {}
    q_dir = ROOT / "questions" / "current"
    if not q_dir.exists():
        dirs = sorted((ROOT / "questions").glob("20*"))
        q_dir = dirs[-1] if dirs else None

    for path in sorted(q_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        q_files[path] = raw

    updated = 0
    for path, items in q_files.items():
        changed = False
        for item in items:
            qid = item["id"]

            # step 1: generate reference answer
            task_prompt = item["prompt"]["system"] + "\n\n"
            for doc in item["prompt"].get("context_documents", []):
                task_prompt += f"[{doc['type']}]\n{doc['content']}\n\n"
            for msg in item["prompt"]["messages"]:
                task_prompt += msg["content"]

            print(f"  {qid} (answer)...", end="", flush=True)
            try:
                resp = call_claude_code(task_prompt, model="opus", effort="max", timeout=600)
                ref_answer = resp["content"]
                item["evaluation"]["reference_answer"] = ref_answer
                print(f"  done ({resp['duration_ms']/1000:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

            # step 2: generate key_facts and disqualifiers from the reference answer
            eval_prompt = (
                "Given this question and reference answer, generate evaluation criteria.\n\n"
                f"## Question\n{task_prompt}\n\n"
                f"## Reference Answer\n{ref_answer}\n\n"
                "Respond with ONLY valid JSON, no other text:\n"
                "{\n"
                '  "key_facts": ["fact that must be covered", ...],\n'
                '  "disqualifiers": ["error that should cap the score at 2", ...]\n'
                "}\n\n"
                "key_facts: 3-5 essential points a good answer must cover.\n"
                "disqualifiers: 1-3 serious errors that indicate a fundamentally wrong answer."
            )

            print(f"  {qid} (eval criteria)...", end="", flush=True)
            try:
                resp2 = call_claude_code(eval_prompt, model="opus", effort="high", timeout=300)
                text = resp2["content"].strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                criteria = json.loads(text)
                item["evaluation"]["key_facts"] = criteria.get("key_facts", [])
                item["evaluation"]["disqualifiers"] = criteria.get("disqualifiers", [])
                changed = True
                updated += 1
                print(f"  done ({resp2['duration_ms']/1000:.1f}s)")
            except Exception as e:
                # still save the reference answer even if criteria fail
                changed = True
                updated += 1
                print(f"  criteria FAILED ({e}), kept reference answer")

        if changed:
            path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")

    print(f"\n  {updated}/{len(questions)} evaluations updated\n")


# ── review references ──────────────────────────────────────────────────
def _review_references(questions: list[Question]) -> None:
    """Review reference answers one by one. Approve, regenerate, or skip."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()

    q_dir = ROOT / "questions" / "current"
    if not q_dir.exists():
        dirs = sorted((ROOT / "questions").glob("20*"))
        q_dir = dirs[-1] if dirs else None

    # load raw files so we can write back
    q_files: dict[Path, list] = {}
    for path in sorted(q_dir.glob("*.json")):
        q_files[path] = json.loads(path.read_text())

    # build lookup: question id -> (file path, index in list)
    lookup: dict[str, tuple[Path, int]] = {}
    for path, items in q_files.items():
        for i, item in enumerate(items):
            lookup[item["id"]] = (path, i)

    approved = 0
    regenerated = 0
    skipped = 0

    console.print(f"\n[bold]Review Reference Answers[/bold]")
    console.print(f"  {len(questions)} questions")
    console.print(f"  Keys: [green]y[/green]=approve  [yellow]r[/yellow]=regenerate (Opus max)  [dim]s[/dim]=skip  [red]q[/red]=quit\n")

    for idx, q in enumerate(questions, 1):
        console.rule(f"[bold]{idx}/{len(questions)}[/bold]  {q.id}")
        console.print()
        console.print(Panel(q.prompt.system, title="System", border_style="dim"))
        for msg in q.prompt.messages:
            console.print(Panel(Markdown(msg.content), title=msg.role, border_style="blue"))
        console.print(Panel(Markdown(q.evaluation.reference_answer), title="Reference Answer", border_style="green"))
        if q.evaluation.key_facts:
            facts = "\n".join(f"- {f}" for f in q.evaluation.key_facts)
            console.print(Panel(facts, title="Key Facts", border_style="dim"))
        if q.evaluation.disqualifiers:
            disqs = "\n".join(f"- {d}" for d in q.evaluation.disqualifiers)
            console.print(Panel(disqs, title="Disqualifiers", border_style="red"))
        console.print()

        choice = Prompt.ask("[green]y[/green]=approve [yellow]r[/yellow]=regenerate [dim]s[/dim]=skip [red]q[/red]=quit", choices=["y", "r", "s", "q"], default="y")

        if choice == "q":
            break
        elif choice == "y":
            approved += 1
            console.print("  [green]Approved[/green]")
        elif choice == "s":
            skipped += 1
            console.print("  [dim]Skipped[/dim]")
        elif choice == "r":
            task_prompt = q.prompt.system + "\n\n"
            for doc in q.prompt.context_documents:
                task_prompt += f"[{doc.type}]\n{doc.content}\n\n"
            for msg in q.prompt.messages:
                task_prompt += msg.content

            # regenerate reference answer
            console.print("  Regenerating answer (Opus max)...", end="", highlight=False)
            try:
                resp = call_claude_code(task_prompt, model="opus", effort="max", timeout=600)
                ref_answer = resp["content"]
                console.print(f"  done ({resp['duration_ms']/1000:.1f}s)")
            except Exception as e:
                console.print(f"  [red]Failed: {e}[/red]")
                continue

            # regenerate key_facts and disqualifiers
            console.print("  Regenerating eval criteria (Opus high)...", end="", highlight=False)
            eval_prompt = (
                "Given this question and reference answer, generate evaluation criteria.\n\n"
                f"## Question\n{task_prompt}\n\n"
                f"## Reference Answer\n{ref_answer}\n\n"
                "Respond with ONLY valid JSON, no other text:\n"
                "{\n"
                '  "key_facts": ["fact that must be covered", ...],\n'
                '  "disqualifiers": ["error that should cap the score at 2", ...]\n'
                "}\n\n"
                "key_facts: 3-5 essential points a good answer must cover.\n"
                "disqualifiers: 1-3 serious errors that indicate a fundamentally wrong answer."
            )
            try:
                resp2 = call_claude_code(eval_prompt, model="opus", effort="high", timeout=300)
                text = resp2["content"].strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                criteria = json.loads(text)
                console.print(f"  done ({resp2['duration_ms']/1000:.1f}s)")
            except Exception as e:
                criteria = {}
                console.print(f"  [yellow]criteria failed ({e}), keeping old[/yellow]")

            # save everything
            path, i = lookup[q.id]
            q_files[path][i]["evaluation"]["reference_answer"] = ref_answer
            if criteria.get("key_facts"):
                q_files[path][i]["evaluation"]["key_facts"] = criteria["key_facts"]
            if criteria.get("disqualifiers"):
                q_files[path][i]["evaluation"]["disqualifiers"] = criteria["disqualifiers"]
            path.write_text(json.dumps(q_files[path], ensure_ascii=False, indent=2) + "\n")
            regenerated += 1

            # show new evaluation
            console.print(Panel(Markdown(ref_answer), title="New Reference Answer", border_style="yellow"))
            if criteria.get("key_facts"):
                console.print(Panel("\n".join(f"- {f}" for f in criteria["key_facts"]), title="New Key Facts", border_style="yellow"))
            if criteria.get("disqualifiers"):
                console.print(Panel("\n".join(f"- {d}" for d in criteria["disqualifiers"]), title="New Disqualifiers", border_style="yellow"))

    total = approved + regenerated + skipped
    console.print(f"\n{'='*60}")
    console.print(f"  [bold]Review Complete[/bold]  ({total} reviewed)\n")
    console.print(f"  Approved:     {approved}")
    console.print(f"  Regenerated:  {regenerated}")
    console.print(f"  Skipped:      {skipped}\n")


# ── main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gauntlet — run benchmark")
    parser.add_argument("--baseline-only", action="store_true", help="Only generate baselines")
    parser.add_argument("--baseline-model", metavar="NAME", help="Run a single baseline (e.g. opus, sonnet, haiku)")
    parser.add_argument("--local-only", action="store_true", help="Only run local models")
    parser.add_argument("--dry-run", action="store_true", help="List questions without running")
    parser.add_argument("--clean", metavar="NAME", help="Remove all data for a model")
    parser.add_argument("--reset", action="store_true", help="Remove ALL data (responses, scores, results, A/B)")
    parser.add_argument("--generate-references", action="store_true", help="Generate reference answers via Opus max effort")
    parser.add_argument("--review-references", action="store_true", help="Review reference answers (approve, regenerate, skip)")
    parser.add_argument("--list", action="store_true", help="List all models with data")
    args = parser.parse_args()

    if args.list:
        models = _list_models()
        if models:
            print(f"\n{len(models)} model(s) with data:\n")
            for m in models:
                resp = len(list((ROOT / "responses" / m).glob("*.json"))) if (ROOT / "responses" / m).exists() else 0
                scored = len(list((ROOT / "scores" / m).glob("*.json"))) if (ROOT / "scores" / m).exists() else 0
                tag = ""
                if _is_primary(m):
                    tag = f"  (baseline, primary, {_baseline_month(m)})"
                elif _is_baseline(m):
                    tag = f"  (baseline, {_baseline_month(m)})"
                print(f"  {m}  ({resp} responses, {scored} scored){tag}")
        else:
            print("\n  No model data found.\n")
        return

    if args.reset:
        _reset()
        return

    if args.clean:
        _clean(args.clean)
        return

    questions = _load_questions()

    if args.generate_references:
        _generate_references(questions)
        return

    if args.review_references:
        _review_references(questions)
        return

    endpoint = _load_endpoint()
    judge_cfg = _load_judge_config()

    if args.dry_run:
        print(f"\n{len(questions)} questions loaded:\n")
        for q in questions:
            cat = CATEGORY_LABELS.get(q.category.value, q.category.value)
            sub = q.subcategory.replace("_", " ").title().replace("Ml ", "ML ")
            print(f"  {q.id}  {cat:10s}  d={q.difficulty.value}  {sub}")
        print(f"\nEndpoint: {endpoint.get('url')}")
        return

    # Baselines
    if not args.local_only:
        baselines = judge_cfg.get("baselines", [])
        if args.baseline_model:
            # run a single baseline by model name
            matched = [bl for bl in baselines if bl["model"] == args.baseline_model]
            if not matched:
                log.error("Baseline '%s' not found in judge.toml", args.baseline_model)
                sys.exit(1)
            is_primary = baselines[0]["model"] == args.baseline_model
            n = run_baseline(questions, matched[0], is_primary=is_primary)
            print(f"\n  ✓ {args.baseline_model}: {n}/{len(questions)} done\n")
        else:
            for i, bl in enumerate(baselines):
                n = run_baseline(questions, bl, is_primary=(i == 0))
                print(f"\n  ✓ {bl['model']}: {n}/{len(questions)} done\n")

    # Local model
    if not args.baseline_only and not args.baseline_model:
        asyncio.run(run_local_model(endpoint, questions))


if __name__ == "__main__":
    main()
