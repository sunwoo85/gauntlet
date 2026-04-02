"""
Gauntlet — judge model responses using Claude Code CLI (latest Opus, max effort).

Reads responses from responses/{model}/, scores each against the question's
reference answer and rubrics, saves results to scores/{model}/.

Designed by SK. Built by Claude.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import tomllib

from .clients import call_claude_code
from .schema import JudgeResult, Question, Scores, composite_score

# ── config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gauntlet.judge")


# ── helpers ─────────────────────────────────────────────────────────────
def _load_questions() -> dict[str, Question]:
    """Load questions indexed by ID."""
    questions = {}
    q_dir = ROOT / "questions" / "current"
    if not q_dir.exists():
        dirs = sorted((ROOT / "questions").glob("20*"))
        q_dir = dirs[-1] if dirs else None
    if not q_dir:
        log.error("No question directories found")
        sys.exit(1)

    for path in sorted(q_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        for item in raw:
            q = Question.model_validate(item)
            questions[q.id] = q
    return questions


def _load_prompt_template(judge_cfg: dict) -> str:
    """Load the judge prompt template from config path."""
    prompt_file = judge_cfg.get("prompt_file", "prompts/judge_v1.md")
    path = ROOT / prompt_file
    return path.read_text()


def _build_judge_prompt(template: str, q: Question, response: str) -> str:
    """Fill the judge prompt template with question and response data."""
    system_prompt = q.prompt.system
    user_message = "\n".join(m.content for m in q.prompt.messages)
    context = "\n".join(f"[{d.type}]\n{d.content}" for d in q.prompt.context_documents) or "(none)"
    key_facts = "\n".join(f"- {f}" for f in q.evaluation.key_facts) or "(none)"
    disqualifiers = "\n".join(f"- {d}" for d in q.evaluation.disqualifiers) or "(none)"

    return (
        template
        .replace("{system_prompt}", system_prompt)
        .replace("{user_message}", user_message)
        .replace("{context_documents}", context)
        .replace("{reference_answer}", q.evaluation.reference_answer)
        .replace("{key_facts}", key_facts)
        .replace("{disqualifiers}", disqualifiers)
        .replace("{model_response}", response)
    )


def _parse_scores(raw: str) -> Scores:
    """Extract JSON scores from judge response."""
    # find JSON block in the response
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # try parsing as-is if no code blocks
    data = json.loads(text)
    return Scores.model_validate(data)


def _save_score(model: str, result: JudgeResult) -> Path:
    """Save a judge result to disk."""
    out_dir = ROOT / "scores" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.question_id}.json"
    path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return path


# ── judge ───────────────────────────────────────────────────────────────
def judge_model(model: str, questions: dict[str, Question], template: str, judge_cfg: dict) -> int:
    """Judge all responses for a model."""
    resp_dir = ROOT / "responses" / model
    if not resp_dir.exists():
        log.error("No responses found for %s", model)
        return 0

    j_model = judge_cfg.get("model", "opus")
    j_effort = judge_cfg.get("effort", "max")
    j_timeout = judge_cfg.get("timeout_sec", 300)

    count = 0
    for resp_path in sorted(resp_dir.glob("*.json")):
        qid = resp_path.stem
        score_path = ROOT / "scores" / model / f"{qid}.json"
        if score_path.exists():
            log.info("  skip %s (cached)", qid)
            count += 1
            continue

        if qid not in questions:
            log.warning("  skip %s (question not found)", qid)
            continue

        q = questions[qid]
        resp_data = json.loads(resp_path.read_text())
        response_text = resp_data.get("content") or ""

        prompt = _build_judge_prompt(template, q, response_text)

        try:
            judge_resp = call_claude_code(prompt, model=j_model, effort=j_effort, timeout=j_timeout)
            scores = _parse_scores(judge_resp["content"])
            comp = composite_score(scores)

            result = JudgeResult(
                question_id=qid,
                model=model,
                scores=scores,
                composite=comp,
            )
            _save_score(model, result)
            count += 1
            log.info("  %s  score=%.1f  %s", qid, comp, model)
        except json.JSONDecodeError as e:
            log.error("  %s  PARSE ERROR: %s", qid, e)
        except Exception as e:
            log.error("  %s  FAILED: %s", qid, e)

    return count


# ── main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gauntlet — judge responses")
    parser.add_argument("--model", help="Judge only this model (directory name under responses/)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be judged")
    args = parser.parse_args()

    questions = _load_questions()

    with open(ROOT / "config" / "judge.toml", "rb") as f:
        judge_cfg = tomllib.load(f).get("judge", {})

    template = _load_prompt_template(judge_cfg)

    log.info("Loaded %d questions, judge prompt %s, model=%s effort=%s",
             len(questions), judge_cfg.get("prompt_version", "v1"),
             judge_cfg.get("model", "opus"), judge_cfg.get("effort", "max"))

    # find models to judge
    resp_root = ROOT / "responses"
    if not resp_root.exists():
        log.error("No responses directory found. Run the runner first.")
        sys.exit(1)

    models = [args.model] if args.model else [d.name for d in sorted(resp_root.iterdir()) if d.is_dir()]

    if args.dry_run:
        for model in models:
            n_resp = len(list((resp_root / model).glob("*.json")))
            n_scored = len(list((ROOT / "scores" / model).glob("*.json"))) if (ROOT / "scores" / model).exists() else 0
            print(f"  {model}: {n_resp} responses, {n_scored} already scored, {n_resp - n_scored} to judge")
        return

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Judging: {model}")
        print(f"{'='*60}\n")
        n = judge_model(model, questions, template, judge_cfg)
        total = len(list((resp_root / model).glob("*.json")))
        print(f"\n  {n}/{total} scored\n")


if __name__ == "__main__":
    main()
