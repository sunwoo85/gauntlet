#!/usr/bin/env bash
# Gauntlet — benchmark runner
# Usage: gauntlet.sh {run|local|baseline|judge|results|ab|list|clean|reset|gen-refs|review-refs|dry-run}
#
# Designed by SK. Built by Claude.

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
[ -f "${DIR}/venv/bin/activate" ] && source "${DIR}/venv/bin/activate"

# kill background processes on exit/interrupt
trap 'pkill -P $$ 2>/dev/null; exit' INT TERM EXIT

# ── commands ────────────────────────────────────────────────────────────
case "${1:-run}" in
    run)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Gauntlet — LLM Quality Benchmark"
        echo "  Can your model run the Gauntlet?"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "Step 1: Running all models in parallel..."

        python3 -m src.runner --baseline-model opus &
        python3 -m src.runner --baseline-model sonnet &
        python3 -m src.runner --baseline-model haiku &
        python3 -m src.runner --local-only &
        wait

        echo "Step 2: Judging responses (Claude Code CLI, latest Opus, max effort)..."
        python3 -m src.judge
        echo ""
        echo "Step 3: Results..."
        python3 -m src.aggregate
        echo ""
        echo "Done."
        ;;
    local)
        python3 -m src.runner --local-only
        ;;
    baseline)
        echo "Running all baselines in parallel..."
        python3 -m src.runner --baseline-model opus &
        python3 -m src.runner --baseline-model sonnet &
        python3 -m src.runner --baseline-model haiku &
        wait
        echo "Done."
        ;;
    judge)
        python3 -m src.judge
        ;;
    results)
        python3 -m src.aggregate
        ;;
    ab)
        shift
        python3 -m src.ab_cli "$@"
        ;;
    list)
        python3 -m src.runner --list
        ;;
    clean)
        [ -z "${2:-}" ] && { echo "Usage: gauntlet.sh clean <model_name|ab>"; exit 1; }
        if [ "$2" = "ab" ]; then
            rm -f ab_results/*.jsonl 2>/dev/null
            echo "  A/B results cleared."
        else
            python3 -m src.runner --clean "$2"
        fi
        python3 -m src.aggregate
        ;;
    reset)
        pkill -f "src.runner" 2>/dev/null || true
        pkill -f "src.judge" 2>/dev/null || true
        python3 -m src.runner --reset
        ;;
    gen-refs)
        python3 -m src.runner --generate-references
        ;;
    review-refs)
        python3 -m src.runner --review-references
        ;;
    dry-run)
        python3 -m src.runner --dry-run
        ;;
    *)
        cat <<USAGE
Usage: gauntlet.sh {run|local|baseline|judge|results|ab|list|clean|reset|gen-refs|review-refs|dry-run}

  run            Full pipeline: run all models in parallel, judge, results
  local          Run local model only
  baseline       Generate all baselines in parallel (Opus, Sonnet, Haiku)
  judge          Judge all responses
  results        Show leaderboard
  ab [--count N] A/B blind comparison (default 10 questions)
  gen-refs       Generate reference answers via Opus max effort
  review-refs    Review reference answers (approve, regenerate, skip)
  list           List all models with data
  clean <name>   Remove all data for a model (or "ab" to clear A/B results)
  reset          Remove ALL data (fresh start)
  dry-run        List questions without running
USAGE
        ;;
esac
