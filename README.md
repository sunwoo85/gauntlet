# Gauntlet

Real-world LLM quality benchmark. Can your model run the Gauntlet?

Designed by SK. Built by Claude.

## Why

Generic benchmarks don't tell you if a model can handle your actual work — drafting investment memos in Korean, triaging biotech partnership emails, debugging GPU infrastructure, or reconciling tax documents. Gauntlet tests what matters: professional tasks drawn from real usage patterns, scored by the latest Opus at maximum effort.

The goal: find local models that beat the latest Claude Opus.

## Quick Start

```bash
git clone https://github.com/sunwoo85/gauntlet.git
cd gauntlet
python3 -m venv venv && source venv/bin/activate
pip install -e .
./gauntlet.sh dry-run
```

## How It Works

```
Questions (45)
    │
    ├──► Local model (Claude Code CLI) ──► responses/{model_id}/*.json
    │
    └──► Baselines (Claude Code CLI) ──► responses/{model_id}/*.json
              │                           (Opus, Sonnet, Haiku)
              ▼
         Judge (Claude Code CLI, latest Opus, max effort)
              │
              ▼
         scores/{model_id}/*.json ──► Leaderboard
```

All models run through Claude Code CLI with identical tool access (config/mcp.json). Built-in tools disabled — only MCP tools if configured.

10 pilot questions across three professional domains, scored on six dimensions by the latest Opus (max effort) as judge. Three dependencies, no bloat.

## Categories

| Category | Questions | What it tests |
|----------|-----------|---------------|
| Investment | 15 | Deal analysis, financial KB extraction, transcript summarization (Korean/English) |
| Biotech | 15 | Email triage, competitive intelligence, partnership communications, clinical data |
| General | 15 | Infrastructure debugging, hardware evaluation, coding, document analysis |

## Scoring

Six dimensions, each 1-5:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Factual Accuracy | 25% | Correctness, no hallucination |
| Reasoning Depth | 20% | Multi-step logic, edge cases, nuance |
| Practical Usefulness | 20% | Actionable output a professional can use as-is |
| Instruction Following | 15% | Format, language, scope constraints |
| Communication Quality | 10% | Structure, tone, conciseness |
| Bilingual Competence | 10% | Korean/English fluency, terminology |

Composite score: weighted sum normalized to 0-100. Weights configurable in `config/metrics.toml`.

## Commands

```bash
./gauntlet.sh              # full pipeline: run, judge, results
./gauntlet.sh local        # run local model only
./gauntlet.sh baseline     # generate all baselines (Opus, Sonnet, Haiku)
./gauntlet.sh judge        # judge all responses
./gauntlet.sh results      # show leaderboard
./gauntlet.sh ab           # A/B blind test (random 10)
./gauntlet.sh ab --count 0 # A/B all questions
./gauntlet.sh gen-refs     # generate reference answers via Opus max effort
./gauntlet.sh review-refs  # review references (approve, regenerate, skip)
./gauntlet.sh list         # list all models with data
./gauntlet.sh clean <name> # remove a model's data (or "ab" for A/B results)
./gauntlet.sh reset        # remove ALL data (fresh start)
./gauntlet.sh dry-run      # list questions without running
```

## Configuration

### Prerequisites

- **Python 3.11+**
- **Claude Code CLI** — installed and authenticated. Used for baselines and judging. No API keys needed — it uses your existing session.
- **A local model** — served behind any OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, LM Studio, etc.)

### Endpoint (`config/models.toml`)

Point at your local model. Copy the example file, then edit. Model ID, max_tokens, and concurrency are auto-detected. Swap the model behind the endpoint and re-run — new model gets its own results automatically.

```bash
cp config/models.example.toml config/models.toml
```

```toml
[endpoint]
url = "http://localhost:1319/v1/chat/completions"
# concurrency = 8              # override auto-detected (default: context_length / max_tokens)
# [endpoint.sampling]
# temperature = 0              # override server default
```

### Baselines & Judge (`config/judge.toml`)

Three Claude models as baselines (Opus, Sonnet, Haiku), all at high effort. The first baseline listed is the primary hurdle — highlighted in the leaderboard. Judge always uses the latest Opus at max effort. Baselines refresh automatically each month.

### MCP Tools (`config/mcp.json`)

Optional. Copy `mcp.example.json` to `mcp.json` and add MCP servers (e.g. web search). Both baselines and local models get the same tools — no built-in tools, only what you configure here.

```bash
cp config/mcp.example.json config/mcp.json
```

### Metrics (`config/metrics.toml`)

Scoring dimension weights and rubric definitions. Adjust weights to emphasize different qualities.

### Output structure

```
responses/{model_id}/      # e.g. responses/Qwen--Qwen3.5-122B-A10B-FP8/
scores/{model_id}/         # Judge scores (one JSON per question)
results/YYYY-MM/           # Leaderboard (JSON + markdown)
ab_results/                # Human A/B session logs
```

All generated data is gitignored.

## Roadmap

- [ ] Full 45-question set (15 per category)
- [ ] Monthly question rotation
- [ ] Community question contributions

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2026.04 | 2026-04-02 | Initial release — 10 pilot questions, automated judge pipeline, A/B CLI |

## License

MIT
