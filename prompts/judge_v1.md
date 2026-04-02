# Gauntlet Judge Prompt v1

You are evaluating an LLM response on a professional task. Score the response on six dimensions using the rubrics below. Be rigorous and calibrated — a score of 5 means genuinely excellent, not just adequate.

## Task

{system_prompt}

{user_message}

{context_documents}

## Reference Answer

{reference_answer}

## Key Facts That Must Be Covered

{key_facts}

## Disqualifiers (automatic score cap at 2 if triggered)

{disqualifiers}

## Model Response to Evaluate

{model_response}

---

## Scoring Rubrics

### Factual Accuracy (25%)
- 5: All facts correct, no hallucination
- 4: Minor inaccuracy that doesn't affect conclusion
- 3: One material error but mostly correct
- 2: Multiple errors, partially unreliable
- 1: Fundamentally incorrect or fabricated

### Reasoning Depth (20%)
- 5: Multi-step reasoning, considers edge cases, identifies nuances
- 4: Sound reasoning with minor gaps
- 3: Adequate reasoning but surface-level
- 2: Shallow or partially flawed logic
- 1: No meaningful reasoning, jumps to conclusion

### Practical Usefulness (20%)
- 5: Directly actionable, saves the professional real time
- 4: Useful with minor additional work needed
- 3: Somewhat helpful but requires significant rework
- 2: Marginally useful, mostly generic
- 1: Not useful for the intended professional task

### Instruction Following (15%)
- 5: Follows all constraints perfectly (format, length, language, scope)
- 4: Follows most constraints, one minor deviation
- 3: Follows some constraints, misses important ones
- 2: Significant deviations from instructions
- 1: Ignores instructions entirely

### Communication Quality (10%)
- 5: Clear structure, appropriate tone, well-organized, concise
- 4: Good communication with minor issues
- 3: Adequate but could be clearer or better organized
- 2: Confusing structure or inappropriate tone
- 1: Incoherent or entirely wrong register

### Bilingual Competence (10%)
- 5: Native-level fluency in target language, correct terminology, natural phrasing
- 4: Near-native with minor awkwardness
- 3: Understandable but clearly non-native patterns
- 2: Frequent errors affecting comprehension
- 1: Wrong language used, or unintelligible
- N/A: English-only question (score as 5)

---

## Output

Respond with ONLY valid JSON, no other text:

```json
{
  "factual_accuracy": {"score": <1-5>, "reasoning": "<brief justification>"},
  "reasoning_depth": {"score": <1-5>, "reasoning": "<brief justification>"},
  "practical_usefulness": {"score": <1-5>, "reasoning": "<brief justification>"},
  "instruction_following": {"score": <1-5>, "reasoning": "<brief justification>"},
  "communication_quality": {"score": <1-5>, "reasoning": "<brief justification>"},
  "bilingual_competence": {"score": <1-5>, "reasoning": "<brief justification>"}
}
```
