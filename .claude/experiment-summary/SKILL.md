---
name: experiment-summary
description: Turn rough RL experiment notes, run logs, or results tables into a short, consistent experiment summary. Use when Bryan pastes notes or logs from training runs (e.g., PPO on LightsOut, sliding puzzle, Sokoban) and asks to summarize, write up, or log an experiment.
---

# Experiment Summary

## Steps
1. Read everything first: notes, logs, configs, numbers.
2. Find the comparison: what was changed versus what it's being compared to.
3. Pull out the setup: env, algorithm, the key hyperparameters that differ, seeds, training budget.
4. Pull out the results, keeping every number exactly as given.
5. Separate what was observed from what Bryan thinks explains it.
6. Fill the template, then check it against the rules.

## Format
**[Env] — [what was tested]** · [date if given]

**Question:** one sentence. What was this run trying to find out?

**Setup**
- Env / reward: …
- Algorithm: …
- Changed vs. baseline: …
- Seeds / budget: …

**Results**
| Config | Metric | Value | Seeds |
|---|---|---|---|

**Observations:** what the data shows, with no explanations.

**Interpretation:** Bryan's hypotheses, labeled as hypotheses.

**Next:** concrete follow-ups from the notes.

## Rules
- Never invent numbers, seeds, or hyperparameters. Write "not recorded" instead.
- Copy numbers exactly, at the precision given. Don't round, average, or convert.
- If there are fewer than 3 seeds, or no variance is reported, flag it in Observations.
- Keep observations and interpretation separate. If a claim isn't in the data, it goes under Interpretation.
- Flag any change in the comparison other than the one being tested (different budget, reward, env version).
- Use Bryan's names for runs, configs, and environments.
- Note sparse vs. dense reward when it matters for reading the results.
- Fit on one screen. Put supporting details in a short "Details" list at the end, only if needed.
