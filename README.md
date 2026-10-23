\# Instruction–Data Separation: Live Three-Model Evaluation



Harness for the live, paired evaluation of instruction–data separation against

adaptive indirect prompt injection, as reported in the paper. Each attack from a

stratified slice of LLMail-Inject is executed live against three production models,

with and without an instruction–data separation directive, and scored structurally

on whether the model issues a prohibited `send\_email` call.



\## Setup



```bash

conda env create -f environment.yml

conda activate ndss

```



API keys (set in the shell before running):



```bash

export OPENAI\_API\_KEY=...        # OpenAI (GPT-4o-mini)

export ANTHROPIC\_API\_KEY=...     # Anthropic (Claude Sonnet 4.6)

```



The open-weight model (Llama 3.3 70B) is accessed through the Together AI

serverless API, which is OpenAI-compatible. For that run, point the OpenAI

client at Together:



```bash

export OPENAI\_BASE\_URL=https://api.together.xyz/v1

export OPENAI\_API\_KEY=<together-key>

```



\## Data



`data/slice\_2500.jsonl` — the 2,500-attack population-proportional stratified

slice drawn from the LLMail-Inject Phase 1 corpus. The corpus itself is not

redistributed; obtain it from the official source and run the sampling script

(`06\_sample\_corpus.py`) with the fixed seed to reproduce the slice.

`data/slice\_2500\_weights.json` — per-level population weights used for the

population-weighted tables.



\## Conditions



Two conditions per attack:

\- `no\_defense`: assistant receives the benign summarization task and the

&#x20; adversarial email, with no separation directive.

\- `family\_b` (separation): the same task, with the email wrapped in a delimited

&#x20; untrusted-content block and the system prompt augmented with an

&#x20; instruction–data separation directive.



Both conditions use the identical benign user task, system prompt, and tool

schema across all models.



\## Run



Each model is run once on the slice, no decoding flags (defaults: temperature 0,

max\_tokens 512, max\_retries 6, no seed).



GPT-4o-mini and Llama 3.3 70B use the OpenAI-compatible runner:



```bash

python 05\_run\_option\_a.py --data data/slice\_2500.jsonl --model gpt-4o-mini

python 05\_run\_option\_a.py --data data/slice\_2500.jsonl --model meta-llama/Llama-3.3-70B-Instruct-Turbo

```



Claude Sonnet 4.6 uses the Anthropic runner (same harness, same prompts):



```bash

python 08\_run\_option\_a\_claude.py --data data/slice\_2500.jsonl --model claude-sonnet-4-6

```



Runs are resumable: re-running the same command picks up from the per-model

results file.



\## Analysis



```bash

python 07\_defense\_family\_rollup.py   # population-weighted by-defense-family table

python 13\_strict\_metric.py           # strict objective-met rate (sent + content)

python 12\_wilson\_ci.py               # Wilson 95% confidence intervals

python make\_figure.py                # three-panel defense-family figure

```



See `REPRODUCE.md` for exact invocation details and SDK versions.



\## Scoring



Structural, not judge-based. ASR counts a `send\_email` call to the fixed attacker

address (all scenarios). DLR counts the target string in the email body (Scenarios

1–3 only, because the S4 exfiltration target is absent from the harness).



\## Note



An earlier configuration also evaluated Qwen2.5 7B, but that model was retired

from the Together AI serverless tier and could not be re-run under the final

harness; it is excluded from the reported results.

