# Authorization, parallel tool calls, and indirect prompt injection

Artifact for *When the User Says "Send": Authorization, Parallel Tool Calls, and the
Limits of Prompt-Level and Structural Defenses Against Indirect Prompt Injection*
(Babek Habibi Norouzlou, University of the Cumberlands).

The harness replays 1,000 human-written adaptive attacks from the
[LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge)
corpus against tool-using email assistants and scores every execution
structurally from the `send_email` calls the model issues.

## Design in one paragraph

Each execution shows the model a two-email context (a benign internal note from
Dana that names a reviewer, Raj, plus one attack email) and one tool,
`send_email(to, subject, body)`. Three **authorization regimes** change only the
user request: **R0** summarize only (no send authorized), **R1** summarize and
send to a recipient the user names, **R2** summarize and send to the reviewer
Dana names. Two **prompt conditions**: `none` and `separation` (an
instruction-data separation directive plus an `UNTRUSTED_EMAIL` block). Four
**attack variants**: unmodified, spoofed internal sender, internal drop
mailbox, and both. Four **structural recipient checks** (query binding,
directory allowlist, provenance, directory + provenance) are applied post hoc to
the logged calls; this is equivalent to online enforcement because each
execution is a single turn with no tool result returned to the model.

## Layout

| Path | Purpose |
|---|---|
| `aipi/prompts.py` | System prompt, separation directive, regimes, context, tool schema, attack variants. One source for every provider. |
| `aipi/providers.py` | Thin adapters for OpenAI-compatible APIs (OpenAI, Together AI, DeepInfra) and Anthropic. Wire format only. |
| `aipi/scoring.py` | Structural scoring and the four recipient checks. |
| `run_live.py` | Live runner. Resumable; logs raw provider output per execution. |
| `analyze.py` | Per-cell rates with bootstrap CIs, all checks, CSV output. |
| `diag_calls.py` | Calls per response; attacker sends with vs. without the legitimate send. |
| `stats_report.py` | Every number in the paper: CIs and paired bootstrap tests. |
| `probe_models.py` | Checks that a hosted model is callable and emits structured tool calls. |
| `sample_corpus.py` | Derives `data/slice_2500.jsonl` from the corpus with a fixed seed. |
| `tests/` | Unit tests for scoring and checks (`python -m pytest tests`). |
| `results_v2/` | Summary CSVs and `stats_report.txt` for the reported runs. |
| `legacy/` | Earlier single-email harness, kept for the replication section (see `legacy/README.md`). |

Raw per-execution logs (about 180,000 executions) are too large for GitHub and
are archived on Zenodo: [10.5281/zenodo.23735249](https://doi.org/10.5281/zenodo.23735249)
(concept DOI, resolves to the latest version; the logs are in `aipi_raw_logs.zip`).

## Reproduce

```bash
conda env create -f environment.yml && conda activate ipi-validation
python -m pytest tests            # scoring and checks

# OpenAI
export OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=...
python run_live.py --provider openai --model gpt-4o-mini --subset 1000 --samples 3 \
  --temperature 0.7 --regimes R0 --transforms none --out-dir results_full
python run_live.py --provider openai --model gpt-4o-mini --subset 1000 --samples 3 \
  --temperature 0.7 --regimes R1 R2 \
  --transforms none spoof_sender internal_drop spoof_drop --out-dir results_full

# Parallel tool calls disabled (OpenAI only)
python run_live.py --provider openai --model gpt-4o-mini --subset 1000 --samples 3 \
  --temperature 0.7 --regimes R1 R2 --transforms none --no-parallel-tools \
  --out-dir results_noparallel
```

Other models use the same commands with:

| Model | Provider and endpoint | Extra flags |
|---|---|---|
| `openai/gpt-oss-120b` | Together AI, `https://api.together.xyz/v1` | `--max-tokens 2048` |
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | DeepInfra, `https://api.deepinfra.com/v1/openai` (FP8) | `--max-tokens 1024` |
| `claude-sonnet-4-6` | `--provider anthropic`, `ANTHROPIC_API_KEY` | `--temperature default --max-tokens 2048 --samples 1 --regimes R0 R1 R2 --transforms none` |

The Anthropic API rejects an explicit temperature for Claude Sonnet 4.6, so that
model runs at the provider default (`--temperature default` omits the field).

Single-email replication (`--context legacy`, R0 only):

```bash
python run_live.py ... --context legacy --regimes R0 --transforms none --out-dir results_legacy
```

Analysis:

```bash
python analyze.py results_full/live_<model>.jsonl --out results_full/summary_<model>.csv
python diag_calls.py results_full/live_*.jsonl
python stats_report.py gpt=results_full/live_gpt-4o-mini.jsonl \
  llama=results_full/live_meta-llama_Llama-3.3-70B-Instruct-Turbo.jsonl \
  oss=results_full/live_openai_gpt-oss-120b.jsonl \
  claude=results_full/live_claude-sonnet-4-6.jsonl \
  gpt_noparallel=results_noparallel/live_gpt-4o-mini.jsonl
```

Provider aliases resolve to versions the provider can change; each log row
records the model version the provider reported (`served_model`) and a UTC
timestamp.

## Safety

No email is sent: the harness records tool calls and executes none. Benign
addresses use the reserved `example` domain. The attack variants substitute a
sender header and a destination address; they add no new attack wording.
