## Reproduction

The three model arms were run through the same harness, invoked once per model
with no decoding flags.

GPT-4o-mini and Llama 3.3 70B (OpenAI-compatible runner):

    python 05_run_option_a.py --data data/slice_2500.jsonl --model gpt-4o-mini
    python 05_run_option_a.py --data data/slice_2500.jsonl --model meta-llama/Llama-3.3-70B-Instruct-Turbo

Claude Sonnet 4.6 (Anthropic runner, identical prompts and scoring):

    python 08_run_option_a_claude.py --data data/slice_2500.jsonl --model claude-sonnet-4-6

With no flags passed, the runner resolves to its argparse defaults:

| Parameter    | Value |
|--------------|-------|
| temperature  | 0.0   |
| max_tokens   | 512   |
| max_retries  | 6     |

All three arms were run at temperature 0 with no decoding seed.

Llama 3.3 70B is accessed through the Together AI serverless API, which is
OpenAI-compatible; set OPENAI_BASE_URL to the Together endpoint and
OPENAI_API_KEY to a Together key for that run. GPT-4o-mini and Claude Sonnet 4.6
use the OpenAI and Anthropic APIs directly.

Models were addressed by provider alias (`gpt-4o-mini`, `claude-sonnet-4-6`,
`meta-llama/Llama-3.3-70B-Instruct-Turbo`).
SDK versions at run time: openai 2.43.0, anthropic 0.111.0.

The 2,500-attack stratified slice is drawn from the LLMail-Inject Phase 1
corpus with a fixed seed; see `06_sample_corpus.py` for the seed and the
scenario-configuration mapping.

An earlier configuration also included Qwen2.5 7B (alias
`Qwen/Qwen2.5-7B-Instruct-Turbo`), retired from Together AI serverless and
therefore excluded from the final results.