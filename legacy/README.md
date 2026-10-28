# Legacy single-email harness (superseded)

These scripts produced an earlier single-email evaluation (R0 only, attack email
alone, no sender header). They are kept because the paper's replication section
compares against those earlier numbers. They are **not** used for any other
result in the paper; the current harness is in the repository root.

Known differences from the current harness, documented so the earlier numbers
are read correctly:

- `08_run_option_a_claude.py` did not set a temperature (Anthropic API default
  applied), used different user queries for levels 3 and 4, a different email
  rendering, and a `to`/`body` tool schema; the OpenAI-compatible runner used a
  `recipient`/`subject`/`body` schema. The current harness uses one shared
  prompt and schema for every provider.
- `results/option_a_results_claude-sonnet-4-6.csv` includes smoke-test rows
  (n = 2,543 / 2,539 rather than 2,500).
- The Llama 3.3 70B run used Together AI's serverless endpoint, which no longer
  serves that model; its per-row log was not retained.
- Earlier runs used temperature 0 and one sample per attack.

The current harness reproduces the earlier GPT-4o-mini single-email result
within two points (`--context legacy`); it does not reproduce the earlier Llama
result, which the paper attributes to the hosting provider.
