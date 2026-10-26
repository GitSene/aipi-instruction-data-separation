"""Authorization-conditioned evaluation of indirect prompt injection defenses.

One harness for every provider: identical system prompt, user query, context
rendering, tool schema, and decoding parameters. Provider adapters differ only
in wire format.
"""
