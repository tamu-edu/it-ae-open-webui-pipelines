# Tests

Unit tests for the manifold pipelines. Run from the repo root:

```sh
pip install -r requirements-test.txt
pytest
```

## What is covered

| File | Covers |
|---|---|
| `test_error_handling.py` | How upstream LiteLLM errors are classified and worded: budget exhaustion, rate limits, context-window overflow, guardrails, auth failures |
| `test_prompt_caching.py` | Where Anthropic `cache_control` breakpoints are placed, and which models get them |

Both areas are ones where a regression is **silent** -- the request still
succeeds, it is just wrong or more expensive -- so the tests assert what must be
*absent* from a result as much as what must be present. Two examples worth
keeping in mind when editing them:

- A budget error must not be reported as a rate limit, and must never echo the
  key owner's email address or key prefix. The incident that prompted these
  tests was found through a public chat share link.
- A non-Anthropic model must come back from the caching pass byte-identical.
  OpenAI models report `supports_prompt_caching: true` but cache automatically
  and have no `cache_control` field.

## Why the dependencies are stubbed

A pipeline imports fastapi, pydantic, httpx, requests and psycopg2 at module
scope. The code under test uses none of them -- the error formatters and the
cache-control placement need only `re`, `json` and `self.valves` -- so
`conftest.py` installs stand-ins before loading a pipeline, and only when the
real package is absent. That keeps the suite fast and offline.

`Pipeline.__init__` reads environment variables and calls LiteLLM, so
`conftest.build_pipeline()` bypasses it and attaches just the valves a test
names. A test that needs real behaviour from one of those libraries should
install the real package rather than widen the stubs.

## What these tests cannot cover

The wire format. These tests assert the payload *this repo* produces; whether
LiteLLM then renders it correctly for a provider is LiteLLM's contract. For
prompt caching that half was checked by hand against the vendored LiteLLM in a
running pod, and is worth re-checking after a LiteLLM upgrade:

```python
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

cfg = AmazonConverseConfig()
cc = {"type": "ephemeral", "ttl": "1h"}
messages = [
    {"role": "system", "content": [{"type": "text", "text": "x " * 50, "cache_control": cc}]},
    {"role": "user", "content": [{"type": "text", "text": "q", "cache_control": cc}]},
]
rest, system_blocks = cfg._transform_system_message(list(messages), model="us.anthropic.claude-opus-4-7")
print(system_blocks)  # expect a {'cachePoint': {'type': 'default', 'ttl': '1h'}} block
```

Two behaviours that check confirmed, and that the unit tests assume:

- `cache_control` becomes a Bedrock `cachePoint` block.
- `ttl` is **silently dropped** for models whose entry in
  `model_prices_and_context_window.json` has no
  `cache_creation_input_token_cost_above_1hr`. Opus 4.7/4.8/5 qualify; Sonnet
  4.5 does not. So a `"1h"` valve degrades to the 5-minute default rather than
  erroring.

In production the ground truth is `usage.cache_read_input_tokens` on the second
and later turns of a conversation. If it stays zero, something upstream is
changing the prefix and no marker placement will help.
