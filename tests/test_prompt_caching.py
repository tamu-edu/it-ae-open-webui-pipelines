"""Placement of Anthropic cache_control breakpoints on chat-completions payloads.

Caching failures are silent and expensive in both directions, which is what
these tests are for:

* Place too few markers, or place them on volatile content, and the whole prefix
  is re-billed at full input price every turn -- the regression that cost a
  measured prod chat ~$7 of a $10 daily budget in nine turns.
* Place a marker on a model that does not take them and the request can 400, or
  silently pay a write premium for bytes nothing ever reads back.

Nothing here asserts a cost; they assert the payload shape that produces the
caching, plus the cases where the pipeline must keep its hands off. The
wire-format half -- that these markers become Bedrock ``cachePoint`` blocks --
is verified against the vendored LiteLLM; see tests/README.md.
"""

import copy

import pytest
from conftest import IMAGE_MANIFOLD, build_pipeline

MODEL = "Claude Opus 4.7"
OTHER_MODEL = "GPT-4o"

# A conversation the way OpenWebUI sends one: the workspace model's system
# prompt (with any injected knowledge-base context) first, then alternating
# turns, the newest last.
CONVERSATION = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "BIG SYSTEM PROMPT " * 100},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ],
}


@pytest.fixture
def pipeline():
    return cache_pipeline()


def cache_pipeline(enabled=True, ttl="1h", cache_models=(MODEL,)):
    pipeline = build_pipeline(
        IMAGE_MANIFOLD, PROMPT_CACHING_ENABLED=enabled, PROMPT_CACHE_TTL=ttl
    )
    pipeline.prompt_cache_model_ids = set(cache_models)
    return pipeline


def breakpoints(payload):
    """Every cache_control in the payload, as (role, control) in order."""
    found = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("cache_control"):
            found.append((message.get("role"), message["cache_control"]))
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("cache_control"):
                    found.append((message.get("role"), block["cache_control"]))
    return found


class TestPlacement:
    def test_two_breakpoints_on_system_and_the_newest_turn(self, pipeline):
        """The stable prefix and the growing tail are the two boundaries.

        System holds the system prompt and injected context. The final message
        is written this turn and read back on the next, so history accrues as
        cache hits instead of being re-billed in full.
        """
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        assert [role for role, _ in breakpoints(payload)] == ["system", "user"]

    def test_the_marked_user_turn_is_the_newest_one(self, pipeline):
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        newest = payload["messages"][-1]["content"]
        assert newest[0]["text"] == "second question"
        assert newest[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_intermediate_turns_are_not_marked(self, pipeline):
        """Anthropic allows four breakpoints; don't spend them on history.

        Earlier turns are already covered by the moving tail marker from the
        turn on which they were newest.
        """
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        for message in payload["messages"][1:-1]:
            assert not message.get("cache_control")

    def test_never_exceeds_the_four_breakpoint_limit(self, pipeline):
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        assert len(breakpoints(payload)) <= 4

    def test_string_content_is_promoted_without_altering_the_text(self, pipeline):
        """A marker only attaches to a content block, so a str becomes one.

        The text must survive byte-for-byte: caching is a prefix match, so any
        rewriting here would invalidate the cache it is meant to create.
        """
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        system = payload["messages"][0]["content"]
        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == CONVERSATION["messages"][0]["content"]

    def test_last_system_message_wins_when_there_are_several(self, pipeline):
        payload = {
            "messages": [
                {"role": "system", "content": "first system"},
                {"role": "system", "content": "second system"},
                {"role": "user", "content": "hello"},
            ]
        }
        pipeline._apply_prompt_caching(payload, MODEL)

        assert not payload["messages"][0].get("cache_control")
        assert payload["messages"][1]["content"][0].get("cache_control")

    def test_no_system_message_marks_only_the_tail(self, pipeline):
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        pipeline._apply_prompt_caching(payload, MODEL)

        assert [role for role, _ in breakpoints(payload)] == ["user"]

    def test_a_system_only_payload_is_marked_once(self, pipeline):
        """System and tail are the same object here; marking twice wastes one."""
        payload = {"messages": [{"role": "system", "content": "sys prompt"}]}
        pipeline._apply_prompt_caching(payload, MODEL)

        assert len(breakpoints(payload)) == 1


class TestMultimodalContent:
    def test_marker_goes_on_the_last_text_block(self, pipeline):
        """A trailing image is a poor breakpoint; the text is the token bulk."""
        payload = {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                },
            ]
        }
        pipeline._apply_prompt_caching(payload, MODEL)

        blocks = payload["messages"][-1]["content"]
        assert blocks[0].get("cache_control")
        assert not blocks[1].get("cache_control")


class TestWhenToKeepHandsOff:
    def test_a_non_anthropic_model_is_untouched(self, pipeline):
        """OpenAI caches automatically and has no cache_control field."""
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, OTHER_MODEL)

        assert payload == CONVERSATION

    def test_an_unknown_model_is_untouched(self, pipeline):
        """Detection is best-effort; an empty set must mean no markers."""
        payload = copy.deepcopy(CONVERSATION)
        cache_pipeline(cache_models=())._apply_prompt_caching(payload, MODEL)

        assert payload == CONVERSATION

    def test_the_valve_can_turn_it_off(self):
        payload = copy.deepcopy(CONVERSATION)
        cache_pipeline(enabled=False)._apply_prompt_caching(payload, MODEL)

        assert payload == CONVERSATION

    def test_existing_markers_are_respected(self, pipeline):
        """Adding to someone else's markers risks blowing the limit."""
        prepared = copy.deepcopy(CONVERSATION)
        prepared["messages"][1]["content"] = [
            {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
        ]
        payload = copy.deepcopy(prepared)
        pipeline._apply_prompt_caching(payload, MODEL)

        assert payload == prepared

    def test_content_with_no_cacheable_text_is_skipped(self, pipeline):
        """A tool-call-only assistant turn has nothing to cache."""
        payload = {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            ]
        }
        pipeline._apply_prompt_caching(payload, MODEL)

        assert [role for role, _ in breakpoints(payload)] == ["system"]
        assert payload["messages"][1]["tool_calls"][0]["id"] == "call_1"

    def test_blank_content_is_not_marked(self, pipeline):
        payload = {
            "messages": [
                {"role": "system", "content": "   "},
                {"role": "user", "content": ""},
            ]
        }
        pipeline._apply_prompt_caching(payload, MODEL)

        assert breakpoints(payload) == []


class TestTtl:
    def test_ttl_is_applied_to_every_breakpoint(self, pipeline):
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        assert all(
            control == {"type": "ephemeral", "ttl": "1h"}
            for _, control in breakpoints(payload)
        )

    def test_an_empty_ttl_omits_the_key_rather_than_sending_a_blank(self):
        payload = copy.deepcopy(CONVERSATION)
        cache_pipeline(ttl="")._apply_prompt_caching(payload, MODEL)

        assert all(
            control == {"type": "ephemeral"} for _, control in breakpoints(payload)
        )

    def test_breakpoints_do_not_share_one_mutable_dict(self, pipeline):
        """Each marker needs its own dict, or editing one edits them all."""
        payload = copy.deepcopy(CONVERSATION)
        pipeline._apply_prompt_caching(payload, MODEL)

        controls = [control for _, control in breakpoints(payload)]
        assert len(controls) == 2
        assert controls[0] is not controls[1]


class TestDegeneratePayloads:
    """pipe() hands over whatever OpenWebUI sent; none of it may raise."""

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"model": MODEL}, id="no-messages-key"),
            pytest.param({"messages": []}, id="empty-messages"),
            pytest.param({"messages": "not-a-list"}, id="messages-not-a-list"),
            pytest.param({"messages": [None, "text"]}, id="non-dict-entries"),
            pytest.param({"messages": [{"role": "user"}]}, id="message-without-content"),
        ],
    )
    def test_does_not_raise(self, pipeline, payload):
        pipeline._apply_prompt_caching(payload, MODEL)


class TestModelDetection:
    """Which models get markers at all.

    ``supports_prompt_caching`` alone is the trap: LiteLLM reports it true for
    OpenAI models too, which cache automatically server-side and have no
    cache_control field. The underlying model must also be an Anthropic one,
    read from litellm_params rather than the operator-chosen dropdown alias.
    """

    @pytest.fixture
    def accepts(self):
        from conftest import load_pipeline_module

        return load_pipeline_module(IMAGE_MANIFOLD).Pipeline._accepts_cache_control

    @pytest.mark.parametrize(
        "underlying",
        [
            "bedrock/us.anthropic.claude-opus-4-7",
            "anthropic/claude-opus-4-7",
            "vertex_ai/claude-opus-4-7",
            "bedrock/eu.anthropic.claude-sonnet-4-5-v1:0",
        ],
    )
    def test_anthropic_models_accepted(self, accepts, underlying):
        assert accepts(
            {"litellm_params": {"model": underlying}},
            {"supports_prompt_caching": True},
        )

    @pytest.mark.parametrize(
        "underlying",
        [
            pytest.param("openai/gpt-4o", id="openai-auto-caches"),
            pytest.param("bedrock/us.amazon.nova-pro-v1:0", id="bedrock-nova"),
            pytest.param("bedrock/meta.llama3-70b-instruct-v1:0", id="bedrock-llama"),
            pytest.param("gemini/gemini-2.0-flash", id="gemini"),
        ],
    )
    def test_non_anthropic_models_rejected(self, accepts, underlying):
        assert not accepts(
            {"litellm_params": {"model": underlying}},
            {"supports_prompt_caching": True},
        )

    def test_capability_flag_is_required(self, accepts):
        assert not accepts(
            {"litellm_params": {"model": "bedrock/us.anthropic.claude-opus-4-7"}},
            {"supports_prompt_caching": False},
        )

    @pytest.mark.parametrize(
        "model", [{}, {"litellm_params": {}}, {"litellm_params": {"model": None}}]
    )
    def test_missing_metadata_is_rejected_not_assumed(self, accepts, model):
        assert not accepts(model, {"supports_prompt_caching": True})

    def test_the_dropdown_alias_alone_never_qualifies(self, accepts):
        """An alias is operator-chosen and carries no provider guarantee."""
        assert not accepts(
            {
                "model_name": "Claude Opus 4.7",
                "litellm_params": {"model": "openai/gpt-4o"},
            },
            {"supports_prompt_caching": True},
        )
