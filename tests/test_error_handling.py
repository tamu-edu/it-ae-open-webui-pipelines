"""How upstream LiteLLM errors are classified and worded for the user.

The regression these guard against: LiteLLM's BudgetExceededError hardcodes
``status_code = 429``, so while the budget check lived under
``if response.status_code == 400`` it could never run. In the image pipeline the
429 branch caught budget errors and told the user they were "sending requests
faster than your account allows"; in the text pipeline, which has no 429 branch,
they fell through to the generic formatter, which echoed LiteLLM's raw message --
including the key owner's email address -- into a shareable chat transcript.

So classification must key on the error's identity, not its HTTP status, and
every assertion below about what is *absent* from the output matters as much as
what is present.
"""

import pytest
from conftest import IMAGE_MANIFOLD, TEXT_MANIFOLD, build_pipeline, litellm_error

# Both variants share the budget-error path, so the shared cases run against
# each. Only the image pipeline has 429 / 401-403 / context-window branches.
BOTH_MANIFOLDS = [IMAGE_MANIFOLD, TEXT_MANIFOLD]

# The exact body prod returned for the reported incident, key prefix and all.
PROD_BUDGET_MESSAGE = (
    "Budget has been exceeded! Key=someone@tamu.edu (sk-...wB8Q) "
    "Current cost: 10.39113075, Max budget: 10.0"
)

# The four wordings LiteLLM emits for budget exhaustion, from the separate
# checks in litellm/proxy/auth/auth_checks.py. The last two use "Limit=" rather
# than "Budget=" and prefix amounts with "$", which an earlier version of the
# parser silently failed to match.
BUDGET_MESSAGE_SHAPES = [
    pytest.param(PROD_BUDGET_MESSAGE, "10.39", "10.00", id="virtual-key-max-budget"),
    pytest.param(
        "ExceededBudget: User=u-123 over budget. Spend=12.5, Budget=10.0",
        "12.50",
        "10.00",
        id="user-budget",
    ),
    pytest.param(
        "ExceededBudget: End User=e-9 over budget. Spend=3.25, Budget=3.0",
        "3.25",
        "3.00",
        id="end-user-budget",
    ),
    pytest.param(
        "ExceededBudget: Key over 1d budget. Spend=$10.3911, Limit=$10.00",
        "10.39",
        "10.00",
        id="key-rolling-window",
    ),
]


@pytest.fixture(params=BOTH_MANIFOLDS, ids=["image", "text"])
def pipeline(request):
    return build_pipeline(request.param)


@pytest.fixture
def image_pipeline():
    return build_pipeline(IMAGE_MANIFOLD)


class TestBudgetExhaustion:
    """Budget errors arrive as 429 and must be reported as budget errors."""

    def test_429_budget_is_not_reported_as_a_rate_limit(self, pipeline):
        result = pipeline._handle_litellm_error(
            litellm_error(429, PROD_BUDGET_MESSAGE, "budget_exceeded")
        )
        assert "Budget Exceeded" in result
        assert "Rate Limit" not in result
        # The old message told users to wait, which cannot help: nothing frees
        # up until the budget period rolls over.
        assert "faster than your account allows" not in result

    def test_the_real_status_code_is_preserved(self, pipeline):
        result = pipeline._handle_litellm_error(
            litellm_error(429, PROD_BUDGET_MESSAGE, "budget_exceeded")
        )
        assert "Error Code: 429" in result

    def test_classified_on_type_even_without_a_message_marker(self, pipeline):
        """The body's type is authoritative; don't rely on message wording."""
        result = pipeline._handle_litellm_error(
            litellm_error(429, "you are out of funds", "budget_exceeded")
        )
        assert "Budget Exceeded" in result

    @pytest.mark.parametrize("message,spend,budget", BUDGET_MESSAGE_SHAPES)
    def test_every_litellm_budget_wording_is_parsed(
        self, pipeline, message, spend, budget
    ):
        result = pipeline._handle_litellm_error(
            litellm_error(429, message, "budget_exceeded")
        )
        assert "Budget Exceeded" in result
        assert f"${spend}" in result
        assert f"${budget}" in result

    def test_a_budget_error_arriving_as_400_still_works(self, pipeline):
        """Belt and braces: classification must not depend on the status."""
        result = pipeline._handle_litellm_error(
            litellm_error(
                400,
                "Budget has been exceeded! Current cost: 5.5, Max budget: 5.0",
                "budget_exceeded",
            )
        )
        assert "Budget Exceeded" in result
        assert "$5.50" in result


class TestBudgetErrorsDoNotLeakCredentials:
    """LiteLLM puts the key owner's address and key prefix in the message.

    Chats get shared -- the incident that prompted this was found via a public
    share link -- so neither may reach the transcript, including when the
    amounts cannot be parsed and the formatter falls back.
    """

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param(PROD_BUDGET_MESSAGE, id="amounts-parse"),
            pytest.param(
                "Budget has been exceeded! Key=someone@tamu.edu (sk-...wB8Q)",
                id="amounts-unparseable",
            ),
        ],
    )
    def test_no_email_or_key_prefix_in_output(self, pipeline, message):
        result = pipeline._handle_litellm_error(
            litellm_error(429, message, "budget_exceeded")
        )
        assert "someone@tamu.edu" not in result
        assert "sk-..." not in result
        assert "Key=" not in result

    def test_fallback_still_explains_the_problem(self, pipeline):
        """Redacting must not leave an empty or useless message."""
        result = pipeline._handle_litellm_error(
            litellm_error(
                429,
                "Budget has been exceeded! Key=someone@tamu.edu (sk-...wB8Q)",
                "budget_exceeded",
            )
        )
        assert "Budget Exceeded" in result
        assert "resets daily" in result


class TestSharedTeamBudget:
    """A team budget is drained by other people's usage.

    Telling such a user they personally spent it sends them looking for their
    own runaway chat, and the fix is an administrator's, not theirs.
    """

    TEAM_MESSAGE = (
        "ExceededBudget: Team=open-webui-users over 30d budget. "
        "Spend=$501.2500, Limit=$500.00"
    )

    def test_described_as_shared_and_routed_to_an_admin(self, pipeline):
        result = pipeline._handle_litellm_error(
            litellm_error(429, self.TEAM_MESSAGE, "budget_exceeded")
        )
        assert "shared" in result
        assert "team" in result.lower()
        assert "administrator" in result
        assert "$501.25" in result

    def test_period_comes_from_the_message_not_the_user_valve(self, pipeline):
        """The window checks name their own period; it outranks the valve.

        The pipeline's valve describes the *per-user* budget (1d here), which
        would be the wrong period to quote for a 30d team budget.
        """
        result = pipeline._handle_litellm_error(
            litellm_error(429, self.TEAM_MESSAGE, "budget_exceeded")
        )
        assert "resets monthly" in result
        assert "resets daily" not in result

    def test_personal_budget_advice_is_not_shown(self, pipeline):
        """"Start a new chat" does nothing about a team-wide ceiling."""
        result = pipeline._handle_litellm_error(
            litellm_error(429, self.TEAM_MESSAGE, "budget_exceeded")
        )
        assert "Starting a new chat" not in result


class TestBudgetPeriodWording:
    @pytest.mark.parametrize(
        "period,expected",
        [("1d", "resets daily"), ("7d", "resets weekly"), ("30d", "resets monthly")],
    )
    def test_known_periods(self, period, expected):
        pipeline = build_pipeline(IMAGE_MANIFOLD, LITELLM_USER_BUDGET_PERIOD=period)
        result = pipeline._handle_litellm_error(
            litellm_error(
                429,
                "Budget has been exceeded! Current cost: 10.5, Max budget: 10.0",
                "budget_exceeded",
            )
        )
        assert expected in result

    def test_unknown_period_promises_nothing_specific(self):
        """Better vague than wrong: don't invent a reset interval."""
        pipeline = build_pipeline(IMAGE_MANIFOLD, LITELLM_USER_BUDGET_PERIOD="7h")
        result = pipeline._handle_litellm_error(
            litellm_error(
                429,
                "Budget has been exceeded! Current cost: 10.5, Max budget: 10.0",
                "budget_exceeded",
            )
        )
        assert "next budget period" in result
        assert "resets daily" not in result


class TestOtherErrorsAreUnchanged:
    """The budget hoist must not swallow the other classifications.

    These branches exist only in the image manifold.
    """

    def test_a_genuine_rate_limit_is_still_a_rate_limit(self, image_pipeline):
        result = image_pipeline._handle_litellm_error(
            litellm_error(
                429,
                "litellm.RateLimitError: Max parallel request limit reached",
                "rate_limit_error",
            )
        )
        assert "Rate Limit Reached" in result
        assert "Budget Exceeded" not in result

    def test_context_window_overflow(self, image_pipeline):
        result = image_pipeline._handle_litellm_error(
            litellm_error(
                400,
                "litellm.ContextWindowExceededError: prompt is too long: "
                "215000 tokens > 200000 maximum",
                "invalid_request_error",
            )
        )
        assert "Conversation Too Long" in result
        assert "215000" in result.replace(",", "")
        assert "Budget Exceeded" not in result

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures(self, image_pipeline, status):
        result = image_pipeline._handle_litellm_error(
            litellm_error(status, "key not allowed to access model", "auth_error")
        )
        assert "Access Denied" in result

    def test_guardrail_block(self, image_pipeline):
        from conftest import FakeResponse

        result = image_pipeline._handle_litellm_error(
            FakeResponse(
                400,
                {
                    "error": {
                        "message": str(
                            {
                                "bedrock_guardrail_response": {
                                    "blockedResponse": "Blocked by policy."
                                }
                            }
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        )
        assert "Content Filtered" in result
        assert "Blocked by policy." in result

    def test_a_non_json_body_does_not_raise(self, pipeline):
        from conftest import FakeResponse

        result = pipeline._handle_litellm_error(
            FakeResponse(502, invalid_json=True)
        )
        assert "Service Error" in result
        assert "502" in result
