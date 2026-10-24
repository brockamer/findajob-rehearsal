"""Cost-logging regression test for scripts/rescore_all.py (#470 update).

After #470 the local score_job duplicate was removed; rescore_all.py
imports the canonical findajob.scoring.score_job which returns the
3-tuple (parsed, latency_ms, completion). cost_log rows now populate
cost_usd via cost_usd_override=completion.cost_usd on the LLM path
(authoritative usage.cost from the wrapper). On the prefilter path
(completion=None, no LLM call made) the caller passes cost_usd_override=0.0
and zeroed token overrides rather than falling through to the chars/4
heuristic over the reconstructed input/output text.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

from findajob.cost_tracking import log_call
from scripts.rescore_all import score_job

VALID_LLM_JSON = (
    '{"relevance_score": 7, "interview_likelihood": 6, '
    '"strengths_alignment": "good fit", "industry_sector": "infra", '
    '"comp_estimate": "competitive", "ai_notes": "ok", '
    '"remote_status": "Remote", "score_status": "scored"}'
)

COST_LOG_SCHEMA = """
CREATE TABLE cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms INTEGER,
    success INTEGER DEFAULT 1,
    error_message TEXT,
    logged_at TEXT DEFAULT (datetime('now')),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL
);
"""


class _MockCompletion:
    """Minimal CompletionResult double — only fields the score_job path reads."""

    def __init__(self, text: str, cost_usd: float = 0.001234) -> None:
        self.text = text
        self.cost_usd = cost_usd
        self.prompt_tokens = 100
        self.completion_tokens = 20
        self.cached_tokens = 0
        self.generation_id = "gen-test"


def test_score_job_returns_three_tuple_on_llm_path() -> None:
    """LLM path: score_job returns (parsed, latency_ms, completion).

    completion is non-None so callers can write cost_usd_override.
    """
    with patch(
        "findajob.scoring.complete",
        return_value=_MockCompletion(VALID_LLM_JSON),
    ):
        result = score_job(
            "Senior Engineer",
            "Acme",
            "Remote",
            "Strong JD with detailed responsibilities " * 50,
            candidate_profile="profile",
        )
    assert len(result) == 3
    parsed, latency_ms, completion = result
    assert parsed.get("relevance_score") == 7
    assert latency_ms >= 0
    assert completion is not None
    assert completion.cost_usd == 0.001234


def test_score_job_returns_three_tuple_on_prefilter_path() -> None:
    """Prefilter path: completion is None (no LLM call was made).

    JD and candidate profile are long and non-empty — proves the zero
    cost below comes from the override, not from empty-string inputs
    that happen to compute to 0.0 under the heuristic.
    """
    long_jd = "Detailed job description with responsibilities and requirements. " * 50
    long_profile = "Candidate profile with extensive background and experience. " * 50
    with patch("findajob.scoring.complete") as mock_complete:
        # A title that hits the hard-reject prefilter — wrapper not called.
        result = score_job("Software Engineer", "Acme", "Remote", long_jd, candidate_profile=long_profile)
    assert len(result) == 3
    parsed, latency_ms, completion = result
    assert latency_ms == 0
    assert completion is None
    assert not mock_complete.called


def test_prefilter_hit_logs_zero_cost_not_heuristic() -> None:
    """A prefilter-hit cost_log row gets cost_usd=0.0, not a chars/4 estimate.

    Mirrors the cost_usd_override/token-override logic in rescore_all.py's
    (and orchestrator.py's) scoring loop: when completion is None, the
    overrides passed to log_call must be 0 / 0.0 — never None, which
    would fall through to the heuristic over the long non-empty text below.
    """
    long_jd = "Detailed job description with responsibilities and requirements. " * 50
    long_profile = "Candidate profile with extensive background and experience. " * 50
    with patch("findajob.scoring.complete") as mock_complete:
        parsed, latency_ms, completion = score_job(
            "Software Engineer", "Acme", "Remote", long_jd, candidate_profile=long_profile
        )
    assert completion is None
    assert not mock_complete.called

    conn = sqlite3.connect(":memory:")
    conn.executescript(COST_LOG_SCHEMA)
    log_call(
        conn,
        job_id="job-1",
        operation="rescore",
        model="openrouter:deepseek/deepseek-v3.2",
        input_text=long_jd + long_profile,
        output_text=str(parsed),
        latency_ms=latency_ms,
        success=True,
        cost_usd_override=(completion.cost_usd if completion is not None else 0.0),
        input_tokens_override=(completion.prompt_tokens if completion is not None else 0),
        output_tokens_override=(completion.completion_tokens if completion is not None else 0),
    )
    conn.commit()
    row = conn.execute("SELECT cost_usd, input_tokens, output_tokens FROM cost_log WHERE job_id = 'job-1'").fetchone()
    assert row == (0.0, 0, 0)
