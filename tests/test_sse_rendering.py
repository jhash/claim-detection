"""Tests for the SSE event-rendering helpers in app/main.py.

These specifically guard against the bug class that motivated the redesign:
data being JSON instead of HTML, or the data containing newlines that
break the SSE wire format.
"""

from __future__ import annotations

from app.main import _render_result_html, _render_status_html, _sse_event


def test_status_html_contains_pill_class():
    out = _render_status_html("queued")
    assert "status-queued" in out
    assert "queued" in out.lower()


def test_status_html_handles_unknown_status():
    out = _render_status_html("weird")
    assert "status-weird" in out  # CSS gracefully falls back to default


def test_result_html_for_claim():
    out = _render_result_html({"is_claim": True, "confidence": 0.95}, "finished")
    assert "CLAIM" in out
    assert "95.0%" in out
    assert 'class="result claim"' in out


def test_result_html_for_not_claim():
    out = _render_result_html({"is_claim": False, "confidence": 0.61}, "finished")
    assert "NOT A CLAIM" in out
    assert "61.0%" in out
    assert "not-claim" in out


def test_result_html_for_failed_job():
    out = _render_result_html({"error": "boom"}, "failed")
    assert "error" in out.lower()
    assert "boom" in out


def test_result_html_for_garbage_payload():
    """Defensive: don't crash if Redis hands us None."""
    out = _render_result_html(None, "failed")
    assert "error" in out.lower()


def test_sse_event_collapses_newlines():
    """The wire format uses \\n as the line delimiter, so embedded
    newlines inside `data:` would split into multiple events. The helper
    must normalize."""
    out = _sse_event("result", "<span>line1\nline2</span>")
    # Exactly two lines (event + data), then blank line terminator.
    assert out.count("\n") == 3
    assert "line1 line2" in out  # newline replaced with space
    assert out.endswith("\n\n")


def test_sse_event_format():
    out = _sse_event("status", "<span>queued</span>")
    assert out.startswith("event: status\n")
    assert "data: <span>queued</span>" in out
    assert out.endswith("\n\n")