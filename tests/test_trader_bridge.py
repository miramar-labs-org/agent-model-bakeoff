"""Unit tests for the stdlib-only parts of evallib/trader_bridge.py.

The `run_*` entrypoints call the real `multi-agent-ai-trader` code + a live
model and are exercised by the harness spike, not here. What is covered here is
the recorded-fixture unwrap and the small date helpers, which is where a silent
regression would corrupt every option_pick replay.
"""

import json
import os

from evallib import trader_bridge as tb


def test_trader_pin_is_a_full_sha():
    assert len(tb.TRADER_PIN) == 40
    assert all(c in "0123456789abcdef" for c in tb.TRADER_PIN)
    assert tb.TRADER_PIN in tb._CONFIG_RAW_URL


def test_install_stub_env_sets_defaults_without_clobbering():
    os.environ.pop("ALPACA_PAPER_API_KEY", None)
    os.environ["OPENAI_API_KEY"] = "already-here"
    tb.install_stub_env()
    assert os.environ["ALPACA_PAPER_API_KEY"] == "stub-not-used"
    assert os.environ["OPENAI_API_KEY"] == "already-here"  # not clobbered
    assert os.environ["SLACK_WEBHOOK_URL2"] == ""


def test_fixture_text_unwraps_alpaca_mcp_envelope():
    payload = {"next_page_token": None, "snapshots": {"MSFT260911C00540000": {"greeks": {"delta": 0.1}}}}
    entry = {"output": [{"type": "text", "text": json.dumps(
        {"_alpaca_mcp_security": {"trust": "untrusted_tool_output"}, "data": payload}
    )}]}
    out = tb._fixture_text(entry)
    assert json.loads(out) == payload  # security wrapper stripped, `data` promoted


def test_fixture_text_passes_through_unwrapped_json():
    raw = json.dumps({"snapshots": {"X": {}}})
    entry = {"output": [{"type": "text", "text": raw}]}
    assert json.loads(tb._fixture_text(entry)) == {"snapshots": {"X": {}}}


def test_fixture_text_tolerates_bare_string_and_non_json():
    assert tb._fixture_text("not json at all") == "not json at all"
    assert tb._fixture_text({"output": [{"type": "text", "text": "boom"}]}) == "boom"


def test_as_date_parses_iso_and_passes_date_through():
    import datetime as dt

    assert tb._as_date("2026-09-11") == dt.date(2026, 9, 11)
    assert tb._as_date("2026-09-11T00:00:00Z") == dt.date(2026, 9, 11)
    assert tb._as_date(dt.date(2026, 9, 11)) == dt.date(2026, 9, 11)
    assert tb._as_date("garbage") is None
    assert tb._as_date(None) is None


def test_freeze_datetime_pins_now_to_reference_date():
    import datetime as dt

    frozen = tb._freeze_datetime(dt.date(2026, 8, 28))
    assert frozen.now().date() == dt.date(2026, 8, 28)
    et = dt.timezone(dt.timedelta(hours=-4))
    assert frozen.now(et).date() == dt.date(2026, 8, 28)
    assert frozen.now(et).tzinfo == et
