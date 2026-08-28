"""Unit tests for evallib/gates_signal.py."""

from evallib import gates_signal as g

_INPUTS = {"symbol": "AAPL"}


def _sig(action="BUY", symbol="AAPL", reasoning="RSI 68 and MACD crossover", **kw):
    return {"symbol": symbol, "action": action, "reasoning": reasoning,
            "size_hint": kw.get("size_hint", 1.0), "confidence": kw.get("confidence", 0.7)}


def test_clean_signal_passes_all():
    assert all(g.dealer_signal_gates(_sig(), _INPUTS, {}).values())


def test_empty_output_fails_nonempty():
    gates = g.dealer_signal_gates({}, _INPUTS, {})
    assert gates["nonempty"] is False
    assert gates["action_valid"] is False


def test_empty_string_action_fails():
    assert g.dealer_signal_gates(_sig(action=""), _INPUTS, {})["nonempty"] is False


def test_bad_action_fails_action_valid():
    assert g.dealer_signal_gates(_sig(action="LONG"), _INPUTS, {})["action_valid"] is False


def test_out_of_range_confidence_fails():
    assert g.dealer_signal_gates(_sig(confidence=1.5), _INPUTS, {})["confidence_in_range"] is False


def test_wrong_symbol_fails_match():
    assert g.dealer_signal_gates(_sig(symbol="MSFT"), _INPUTS, {})["symbol_matches"] is False


def test_symbol_match_skipped_when_case_has_no_symbol():
    assert g.dealer_signal_gates(_sig(symbol="MSFT"), {}, {})["symbol_matches"] is True


def test_hold_rate_computation():
    rows = [{"output": _sig(action="HOLD")}, {"output": _sig(action="BUY")},
            {"output": _sig(action="HOLD")}, {"output": _sig(action="SELL")}]
    assert g.hold_rate(rows) == 0.5


def test_hold_rate_ignores_errored_rows():
    rows = [{"output": _sig(action="HOLD")}, {"output": {}}, {"output": None}]
    assert g.hold_rate(rows) == 1.0


def test_hold_rate_ok_band():
    all_hold = [{"output": _sig(action="HOLD")} for _ in range(10)]
    assert g.hold_rate_ok(all_hold, {"hold_rate_min": 0.05, "hold_rate_max": 0.95}) is False
    mixed = [{"output": _sig(action="HOLD")}, {"output": _sig(action="BUY")}]
    assert g.hold_rate_ok(mixed, {}) is True
