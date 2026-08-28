"""Unit tests for evallib/gates_option.py."""

from evallib import gates_option as g

_CFG = {"dte_min": 14, "dte_max": 45, "delta_min": 0.30, "delta_max": 0.60, "max_rounds": 6}
_INPUTS = {"signal": {"action": "BUY"}}


def _pick(**kw):
    base = {"contract_symbol": "AAPL250117C00190000", "strike": 190.0,
            "expiration": "2026-09-20", "right": "call", "delta": 0.45,
            "premium": 3.20, "reasoning": "OI 4k, tight spread, delta mid-band"}
    base.update(kw)
    return base


def _result(**kw):
    base = {"pick": _pick(), "rounds": 3, "used_fallback": False,
            "reference_date": "2026-08-28"}
    base.update(kw)
    return base


def test_clean_pick_passes_all():
    gates = g.option_pick_gates(_result(), _INPUTS, _CFG)
    assert all(gates.values()), gates


def test_missing_pick_fails_nonempty_and_fallback():
    gates = g.option_pick_gates(_result(pick=None), _INPUTS, _CFG)
    assert gates["nonempty_pick"] is False
    assert gates["not_fallback"] is False


def test_fallback_flag_fails_not_fallback():
    assert g.option_pick_gates(_result(used_fallback=True), _INPUTS, _CFG)["not_fallback"] is False


def test_too_many_rounds_fails():
    assert g.option_pick_gates(_result(rounds=7), _INPUTS, _CFG)["within_max_rounds"] is False


def test_dte_outside_window_fails():
    near = g.option_pick_gates(_result(pick=_pick(expiration="2026-09-05")), _INPUTS, _CFG)
    assert near["dte_in_window"] is False  # 8 DTE
    far = g.option_pick_gates(_result(pick=_pick(expiration="2026-11-01")), _INPUTS, _CFG)
    assert far["dte_in_window"] is False


def test_delta_outside_band_fails():
    assert g.option_pick_gates(_result(pick=_pick(delta=0.15)), _INPUTS, _CFG)["delta_in_band"] is False
    assert g.option_pick_gates(_result(pick=_pick(delta=-0.75)), _INPUTS, _CFG)["delta_in_band"] is False


def test_negative_delta_within_band_ok():
    assert g.option_pick_gates(_result(pick=_pick(delta=-0.45)), _INPUTS, _CFG)["delta_in_band"] is True


def test_wrong_right_for_sell_signal():
    gates = g.option_pick_gates(_result(), {"signal": {"action": "SELL"}}, _CFG)
    assert gates["right_matches_signal"] is False  # call on a SELL


def test_put_matches_sell_signal():
    gates = g.option_pick_gates(
        _result(pick=_pick(right="put", delta=-0.45)), {"signal": {"action": "SELL"}}, _CFG
    )
    assert gates["right_matches_signal"] is True


def test_unparseable_expiration_fails():
    gates = g.option_pick_gates(_result(pick=_pick(expiration="Jan 17 2026")), _INPUTS, _CFG)
    assert gates["expiration_parseable"] is False
    assert gates["dte_in_window"] is False


def test_zero_premium_fails():
    assert g.option_pick_gates(_result(pick=_pick(premium=0)), _INPUTS, _CFG)["premium_positive"] is False


def test_expected_right_mapping():
    assert g.expected_right({"signal": {"action": "BUY"}}) == "call"
    assert g.expected_right({"signal": {"action": "SELL"}}) == "put"
    assert g.expected_right({"signal": {"action": "HOLD"}}) == ""
