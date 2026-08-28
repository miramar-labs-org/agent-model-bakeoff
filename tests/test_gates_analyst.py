"""Unit tests for evallib/gates_analyst.py."""

from evallib import gates_analyst as g

_CFG = {"min_universe": 3, "max_universe": 5}


def _pick(sym, budget=5000.0, rationale="RSI 68, above 50-DMA"):
    return {"symbol": sym, "exchange": "stocks", "budget": budget,
            "indicators": ["rsi"], "rationale": rationale}


def _output(*syms):
    return {"symbols": [_pick(s) for s in syms]}


_INPUTS = {"raw_candidates": ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOG"]}


def test_candidate_symbols_from_bare_list():
    assert g.candidate_symbols({"raw_candidates": ["aapl", "MSFT"]}) == {"AAPL", "MSFT"}


def test_candidate_symbols_from_dicts():
    assert g.candidate_symbols({"raw_candidates": [{"symbol": "aapl"}, {"ticker": "msft"}]}) == {
        "AAPL",
        "MSFT",
    }


def test_all_gates_pass_on_clean_book():
    gates = g.analyst_universe_gates(_output("AAPL", "MSFT", "NVDA", "TSLA"), _INPUTS, _CFG)
    assert all(gates.values()), gates


def test_empty_selection_fails_nonempty_and_min():
    gates = g.analyst_universe_gates({"symbols": []}, _INPUTS, _CFG)
    assert gates["nonempty"] is False
    assert gates["min_universe"] is False


def test_hallucinated_ticker_fails_candidate_list_gate():
    gates = g.analyst_universe_gates(_output("AAPL", "MSFT", "FAKE"), _INPUTS, _CFG)
    assert gates["picks_in_candidate_list"] is False


def test_collapsed_universe_fails_min():
    gates = g.analyst_universe_gates(_output("AAPL", "MSFT"), _INPUTS, _CFG)
    assert gates["min_universe"] is False


def test_over_cap_fails_max():
    gates = g.analyst_universe_gates(
        _output("AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOG"), _INPUTS, _CFG
    )
    assert gates["within_max_universe"] is False


def test_duplicate_picks_flagged():
    gates = g.analyst_universe_gates(_output("AAPL", "AAPL", "MSFT", "NVDA"), _INPUTS, _CFG)
    assert gates["no_duplicate_picks"] is False


def test_bad_budget_fails_schema():
    out = {"symbols": [_pick("AAPL", budget=0), _pick("MSFT"), _pick("NVDA")]}
    gates = g.analyst_universe_gates(out, _INPUTS, _CFG)
    assert gates["schema_valid"] is False


def test_missing_rationale_fails_schema():
    out = {"symbols": [_pick("AAPL", rationale=""), _pick("MSFT"), _pick("NVDA")]}
    assert g.analyst_universe_gates(out, _INPUTS, _CFG)["schema_valid"] is False
