"""Deterministic gates for the `option_pick` task.

The Dealer's real entrypoint is `src.dealer.graph._select_option_contract_async`
— a bounded agentic loop (<= `_MAX_TOOL_CALL_ROUNDS` = 6 tool-call rounds against
the Alpaca options MCP) followed by one `.with_structured_output(OptionContractPick)`
call, with `_fallback_pick` as a deterministic backstop.

`OptionContractPick` is `{"contract_symbol", "strike", "expiration" (YYYY-MM-DD),
"right" in {call,put}, "delta", "premium" > 0, "reasoning"}`.

The regressions this guards (qwen3.6:35b-a3b, observed 2026-08-28):
  - `.with_structured_output` returns `''` every time -> `_fallback_pick` makes
    every real contract choice.
  - the loop can churn all 6 rounds without converging.

The harness's `run_case` is expected to return a dict shaped like:
    {"pick": {<OptionContractPick fields>} | None,
     "rounds": <int tool-call rounds used>,
     "used_fallback": <bool>,          # True iff _fallback_pick produced `pick`
     "reference_date": "YYYY-MM-DD"}   # the recorded "today" for DTE math

stdlib-only — safe to `# inline: evallib/gates_option.py` into a component body.
"""

from datetime import date, datetime

_RIGHTS = {"call", "put"}
# Dealer signal action -> the option right the trader would open.
_ACTION_RIGHT = {"BUY": "call", "SELL": "put"}


def _as_date(value):
    """Parse a 'YYYY-MM-DD' string (or pass a date through). None on failure."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def expected_right(inputs) -> str:
    """The option right implied by the Dealer signal on this case, or ''.

    `_select_option_contract_async` is only reached for a BUY or SELL signal;
    BUY -> call, SELL -> put.
    """
    if not isinstance(inputs, dict):
        return ""
    sig = inputs.get("signal")
    action = ""
    if isinstance(sig, dict):
        action = str(sig.get("action", "")).strip().upper()
    if not action:
        action = str(inputs.get("action", "")).strip().upper()
    return _ACTION_RIGHT.get(action, "")


def dte_days(pick: dict, reference_date) -> "int | None":
    """Days to expiration relative to `reference_date` (the recorded 'today')."""
    exp = _as_date(pick.get("expiration")) if isinstance(pick, dict) else None
    ref = _as_date(reference_date)
    if exp is None or ref is None:
        return None
    return (exp - ref).days


def option_pick_gates(result, inputs, gate_cfg: dict) -> dict:
    """Return {gate_name: bool}. `gate_pass` for the case is `all(...)`.

    gate_cfg keys (from config.yaml gate_thresholds.option_pick):
      dte_min / dte_max     — trader options_trading.dte_min/.dte_max (14 / 45)
      delta_min / delta_max — |delta| window (0.30 / 0.60)
      max_rounds            — _MAX_TOOL_CALL_ROUNDS (6)
    """
    dte_min = int(gate_cfg.get("dte_min", 14))
    dte_max = int(gate_cfg.get("dte_max", 45))
    delta_min = float(gate_cfg.get("delta_min", 0.30))
    delta_max = float(gate_cfg.get("delta_max", 0.60))
    max_rounds = int(gate_cfg.get("max_rounds", 6))

    result = result if isinstance(result, dict) else {}
    pick = result.get("pick")
    pick = pick if isinstance(pick, dict) else {}
    has_pick = bool(pick) and bool(str(pick.get("contract_symbol", "")).strip())

    rounds = result.get("rounds")
    try:
        rounds_ok = rounds is not None and 0 <= int(rounds) <= max_rounds
    except (TypeError, ValueError):
        rounds_ok = False

    dte = dte_days(pick, result.get("reference_date"))
    try:
        delta_abs = abs(float(pick.get("delta")))
    except (TypeError, ValueError):
        delta_abs = None
    try:
        premium_ok = float(pick.get("premium")) > 0
    except (TypeError, ValueError):
        premium_ok = False

    want_right = expected_right(inputs)
    got_right = str(pick.get("right", "")).strip().lower()

    return {
        # The '' regression: a structured pick actually came back...
        "nonempty_pick": has_pick,
        # ...and it was the LLM's structured pick, not the deterministic backstop.
        "not_fallback": has_pick and not bool(result.get("used_fallback")),
        "within_max_rounds": rounds_ok,
        "expiration_parseable": _as_date(pick.get("expiration")) is not None,
        "dte_in_window": dte is not None and dte_min <= dte <= dte_max,
        "delta_in_band": delta_abs is not None and delta_min <= delta_abs <= delta_max,
        "right_valid": got_right in _RIGHTS,
        "right_matches_signal": (not want_right) or (got_right == want_right),
        "premium_positive": premium_ok,
    }
