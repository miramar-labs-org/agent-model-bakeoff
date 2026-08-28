"""Deterministic gates for the `dealer_signal` task.

The Dealer's real entrypoint is `src.dealer.graph.llm_call` — one
`.with_structured_output(Signal)` call. `Signal` is
`{"symbol", "action" in {BUY,HOLD,SELL}, "reasoning", "size_hint" 0-1,
"confidence" 0-1}`.

The regression this guards: `.with_structured_output` returning `''` /
`{}` on the first attempt (observed with qwen3.6:35b-a3b).

stdlib-only — safe to `# inline: evallib/gates_signal.py` into a component body.
"""

_ACTIONS = {"BUY", "SELL", "HOLD"}


def requested_symbol(inputs) -> str:
    """The symbol the Dealer was asked to judge, UPPERCASE, or ''."""
    if not isinstance(inputs, dict):
        return ""
    for key in ("symbol", "ticker"):
        v = inputs.get(key)
        if v:
            return str(v).strip().upper()
    cand = inputs.get("candidate") or inputs.get("pick")
    if isinstance(cand, dict) and cand.get("symbol"):
        return str(cand["symbol"]).strip().upper()
    return ""


def _in_unit_interval(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return 0.0 <= f <= 1.0


def dealer_signal_gates(output, inputs, gate_cfg: dict) -> dict:
    """Return {gate_name: bool}. `gate_pass` for the case is `all(...)`.

    No tunable thresholds here — the HOLD-rate band in
    gate_thresholds.dealer_signal is an *aggregate* check, applied by
    `hold_rate_ok` over the whole slice, not per case.
    """
    is_obj = isinstance(output, dict) and bool(output)
    action = str(output.get("action", "")).strip().upper() if is_obj else ""
    want_sym = requested_symbol(inputs)
    got_sym = str(output.get("symbol", "")).strip().upper() if is_obj else ""

    return {
        # First-attempt structured output was a real object, not '' / {}.
        "nonempty": is_obj and action != "",
        "action_valid": action in _ACTIONS,
        "confidence_in_range": is_obj and _in_unit_interval(output.get("confidence", 1.0)),
        "size_hint_in_range": is_obj and _in_unit_interval(output.get("size_hint", 1.0)),
        "reasoning_nonempty": is_obj and bool(str(output.get("reasoning", "")).strip()),
        # If the case names a symbol, the signal must be about that symbol.
        "symbol_matches": (not want_sym) or (got_sym == want_sym),
    }


def hold_rate(rows: list) -> float:
    """Fraction of `dealer_signal` rows whose model output action is HOLD.

    Rows are the harness's per-case output dicts (or full result rows with an
    `output` key). Cases that errored / produced no action don't count toward
    the denominator.
    """
    actions = []
    for r in rows:
        out = r.get("output", r) if isinstance(r, dict) else None
        if not isinstance(out, dict):
            continue
        a = str(out.get("action", "")).strip().upper()
        if a in _ACTIONS:
            actions.append(a)
    if not actions:
        return 0.0
    return sum(1 for a in actions if a == "HOLD") / len(actions)


def hold_rate_ok(rows: list, gate_cfg: dict) -> bool:
    """Aggregate sanity band: not ~0% HOLD (reckless) or ~100% HOLD (inert)."""
    lo = float(gate_cfg.get("hold_rate_min", 0.05))
    hi = float(gate_cfg.get("hold_rate_max", 0.95))
    return lo <= hold_rate(rows) <= hi
