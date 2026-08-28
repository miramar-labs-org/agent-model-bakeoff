"""Deterministic gates for the `analyst_universe` task.

The Analyst's real entrypoint is `src.analyst.graph.llm_select` — one
`.with_structured_output(PortfolioSelection)` call. `PortfolioSelection` is
`{"symbols": [{"symbol", "exchange", "budget", "indicators", "rationale"}, ...]}`.

These gates encode the failure modes that motivated the bakeoff
(qwen3.6:35b-a3b, observed 2026-08-28): picks hallucinated outside the
candidate list, and the universe collapsing toward the mega-cap fallback.

stdlib-only — safe to `# inline: evallib/gates_analyst.py` into a component body.
"""


def candidate_symbols(inputs) -> set:
    """Pull the screener candidate tickers out of a case's `inputs` blob.

    `llm_select`'s node state carries `raw_candidates`, which over the trace
    history has been either a list of bare tickers or a list of
    `{"symbol": ...}`-ish dicts. Tolerate both, plus a pre-dug list.
    """
    raw = None
    if isinstance(inputs, dict):
        raw = inputs.get("raw_candidates", inputs.get("candidates"))
    elif isinstance(inputs, (list, tuple)):
        raw = inputs
    out = set()
    for item in raw or []:
        if isinstance(item, str):
            sym = item
        elif isinstance(item, dict):
            sym = item.get("symbol") or item.get("ticker") or ""
        else:
            sym = ""
        sym = str(sym).strip().upper()
        if sym:
            out.add(sym)
    return out


def picked_symbols(output) -> list:
    """Ordered list of UPPERCASE tickers the model picked (may contain dups)."""
    rows = output.get("symbols") if isinstance(output, dict) else None
    out = []
    for r in rows or []:
        sym = ""
        if isinstance(r, dict):
            sym = r.get("symbol") or ""
        elif isinstance(r, str):
            sym = r
        sym = str(sym).strip().upper()
        if sym:
            out.append(sym)
    return out


def _picks_well_formed(output) -> bool:
    rows = output.get("symbols") if isinstance(output, dict) else None
    if not rows or not isinstance(rows, list):
        return False
    for r in rows:
        if not isinstance(r, dict):
            return False
        if not str(r.get("symbol", "")).strip():
            return False
        try:
            if float(r.get("budget")) <= 0:
                return False
        except (TypeError, ValueError):
            return False
        if not str(r.get("rationale", "")).strip():
            return False
    return True


def analyst_universe_gates(output, inputs, gate_cfg: dict) -> dict:
    """Return {gate_name: bool}. `gate_pass` for the case is `all(...)`.

    gate_cfg keys (from config.yaml gate_thresholds.analyst_universe):
      min_universe   — reject a collapsed book (default 8)
      max_universe   — the trader's `analyst.max_universe_size` (default 10)
    """
    min_universe = int(gate_cfg.get("min_universe", 8))
    max_universe = int(gate_cfg.get("max_universe", 10))

    picks = picked_symbols(output)
    candidates = candidate_symbols(inputs)

    return {
        # The '' / empty-structured-output regression: a real book came back.
        "nonempty": len(picks) >= 1,
        "schema_valid": _picks_well_formed(output),
        # Every pick was actually on the screener's candidate list.
        "picks_in_candidate_list": bool(candidates)
        and all(p in candidates for p in picks),
        "no_duplicate_picks": len(picks) == len(set(picks)),
        # Universe neither collapsed to the fallback nor over the trader's cap.
        "min_universe": len(set(picks)) >= min_universe,
        "within_max_universe": len(set(picks)) <= max_universe,
    }
