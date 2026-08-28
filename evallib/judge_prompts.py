"""Task-specific LLM-as-judge prompt builders.

Each `*_judge_prompt(inputs, output)` returns the *task* portion of the judge
call — `evallib.rubric.wrap_judge_prompt` wraps it in the fixed 1-5 rubric.
Return `None` to skip judging a case (e.g. nothing to ground against).

The judge scores one axis only: **groundedness** — did the model's prose cite
the specific evidence it was given, rather than generic market boilerplate?
The deterministic gates already cover structure/ranges; the judge is the
"ungrounded rationale" failure mode from the bakeoff motivation.

stdlib-only — safe to `# inline: evallib/judge_prompts.py` into a component body.
"""

import json

_MAX_CHARS = 6000


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n...[truncated]"


def _dump(obj) -> str:
    try:
        return _clip(json.dumps(obj, indent=2, default=str))
    except (TypeError, ValueError):
        return _clip(str(obj))


def analyst_universe_judge_prompt(inputs, output):
    """Grade whether each pick's rationale is grounded in the indicator text."""
    indicator_text = ""
    if isinstance(inputs, dict):
        indicator_text = inputs.get("indicator_text") or inputs.get("indicators") or ""
    rows = output.get("symbols") if isinstance(output, dict) else None
    if not rows:
        return None
    rationales = "\n".join(
        f"- {r.get('symbol', '?')}: {r.get('rationale', '')}"
        for r in rows
        if isinstance(r, dict)
    )
    return (
        "TASK: The Analyst agent picked a trading universe and gave each pick a "
        "one-line rationale. Judge ONLY whether the rationales are grounded in the "
        "technical-indicator values the Analyst was given — not whether the picks "
        "are good trades.\n\n"
        "A rationale is grounded if it references specific values (RSI, MACD, "
        "moving averages, % move, volume) that are consistent with the indicator "
        "text below. It is NOT grounded if it cites numbers absent from the text, "
        "contradicts the text, or is generic sentiment with no data.\n\n"
        f"INDICATOR TEXT PROVIDED TO THE ANALYST:\n{_clip(str(indicator_text))}\n\n"
        f"THE ANALYST'S PICKS + RATIONALES:\n{_clip(rationales)}\n\n"
        "Score 5 only if every rationale that could be grounded is; 1 if rationales "
        "cite fabricated or contradictory numbers."
    )


def dealer_signal_judge_prompt(inputs, output):
    """Grade whether the BUY/SELL/HOLD reasoning cites the evidence provided."""
    if not isinstance(output, dict) or not output:
        return None
    context = {}
    if isinstance(inputs, dict):
        for k in ("indicator_text", "indicators", "news_text", "news", "research_text"):
            if inputs.get(k):
                context[k] = inputs[k]
    return (
        "TASK: The Dealer agent emitted a BUY/SELL/HOLD signal on one symbol with a "
        "reasoning string. Judge ONLY whether the reasoning is grounded in the "
        "indicator / news context below — does it cite specific indicator values or "
        "named news items, and are they consistent with the context? Generic "
        "reasoning with no reference to the provided data scores low. Do not judge "
        "whether the call itself is correct.\n\n"
        f"CONTEXT PROVIDED TO THE DEALER:\n{_dump(context)}\n\n"
        f"SIGNAL: {output.get('action', '?')}  (confidence {output.get('confidence', '?')})\n"
        f"REASONING:\n{_clip(str(output.get('reasoning', '')))}\n\n"
        "Score 5 if the reasoning cites specific, consistent evidence; 1 if it is "
        "boilerplate or cites evidence not in the context."
    )


def option_pick_judge_prompt(inputs, output):
    """Grade whether the contract pick + reasoning follow from the signal + chain."""
    result = output if isinstance(output, dict) else {}
    pick = result.get("pick") if isinstance(result.get("pick"), dict) else result
    if not pick or not str(pick.get("contract_symbol", "")).strip():
        return None
    signal = inputs.get("signal") if isinstance(inputs, dict) else None
    return (
        "TASK: The Dealer agent ran a tool-calling loop over an options chain and "
        "picked one contract, with a reasoning string. Judge ONLY whether the "
        "reasoning is grounded: does it justify the strike / expiration / delta "
        "choice using chain data (open interest, volume, bid/ask, greeks) and is it "
        "consistent with the upstream signal direction? Do not judge whether the "
        "trade will be profitable.\n\n"
        f"UPSTREAM SIGNAL:\n{_dump(signal)}\n\n"
        f"PICKED CONTRACT:\n{_dump({k: pick.get(k) for k in ('contract_symbol', 'strike', 'expiration', 'right', 'delta', 'premium')})}\n\n"
        f"REASONING:\n{_clip(str(pick.get('reasoning', '')))}\n\n"
        "Score 5 if the reasoning ties the specific contract to concrete chain "
        "evidence and the signal; 1 if it is generic or ignores the chain."
    )


BY_TASK = {
    "analyst_universe": analyst_universe_judge_prompt,
    "dealer_signal": dealer_signal_judge_prompt,
    "option_pick": option_pick_judge_prompt,
}
