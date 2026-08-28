"""Thin bridge to the real `multi-agent-ai-trader` agent entrypoints.

This is **not** a re-implementation of the agents. It only does the plumbing a
harness needs to call the production code directly:

  1. `install_stub_env()` — sets placeholder env vars so `import src.*` and the
     entrypoints run without real Alpaca / Slack / LangSmith credentials.
  2. `make_cfg(base_url, model, ...)` — builds the `OmegaConf` object the
     entrypoints take as their `cfg` argument, pinned to the same
     `multi-agent-ai-trader` commit the wheel is installed from, with the
     candidate model's endpoint patched in and the advisory DB / tracing
     features disabled.
  3. `run_analyst_case` / `run_dealer_signal_case` — reconstruct the LangGraph
     node state from a dataset case's `inputs` and invoke `llm_select` /
     `llm_call`.
  4. `build_fixture_tools` / `run_option_pick_case` — replay a dataset case's
     recorded Alpaca MCP tool outputs through the real
     `_select_option_contract_async` loop, with `get_options_tools` and the
     module clock monkeypatched.

`multi-agent-ai-trader`'s imports are side-effect-free (lazy Alpaca client,
guarded Slack webhook), so importing it under stub env is safe.

Not stdlib-only (needs `omegaconf` + `requests`, both transitive deps of the
`multi-agent-ai-trader` wheel), but still `# inline:`-splice-safe — every
import is re-indentable into a component method body.
"""

import asyncio
import contextlib
import datetime as _dt
import os

# The multi-agent-ai-trader#21 squash commit — keep in lock-step with the pin in
# requirements.txt and every harness cell's `packages_to_install`.
TRADER_PIN = "174e86b6342d6f7b91e4895c5b4ccc9fe91464b7"
_CONFIG_RAW_URL = (
    "https://raw.githubusercontent.com/miramar-labs-org/multi-agent-ai-trader/"
    f"{TRADER_PIN}/config.yaml"
)

# Placeholder credentials — only filled in if unset, so a caller can still point
# the harness at a real paper account. Never actually used by the current
# harnesses (analyst / dealer-signal never touch Alpaca; option-pick replays
# recorded tool output).
_STUB_CREDENTIALS = {
    "ALPACA_PAPER_API_KEY": "stub-not-used",   # live_account_env_names() defaults
    "ALPACA_PAPER_API_SECRET": "stub-not-used",
    "OPENAI_API_KEY": "not-needed",            # ChatOpenAI init demands *some* key
}

# Safety vars — forced regardless of the ambient environment so a harness run
# can never post to Slack or emit LangSmith traces, even from a dev shell that
# has the real values exported.
_FORCED_SAFETY_ENV = {
    "SLACK_WEBHOOK_URL2": "",       # src/common/slack.py: empty => _post() no-ops
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
}


def install_stub_env() -> None:
    """Fill placeholder credentials (if unset) and force the safety vars off."""
    for key, value in _STUB_CREDENTIALS.items():
        os.environ.setdefault(key, value)
    os.environ.update(_FORCED_SAFETY_ENV)


def make_cfg(base_url: str, model: str, *, temperature: float = 0.0,
             request_timeout_s: int = 120):
    """Build the `cfg` object the trader entrypoints expect.

    Starts from `config.yaml` at `TRADER_PIN` (so it matches the installed
    wheel), then overrides the model endpoint and disables the advisory
    features a harness has no backing services for:

      * `llm.base_url` / `llm.model` / `llm.temperature` / `llm.request_timeout_s`
      * `strategy.dealer_memory.enabled = false`  (no Postgres)
      * `langsmith.enabled = false`               (no tracing)
    """
    import requests
    from omegaconf import OmegaConf

    resp = requests.get(_CONFIG_RAW_URL, timeout=15)
    resp.raise_for_status()
    cfg = OmegaConf.create(resp.text)

    cfg.llm.base_url = base_url
    cfg.llm.model = model
    cfg.llm.temperature = float(temperature)
    cfg.llm.request_timeout_s = int(request_timeout_s)

    OmegaConf.update(cfg, "strategy.dealer_memory.enabled", False, force_add=True)
    OmegaConf.update(cfg, "langsmith.enabled", False, force_add=True)
    return cfg


# --------------------------------------------------------------------------- #
# analyst_universe / dealer_signal — single structured call, no tools
# --------------------------------------------------------------------------- #

def run_analyst_case(inputs: dict, cfg) -> dict:
    """Invoke the real Analyst universe selector on a dataset case's `inputs`.

    Returns the `PortfolioSelection` dict (`llm_select`'s `state["selection"]`).
    """
    from src.analyst.graph import llm_select

    state = dict(inputs)
    state.setdefault("research_text", "")
    state.setdefault("indicator_text", "")
    state.setdefault("track_record_text", "")
    state.setdefault("pnl_text", "")
    out = llm_select(state, cfg)
    return out.get("selection")


def run_dealer_signal_case(inputs: dict, cfg) -> dict:
    """Invoke the real Dealer BUY/SELL/HOLD signal node on a case's `inputs`.

    Returns the `Signal` dict (`llm_call`'s `state["signal"]`).
    """
    from src.dealer.graph import llm_call

    state = dict(inputs)
    state.setdefault("indicators_text", "")
    state.setdefault("ohlcv_features_text", "")
    out = llm_call(state, cfg)
    return out.get("signal")


# --------------------------------------------------------------------------- #
# option_pick — replay recorded Alpaca MCP tool output through the real loop
# --------------------------------------------------------------------------- #

def _fixture_text(entry) -> str:
    """Pull the recorded text payload out of one fixture entry and unwrap it.

    Recorded shape: `{"output": [{"type": "text", "text": "<json>"}]}` where the
    JSON is the alpaca-mcp-server envelope
    `{"_alpaca_mcp_security": {...}, "data": <payload>}` (added by that server's
    output-security middleware). At runtime the trader's `parse_option_chain` /
    `compact_tool_result` are handed this same wrapped string and look for a
    top-level `snapshots` key — which the wrapper hides. To make the option-pick
    eval measure the *model* rather than that envelope handling, this returns the
    unwrapped `data` payload (the shape `parse_option_chain` expects). The
    envelope-unwrap gap in the trader itself is tracked as a separate finding.
    """
    if isinstance(entry, str):
        text = entry
    else:
        blocks = entry.get("output") if isinstance(entry, dict) else entry
        if isinstance(blocks, str):
            text = blocks
        else:
            parts = []
            for block in blocks or []:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                elif isinstance(block, str):
                    parts.append(block)
            text = "\n".join(parts)

    import json

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(parsed, dict) and "_alpaca_mcp_security" in parsed and "data" in parsed:
        return json.dumps(parsed["data"])
    return text


def _make_fixture_tool(name: str, outputs: list):
    """A real LangChain `StructuredTool` that replays `outputs` in call order.

    Has to be a genuine tool object (not a duck-typed shim) because the option
    loop does `llm.bind_tools(tools)`, which rejects anything it can't convert to
    a function schema. Args are accepted but ignored — Alpaca MCP fixtures carry
    no arg-to-output matching.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    class _AnyArgs(BaseModel):
        model_config = ConfigDict(extra="allow")

    state = {"i": 0}
    payloads = list(outputs)

    async def _replay(**_kwargs) -> str:
        if not payloads:
            return "{}"
        out = payloads[min(state["i"], len(payloads) - 1)]
        state["i"] += 1
        return out

    return StructuredTool.from_function(
        coroutine=_replay,
        name=name,
        description=f"Replays recorded Alpaca MCP `{name}` output for an offline eval.",
        args_schema=_AnyArgs,
    )


def build_fixture_tools(fixtures: dict) -> list:
    """`{tool_name: [entry, ...]}` -> list of replay `StructuredTool`s."""
    tools = []
    for name, entries in (fixtures or {}).items():
        tools.append(_make_fixture_tool(name, [_fixture_text(e) for e in entries or []]))
    return tools


def _freeze_datetime(ref_date: _dt.date):
    """A `datetime` subclass whose `.now(tz)` is pinned to noon on `ref_date`.

    The option loop computes `today = datetime.now(et).date()` internally and
    derives its DTE window from it. For an offline replay the recorded chain was
    pulled on the case's own date, so the window has to be anchored there rather
    than at wall-clock now.
    """

    class _Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            base = _dt.datetime(ref_date.year, ref_date.month, ref_date.day, 12, 0)
            return base.replace(tzinfo=tz) if tz is not None else base

    return _Frozen


@contextlib.contextmanager
def _patched_option_loop(fixture_tools: list, ref_date):
    """Monkeypatch `src.dealer.graph` for an offline option-pick replay:

      * `get_options_tools` -> async no-op returning `fixture_tools`
      * `datetime`          -> frozen to `ref_date` (if given)
      * `_trim_history`     -> call-counting wrapper (loop rounds = calls - 1)
    """
    import src.dealer.graph as g

    saved = {"get_options_tools": g.get_options_tools, "datetime": g.datetime,
             "_trim_history": g._trim_history}
    counter = {"trim_calls": 0}

    async def _fake_get_options_tools():
        return fixture_tools

    real_trim = g._trim_history

    def _counting_trim(messages, cap):
        counter["trim_calls"] += 1
        return real_trim(messages, cap)

    g.get_options_tools = _fake_get_options_tools
    g._trim_history = _counting_trim
    if ref_date is not None:
        g.datetime = _freeze_datetime(ref_date)
    try:
        yield counter
    finally:
        for name, value in saved.items():
            setattr(g, name, value)


def _tb_parse_date(value):
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def run_option_pick_case(inputs: dict, fixtures: dict, cfg, *,
                         reference_date=None, timeout_s: int = 300) -> dict:
    """Replay one `option_pick` case through the real `_select_option_contract_async`.

    `reference_date` anchors the loop's DTE window (defaults to the earliest
    expiration in the recorded chain minus `dte_min`, so the window brackets it).

    Returns the shape `evallib.gates_option.option_pick_gates` expects::

        {"pick": <OptionContractPick dict>|None, "rounds": int,
         "used_fallback": bool, "reference_date": "YYYY-MM-DD"}
    """
    from src.dealer.graph import _select_option_contract_async
    from src.dealer.option_chain import parse_option_chain

    signal = inputs.get("signal") or {}
    state = dict(inputs)
    state.setdefault("symbol", signal.get("symbol", ""))

    fixture_tools = build_fixture_tools(fixtures)

    ref = _tb_parse_date(reference_date)
    if ref is None:
        exps = []
        for entries in (fixtures or {}).values():
            for entry in entries or []:
                for row in parse_option_chain(_fixture_text(entry)):
                    d = _tb_parse_date(row.get("expiration"))
                    if d is not None:
                        exps.append(d)
        if exps:
            dte_min = int(cfg.options_trading.dte_min)
            ref = min(exps) - _dt.timedelta(days=dte_min)

    async def _run():
        return await asyncio.wait_for(
            _select_option_contract_async(state, cfg, signal), timeout=timeout_s
        )

    with _patched_option_loop(fixture_tools, ref) as counter:
        pick = asyncio.run(_run())

    pick_dict = pick.model_dump() if pick is not None else None
    used_fallback = bool(
        pick_dict and str(pick_dict.get("reasoning", "")).startswith("deterministic fallback")
    )
    return {
        "pick": pick_dict,
        "rounds": max(counter["trim_calls"] - 1, 0),
        "used_fallback": used_fallback,
        "reference_date": ref.isoformat() if ref is not None else None,
    }
