"""Unit tests for evallib/judge_prompts.py — prompt builders + skip behaviour."""

from evallib import judge_prompts as jp


def test_analyst_prompt_includes_indicator_text_and_rationales():
    inputs = {"indicator_text": "AAPL RSI 74.9, MACD +1.2"}
    output = {"symbols": [{"symbol": "AAPL", "rationale": "RSI hot at 74.9"}]}
    p = jp.analyst_universe_judge_prompt(inputs, output)
    assert "RSI 74.9" in p and "RSI hot at 74.9" in p


def test_analyst_prompt_none_when_no_picks():
    assert jp.analyst_universe_judge_prompt({"indicator_text": "x"}, {"symbols": []}) is None


def test_dealer_prompt_includes_reasoning_and_context():
    inputs = {"news_text": "AAPL beat earnings"}
    output = {"action": "BUY", "confidence": 0.8, "reasoning": "earnings beat + RSI"}
    p = jp.dealer_signal_judge_prompt(inputs, output)
    assert "AAPL beat earnings" in p and "earnings beat + RSI" in p


def test_dealer_prompt_none_on_empty_output():
    assert jp.dealer_signal_judge_prompt({"news_text": "x"}, {}) is None


def test_option_prompt_includes_contract_and_signal():
    inputs = {"signal": {"action": "BUY"}}
    result = {"pick": {"contract_symbol": "AAPL2026C", "strike": 190,
                       "reasoning": "tight spread, OI 4k"}}
    p = jp.option_pick_judge_prompt(inputs, result)
    assert "AAPL2026C" in p and "tight spread, OI 4k" in p and "BUY" in p


def test_option_prompt_none_when_no_contract():
    assert jp.option_pick_judge_prompt({"signal": {}}, {"pick": None}) is None


def test_by_task_registry_covers_all_three():
    assert set(jp.BY_TASK) == {"analyst_universe", "dealer_signal", "option_pick"}


def test_long_text_is_clipped():
    huge = "x" * 20000
    p = jp.dealer_signal_judge_prompt({"news_text": huge}, {"action": "HOLD", "reasoning": "y"})
    assert "[truncated]" in p
