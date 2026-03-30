import pytest

try:
    from src.execution import OrderManager
    from src.markets import Market, TokenInfo
    from src.signals.base import TradingSide
    EXECUTION_AVAILABLE = True
except ImportError:
    EXECUTION_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not EXECUTION_AVAILABLE,
    reason="src/execution.py not yet implemented",
)


class Dummy:
    execution = {}


def test_pick_token_id_yes_no_order_independent():
    m = Market(
        id="1",
        question="Q",
        description="",
        category="Other",
        end_date=None,
        volume_24h=0.0,
        liquidity=0.0,
        tokens={
            "No": TokenInfo(token_id="no_tok", outcome="No", price=0.7, volume_24h=0.0),
            "Yes": TokenInfo(
                token_id="yes_tok", outcome="Yes", price=0.3, volume_24h=0.0
            ),
        },
    )

    om = OrderManager(config=Dummy(), client=None, portfolio=None)

    assert om.pick_token_id(m, TradingSide.BUY_YES) == "yes_tok"
    assert om.pick_token_id(m, TradingSide.BUY_NO) == "no_tok"


def test_pick_token_id_requires_explicit_outcomes():
    m = Market(
        id="1",
        question="Q",
        description="",
        category="Other",
        end_date=None,
        volume_24h=0.0,
        liquidity=0.0,
        tokens={
            "A": TokenInfo(token_id="a", outcome="A", price=0.5, volume_24h=0.0),
            "B": TokenInfo(token_id="b", outcome="B", price=0.5, volume_24h=0.0),
        },
    )

    om = OrderManager(config=Dummy(), client=None, portfolio=None)
    assert om.pick_token_id(m, TradingSide.BUY_YES) is None
    assert om.pick_token_id(m, TradingSide.BUY_NO) is None
