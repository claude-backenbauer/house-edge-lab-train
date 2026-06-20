"""Closed-form economics model for a market creator / liquidity provider.

This answers, on paper, the central research question: *can a creator/LP earn
positive expected value once adverse selection and costs are accounted for?*

All quantities are abstract "units" (USD-equivalent). Fees are fractions.

Revenue
-------
    creator_fee_revenue = creator_fee * volume
    lp_fee_revenue      = lp_fee * volume * liquidity_share
    total_fee_revenue   = creator_fee_revenue + lp_fee_revenue

Costs / losses
--------------
    expected_loss_from_mispricing
        Loss from the creator's own probability estimate being off. Modelled
        as the average pricing error times the volume that trades against the
        creator's book, scaled by how much of the book the creator holds.

    adverse_selection_loss
        Loss to better-informed ("toxic") flow. Informed traders only trade
        when they have an edge, so this is a structural cost of being the LP.

    gas_and_ops_cost
        Fixed + per-trade operational overhead (gas, infra, monitoring).

Bottom line
-----------
    expected_net_profit = total_fee_revenue
                          - expected_loss_from_mispricing
                          - adverse_selection_loss
                          - gas_and_ops_cost

    break_even_volume   -- volume at which expected_net_profit == 0
    max_loss            -- worst-case loss (bounded by liquidity at risk)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EconomicsInputs:
    """Inputs to the economics model (all per-market, per-resolution-cycle)."""

    volume: float
    creator_fee: float = 0.01
    lp_fee: float = 0.02

    # Fraction of total market liquidity supplied by *us*. 1.0 == sole LP.
    liquidity_share: float = 1.0

    # Capital we have at risk in the market (bounds max loss).
    liquidity_at_risk: float = 0.0

    # Average absolute pricing error of our model, as a probability (0..1).
    # e.g. 0.03 means our prices are off by ~3 percentage points on average.
    avg_pricing_error: float = 0.03

    # Fraction of volume that is *net directional* (systematically corrects our
    # price toward truth). Uninformed/noise flow is roughly two-sided and nets
    # out, earning us fees without causing a mispricing loss; only this slice
    # turns pricing error into realised loss. Defaults to informed_flow_fraction.
    directional_fraction: float | None = None

    # Fraction of volume that effectively trades against *our* book.
    # With multiple LPs we only absorb our share.
    book_capture: float = field(default=0.0)

    # Fraction of volume that is "informed"/toxic (adverse selection).
    informed_flow_fraction: float = 0.15

    # Average edge (as a fraction of notional) that informed traders extract
    # when they trade against us.
    informed_edge: float = 0.05

    # Operational costs.
    fixed_ops_cost: float = 5.0  # one-off per market (infra/gas to create)
    per_trade_cost: float = 0.02  # gas/ops per trade
    avg_trade_size: float = 25.0  # used to convert volume -> trade count

    def __post_init__(self) -> None:
        if not self.book_capture:
            # Default: we absorb flow in proportion to our liquidity share.
            self.book_capture = self.liquidity_share
        if self.directional_fraction is None:
            # Only the informed/correcting slice of flow realises mispricing loss.
            self.directional_fraction = self.informed_flow_fraction


@dataclass
class EconomicsResult:
    creator_fee_revenue: float
    lp_fee_revenue: float
    total_fee_revenue: float
    expected_loss_from_mispricing: float
    adverse_selection_loss: float
    gas_and_ops_cost: float
    expected_net_profit: float
    break_even_volume: float | None
    max_loss: float

    def to_dict(self) -> dict:
        return {
            "creator_fee_revenue": self.creator_fee_revenue,
            "lp_fee_revenue": self.lp_fee_revenue,
            "total_fee_revenue": self.total_fee_revenue,
            "expected_loss_from_mispricing": self.expected_loss_from_mispricing,
            "adverse_selection_loss": self.adverse_selection_loss,
            "gas_and_ops_cost": self.gas_and_ops_cost,
            "expected_net_profit": self.expected_net_profit,
            "break_even_volume": self.break_even_volume,
            "max_loss": self.max_loss,
        }


def _gas_and_ops(inp: EconomicsInputs, volume: float) -> float:
    trades = volume / inp.avg_trade_size if inp.avg_trade_size > 0 else 0.0
    return inp.fixed_ops_cost + trades * inp.per_trade_cost


def compute_economics(inp: EconomicsInputs) -> EconomicsResult:
    """Compute the full economics breakdown for the given inputs."""

    v = max(0.0, inp.volume)

    # --- Revenue ---------------------------------------------------------- #
    creator_fee_revenue = inp.creator_fee * v
    lp_fee_revenue = inp.lp_fee * v * inp.liquidity_share
    total_fee_revenue = creator_fee_revenue + lp_fee_revenue

    # --- Mispricing loss -------------------------------------------------- #
    # We only lose on the net directional slice of volume that trades against
    # our book, and only to the extent our prices are wrong. Noise/two-sided
    # flow nets out and earns fees instead.
    expected_loss_from_mispricing = (
        inp.avg_pricing_error * v * inp.directional_fraction * inp.book_capture
    )

    # --- Adverse selection ------------------------------------------------ #
    # Informed flow extracts `informed_edge` on its share of the volume that
    # hits our book.
    adverse_selection_loss = (
        inp.informed_flow_fraction * inp.informed_edge * v * inp.book_capture
    )

    # --- Costs ------------------------------------------------------------ #
    gas_and_ops_cost = _gas_and_ops(inp, v)

    # --- Net -------------------------------------------------------------- #
    expected_net_profit = (
        total_fee_revenue
        - expected_loss_from_mispricing
        - adverse_selection_loss
        - gas_and_ops_cost
    )

    # --- Break-even volume ------------------------------------------------ #
    # Per-unit-of-volume margin:
    #   margin = creator_fee + lp_fee*share
    #            - avg_pricing_error*capture
    #            - informed_fraction*informed_edge*capture
    #            - per_trade_cost/avg_trade_size
    per_unit_cost = (
        inp.per_trade_cost / inp.avg_trade_size if inp.avg_trade_size > 0 else 0.0
    )
    margin_per_unit = (
        inp.creator_fee
        + inp.lp_fee * inp.liquidity_share
        - inp.avg_pricing_error * inp.directional_fraction * inp.book_capture
        - inp.informed_flow_fraction * inp.informed_edge * inp.book_capture
        - per_unit_cost
    )
    if margin_per_unit > 1e-9:
        break_even_volume = inp.fixed_ops_cost / margin_per_unit
    else:
        # No volume can overcome fixed costs -- the per-unit margin is <= 0.
        break_even_volume = None

    # --- Max loss --------------------------------------------------------- #
    # Worst case: we lose our liquidity at risk minus whatever fees we banked,
    # plus fixed ops cost. Capital at risk bounds the downside (LMSR/AMM books
    # have bounded loss; a bookmaker is bounded by max exposure).
    max_loss = inp.liquidity_at_risk + inp.fixed_ops_cost - total_fee_revenue
    max_loss = max(0.0, max_loss)

    return EconomicsResult(
        creator_fee_revenue=creator_fee_revenue,
        lp_fee_revenue=lp_fee_revenue,
        total_fee_revenue=total_fee_revenue,
        expected_loss_from_mispricing=expected_loss_from_mispricing,
        adverse_selection_loss=adverse_selection_loss,
        gas_and_ops_cost=gas_and_ops_cost,
        expected_net_profit=expected_net_profit,
        break_even_volume=break_even_volume,
        max_loss=max_loss,
    )
