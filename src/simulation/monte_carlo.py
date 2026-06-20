"""Monte Carlo simulator.

For a candidate market we run many independent trading runs. Each run:

  1. draws a "true" probability for Yes (the creator's model has some error
     relative to this);
  2. steps through time, letting a population of trader agents arrive and trade
     against an LMSR market maker;
  3. records the realised price path and volume;
  4. samples the final outcome from the true probability;
  5. settles the market maker and computes P&L.

Aggregating across runs gives a risk picture: mean / median profit, the worst
5% (CVaR-style tail), max drawdown, probability of loss, expected volume and a
decomposition of fee revenue vs adverse-selection loss.

Everything is offline simulation.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

from src.agents.traders import TraderAgent, TraderType, default_trader_population
from src.market_makers.lmsr import LMSRMarket
from src.models.candidate_market import CandidateMarket


@dataclass
class SimulationConfig:
    runs: int = 1000
    steps: int = 50  # trading steps per run (proxy for market lifetime)
    seed: int | None = 42
    # Creator's pricing error: std-dev of the gap between model and truth.
    creator_pricing_sigma: float = 0.05
    # LMSR liquidity parameter is derived from initial_liquidity unless set.
    liquidity_b: float | None = None
    population_scale: int = 1


@dataclass
class RunResult:
    pnl: float
    volume: float
    fees: float
    adverse_selection_loss: float
    max_drawdown: float
    final_price: float
    outcome_yes: bool


@dataclass
class SimulationResult:
    market_id: str
    runs: int
    mean_profit: float
    median_profit: float
    worst_5pct_profit: float  # mean of the worst 5% (expected shortfall)
    max_drawdown: float  # worst single-run drawdown observed
    prob_loss: float
    expected_volume: float
    fee_revenue: float  # mean fees across runs
    adverse_selection_loss: float  # mean across runs
    std_profit: float = 0.0
    best_profit: float = 0.0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "runs": self.runs,
            "mean_profit": self.mean_profit,
            "median_profit": self.median_profit,
            "worst_5pct_profit": self.worst_5pct_profit,
            "max_drawdown": self.max_drawdown,
            "prob_loss": self.prob_loss,
            "expected_volume": self.expected_volume,
            "fee_revenue": self.fee_revenue,
            "adverse_selection_loss": self.adverse_selection_loss,
            "std_profit": self.std_profit,
            "best_profit": self.best_profit,
            **self.extras,
        }


def _derive_b(market: CandidateMarket, cfg: SimulationConfig) -> float:
    if cfg.liquidity_b is not None:
        return max(1.0, cfg.liquidity_b)
    # Heuristic: deeper book for more seeded liquidity. b ~ liquidity / ln2-ish.
    return max(10.0, market.initial_liquidity / 4.0)


def _single_run(
    market: CandidateMarket,
    cfg: SimulationConfig,
    population: list[TraderAgent],
    rng: random.Random,
) -> RunResult:
    # True probability of Yes for this run.
    true_p = rng.random()
    # Creator's model price is the truth plus pricing error.
    model_p = min(0.99, max(0.01, true_p + rng.gauss(0, cfg.creator_pricing_sigma)))

    b = _derive_b(market, cfg)
    mm = LMSRMarket(b=b, fee=market.lp_fee + market.creator_fee, p_init=model_p)

    # Shares the maker seeded itself with (not sold to traders).
    seed_yes = mm.q_yes
    seed_no = mm.q_no

    def mark_to_market() -> float:
        """Maker P&L if the book were settled at current prices."""
        cash = (mm._cost() - mm._c0) + mm.collected_fees
        traded_yes = max(0.0, mm.q_yes - seed_yes)
        traded_no = max(0.0, mm.q_no - seed_no)
        owed = traded_yes * mm.price_yes() + traded_no * mm.price_no()
        return cash - owed

    price_history: list[float] = [mm.price_yes()]
    peak_value = mark_to_market()  # mark-to-market P&L peak, for drawdown
    max_dd = 0.0
    adverse_loss = 0.0

    for _ in range(cfg.steps):
        rng.shuffle(population)
        for trader in population:
            decision = trader.decide(
                true_p=true_p,
                market_p=mm.price_yes(),
                category=market.category,
                price_history=price_history,
                rng=rng,
            )
            if decision is None:
                continue
            side, notional = decision
            # Convert notional into a target-belief trade, capped by size.
            belief = trader.belief(true_p, mm.price_yes(), price_history, rng)
            max_shares = notional  # 1 unit notional ~ 1 share near mid
            before_price = mm.price_yes()
            rec = mm.trade_to_belief(belief, max_shares)

            # Track adverse selection: informed/arb traders moving price toward
            # truth are extracting value from the maker.
            if trader.trader_type in (TraderType.INFORMED, TraderType.ARBITRAGE):
                moved_toward_truth = (true_p - before_price) * (
                    mm.price_yes() - before_price
                )
                if moved_toward_truth > 0:
                    adverse_loss += abs(rec.get("cost", 0.0)) * 0.5

            price_history.append(mm.price_yes())
            mtm = mark_to_market()
            peak_value = max(peak_value, mtm)
            max_dd = max(max_dd, peak_value - mtm)

    outcome_yes = rng.random() < true_p
    pnl = mm.maker_pnl(outcome_yes)

    return RunResult(
        pnl=pnl,
        volume=mm.volume,
        fees=mm.collected_fees,
        adverse_selection_loss=adverse_loss,
        max_drawdown=max_dd,
        final_price=mm.price_yes(),
        outcome_yes=outcome_yes,
    )


def simulate_market(
    market: CandidateMarket,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Run the Monte Carlo simulation for a single candidate market."""

    cfg = config or SimulationConfig()
    rng = random.Random(cfg.seed)

    profits: list[float] = []
    volumes: list[float] = []
    fees: list[float] = []
    adverse: list[float] = []
    drawdowns: list[float] = []

    for _ in range(cfg.runs):
        # Fresh population each run (state-free agents, but reset bankroll lean).
        population = default_trader_population(rng, scale=cfg.population_scale)
        result = _single_run(market, cfg, population, rng)
        profits.append(result.pnl)
        volumes.append(result.volume)
        fees.append(result.fees)
        adverse.append(result.adverse_selection_loss)
        drawdowns.append(result.max_drawdown)

    profits_sorted = sorted(profits)
    tail_n = max(1, len(profits_sorted) // 20)  # worst 5%
    worst_5 = statistics.fmean(profits_sorted[:tail_n])
    prob_loss = sum(1 for p in profits if p < 0) / len(profits)

    return SimulationResult(
        market_id=market.id,
        runs=cfg.runs,
        mean_profit=statistics.fmean(profits),
        median_profit=statistics.median(profits),
        worst_5pct_profit=worst_5,
        max_drawdown=max(drawdowns) if drawdowns else 0.0,
        prob_loss=prob_loss,
        expected_volume=statistics.fmean(volumes),
        fee_revenue=statistics.fmean(fees),
        adverse_selection_loss=statistics.fmean(adverse),
        std_profit=statistics.pstdev(profits) if len(profits) > 1 else 0.0,
        best_profit=max(profits) if profits else 0.0,
    )
