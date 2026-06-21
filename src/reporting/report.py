"""Per-run markdown reporting.

Ties together validation, the closed-form economics model and the Monte Carlo
simulator into a single per-market analysis, then renders a markdown report
covering top candidates, rejected markets, expected revenue/profit, worst-case
loss, probability of loss, recommended max liquidity, the allowed/risky/blocked
verdict, and the assumptions used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.economics.model import EconomicsInputs, EconomicsResult, compute_economics
from src.models.candidate_market import CandidateMarket
from src.models.platform_profile import get_platform_profile
from src.scoring.market_scorer import MarketScorer, ScoreResult
from src.simulation.monte_carlo import (
    SimulationConfig,
    SimulationResult,
    simulate_market,
)
from src.validation.validator import RiskLevel, ValidationResult, validate_market


@dataclass
class MarketAnalysis:
    market: CandidateMarket
    validation: ValidationResult
    economics: EconomicsResult | None = None
    simulation: SimulationResult | None = None
    lp_score: ScoreResult | None = None
    recommended_max_liquidity: float = 0.0
    verdict: str = "blocked"  # create / risky / avoid / blocked
    extras: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Ranking score (higher is better): the LP attractiveness score."""
        if self.lp_score is not None:
            return float(self.lp_score.score)
        if self.simulation is None or not self.validation.allowed:
            return float("-inf")
        s = self.simulation
        return s.mean_profit - 0.5 * abs(s.worst_5pct_profit) - 100.0 * s.prob_loss


def _recommended_max_liquidity(
    market: CandidateMarket, sim: SimulationResult | None
) -> float:
    """A conservative cap on capital to commit.

    Caps liquidity so that the simulated worst-5% loss stays within a tolerable
    fraction of capital. Falls back to the seeded liquidity if no sim.
    """
    base = market.initial_liquidity or 0.0
    if sim is None:
        return base
    tail = abs(min(0.0, sim.worst_5pct_profit))
    if tail <= 0:
        # No meaningful tail loss -> allow scaling up modestly.
        return round(max(base, base * 1.5 + sim.mean_profit), 2)
    # Size so the tail loss is ~20% of committed capital.
    cap = tail / 0.20
    return round(min(max(base, 0.0) or cap, cap), 2)


def analyze_market(
    market: CandidateMarket,
    *,
    run_simulation: bool = True,
    sim_config: SimulationConfig | None = None,
) -> MarketAnalysis:
    """Run the full analysis pipeline for a single market."""

    validation = validate_market(market)

    economics: EconomicsResult | None = None
    simulation: SimulationResult | None = None

    if validation.allowed:
        # Closed-form economics using the market's own parameters.
        econ_inputs = EconomicsInputs(
            volume=market.expected_volume,
            creator_fee=market.creator_fee,
            lp_fee=market.lp_fee,
            liquidity_share=1.0,
            liquidity_at_risk=market.initial_liquidity,
        )
        economics = compute_economics(econ_inputs)

        if run_simulation:
            simulation = simulate_market(market, sim_config)

    # The market-selection scorer gives the decision-relevant verdict
    # (create/risky/avoid) and recommended fees/liquidity from the fees-vs-
    # adverse-selection economics, grounded in real-money data.
    lp_score = MarketScorer().score(market)
    verdict = lp_score.verdict
    rec_liq = (lp_score.recommended_max_liquidity if validation.allowed
               else _recommended_max_liquidity(market, simulation))

    return MarketAnalysis(
        market=market,
        validation=validation,
        economics=economics,
        simulation=simulation,
        lp_score=lp_score,
        recommended_max_liquidity=rec_liq,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt(x: float | None, money: bool = False) -> str:
    if x is None:
        return "n/a"
    if money:
        return f"{x:,.2f}"
    return f"{x:,.2f}"


def _assumptions_block(sim_config: SimulationConfig | None) -> str:
    cfg = sim_config or SimulationConfig()
    lines = [
        "## Assumptions",
        "",
        "- **Not real money.** All values are abstract units; no bets, markets, "
        "wallets, or APIs are touched.",
        "- Fees are fractions of traded volume; revenue is fee-on-volume.",
        "- Market maker in simulation: binary **LMSR** with bounded loss "
        "`b * ln 2`; `b` derived from seeded liquidity unless overridden.",
        f"- Monte Carlo: **{cfg.runs} runs**, {cfg.steps} steps each, "
        f"seed={cfg.seed}.",
        f"- Creator pricing error sigma: {cfg.creator_pricing_sigma:.3f} "
        "(model price vs. true probability).",
        "- 'True' outcome probability drawn uniformly per run; outcome sampled "
        "from it.",
        "- Adverse selection attributed to informed/arbitrage flow that moves "
        "price toward truth.",
        "- Recommended max liquidity sizes capital so the simulated worst-5% "
        "loss is ~20% of committed capital.",
        "- Forecasting baseline is conservative and uses **no external LLM/API**.",
        "- GPU training is a documented stub; **no model is trained** in v1.",
    ]
    return "\n".join(lines)


def _analysis_row(a: MarketAnalysis) -> str:
    lp = a.lp_score
    s = a.simulation
    score = f"{lp.score}" if lp else "n/a"
    conf = lp.confidence if lp else "n/a"
    rec_fees = f"{lp.recommended_creator_fee + lp.recommended_lp_fee:.1%}" if lp else "n/a"
    ploss = f"{s.prob_loss:.0%}" if s else "n/a"
    return (
        f"| {a.market.id} | **{a.verdict}** | {score} | {conf} | {rec_fees} | "
        f"{ploss} | {_fmt(a.recommended_max_liquidity)} |"
    )


def build_report(
    analyses: list[MarketAnalysis],
    *,
    sim_config: SimulationConfig | None = None,
    title: str = "house-edge-lab run report",
) -> str:
    """Render a full markdown report for a list of analyses."""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    allowed = [a for a in analyses if a.validation.allowed]
    rejected = [a for a in analyses if not a.validation.allowed]

    ranked = sorted(allowed, key=lambda a: a.score, reverse=True)

    out: list[str] = []
    out.append(f"# {title}")
    out.append("")
    out.append(f"_Generated {now}. Simulation only — no real-money activity._")
    out.append("")
    out.append(
        f"**{len(analyses)} candidate market(s)** analysed — "
        f"{len(allowed)} allowed, {len(rejected)} rejected."
    )
    out.append("")

    # --- Top candidates --------------------------------------------------- #
    out.append("## Top candidate markets")
    out.append("")
    if ranked:
        out.append(
            "| id | verdict | LP score | confidence | rec. fees | "
            "P(loss) | rec. max liq |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for a in ranked:
            out.append(_analysis_row(a))
    else:
        out.append("_No markets passed validation._")
    out.append("")

    # --- Per-market detail ------------------------------------------------ #
    out.append("## Candidate detail")
    out.append("")
    for a in ranked:
        m = a.market
        out.append(f"### {m.id} — {m.question}")
        out.append("")
        out.append(f"- Platform: `{m.platform}` | Category: `{m.category}` | "
                   f"Outcomes: {m.num_outcomes}")
        out.append(f"- Verdict: **{a.verdict}** "
                   f"(risk: {a.validation.risk_level.value})")
        out.append(f"- Creator fee: {m.creator_fee:.2%} | LP fee: {m.lp_fee:.2%} | "
                   f"Seeded liquidity: {_fmt(m.initial_liquidity)}")
        if a.lp_score:
            lp = a.lp_score
            out.append(
                f"- **LP decision:** {lp.verdict} (score {lp.score}/100, "
                f"{lp.confidence} confidence) — adverse-selection "
                f"{lp.adverse_selection_rate:.1%} of volume vs fees "
                f"{lp.fee_rate:.1%} → net {lp.net_margin_rate:+.1%}"
            )
            out.append(
                f"- **Recommended fees:** creator {lp.recommended_creator_fee:.1%}"
                f" + LP {lp.recommended_lp_fee:.1%} "
                f"(net {lp.net_margin_at_recommended:+.1%} at those fees)"
            )
        if a.economics:
            e = a.economics
            be = _fmt(e.break_even_volume) if e.break_even_volume else "unreachable"
            out.append(
                f"- Economics (closed-form): fee revenue {_fmt(e.total_fee_revenue)}, "
                f"adverse-selection {_fmt(e.adverse_selection_loss)}, "
                f"net {_fmt(e.expected_net_profit)}, break-even volume {be}, "
                f"max loss {_fmt(e.max_loss)}"
            )
        if a.simulation:
            s = a.simulation
            out.append(
                f"- Simulation ({s.runs} runs): mean {_fmt(s.mean_profit)}, "
                f"median {_fmt(s.median_profit)}, worst-5% {_fmt(s.worst_5pct_profit)}, "
                f"max drawdown {_fmt(s.max_drawdown)}, P(loss) {s.prob_loss:.0%}, "
                f"exp. volume {_fmt(s.expected_volume)}, "
                f"fee revenue {_fmt(s.fee_revenue)}, "
                f"adverse-selection {_fmt(s.adverse_selection_loss)}"
            )
        out.append(f"- **Recommended max liquidity: {_fmt(a.recommended_max_liquidity)}**")
        if a.validation.reasons:
            out.append(f"- Notes: {'; '.join(a.validation.reasons)}")
        out.append("")

    # --- Rejected --------------------------------------------------------- #
    out.append("## Rejected markets")
    out.append("")
    if rejected:
        out.append("| id | risk | reasons |")
        out.append("|---|---|---|")
        for a in rejected:
            reasons = "; ".join(a.validation.reasons)
            out.append(
                f"| {a.market.id} | {a.validation.risk_level.value} | {reasons} |"
            )
    else:
        out.append("_No markets rejected._")
    out.append("")

    # --- Portfolio summary ------------------------------------------------ #
    if allowed:
        total_fee = sum(
            a.economics.total_fee_revenue for a in allowed if a.economics
        )
        total_net_sim = sum(
            a.simulation.mean_profit for a in allowed if a.simulation
        )
        worst = min(
            (a.simulation.worst_5pct_profit for a in allowed if a.simulation),
            default=0.0,
        )
        out.append("## Portfolio summary (allowed markets)")
        out.append("")
        out.append(f"- Expected fee revenue (closed-form): {_fmt(total_fee)}")
        out.append(f"- Expected net profit (mean of sims): {_fmt(total_net_sim)}")
        out.append(f"- Worst single-market worst-5% loss: {_fmt(worst)}")
        out.append("")

    out.append(_assumptions_block(sim_config))
    out.append("")
    return "\n".join(out)


def write_report(report_md: str, path: str) -> str:
    """Write the markdown report to `path` and return the path."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    return path
