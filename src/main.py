"""Command-line interface for house-edge-lab.

Commands
--------
    validate  -- run the validator over a markets file and print verdicts
    simulate  -- Monte Carlo simulate each market, print risk/profit metrics
    report    -- run the full pipeline and write a markdown report

Examples
--------
    python -m src.main validate --data src/markets/sample_markets.json
    python -m src.main simulate --data src/markets/sample_markets.json --runs 1000
    python -m src.main report   --data src/markets/sample_markets.json

This tool is offline and never touches real money, markets, wallets or APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from src.data.collectors import COLLECTORS
from src.data.sources import print_sources
from src.data.store import DatasetStore
from src.models.candidate_market import CandidateMarket
from src.predictors.base import BaselinePredictor
from src.reporting.report import analyze_market, build_report, write_report
from src.scoring.market_scorer import MarketScorer
from src.simulation.monte_carlo import SimulationConfig, simulate_market
from src.telemetry.tracker import TelemetryTracker
from src.validation.validator import validate_market


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_markets(path: str) -> list[CandidateMarket]:
    if not os.path.exists(path):
        raise SystemExit(f"error: data file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("markets", [])
    if not isinstance(raw, list):
        raise SystemExit("error: data file must be a JSON list of markets")
    return [CandidateMarket.from_dict(item) for item in raw]


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_validate(args: argparse.Namespace) -> int:
    markets = load_markets(args.data)
    results = []
    print(f"Validating {len(markets)} market(s) from {args.data}\n")
    for m in markets:
        v = validate_market(m)
        flag = "ALLOWED" if v.allowed else "REJECTED"
        print(f"[{flag:8}] {m.id}  risk={v.risk_level.value:7}  {m.question[:60]}")
        for r in v.reasons:
            print(f"            - {r}")
        results.append({"id": m.id, **v.to_dict()})
        print()
    allowed = sum(1 for r in results if r["allowed"])
    print(f"Summary: {allowed}/{len(results)} allowed.")
    if args.json:
        _maybe_write_json(args.json, results)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    markets = load_markets(args.data)
    cfg = SimulationConfig(
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
    )
    out = []
    print(f"Simulating {len(markets)} market(s), {cfg.runs} runs each\n")
    for m in markets:
        v = validate_market(m)
        if not v.allowed:
            print(f"[SKIP   ] {m.id}  (blocked: {v.reasons[0]})")
            continue
        s = simulate_market(m, cfg)
        print(f"[{m.id}] {m.question[:55]}")
        print(
            f"    mean={s.mean_profit:,.2f}  median={s.median_profit:,.2f}  "
            f"worst5%={s.worst_5pct_profit:,.2f}  P(loss)={s.prob_loss:.0%}"
        )
        print(
            f"    exp_volume={s.expected_volume:,.0f}  fees={s.fee_revenue:,.2f}  "
            f"adverse_sel={s.adverse_selection_loss:,.2f}  "
            f"max_dd={s.max_drawdown:,.2f}"
        )
        print()
        out.append(s.to_dict())
    if args.json:
        _maybe_write_json(args.json, out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    markets = load_markets(args.data)
    cfg = SimulationConfig(runs=args.runs, steps=args.steps, seed=args.seed)
    run_sim = not args.no_sim
    analyses = [
        analyze_market(m, run_simulation=run_sim, sim_config=cfg) for m in markets
    ]
    report_md = build_report(analyses, sim_config=cfg)

    out_path = args.out
    if not out_path:
        os.makedirs("reports", exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("reports", f"report-{stamp}.md")
    write_report(report_md, out_path)
    print(f"Wrote report to {out_path}")
    if args.stdout:
        print("\n" + report_md)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    markets = load_markets(args.data)
    scorer = MarketScorer()
    results = [scorer.score(m) for m in markets]
    # Best LP markets first.
    order = {"create": 0, "risky": 1, "avoid": 2, "blocked": 3}
    results.sort(key=lambda r: (order.get(r.verdict, 9), -r.score))
    print(f"Scoring {len(markets)} market(s) for LP attractiveness "
          f"(edge = fees beating adverse selection)\n")
    for r in results:
        print(f"[{r.verdict:7}] score {r.score:3}/100  {r.market_id}")
        for reason in r.reasons:
            print(f"            - {reason}")
        if r.verdict != "blocked":
            print(f"            > recommend: creator {r.recommended_creator_fee:.1%}"
                  f" + LP {r.recommended_lp_fee:.1%}, "
                  f"max liquidity {r.recommended_max_liquidity:,.0f}")
        print()
    create = sum(1 for r in results if r.verdict == "create")
    print(f"Summary: {create} worth creating, "
          f"{sum(1 for r in results if r.verdict=='risky')} risky, "
          f"{sum(1 for r in results if r.verdict=='avoid')} avoid, "
          f"{sum(1 for r in results if r.verdict=='blocked')} blocked.")
    if args.json:
        _maybe_write_json(args.json, [r.to_dict() for r in results])
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    print_sources()
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    if args.source not in COLLECTORS:
        raise SystemExit(f"unknown source '{args.source}'. "
                         f"choices: {', '.join(COLLECTORS)}")
    collector = COLLECTORS[args.source]()
    print(f"Collecting up to {args.limit} RELIABLE resolved markets from "
          f"'{args.source}' (read-only, public API, quality-swept)...")
    kwargs = {"limit": args.limit}
    if args.source == "manifold":
        kwargs["with_history"] = args.history
        kwargs["min_volume"] = args.min_volume
    elif args.source == "polymarket":
        kwargs["with_history"] = args.history
    examples = collector.collect(**kwargs)

    if args.fresh and os.path.exists(args.out):
        os.remove(args.out)
        print(f"  (cleared old {args.out})")
    store = DatasetStore(args.out)
    n = store.extend(examples)

    stats = getattr(collector, "last_stats", {})
    if stats:
        print(f"\nReliability sweep: scanned {stats.get('scanned')}, "
              f"kept {stats.get('kept')}.")
        dropped = stats.get("dropped", {})
        if dropped:
            print("  weeded out:")
            for reason, count in sorted(dropped.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")
    print(f"\nWrote {n} reliable examples -> {args.out}")
    if examples:
        ex = examples[0]
        print(f"  sample: [{ex.platform}] {ex.question[:60]}  "
              f"-> {ex.final_outcome}  (src: {ex.provenance.get('url','')})")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    markets = load_markets(args.data)
    predictor = BaselinePredictor()
    tracker = TelemetryTracker(log_path=args.telemetry, model=predictor.name) \
        if args.telemetry else None
    print(f"Predicting {len(markets)} market(s) with '{predictor.name}'\n")
    for m in markets:
        v = validate_market(m)
        if not v.allowed:
            print(f"[SKIP   ] {m.id}  (blocked: {v.reasons[0]})")
            continue
        pred = predictor.predict(m)
        top, p = pred.top_outcome()
        print(f"[{m.id}] {m.question[:55]}")
        probs = ", ".join(
            f"{o}={pr:.0%}" for o, pr in zip(pred.outcomes, pred.probabilities)
        )
        print(f"    {probs}")
        print(f"    most likely: {top} ({p:.0%})  confidence={pred.confidence:.0%}")
        print()
        if tracker:
            tracker.record_prediction(pred)
    if tracker:
        print(f"Logged predictions to {args.telemetry}")
        print("Once events resolve, record outcomes to watch the scoreboard:")
        print("  TelemetryTracker(log_path=...).resolve(market_id, winner)")
    return 0


def _maybe_write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote JSON to {path}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="house-edge-lab",
        description=(
            "Prediction-market design simulation lab. Offline only — never "
            "places real bets or touches real-money systems."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data",
        required=True,
        help="path to a JSON file of candidate markets",
    )

    v = sub.add_parser("validate", parents=[common], help="validate markets")
    v.add_argument("--json", help="optional path to write JSON results")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("simulate", parents=[common], help="Monte Carlo simulate")
    s.add_argument("--runs", type=int, default=1000, help="runs per market")
    s.add_argument("--steps", type=int, default=50, help="trading steps per run")
    s.add_argument("--seed", type=int, default=42, help="RNG seed")
    s.add_argument("--json", help="optional path to write JSON results")
    s.set_defaults(func=cmd_simulate)

    r = sub.add_parser("report", parents=[common], help="write a markdown report")
    r.add_argument("--runs", type=int, default=500, help="runs per market")
    r.add_argument("--steps", type=int, default=50, help="trading steps per run")
    r.add_argument("--seed", type=int, default=42, help="RNG seed")
    r.add_argument("--out", help="output markdown path (default reports/report-*.md)")
    r.add_argument("--no-sim", action="store_true", help="skip Monte Carlo")
    r.add_argument("--stdout", action="store_true", help="also print report")
    r.set_defaults(func=cmd_report)

    pr = sub.add_parser("predict", parents=[common],
                        help="run a predictor over markets")
    pr.add_argument("--telemetry", help="JSONL log to record predictions into")
    pr.set_defaults(func=cmd_predict)

    sc = sub.add_parser("score", parents=[common],
                        help="score markets for LP attractiveness (fee edge)")
    sc.add_argument("--json", help="optional path to write JSON results")
    sc.set_defaults(func=cmd_score)

    so = sub.add_parser("sources", help="list recommended data sources")
    so.set_defaults(func=cmd_sources)

    co = sub.add_parser("collect",
                        help="collect resolved markets into a dataset (read-only)")
    co.add_argument("--source", required=True, choices=list(COLLECTORS),
                    help="which public market source to read")
    co.add_argument("--limit", type=int, default=50,
                    help="max resolved markets to collect")
    co.add_argument("--out", default="data/markets.jsonl",
                    help="dataset file to append to")
    co.add_argument("--history", action="store_true",
                    help="(manifold) also pull price history per market")
    co.add_argument("--min-volume", type=float, default=250.0,
                    dest="min_volume",
                    help="(manifold) skip markets below this volume (junk filter)")
    co.add_argument("--fresh", action="store_true",
                    help="overwrite the output file instead of appending")
    co.set_defaults(func=cmd_collect)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
