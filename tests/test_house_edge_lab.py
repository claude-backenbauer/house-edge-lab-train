"""Test suite for house-edge-lab (stdlib unittest, no external deps)."""

import math
import os
import tempfile
import unittest

from src.data.schema import PricePoint, TrainingExample
from src.data.sources import DATA_SOURCES, recommended_sources
from src.data.store import DatasetStore
from src.economics.model import EconomicsInputs, compute_economics
from src.predictors.base import BaselinePredictor, ExternalPredictor, Prediction
from src.telemetry.tracker import TelemetryTracker
from src.training.featurize import FEATURE_NAMES, build_xy, featurize, label_yes
from src.forecasting.baseline import BaselineForecaster, ForecastFeatures
from src.market_makers.bookmaker import BookmakerModel
from src.market_makers.lmsr import LMSRMarket
from src.models.candidate_market import CandidateMarket
from src.models.platform_profile import PolkamarketsProfile, get_platform_profile
from src.simulation.monte_carlo import SimulationConfig, simulate_market
from src.training.gpu_stub import describe_training_plan, train
from src.validation.validator import RiskLevel, validate_market


def _good_market(**over):
    base = dict(
        id="t1",
        question="Will BTC close above 100k on 2026-12-31 per CoinGecko?",
        description="objective",
        category="crypto",
        outcomes=["Yes", "No"],
        close_time="2026-12-31T23:00:00Z",
        event_time="2026-12-31T23:59:00Z",
        resolution_source="CoinGecko",
        platform="polkamarkets",
        creator_fee=0.01,
        lp_fee=0.02,
        initial_liquidity=5000,
        expected_volume=80000,
    )
    base.update(over)
    return CandidateMarket.from_dict(base)


class TestCandidateMarket(unittest.TestCase):
    def test_time_parsing_and_horizon(self):
        m = _good_market()
        self.assertTrue(m.is_binary)
        self.assertEqual(m.num_outcomes, 2)
        self.assertIsNotNone(m.close_time)
        self.assertLess(m.close_time, m.event_time)

    def test_roundtrip(self):
        m = _good_market()
        d = m.to_dict()
        m2 = CandidateMarket.from_dict(d)
        self.assertEqual(m2.id, m.id)
        self.assertEqual(m2.outcomes, m.outcomes)


class TestPlatformProfile(unittest.TestCase):
    def test_polkamarkets_defaults(self):
        p = PolkamarketsProfile()
        self.assertTrue(p.allows_market_creation)
        self.assertEqual(p.max_creator_fee, 0.05)
        self.assertEqual(p.max_outcomes, 32)
        self.assertTrue(p.requires_initial_liquidity)
        self.assertFalse(p.supports_api)  # no live API in v1

    def test_fee_limits(self):
        p = PolkamarketsProfile()
        self.assertEqual(p.fee_within_limits(0.01, 0.02), [])
        self.assertTrue(p.fee_within_limits(0.20, 0.02))

    def test_registry(self):
        self.assertEqual(get_platform_profile("polkamarkets").name, "polkamarkets")
        self.assertEqual(get_platform_profile("unknown").name, "generic")


class TestValidator(unittest.TestCase):
    def test_good_market_allowed(self):
        r = validate_market(_good_market())
        self.assertTrue(r.allowed)
        self.assertEqual(r.risk_level, RiskLevel.LOW)

    def test_hard_blocked_category_always_blocked(self):
        # Assassination is always blocked, even in permissive mode.
        r = validate_market(_good_market(category="assassination"))
        self.assertFalse(r.allowed)
        self.assertEqual(r.risk_level, RiskLevel.BLOCKED)

    def test_hard_blocked_keyword(self):
        r = validate_market(
            _good_market(category="politics",
                         question="Will there be an assassination of a leader?")
        )
        self.assertFalse(r.allowed)

    def test_restricted_permissive_vs_strict(self):
        # 'death' is allowed-but-high-risk by default (permissive)...
        r = validate_market(_good_market(category="death"))
        self.assertTrue(r.allowed)
        self.assertGreaterEqual(r.risk_level.rank, RiskLevel.HIGH.rank)
        # ...but blocked under strict policy.
        from src.validation.validator import ValidationPolicy
        r2 = validate_market(_good_market(category="death"),
                             policy=ValidationPolicy(mode="strict"))
        self.assertFalse(r2.allowed)

    def test_missing_resolution_source(self):
        r = validate_market(_good_market(resolution_source=""))
        self.assertFalse(r.allowed)
        self.assertTrue(any("resolution source" in x for x in r.reasons))

    def test_close_after_event(self):
        r = validate_market(
            _good_market(
                close_time="2026-12-31T23:59:00Z",
                event_time="2026-12-31T23:00:00Z",
            )
        )
        self.assertFalse(r.allowed)

    def test_too_few_outcomes(self):
        r = validate_market(_good_market(outcomes=["Yes"]))
        self.assertFalse(r.allowed)

    def test_too_many_outcomes(self):
        r = validate_market(_good_market(outcomes=[f"o{i}" for i in range(33)]))
        self.assertFalse(r.allowed)

    def test_extreme_fee_blocked(self):
        r = validate_market(_good_market(creator_fee=0.20))
        self.assertFalse(r.allowed)

    def test_ambiguous_wording(self):
        r = validate_market(
            _good_market(question="Will the project be popular soon enough?")
        )
        self.assertGreaterEqual(r.risk_level.rank, RiskLevel.HIGH.rank)

    def test_low_liquidity(self):
        r = validate_market(_good_market(initial_liquidity=50))
        self.assertGreaterEqual(r.risk_level.rank, RiskLevel.HIGH.rank)


class TestEconomics(unittest.TestCase):
    def test_revenue_breakdown(self):
        res = compute_economics(
            EconomicsInputs(volume=100000, creator_fee=0.01, lp_fee=0.02,
                            liquidity_share=1.0, liquidity_at_risk=5000)
        )
        self.assertAlmostEqual(res.creator_fee_revenue, 1000.0)
        self.assertAlmostEqual(res.lp_fee_revenue, 2000.0)
        self.assertAlmostEqual(res.total_fee_revenue, 3000.0)
        self.assertGreater(res.expected_net_profit, 0)

    def test_break_even_volume_positive(self):
        res = compute_economics(EconomicsInputs(volume=100000))
        self.assertIsNotNone(res.break_even_volume)
        self.assertGreater(res.break_even_volume, 0)

    def test_max_loss_nonnegative(self):
        res = compute_economics(EconomicsInputs(volume=10, liquidity_at_risk=5000))
        self.assertGreaterEqual(res.max_loss, 0)


class TestMarketMakers(unittest.TestCase):
    def test_lmsr_prices_sum_to_one(self):
        m = LMSRMarket(b=100, p_init=0.3)
        p = m.prices()
        self.assertAlmostEqual(p["yes"] + p["no"], 1.0, places=9)
        self.assertAlmostEqual(m.price_yes(), 0.3, places=6)

    def test_lmsr_bounded_loss(self):
        m = LMSRMarket(b=100)
        self.assertAlmostEqual(m.max_loss, 100 * math.log(2), places=6)

    def test_lmsr_buy_moves_price(self):
        m = LMSRMarket(b=100, p_init=0.5)
        before = m.price_yes()
        m.buy("yes", 50)
        self.assertGreater(m.price_yes(), before)

    def test_bookmaker_demand(self):
        bk = BookmakerModel(model_probability=0.6, spread=0.03, fee=0.02,
                            max_exposure=1000)
        summary = bk.simulate_demand(
            [("yes", 100), ("no", 50), ("yes", 80)], outcome_yes=True
        )
        self.assertGreater(summary["volume"], 0)
        self.assertGreaterEqual(summary["fees"], 0)
        self.assertIn("pnl", summary)


class TestSimulation(unittest.TestCase):
    def test_simulation_runs_and_is_deterministic(self):
        m = _good_market()
        cfg = SimulationConfig(runs=50, steps=10, seed=7)
        a = simulate_market(m, cfg)
        b = simulate_market(m, cfg)
        self.assertEqual(a.runs, 50)
        self.assertEqual(a.mean_profit, b.mean_profit)  # deterministic by seed
        self.assertTrue(0.0 <= a.prob_loss <= 1.0)
        self.assertGreaterEqual(a.expected_volume, 0.0)
        self.assertLessEqual(a.worst_5pct_profit, a.mean_profit + 1e-6)


class TestForecaster(unittest.TestCase):
    def test_probabilities_normalised(self):
        f = BaselineForecaster()
        r = f.forecast(ForecastFeatures(num_outcomes=3))
        self.assertAlmostEqual(sum(r.probabilities), 1.0, places=9)
        self.assertEqual(len(r.probabilities), 3)

    def test_uncertainty_increases_with_horizon(self):
        f = BaselineForecaster()
        short = f.forecast(ForecastFeatures(event_horizon_days=1))
        long = f.forecast(ForecastFeatures(event_horizon_days=720))
        self.assertGreater(long.uncertainty, short.uncertainty)
        self.assertIn("baseline", short.explanation.lower())


class TestTrainingStub(unittest.TestCase):
    def test_plan_described(self):
        plan = describe_training_plan()
        self.assertEqual(plan["status"], "not_implemented")
        self.assertIn("market_question_text", plan["dataset_features"])
        self.assertIn("calibrated_event_probability", plan["prediction_targets"])

    def test_train_raises(self):
        with self.assertRaises(NotImplementedError):
            train()


class TestPredictors(unittest.TestCase):
    def test_baseline_predictor(self):
        p = BaselinePredictor()
        pred = p.predict(_good_market())
        self.assertEqual(pred.source, "baseline")
        self.assertAlmostEqual(sum(pred.probabilities), 1.0, places=6)
        self.assertEqual(len(pred.probabilities), 2)

    def test_external_predictor_normalises(self):
        ext = ExternalPredictor("mirofish", lambda m: [3.0, 1.0])
        pred = ext.predict(_good_market())
        self.assertAlmostEqual(sum(pred.probabilities), 1.0, places=6)
        self.assertAlmostEqual(pred.probabilities[0], 0.75, places=6)


class TestTelemetry(unittest.TestCase):
    def _pred(self, mid, probs):
        return Prediction(market_id=mid, outcomes=["Yes", "No"],
                          probabilities=probs, confidence=max(probs),
                          source="m")

    def test_tracks_accuracy_and_brier(self):
        t = TelemetryTracker(model="m")
        t.record_prediction(self._pred("a", [0.9, 0.1]), stake=10)
        t.record_prediction(self._pred("b", [0.2, 0.8]), stake=10)
        t.resolve("a", "Yes", payout=11)   # correct
        t.resolve("b", "Yes", payout=0)    # wrong
        s = t.summary()
        self.assertEqual(s.n_resolved, 2)
        self.assertAlmostEqual(s.accuracy, 0.5)
        self.assertGreater(s.brier_score, 0)
        self.assertEqual(s.total_profit, 11 - 10 - 10)  # +1 -10 stake net

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "telem.jsonl")
            t = TelemetryTracker(log_path=path, model="m")
            t.record_prediction(self._pred("a", [0.7, 0.3]))
            t.resolve("a", "Yes")
            t2 = TelemetryTracker(log_path=path, model="m")
            self.assertEqual(t2.summary().n_resolved, 1)


class TestDataLayer(unittest.TestCase):
    def _example(self, won_yes=True):
        return TrainingExample(
            market_id="x1",
            question="Will team A win the final?",
            category="sports",
            outcomes=["Yes", "No"],
            close_time="2026-07-19T18:00:00Z",
            event_time="2026-07-19T22:00:00Z",
            resolution_source="FIFA",
            price_series=[PricePoint(t="2026-07-01T00:00:00Z", prices=[0.4, 0.6],
                                     volume=100)],
            final_volume=5000,
            liquidity=8000,
            final_outcome="Yes" if won_yes else "No",
            final_outcome_index=0 if won_yes else 1,
        )

    def test_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = DatasetStore(os.path.join(d, "ds.jsonl"))
            store.append(self._example(True))
            store.append(self._example(False))
            self.assertEqual(len(store), 2)
            self.assertEqual(len(store.labeled()), 2)

    def test_featurize_shape_and_label(self):
        ex = self._example(True)
        feats = featurize(ex)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertEqual(label_yes(ex), 1.0)
        self.assertEqual(label_yes(self._example(False)), 0.0)

    def test_build_xy(self):
        X, y = build_xy([self._example(True), self._example(False)])
        self.assertEqual(len(X), 2)
        self.assertEqual(len(X[0]), len(FEATURE_NAMES))
        self.assertEqual(y, [1.0, 0.0])

    def test_sources_registry(self):
        self.assertTrue(any(s.kind == "markets" for s in DATA_SOURCES))
        recs = recommended_sources()
        self.assertIn("world_cup_2026", recs)
        self.assertTrue(recs["train_the_model_first"])


class TestCollectorsMapping(unittest.TestCase):
    """Mapping logic only -- no network calls."""

    def test_manifold_mapping(self):
        from src.data.collectors import ManifoldCollector
        m = {
            "id": "abc", "outcomeType": "BINARY", "isResolved": True,
            "resolution": "YES", "probability": 0.82, "question": "Will X?",
            "textDescription": "desc", "volume": 1234,
            "closeTime": 1700000000000, "resolutionTime": 1700100000000,
            "pool": {"YES": 100, "NO": 50},
        }
        ex = ManifoldCollector()._to_example(m, with_history=False)
        self.assertIsNotNone(ex)
        self.assertEqual(ex.final_outcome, "YES")
        self.assertEqual(ex.final_outcome_index, 0)
        self.assertEqual(ex.platform, "manifold")
        self.assertTrue(ex.is_labeled())

    def test_manifold_skips_unresolved_and_nonbinary(self):
        from src.data.collectors import ManifoldCollector
        c = ManifoldCollector()
        self.assertIsNone(c._to_example(
            {"outcomeType": "BINARY", "isResolved": False}, False))
        self.assertIsNone(c._to_example(
            {"outcomeType": "MULTIPLE_CHOICE", "isResolved": True,
             "resolution": "YES"}, False))
        self.assertIsNone(c._to_example(
            {"outcomeType": "BINARY", "isResolved": True,
             "resolution": "CANCEL"}, False))

    def test_polymarket_mapping_infers_winner(self):
        from src.data.collectors import PolymarketCollector
        m = {
            "id": "p1", "question": "Will Real Madrid win the 2024 final?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1", "0"]',
            "volumeNum": 5000, "closed": True,
            "closedTime": "2024-01-01T00:00:00Z",
        }
        ex, reason = PolymarketCollector()._to_example(m)
        self.assertEqual(reason, "ok")
        self.assertEqual(ex.final_outcome, "Yes")
        self.assertEqual(ex.final_outcome_index, 0)
        self.assertEqual(ex.provenance["source"], "polymarket")

    def test_polymarket_skips_ambiguous(self):
        from src.data.collectors import PolymarketCollector
        m = {"id": "p2", "question": "Will the long enough question resolve yes?",
             "outcomes": '["Yes","No"]',
             "outcomePrices": '["0.5","0.5"]', "closed": True, "volumeNum": 5000}
        ex, reason = PolymarketCollector()._to_example(m)
        self.assertIsNone(ex)
        self.assertEqual(reason, "ambiguous-resolution")


class TestDataQuality(unittest.TestCase):
    def test_junk_questions_rejected(self):
        from src.data.quality import question_looks_real
        for junk in ["Hi", "Hello", "👋", "?", "test", "yo", "abc"]:
            self.assertFalse(question_looks_real(junk), junk)

    def test_real_questions_accepted(self):
        from src.data.quality import question_looks_real
        for ok in ["Will France win the 2026 World Cup final?",
                   "Will BTC close above $100k by year end?",
                   "Is the Fed going to cut rates in September 2026?"]:
            self.assertTrue(question_looks_real(ok), ok)

    def test_manifold_reliable_gate(self):
        from src.data.quality import manifold_reliable
        good = {"outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
                "question": "Will France win the 2026 World Cup final?",
                "uniqueBettorCount": 40, "volume": 5000}
        keep, reason = manifold_reliable(good)
        self.assertTrue(keep)
        self.assertEqual(reason, "ok")
        junk = {"outcomeType": "BINARY", "isResolved": True, "resolution": "NO",
                "question": "Hi", "uniqueBettorCount": 2, "volume": 300}
        keep2, reason2 = manifold_reliable(junk)
        self.assertFalse(keep2)
        self.assertEqual(reason2, "low-quality-question")

    def test_manifold_rejects_too_few_traders(self):
        from src.data.quality import manifold_reliable
        m = {"outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
             "question": "Will Russia and Ukraine sign a treaty by 2027?",
             "uniqueBettorCount": 3, "volume": 5000}
        keep, reason = manifold_reliable(m)
        self.assertFalse(keep)
        self.assertEqual(reason, "too-few-traders")

    def test_topic_filter_drops_meta_and_personal(self):
        from src.data.quality import is_forecastable_event
        self.assertFalse(is_forecastable_event(
            "Will 40 or more unique traders participate in this market?")[0])
        self.assertEqual(is_forecastable_event(
            "Will I get a girlfriend before 2027?")[1], "personal-market")
        # Dropped (off-topic personal trivia), regardless of exact reason.
        self.assertFalse(is_forecastable_event(
            "Will using paper towels clog my toilet?")[0])

    def test_topic_filter_keeps_real_events(self):
        from src.data.quality import is_forecastable_event
        for q in ["Will the Fed cut interest rates in September 2026?",
                  "Will France win the 2026 World Cup?",
                  "Will Bitcoin close above $100k by 2027?"]:
            self.assertTrue(is_forecastable_event(q)[0], q)

    def test_infer_topic(self):
        from src.data.quality import infer_topic
        self.assertEqual(infer_topic("Will the Fed cut interest rates?"),
                         "economics")
        self.assertEqual(infer_topic("Will Bitcoin hit $100k?"), "crypto")
        self.assertEqual(infer_topic("Will France win the World Cup?"), "sports")

    def test_manifold_serious_filter_integrated(self):
        from src.data.quality import manifold_reliable
        meta = {"outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
                "question": "Will this market reach 50 unique traders?",
                "uniqueBettorCount": 60, "volume": 5000}
        keep, reason = manifold_reliable(meta)
        self.assertFalse(keep)
        self.assertEqual(reason, "meta-market")


if __name__ == "__main__":
    unittest.main()
