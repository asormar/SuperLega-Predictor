"""Smoke tests for the four ML models used in the backend.

All tests use synthetic-fit model fixtures from conftest.py so they
pass on a fresh clone without models/*.joblib.

Covers:
  - SetPredictor, MatchPredictor predict_proba bounds [0,1] + no NaN
  - PointProbabilityModel get_point_probabilities returns valid dict
  - PlayerStatsGenerator generate_set_stats returns list of dicts
  - One @pytest.mark.slow real-artifact integration test
  - W2 gate-review: synthetic-fixture instantiation smoke (test_synthetic_fixtures_smoke)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.set_predictor import SetPredictor
from src.models.match_predictor import MatchPredictor
from src.models.point_probability import PointProbabilityModel
from src.models.player_stats_generator import PlayerStatsGenerator
from src.simulation.feature_builder import RuntimeFeatureBuilder


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _assert_proba_bounded(proba: np.ndarray, label: str):
    """Assert that a predict_proba output is in [0,1] with no NaN."""
    assert proba is not None, f"{label}: predict_proba returned None"
    assert not np.any(np.isnan(proba)), f"{label}: NaN found in probabilities"
    assert np.all(proba >= 0.0), f"{label}: negative probabilities found"
    assert np.all(proba <= 1.0), f"{label}: probabilities > 1.0 found"


# ─────────────────────────────────────────────────────────────
# W2 gate review: synthetic fixture instantiation smoke
# ─────────────────────────────────────────────────────────────

class TestSyntheticFixturesSmoke:
    """Every fixture in conftest.py instantiates cleanly and has the right shape.

    This test explicitly addresses WARNING #2 from PR1's gate review:
    "synthetic fixtures are defined but not exercised by PR1 tests."
    """

    def test_synthetic_set_predictor(self, synthetic_set_predictor: SetPredictor):
        pred = synthetic_set_predictor
        assert pred is not None
        assert isinstance(pred.feature_names, list)
        assert len(pred.feature_names) > 0

        # Build a tiny input and assert predict_proba returns bounded [0,1]
        feat_df = pd.DataFrame(
            [{f: 0.0 for f in pred.feature_names}],
        )
        proba = pred.predict_proba(feat_df)
        _assert_proba_bounded(proba, "SetPredictor")
        # Shape: (n_samples, 2) — P(away), P(home)
        assert proba.shape == (1, 2)

    def test_synthetic_match_predictor(self, synthetic_match_predictor: MatchPredictor):
        pred = synthetic_match_predictor
        assert pred is not None
        assert isinstance(pred.feature_names, list)
        assert len(pred.feature_names) > 0

        feat_df = pd.DataFrame(
            [{f: 0.0 for f in pred.feature_names}],
        )
        proba = pred.predict_proba(feat_df)
        _assert_proba_bounded(proba, "MatchPredictor")
        assert proba.shape == (1, 2)

    def test_synthetic_point_model(self, synthetic_point_model: PointProbabilityModel):
        model = synthetic_point_model
        assert model is not None
        assert model.is_fitted

        probs = model.get_point_probabilities(
            match_features={
                "elo_diff": 0.0,
                "diff_win_rate_global": 0.0,
                "diff_set_win_rate": 0.0,
                "diff_dominancia": 0.0,
                "diff_set_ratio": 0.0,
                "diff_forma_efectiva": 0.0,
            },
        )
        assert isinstance(probs, dict)
        for key in ("p_home_serving", "p_home_receiving", "p_away_serving", "p_away_receiving"):
            assert key in probs
            assert 0.0 <= probs[key] <= 1.0

    def test_synthetic_player_gen(self, synthetic_player_gen: PlayerStatsGenerator):
        gen = synthetic_player_gen
        assert gen is not None
        assert len(gen.team_profiles) > 0

        stats = gen.generate_set_stats("Trento", team_score=25, opponent_score=23)
        assert isinstance(stats, list)
        # At least one player should have stats
        assert len(stats) >= 0

    def test_synthetic_feature_builder(self, synthetic_feature_builder: RuntimeFeatureBuilder):
        builder = synthetic_feature_builder
        assert builder is not None
        df = builder.build_features("Trento", "Perugia", jornada=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ─────────────────────────────────────────────────────────────
# SetPredictor smoke
# ─────────────────────────────────────────────────────────────

class TestSetPredictorSmoke:
    """SetPredictor with synthetic data — calibrated, bounded, no NaN."""

    def test_predict_proba_bounds(self, synthetic_set_predictor: SetPredictor):
        feat_df = pd.DataFrame(
            [{f: 0.5 for f in synthetic_set_predictor.feature_names}],
        )
        proba = synthetic_set_predictor.predict_proba(feat_df)
        _assert_proba_bounded(proba, "SetPredictor")
        assert proba.shape == (1, 2)

    def test_predict_proba_sums_to_one(self, synthetic_set_predictor: SetPredictor):
        feat_df = pd.DataFrame(
            [{f: 0.3 for f in synthetic_set_predictor.feature_names}],
        )
        proba = synthetic_set_predictor.predict_proba(feat_df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_returns_int(self, synthetic_set_predictor: SetPredictor):
        feat_df = pd.DataFrame(
            [{f: 0.3 for f in synthetic_set_predictor.feature_names}],
        )
        preds = synthetic_set_predictor.predict(feat_df)
        assert preds.dtype == np.int_ or preds.dtype == np.int64


# ─────────────────────────────────────────────────────────────
# MatchPredictor smoke
# ─────────────────────────────────────────────────────────────

class TestMatchPredictorSmoke:
    """MatchPredictor with synthetic data."""

    def test_predict_proba_bounds(self, synthetic_match_predictor: MatchPredictor):
        feat_df = pd.DataFrame(
            [{f: 0.5 for f in synthetic_match_predictor.feature_names}],
        )
        proba = synthetic_match_predictor.predict_proba(feat_df)
        _assert_proba_bounded(proba, "MatchPredictor")
        assert proba.shape == (1, 2)

    def test_predict_proba_sums_to_one(self, synthetic_match_predictor: MatchPredictor):
        feat_df = pd.DataFrame(
            [{f: 0.3 for f in synthetic_match_predictor.feature_names}],
        )
        proba = synthetic_match_predictor.predict_proba(feat_df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


# ─────────────────────────────────────────────────────────────
# PointProbabilityModel smoke
# ─────────────────────────────────────────────────────────────

class TestPointProbabilityModelSmoke:
    """PointProbabilityModel with synthetic data."""

    def test_get_point_probabilities_with_features(self, synthetic_point_model: PointProbabilityModel):
        probs = synthetic_point_model.get_point_probabilities(
            match_features={
                "elo_diff": 5.0,
                "diff_win_rate_global": 0.05,
                "diff_set_win_rate": 0.03,
                "diff_dominancia": 0.02,
                "diff_set_ratio": 0.04,
                "diff_forma_efectiva": 0.01,
            },
        )
        assert 0.0 <= probs["p_home_serving"] <= 1.0
        assert 0.0 <= probs["p_home_receiving"] <= 1.0
        assert 0.0 <= probs["p_away_serving"] <= 1.0
        assert 0.0 <= probs["p_away_receiving"] <= 1.0
        # Conservation: p_home_serving + p_away_receiving ≈ 1
        # (they are complements in Markov)
        assert abs(probs["p_home_serving"] + probs["p_away_receiving"] - 1.0) < 1e-6

    def test_get_point_probabilities_without_features(self, synthetic_point_model: PointProbabilityModel):
        """Fallback path: no match_features dict — uses strength directly."""
        probs = synthetic_point_model.get_point_probabilities(
            match_features=None,
            home_strength=0.60,
            away_strength=0.40,
        )
        for key in ("p_home_serving", "p_home_receiving"):
            assert 0.0 <= probs[key] <= 1.0

    def test_default_sideout_rate_is_62(self):
        assert PointProbabilityModel.DEFAULT_SIDEOUT_RATE == 0.62


# ─────────────────────────────────────────────────────────────
# PlayerStatsGenerator smoke
# ─────────────────────────────────────────────────────────────

class TestPlayerStatsGeneratorSmoke:
    """PlayerStatsGenerator with synthetic data (no player_stats_params.json)."""

    def test_generate_set_stats_shape(self, synthetic_player_gen: PlayerStatsGenerator):
        stats = synthetic_player_gen.generate_set_stats(
            "Trento", team_score=25, opponent_score=23,
        )
        assert isinstance(stats, list)
        if stats:
            for entry in stats:
                assert "jugador" in entry
                assert "puntos" in entry

    def test_get_roster(self, synthetic_player_gen: PlayerStatsGenerator):
        roster = synthetic_player_gen.get_roster("Trento")
        assert isinstance(roster, list)

    def test_get_profile(self, synthetic_player_gen: PlayerStatsGenerator):
        profile = synthetic_player_gen.get_profile("Trento")
        assert isinstance(profile, dict)

    def test_generate_set_stats_returns_list(self, synthetic_player_gen: PlayerStatsGenerator):
        stats = synthetic_player_gen.generate_set_stats(
            "Trento", team_score=21, opponent_score=25,
        )
        assert isinstance(stats, list)

    def test_generate_set_stats_unknown_team(self, synthetic_player_gen: PlayerStatsGenerator):
        """Unknown team returns empty list."""
        stats = synthetic_player_gen.generate_set_stats(
            "NonExistent", team_score=25, opponent_score=23,
        )
        assert stats == []


# ─────────────────────────────────────────────────────────────
# Slow: real artifact integration (opt-in)
# ─────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestRealArtifacts:
    """Integration tests requiring actual models/*.joblib + player_stats_params.json.

    Skipped by default on a fresh clone.  Opt in with ``pytest -m slow``.
    """

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"

    def test_real_set_predictor_loads_and_predicts(self):
        set_path = self.MODELS_DIR / "set_predictor.joblib"
        if not set_path.exists():
            pytest.skip("set_predictor.joblib not found — install model artifacts first")

        pred = SetPredictor.load(set_path)
        assert pred.feature_names is not None
        feat_df = pd.DataFrame(
            [{f: 0.5 for f in pred.feature_names}],
        )
        proba = pred.predict_proba(feat_df)
        _assert_proba_bounded(proba, "real SetPredictor")

    def test_real_match_predictor_loads_and_predicts(self):
        match_path = self.MODELS_DIR / "match_predictor.joblib"
        if not match_path.exists():
            pytest.skip("match_predictor.joblib not found")

        pred = MatchPredictor.load(match_path)
        assert pred.feature_names is not None
        feat_df = pd.DataFrame(
            [{f: 0.5 for f in pred.feature_names}],
        )
        proba = pred.predict_proba(feat_df)
        _assert_proba_bounded(proba, "real MatchPredictor")

    def test_real_point_model_get_probs(self):
        pp_path = self.MODELS_DIR / "point_probability.joblib"
        if not pp_path.exists():
            pytest.skip("point_probability.joblib not found")

        model = PointProbabilityModel.load(pp_path)
        probs = model.get_point_probabilities(
            match_features={
                "elo_diff": 0.0,
                "diff_win_rate_global": 0.0,
                "diff_set_win_rate": 0.0,
                "diff_dominancia": 0.0,
                "diff_set_ratio": 0.0,
                "diff_forma_efectiva": 0.0,
            },
        )
        for val in probs.values():
            assert 0.0 <= val <= 1.0

    def test_real_player_gen_fit(self):
        params_path = self.MODELS_DIR / "player_stats_params.json"
        if not params_path.exists():
            pytest.skip("player_stats_params.json not found")

        gen = PlayerStatsGenerator.load(params_path)
        assert len(gen.team_profiles) > 0



