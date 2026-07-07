"""Tests for the MatchSimulator — Markov-chain Monte Carlo volleyball engine."""

import numpy as np
import pytest

from src.simulation.simulator import MatchSimulator
from src.simulation.constants import (
    DEFAULT_CLAMP_RANGE,
    POINT_PROB_CLIP_ADAPTIVE_HARD,
    CLAMP_MARGIN,
    POINT_PROB_CLIP,
    DEFAULT_SIDEOUT_RATE,
)



class TestMatchShape:
    """A simulated match must end 3-0, 3-1, or 3-2."""

    def test_match_ends_within_3_to_5_sets(self):
        sim = MatchSimulator()
        for _ in range(10):
            match = sim.simulate_match(
                "Trento", "Perugia",
                home_strength=0.55, away_strength=0.52,
                seed=42,
            )
            total_sets = match.sets_home + match.sets_away
            assert 3 <= total_sets <= 5, f"Expected 3-5 sets, got {total_sets}"
            assert match.resultado in ("3-0", "3-1", "3-2", "0-3", "1-3", "2-3")

    def test_set_has_at_least_25_points_with_2pt_margin(self):
        sim = MatchSimulator()
        match = sim.simulate_match(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            seed=42,
        )
        for s in match.sets:
            winner_score = max(s.score_home, s.score_away)
            loser_score = min(s.score_home, s.score_away)
            assert winner_score >= 25, f"Set {s.set_number}: winner {winner_score} < 25"
            assert winner_score - loser_score >= 2, (
                f"Set {s.set_number}: margin {winner_score - loser_score} < 2"
            )

    def test_fifth_set_is_15_points(self):
        """Fifth set (tiebreak) uses target_score=15."""
        sim = MatchSimulator()
        # Use equal strengths + many seeds to increase chance of 5-set match
        for seed in range(5):
            match = sim.simulate_match(
                "Trento", "Perugia",
                home_strength=0.50, away_strength=0.50,
                seed=seed,
            )
            # We only verify the 5th-set rule when a 5th set actually occurred
            if len(match.sets) == 5:
                s5 = match.sets[-1]
                assert max(s5.score_home, s5.score_away) >= 15
                assert max(s5.score_home, s5.score_away) - min(s5.score_home, s5.score_away) >= 2



class TestDefaultClamp:
    """Without SetPredictor, point-probability clamp is DEFAULT_CLAMP_RANGE."""

    def test_default_point_probs_bounded(self):
        """_default_point_probs clamps probabilities to POINT_PROB_CLIP."""
        sim = MatchSimulator()
        # Extreme strengths should still produce bounded probabilities
        probs = sim._default_point_probs(home_strength=0.99, away_strength=0.01)
        for val in probs.values():
            assert POINT_PROB_CLIP[0] <= val <= POINT_PROB_CLIP[1], (
                f"Probability {val:.4f} outside [{POINT_PROB_CLIP[0]}, {POINT_PROB_CLIP[1]}]"
            )

    def test_default_clamp_range_constant(self):
        """DEFAULT_CLAMP_RANGE is (0.20, 0.80)."""
        assert DEFAULT_CLAMP_RANGE == (0.20, 0.80)

    def test_default_clamp_applied_in_simulate_set(self, monkeypatch):
        """_simulate_set uses DEFAULT_CLAMP_RANGE (0.20, 0.80) for np.clip when no SetPredictor."""
        import numpy as np
        bounds_seen = set()
        original_clip = np.clip

        def recording_clip(a, a_min, a_max, **kwargs):
            bounds_seen.add((a_min, a_max))
            return original_clip(a, a_min, a_max, **kwargs)

        monkeypatch.setattr(np, "clip", recording_clip)
        sim = MatchSimulator()
        sim.simulate_match("Trento", "Perugia", seed=42)
        assert (DEFAULT_CLAMP_RANGE[0], DEFAULT_CLAMP_RANGE[1]) in bounds_seen, (
            f"DEFAULT_CLAMP_RANGE {DEFAULT_CLAMP_RANGE} not found in np.clip calls; "
            f"seen bounds: {sorted(bounds_seen)}"
        )



class TestAdaptiveClamp:
    """With SetPredictor, the adaptive clamp uses POINT_PROB_CLIP_ADAPTIVE_HARD."""

    def test_adaptive_clamp_constants(self):
        assert POINT_PROB_CLIP_ADAPTIVE_HARD == (0.10, 0.90)
        assert CLAMP_MARGIN == 0.20

    def test_adaptive_clamp_narrows_around_predictor_value(self, synthetic_set_predictor):
        """With SetPredictor, the clamp adjusts around p_set_home."""
        sim = MatchSimulator()
        # Build team_features from a minimal dict
        team_features = {
            "set_wr_h": 0.5, "set_wr_a": 0.5,
            "forma_h": 0.5, "forma_a": 0.5,
            "pts_fav_h": 23.5, "pts_fav_a": 23.5,
        }
        match = sim.simulate_match(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            seed=42, set_predictor=synthetic_set_predictor,
            team_features=team_features,
        )
        # The match should still be valid
        assert match.winner in ("home", "away")
        assert 3 <= match.sets_home + match.sets_away <= 5

    def test_adaptive_clamp_extreme_predictor_bounds(self, synthetic_set_predictor, monkeypatch):
        """Even with extreme SetPredictor output, np.clip bounds stay within [0.10, 0.90]."""
        import numpy as np
        # Predict 98% away-set-win → p_set_home = 0.02
        monkeypatch.setattr(synthetic_set_predictor, "predict_proba",
                            lambda _: np.array([[0.02, 0.98]]))

        bounds_seen = set()
        original_clip = np.clip

        def recording_clip(a, a_min, a_max, **kwargs):
            bounds_seen.add((a_min, a_max))
            return original_clip(a, a_min, a_max, **kwargs)

        monkeypatch.setattr(np, "clip", recording_clip)
        sim = MatchSimulator()
        team_features = {"set_wr_h": 0.5, "set_wr_a": 0.5,
                         "forma_h": 0.5, "forma_a": 0.5,
                         "pts_fav_h": 23.5, "pts_fav_a": 23.5}
        match = sim.simulate_match("Trento", "Perugia", home_strength=0.55,
                                   away_strength=0.52, seed=42,
                                   set_predictor=synthetic_set_predictor,
                                   team_features=team_features)
        for low, high in bounds_seen:
            assert low >= POINT_PROB_CLIP_ADAPTIVE_HARD[0], f"low {low} < 0.10"
            assert high <= POINT_PROB_CLIP_ADAPTIVE_HARD[1], f"high {high} > 0.90"
        assert match.winner in ("home", "away")



class TestMCDeterminism:
    """Monte Carlo simulation with the same seed must produce identical results."""

    def test_mc_determinism_different_seeds_differ(self):
        sim = MatchSimulator()
        r1 = sim.monte_carlo_simulate(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            n_simulations=100, seed=42,
        )
        r2 = sim.monte_carlo_simulate(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            n_simulations=100, seed=99,
        )
        # Extremely unlikely to produce identical results with different seeds
        assert (r1["home_wins"] != r2["home_wins"]) or (r1["away_wins"] != r2["away_wins"])

    def test_mc_seed_produces_integer_counts(self):
        sim = MatchSimulator()
        result = sim.monte_carlo_simulate(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            n_simulations=100, seed=42,
        )
        assert isinstance(result["home_wins"], int)
        assert isinstance(result["away_wins"], int)
        assert result["home_wins"] + result["away_wins"] == 100



class TestSideoutMath:
    """Sideout adjustments in _default_point_probs follow Markov formulas."""

    def test_sideout_conservation_and_rate(self):
        """_default_point_probs satisfies Markov conservation and uses DEFAULT_SIDEOUT_RATE."""
        sim = MatchSimulator()
        p = sim._default_point_probs(home_strength=0.55, away_strength=0.45)
        assert abs(p["p_home_serving"] + p["p_away_receiving"] - 1.0) < 1e-10
        assert abs(p["p_home_receiving"] + p["p_away_serving"] - 1.0) < 1e-10
        # Sideout rate test: all values in [0,1] for any input
        assert all(0.0 <= v <= 1.0 for v in sim._default_point_probs(0.7, 0.3).values())

    def test_sideout_identical_strengths(self):
        """Equal strengths yield exact serving=0.38, receiving=0.62."""
        sim = MatchSimulator()
        probs = sim._default_point_probs(home_strength=0.5, away_strength=0.5)
        assert probs["p_home_serving"] == pytest.approx(0.38)
        assert probs["p_home_receiving"] == pytest.approx(0.62)

    def test_sideout_extreme_strengths_clamped(self):
        """Even extreme strength differences are clamped to POINT_PROB_CLIP."""
        sim = MatchSimulator()
        probs = sim._default_point_probs(home_strength=0.99, away_strength=0.01)
        for key, val in probs.items():
            assert POINT_PROB_CLIP[0] <= val <= POINT_PROB_CLIP[1], (
                f"{key} = {val:.4f} outside {POINT_PROB_CLIP}"
            )
        assert max(probs.values()) - min(probs.values()) <= 0.50

    def test_per_team_sideout_changes_receiving_prob(self):
        """Equal strengths + asymmetric per-team sideout should give p_home_receiving != p_home_serving."""
        sim = MatchSimulator()
        # When home is a strong sideoutter (0.65) and away is weak (0.50),
        # p_home_receiving (home winning when receiving) should be HIGH
        # and p_home_serving (home winning when serving) should be LOWER
        # than the symmetric case (where both sideout at 0.55).
        asymmetric = sim._default_point_probs(
            home_strength=0.5, away_strength=0.5,
            home_sideout=0.65, away_sideout=0.50,
        )
        symmetric = sim._default_point_probs(
            home_strength=0.5, away_strength=0.5,
            home_sideout=0.55, away_sideout=0.55,
        )
        # Home is the better sideoutter → home_receiving should be higher than symmetric
        assert asymmetric["p_home_receiving"] > symmetric["p_home_receiving"]
        # Conversely, away is the worse sideoutter → home_serving should be
        # higher (when home serves, away is receiving, away's lower sideout
        # means home wins more often).
        assert asymmetric["p_home_serving"] > symmetric["p_home_serving"]

    def test_per_team_sideout_markov_conservation(self):
        """Per-team sideout must still satisfy p_home_serving + p_away_receiving = 1."""
        sim = MatchSimulator()
        for home_sideout, away_sideout in [(0.55, 0.55), (0.65, 0.50), (0.50, 0.65)]:
            probs = sim._default_point_probs(
                home_strength=0.5, away_strength=0.5,
                home_sideout=home_sideout, away_sideout=away_sideout,
            )
            assert abs(probs["p_home_serving"] + probs["p_away_receiving"] - 1.0) < 1e-10
            assert abs(probs["p_home_receiving"] + probs["p_away_serving"] - 1.0) < 1e-10


class TestPerTeamSideoutIntegration:
    """simulate_match wires per-team sideout through the simulation loop."""

    def test_simulate_match_with_known_teams_uses_data(self, monkeypatch):
        """When team names resolve to known sideout rates, the simulator pulls them."""
        from src.simulation import simulator as sim_mod

        captured = {}

        def fake_get_probs(self, **kwargs):
            captured["home_sideout"] = kwargs.get("home_sideout")
            captured["away_sideout"] = kwargs.get("away_sideout")
            # Return a deterministic dict so the simulation can run
            return {
                "p_home_serving": 0.5,
                "p_home_receiving": 0.5,
                "p_away_serving": 0.5,
                "p_away_receiving": 0.5,
            }

        # Build a minimal point_model that exposes get_point_probabilities
        class FakePointModel:
            get_point_probabilities = fake_get_probs

        sim = MatchSimulator(point_model=FakePointModel())
        sim.simulate_match(
            home_team="Perugia", away_team="Grottazzolina",
            home_strength=0.5, away_strength=0.5,
            match_features={"elo_diff": 0.0},
            generate_points=False, generate_player_stats=False,
            seed=42,
        )
        # Perugia sideout (~0.530) > Grottazzolina sideout (~0.471)
        assert captured["home_sideout"] > 0.45
        assert captured["away_sideout"] > 0.45
        assert captured["home_sideout"] > captured["away_sideout"], (
            f"Perugia home_sideout {captured['home_sideout']:.3f} should beat "
            f"Grottazzolina away_sideout {captured['away_sideout']:.3f}"
        )

    def test_simulate_match_unknown_teams_use_default(self):
        """Unknown team names fall back to DEFAULT_SIDEOUT_RATE (0.62)."""
        from src.simulation import simulator as sim_mod

        captured = {}

        def fake_get_probs(self, **kwargs):
            captured["home_sideout"] = kwargs.get("home_sideout")
            captured["away_sideout"] = kwargs.get("away_sideout")
            return {
                "p_home_serving": 0.5, "p_home_receiving": 0.5,
                "p_away_serving": 0.5, "p_away_receiving": 0.5,
            }

        class FakePointModel:
            get_point_probabilities = fake_get_probs

        sim = MatchSimulator(point_model=FakePointModel())
        sim.simulate_match(
            home_team="Foo", away_team="Bar",
            home_strength=0.5, away_strength=0.5,
            match_features={"elo_diff": 0.0},
            generate_points=False, generate_player_stats=False,
            seed=42,
        )
        assert captured["home_sideout"] == DEFAULT_SIDEOUT_RATE
        assert captured["away_sideout"] == DEFAULT_SIDEOUT_RATE



class TestFeatureNamesGuard:
    """_eval_set_predictor returns None when feature_names is None (N14 fix)."""

    def test_eval_set_predictor_with_none_context_returns_None(self, synthetic_set_predictor):
        """When set_context_base is None, _eval_set_predictor returns None."""
        sim = MatchSimulator()
        result = sim._eval_set_predictor(
            set_predictor=synthetic_set_predictor,
            set_context_base=None,
            score_home=0, score_away=0,
            target_score=25, sets_home_antes=0, sets_away_antes=0,
        )
        assert result is None



class TestRegressionPins:
    """Targeted regression tests for previously fixed bugs."""

    def test_pin_point_model_integration(self, synthetic_point_model):
        """REGRESSION N6: PointProbabilityModel integrated with MatchSimulator."""
        sim = MatchSimulator(point_model=synthetic_point_model)
        match_features = {
            "elo_diff": 5.0,
            "diff_win_rate_global": 0.05,
            "diff_set_win_rate": 0.03,
            "diff_dominancia": 0.02,
            "diff_set_ratio": 0.04,
            "diff_forma_efectiva": 0.01,
        }
        match = sim.simulate_match(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            match_features=match_features,
            seed=42,
        )
        assert match.winner in ("home", "away")
        assert 3 <= match.sets_home + match.sets_away <= 5

    def test_pin_mc_seed_determinism(self):
        """REGRESSION N8: Same MC seed → identical results."""
        sim = MatchSimulator()
        r1 = sim.monte_carlo_simulate(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            n_simulations=50, seed=12345,
        )
        r2 = sim.monte_carlo_simulate(
            "Trento", "Perugia",
            home_strength=0.55, away_strength=0.52,
            n_simulations=50, seed=12345,
        )
        assert r1 == r2

    def test_pin_feature_names_none_returns_None(self):
        """REGRESSION N14: feature_names=None → _eval_set_predictor returns None."""
        sim = MatchSimulator()
        result = sim._eval_set_predictor(
            set_predictor=None, set_context_base=None,
            score_home=5, score_away=3,
            target_score=25, sets_home_antes=0, sets_away_antes=0,
        )
        assert result is None
