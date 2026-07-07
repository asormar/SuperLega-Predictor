"""Regression tests for feature builder and point-probability feature helpers.

Covers win_rate asymmetry, pts_fav_exp formula, ASSUMED_REST_DAYS=7,
build_features schema, and elo_diff * 200 scaling.

3 regression pins:
  - feature-builder bugs (build_features schema)
  - elo_diff x200 (build_features_from_strengths)
  - thread-safety (RuntimeFeatureBuilder._lock)
"""

import threading

import numpy as np
import pandas as pd
import pytest

from src.models.point_probability import build_features_from_strengths
from src.simulation.feature_builder import (
    RuntimeFeatureBuilder,
    ASSUMED_REST_DAYS,
    ELO_BASE,
)

from src.data.feature_store import MATCH_FEATURE_COLS

# Collect all columns that _load_static_profiles iterates over — these MUST be
# present in any CSV passed to RuntimeFeatureBuilder.
_ALL_H_A_COLS = [c for c in MATCH_FEATURE_COLS if c.startswith("h_") or c.startswith("a_")]


def _make_minimal_csv(tmp_path, extra_cols=None):
    """Write a minimal CSV with all required MATCH_FEATURE_COLS h_/a_ columns."""
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    try:
        # Build header row
        base_cols = ["local", "visitante", "gana_local", "temporada"]
        req_cols = list(_ALL_H_A_COLS)
        if extra_cols:
            req_cols = list(set(req_cols + extra_cols))
        header = base_cols + req_cols
        tmp.write(",".join(header) + "\n")

        # Data row
        vals = ["Trento", "Perugia", "1", "2024/2025"] + ["0.5"] * len(req_cols)
        tmp.write(",".join(vals) + "\n")
        tmp.close()
        return tmp.name
    except:
        os.unlink(tmp.name)
        raise


# ─────────────────────────────────────────────────────────────
# build_features_from_strengths — elo_diff scaling (R9)
# ─────────────────────────────────────────────────────────────

class TestBuildFeaturesFromStrengths:
    """build_features_from_strengths constructs the 6-feature dict for PointProbabilityModel."""

    def test_elo_diff_scaled_by_200(self):
        """elo_diff = diff * 200 while other diff_* features are unscaled."""
        feats = build_features_from_strengths(0.55, 0.45)
        diff = 0.55 - 0.45  # 0.10
        assert feats["elo_diff"] == pytest.approx(diff * 200)
        assert feats["diff_win_rate_global"] == pytest.approx(diff)
        assert feats["diff_set_win_rate"] == pytest.approx(diff)
        assert feats["diff_dominancia"] == pytest.approx(diff)
        assert feats["diff_set_ratio"] == pytest.approx(diff)
        assert feats["diff_forma_efectiva"] == pytest.approx(diff)

    def test_pin_elo_diff_x200(self):
        """REGRESSION #7: elo_diff carries the *200 scaling."""
        feats = build_features_from_strengths(0.60, 0.40)
        assert feats["elo_diff"] == pytest.approx(0.20 * 200)

    def test_negative_diff_negative_elo_diff(self):
        """When away is stronger, elo_diff is negative (still *200)."""
        feats = build_features_from_strengths(0.40, 0.60)
        assert feats["elo_diff"] == pytest.approx(-0.20 * 200)

    def test_equal_strengths_zero_diff(self):
        """Equal strengths produce all-diff-zero features."""
        feats = build_features_from_strengths(0.50, 0.50)
        assert feats["elo_diff"] == pytest.approx(0.0)
        for key, val in feats.items():
            assert val == pytest.approx(0.0), f"{key} should be 0 for equal strengths"


# ─────────────────────────────────────────────────────────────
# ASSUMED_REST_DAYS
# ─────────────────────────────────────────────────────────────

class TestAssumedRestDays:
    """ASSUMED_REST_DAYS is pinned at 7 in the feature builder module."""

    def test_assumed_rest_days_constant_is_7(self):
        assert ASSUMED_REST_DAYS == 7

    def test_pin_assumed_rest_days_7(self):
        """REGRESSION #10: ASSUMED_REST_DAYS must always be 7."""
        from src.simulation.constants import AVG_POINTS_PER_SET
        assert ASSUMED_REST_DAYS == 7
        assert AVG_POINTS_PER_SET == 23.5


# ─────────────────────────────────────────────────────────────
# RuntimeFeatureBuilder — schema & thread-safety
# ─────────────────────────────────────────────────────────────

class TestRuntimeFeatureBuilder:
    """RuntimeFeatureBuilder construction, schema, and lock presence."""

    def test_builder_has_lock(self, tmp_path):
        """RuntimeFeatureBuilder must have a threading.Lock for thread-safety."""
        csv_path = _make_minimal_csv(tmp_path)
        try:
            builder = RuntimeFeatureBuilder(csv_path=csv_path)
            assert hasattr(builder, "_lock")
            # threading.Lock is a factory function, not a type — use type() to
            # create a reference instance for isinstance.
            lock_type = type(threading.Lock())
            assert isinstance(builder._lock, lock_type)
        finally:
            import os; os.unlink(csv_path)

    def test_pin_runtime_lock_present(self, tmp_path):
        """REGRESSION #8: RuntimeFeatureBuilder has a threading.Lock."""
        csv_path = _make_minimal_csv(tmp_path)
        try:
            builder = RuntimeFeatureBuilder(csv_path=csv_path)
            lock_type = type(threading.Lock())
            assert isinstance(builder._lock, lock_type)
        finally:
            import os; os.unlink(csv_path)

    def test_build_features_returns_dataframe(self, synthetic_feature_builder):
        """build_features returns a non-empty DataFrame with expected columns."""
        df = synthetic_feature_builder.build_features("Trento", "Perugia", jornada=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # single row
        assert "elo_diff" in df.columns
        assert "h_win_rate_global" in df.columns
        assert "a_win_rate_global" in df.columns

    def test_pin_build_features_schema(self, tmp_path):
        """REGRESSION #5: build_features returns stable schema."""
        csv_path = _make_minimal_csv(tmp_path)
        try:
            builder = RuntimeFeatureBuilder(csv_path=csv_path)
            df = builder.build_features("Trento", "Perugia", jornada=1)
            assert len(df) == 1
            assert df.iloc[0]["elo_h"] == ELO_BASE
            assert df.iloc[0]["elo_a"] == ELO_BASE
            assert df.iloc[0]["diff_win_rate_global"] == 0.0
            assert df.iloc[0]["elo_diff"] == 0.0
        finally:
            import os; os.unlink(csv_path)


# ─────────────────────────────────────────────────────────────
# Win rate asymmetry and pts_fav_exp defaults
# ─────────────────────────────────────────────────────────────

class TestWinRateAsymmetry:
    """With no prior results, win_rates fall back to 0.5 and pts_fav_exp to AVG."""

    def test_no_results_default_win_rates(self, synthetic_feature_builder):
        """Before any matches, h_win_rate_global == a_win_rate_global == 0.5."""
        df = synthetic_feature_builder.build_features("Trento", "Perugia", jornada=1)
        assert df.iloc[0]["h_win_rate_global"] == 0.5
        assert df.iloc[0]["a_win_rate_global"] == 0.5

    def test_no_results_default_pts_fav_exp(self, synthetic_feature_builder):
        """Before any matches, pts_fav_exp falls back to AVG_POINTS_PER_SET."""
        from src.simulation.constants import AVG_POINTS_PER_SET
        df = synthetic_feature_builder.build_features("Trento", "Perugia", jornada=1)
        assert df.iloc[0]["h_pts_fav_exp"] == AVG_POINTS_PER_SET
        assert df.iloc[0]["a_pts_fav_exp"] == AVG_POINTS_PER_SET

    def test_asymmetric_win_rate_after_update(self, synthetic_feature_builder):
        """After one match, win_rates diverge (home≠away)."""
        builder = synthetic_feature_builder
        # Simulate a Trento home win
        builder.update("Trento", "Perugia",
                       sets_local=3, sets_visitante=1,
                       winner="home",
                       points_local=75, points_visitante=60)

        df = builder.build_features("Trento", "Perugia", jornada=2)
        h_wr = df.iloc[0]["h_win_rate_global"]
        a_wr = df.iloc[0]["a_win_rate_global"]
        assert h_wr == 1.0  # Trento won their only match
        assert a_wr == 0.0  # Perugia lost their only match
        # Asymmetry: home≠away
        assert h_wr != a_wr
