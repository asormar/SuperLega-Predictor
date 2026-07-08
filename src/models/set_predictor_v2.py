"""
set_predictor_v2.py — Adapter for the v2 LogReg SetPredictor.

The v2 model is stored as a joblib dict (not a SetPredictor class instance)
with keys: type, model, features, recency_halflife, train_seasons.

This adapter wraps that dict and exposes the duck-typed interface that
MatchSimulator and SeasonSimulator expect:
  - .feature_names -> list[str]
  - .predict_proba(df) -> np.ndarray shape [n, 2]
  - .predict(df) -> np.ndarray int 0/1
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Union


class LogRegSetPredictor:
    """Adapter for the v2 LogisticRegression SetPredictor (no scaler, no calibration).

    The v2 was trained on raw features (21 columns from SET_FEATURE_COLS) with
    recency weighting (half-life = 2.0 seasons), C = 0.5, max_iter = 2000.
    No feature scaling was applied at training time, so none is needed at inference.
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        recency_halflife: Optional[float] = None,
        train_seasons: Optional[list[int]] = None,
        type_: Optional[str] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.recency_halflife = recency_halflife
        self.train_seasons = train_seasons
        self.type_ = type_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities P(away), P(home).

        The input DataFrame may have columns in any order, or may have extra
        columns.  Only the columns listed in ``self.feature_names`` are used;
        any missing column is treated as 0.0.  This makes the adapter
        order-agnostic and robust at runtime.

        Returns:
            np.ndarray of shape (n, 2) — standard sklearn convention:
            column 0 = P(away), column 1 = P(home).
        """
        # Build a DataFrame with columns in the exact order the model expects
        row = {f: X.get(f, pd.Series(0.0, index=X.index)) for f in self.feature_names}
        X_reordered = pd.DataFrame(row, columns=self.feature_names)
        return self.model.predict_proba(X_reordered.fillna(0.0))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class label: 0 = away wins, 1 = home wins."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def __repr__(self) -> str:
        return (
            f"LogRegSetPredictor(type={self.type_}, "
            f"n_features={len(self.feature_names)}, "
            f"halflife={self.recency_halflife}, "
            f"train_seasons={self.train_seasons})"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogRegSetPredictor":
        """Load the v2 adapter from a joblib dict file."""
        data = joblib.load(str(path))
        return cls(
            model=data["model"],
            feature_names=data["features"],
            recency_halflife=data.get("recency_halflife"),
            train_seasons=data.get("train_seasons"),
            type_=data.get("type"),
        )

    @classmethod
    def try_load_v2(
        cls,
        v2_path: Union[str, Path],
        legacy_path: Optional[Union[str, Path]] = None,
    ) -> tuple:
        """Try to load the v2 predictor; fall back to the legacy SetPredictor.

        Args:
            v2_path: Path to ``set_predictor_v2.joblib``.
            legacy_path: Path to the legacy ``set_predictor.joblib`` (ExtraTrees).

        Returns:
            (predictor, source_label) where source_label is one of
            ``"logreg_v2"``, ``"extra_trees_v1"``, or ``"none"``.

        The returned predictor is duck-typed to the same interface:
        ``.feature_names``, ``.predict_proba(df)`` → ``[n, 2]``.
        """
        v2_path = Path(v2_path)
        if v2_path.exists():
            predictor = cls.load(v2_path)
            return predictor, "logreg_v2"

        if legacy_path is not None:
            from src.models.set_predictor import SetPredictor as LegacySetPredictor
            legacy_path = Path(legacy_path)
            if legacy_path.exists():
                return LegacySetPredictor.load(legacy_path), "extra_trees_v1"

        return None, "none"
