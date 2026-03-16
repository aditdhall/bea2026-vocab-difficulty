"""XGBoost feature-only baseline. One model per L1."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats


class XGBoostBaseline:
    """Feature-based XGBoost regressor. Uses fitted FeaturePipeline for transform."""

    def __init__(self, l1: str, feature_pipeline):
        self.l1 = l1
        self.pipeline = feature_pipeline
        self.model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    def train(self, train_df: pd.DataFrame) -> "XGBoostBaseline":
        """Fit pipeline on train, transform, then train XGBRegressor."""
        self.pipeline.fit(train_df)
        X = self.pipeline.transform(train_df)
        if "GLMM_score" not in X.columns:
            raise ValueError("train_df must contain GLMM_score")
        y = X["GLMM_score"]
        self._feature_cols = [c for c in X.columns if c != "GLMM_score"]
        X = X[self._feature_cols]
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Transform with fitted pipeline and predict."""
        X = self.pipeline.transform(df)
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0
        return self.model.predict(X[self._feature_cols])

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Predict and return RMSE and Pearson r."""
        if "GLMM_score" not in df.columns:
            return {"rmse": None, "pearson_r": None, "pearson_p": None}
        y_true = df["GLMM_score"].values
        y_pred = self.predict(df)
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        r, p = stats.pearsonr(y_true, y_pred)
        return {"rmse": rmse, "pearson_r": float(r), "pearson_p": float(p)}

    def save_predictions(self, df: pd.DataFrame, output_path: str) -> None:
        """Save CSV with columns item_id, prediction."""
        preds = self.predict(df)
        out = pd.DataFrame({"item_id": df["item_id"].values, "prediction": preds})
        out.to_csv(output_path, index=False)
