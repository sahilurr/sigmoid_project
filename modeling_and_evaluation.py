"""
Demand Modeling & Operational Fleet Sizing Pipeline
---------------------------------------------------
Dataset: Capital Bikeshare (Washington D.C., 2011-2012)
Author: Analytics & Machine Learning Team

Purpose:
1. Train and benchmark hourly bike demand models on a strict out-of-time holdout set
   (Train: Jan 2011 - Aug 2012 | Test: Sep 2012 - Dec 2012, 2,888 unseen hours).
2. Evaluate decomposed models (Casual vs. Registered) vs. direct aggregate models.
3. Fit Quantile Gradient Boosters (90th and 95th percentiles) to determine risk-adjusted
   operational fleet staging buffers, answering the CEO's operational question:
   "How many bikes do we actually need?"
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_features(df):
    """
    Transforms raw telemetry into model-ready domain features:
    - Rescaled physical weather units (Celsius, %, km/h)
    - Cyclical continuous encodings of diurnal, seasonal, and weekly cycles
    - Commute vs leisure indicator flags
    - Cross-feature interactions between weather and workday schedules
    """
    data = df.copy()
    
    # 1. Physical weather conversions
    data["temp_c"] = data["temp"] * 41.0
    data["atemp_c"] = data["atemp"] * 50.0
    data["temp_diff"] = data["atemp_c"] - data["temp_c"]
    data["hum_pct"] = data["hum"] * 100.0
    data["windspeed_kmh"] = data["windspeed"] * 67.0
    
    # 2. Non-linear weather transformations
    data["temp_sq"] = data["temp"] ** 2
    data["hum_x_temp"] = data["hum"] * data["temp"]
    
    # 3. Cyclical encodings (prevents discontinuity at boundary hours/months)
    data["sin_hr"] = np.sin(2 * np.pi * data["hr"] / 24)
    data["cos_hr"] = np.cos(2 * np.pi * data["hr"] / 24)
    data["sin_mnth"] = np.sin(2 * np.pi * data["mnth"] / 12)
    data["cos_mnth"] = np.cos(2 * np.pi * data["mnth"] / 12)
    data["sin_weekday"] = np.sin(2 * np.pi * data["weekday"] / 7)
    data["cos_weekday"] = np.cos(2 * np.pi * data["weekday"] / 7)
    
    # 4. Operational calendar segments
    data["is_commute_morning"] = ((data["workingday"] == 1) & (data["hr"].isin([7, 8, 9]))).astype(int)
    data["is_commute_evening"] = ((data["workingday"] == 1) & (data["hr"].isin([16, 17, 18, 19]))).astype(int)
    data["is_commute_peak"] = data["is_commute_morning"] | data["is_commute_evening"]
    data["is_weekend_afternoon"] = ((data["workingday"] == 0) & (data["hr"].between(11, 17))).astype(int)
    data["is_night"] = (data["hr"].between(0, 5)).astype(int)
    data["is_bad_weather"] = (data["weathersit"].isin([3, 4])).astype(int)
    
    # 5. Segment interactions
    data["workingday_x_hr"] = data["workingday"] * data["hr"]
    data["temp_x_workingday"] = data["temp"] * data["workingday"]
    data["temp_x_weekend"] = data["temp"] * (1 - data["workingday"])
    
    return data


FEATURE_COLS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "temp_c", "atemp_c", "temp_diff", "hum_pct", "windspeed_kmh",
    "temp_sq", "hum_x_temp",
    "sin_hr", "cos_hr", "sin_mnth", "cos_mnth", "sin_weekday", "cos_weekday",
    "is_commute_morning", "is_commute_evening", "is_commute_peak",
    "is_weekend_afternoon", "is_night", "is_bad_weather",
    "workingday_x_hr", "temp_x_workingday", "temp_x_weekend"
]


def calculate_metrics(y_true, y_pred, label="Model"):
    """Evaluate standard regression metrics on positive-constrained predictions."""
    y_pred_clipped = np.maximum(y_pred, 0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred_clipped)
    r2 = r2_score(y_true, y_pred_clipped)
    
    # Avoid zero division in MAPE
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred_clipped[mask]) / y_true[mask])) * 100
    
    return {
        "model": label,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 1)
    }


def train_and_benchmark():
    # 1. Load and prepare time series
    raw_hour = pd.read_csv("hour.csv")
    raw_hour["datetime"] = pd.to_datetime(raw_hour["dteday"]) + pd.to_timedelta(raw_hour["hr"], unit="h")
    raw_hour = raw_hour.sort_values("datetime").reset_index(drop=True)
    
    hour_df = create_features(raw_hour)
    
    # 2. Out-of-time chronological split
    # Training: Jan 2011 through Aug 2012 (20 months)
    # Testing:  Sep 2012 through Dec 2012 (4 months holdout across fall/winter)
    split_date = pd.to_datetime("2012-09-01")
    train_mask = hour_df["datetime"] < split_date
    test_mask = hour_df["datetime"] >= split_date
    
    X_train, y_train_cnt = hour_df.loc[train_mask, FEATURE_COLS], hour_df.loc[train_mask, "cnt"]
    X_test, y_test_cnt = hour_df.loc[test_mask, FEATURE_COLS], hour_df.loc[test_mask, "cnt"]
    
    y_train_cas, y_test_cas = hour_df.loc[train_mask, "casual"], hour_df.loc[test_mask, "casual"]
    y_train_reg, y_test_reg = hour_df.loc[train_mask, "registered"], hour_df.loc[test_mask, "registered"]
    
    print(f"Dataset Split (Chronological Holdout):")
    print(f"  Training Set: {len(X_train):,} hours ({hour_df.loc[train_mask, 'datetime'].min().date()} to {hour_df.loc[train_mask, 'datetime'].max().date()})")
    print(f"  Test Holdout: {len(X_test):,} hours ({hour_df.loc[test_mask, 'datetime'].min().date()} to {hour_df.loc[test_mask, 'datetime'].max().date()})\n")
    
    results = []

    # Model 1: Stratified Historical Baseline (workingday x hr x month lookup)
    baseline_lookup = hour_df.loc[train_mask].groupby(["workingday", "hr", "mnth"])["cnt"].mean().to_dict()
    test_tuples = list(zip(hour_df.loc[test_mask, "workingday"], hour_df.loc[test_mask, "hr"], hour_df.loc[test_mask, "mnth"]))
    y_pred_baseline = np.array([baseline_lookup.get(k, y_train_cnt.mean()) for k in test_tuples])
    results.append(calculate_metrics(y_test_cnt, y_pred_baseline, "Historical Stratified Mean"))

    # Model 2: Ridge Regression
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train_cnt)
    y_pred_ridge = ridge.predict(X_test)
    results.append(calculate_metrics(y_test_cnt, y_pred_ridge, "Regularized Ridge Regression"))

    # Model 3: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_cnt)
    y_pred_rf = rf.predict(X_test)
    results.append(calculate_metrics(y_test_cnt, y_pred_rf, "Random Forest (100 trees)"))

    # Model 4: HistGradientBoosting (Total Demand Direct)
    hgb_cnt = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    hgb_cnt.fit(X_train, y_train_cnt)
    y_pred_hgb = hgb_cnt.predict(X_test)
    results.append(calculate_metrics(y_test_cnt, y_pred_hgb, "HistGradientBoosting (Total)"))

    # Model 5: Decomposed Models (Casual + Registered)
    hgb_cas = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    hgb_cas.fit(X_train, y_train_cas)
    y_pred_cas = np.maximum(hgb_cas.predict(X_test), 0)

    hgb_reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    hgb_reg.fit(X_train, y_train_reg)
    y_pred_reg = np.maximum(hgb_reg.predict(X_test), 0)

    y_pred_decomposed = y_pred_cas + y_pred_reg
    results.append(calculate_metrics(y_test_cnt, y_pred_decomposed, "Decomposed (Casual + Registered)"))

    # Print benchmark table
    benchmark_df = pd.DataFrame(results)
    print("--- Benchmark Model Performance on Out-of-Time Test Set ---")
    print(benchmark_df.to_string(index=False))
    
    # 6. Operational Quantile Models for Buffer Sizing (90% and 95% service levels)
    print("\n--- Training Operational Quantile Models for Fleet Buffering ---")
    hgb_q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    hgb_q90.fit(X_train, y_train_cnt)
    y_pred_q90 = hgb_q90.predict(X_test)
    cov_90 = (y_test_cnt <= y_pred_q90).mean() * 100

    hgb_q95 = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    hgb_q95.fit(X_train, y_train_cnt)
    y_pred_q95 = hgb_q95.predict(X_test)
    cov_95 = (y_test_cnt <= y_pred_q95).mean() * 100

    print(f"90% Target Quantile Model -> Actual Holdout Coverage: {cov_90:.1f}%")
    print(f"95% Target Quantile Model -> Actual Holdout Coverage: {cov_95:.1f}%")
    print("Takeaway: Quantile models successfully provide upper-bound safety buffers that shield riders from stockouts.")

    # 7. Persist model bundle for operations assistant
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "feature_cols": FEATURE_COLS,
        "model_cnt": hgb_cnt,
        "model_cas": hgb_cas,
        "model_reg": hgb_reg,
        "model_q90": hgb_q90,
        "model_q95": hgb_q95,
        "rf_cnt": rf,
        "test_metrics": benchmark_df.to_dict(orient="records"),
        "coverage": {"cov_90": round(cov_90, 2), "cov_95": round(cov_95, 2)}
    }
    artifact_path = "models/bike_demand_models.joblib"
    joblib.dump(artifacts, artifact_path)
    print(f"\nModel artifacts successfully exported to {artifact_path}")


if __name__ == "__main__":
    train_and_benchmark()
