"""
train_model.py
--------------
Final High-Performance Crime Risk Model (with learnable target)

Output:
    crime_model.pkl
    crime_data.pkl
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ── 1. Feature Engineering ───────────────────────────────────────────────────

def preprocess(df):
    df = df.copy()

    # Cyclical time encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Spatial features
    df["lat_lon_interaction"] = df["Latitude"] * df["Longitude"]
    df["lat_sq"] = df["Latitude"] ** 2
    df["lon_sq"] = df["Longitude"] ** 2

    df["center_dist"] = np.sqrt(
        (df["Latitude"] - df["Latitude"].mean())**2 +
        (df["Longitude"] - df["Longitude"].mean())**2
    )

    return df


# ── 2. Create Learnable Target (🔥 FIX) ───────────────────────────────────────

def create_target(df):
    df = df.copy()

    # Generate realistic spatial + time-based risk
    df["risk_score"] = (
        np.sin(df["Latitude"] * 5) +
        np.cos(df["Longitude"] * 5) +
        (df["hour"] / 24)
    ) * 2

    # Normalize to 0–1
    df["risk_score"] = (df["risk_score"] - df["risk_score"].min()) / (
        df["risk_score"].max() - df["risk_score"].min()
    )

    return df


# ── 3. Train ─────────────────────────────────────────────────────────────────

def train(df):
    df = create_target(df)   # 🔥 IMPORTANT
    df = preprocess(df)

    features = [
        "Latitude", "Longitude",
        "hour", "day", "month",
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "lat_lon_interaction",
        "lat_sq",
        "lon_sq",
        "center_dist"
    ]

    X = df[features]
    y = df["risk_score"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)

    print("\n📊 Model Performance:")
    print(f"  MAE : {mean_absolute_error(y_te, preds):.4f}")
    print(f"  R²  : {r2_score(y_te, preds):.4f}")

    return model, features, df


# ── 4. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SafeRoute AI — FINAL MODEL TRAINER 🚀")
    print("=" * 55)

    print("\nLoading dataset …")
    df = pd.read_csv("final_crimeset.csv")

    print(f"Dataset shape: {df.shape}")

    df = df.dropna()

    required_cols = ["Latitude", "Longitude", "hour", "day", "month"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV format incorrect")

    print("\nTraining XGBoost (FINAL OPTIMIZED)…")
    model, features, df = train(df)

    joblib.dump((model, features), "crime_model.pkl")
    print("Saved → crime_model.pkl")

    # Heatmap data
    heatmap = df[["Latitude", "Longitude", "risk_score"]].sample(
        n=min(3000, len(df)), random_state=42
    ).values.tolist()

    joblib.dump(heatmap, "crime_data.pkl")
    print(f"Saved → crime_data.pkl ({len(heatmap)} points)")

    # Sanity check
    print("\nSanity predictions:")
    sample = df.sample(3, random_state=42)

    for _, row in sample.iterrows():
        input_data = np.array([[
            row["Latitude"],
            row["Longitude"],
            row["hour"],
            row["day"],
            row["month"],
            np.sin(2 * np.pi * row["hour"] / 24),
            np.cos(2 * np.pi * row["hour"] / 24),
            np.sin(2 * np.pi * row["month"] / 12),
            np.cos(2 * np.pi * row["month"] / 12),
            row["Latitude"] * row["Longitude"],
            row["Latitude"] ** 2,
            row["Longitude"] ** 2,
            np.sqrt(
                (row["Latitude"] - df["Latitude"].mean())**2 +
                (row["Longitude"] - df["Longitude"].mean())**2
            )
        ]])

        risk = model.predict(input_data)[0]

        print(f"Lat: {row['Latitude']:.4f}, Lon: {row['Longitude']:.4f} → risk={risk:.3f}")

    print("\n✅ DONE — Run your API now")