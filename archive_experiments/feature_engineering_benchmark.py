import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from catboost import CatBoostRegressor

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "="*60)
print("LOADING DATASET")
print("="*60 + "\n")

df = pd.read_csv("data/Oxi_Cycle.csv")

print(df.head())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "="*60)
print("GENERATING ENGINEERED FEATURES")
print("="*60 + "\n")

# LOG TIME
df["log_time"] = np.log1p(df["Time"])

# TIME SQUARED
df["time_squared"] = df["Time"] ** 2

# TEMPERATURE × TIME
df["temp_time_interaction"] = (
    df["Temperature"] * df["Time"]
)

# Cr / Fe
df["cr_fe_ratio"] = (
    df["Cr"] / (df["Fe"] + 1e-6)
)

# Ni / Cr
df["ni_cr_ratio"] = (
    df["Ni"] / (df["Cr"] + 1e-6)
)

# TOTAL ALLOYING CONTENT
df["total_alloying"] = (
    df["Cr"] + df["Ni"]
)

# Cr × Temperature
df["cr_temp_interaction"] = (
    df["Cr"] * df["Temperature"]
)

# Ni × Temperature
df["ni_temp_interaction"] = (
    df["Ni"] * df["Temperature"]
)

print(df.head())

# ============================================================
# FEATURES / TARGET
# ============================================================

feature_columns = [

    "Fe",
    "Ni",
    "Cr",
    "Temperature",
    "Time",

    "log_time",
    "time_squared",
    "temp_time_interaction",
    "cr_fe_ratio",
    "ni_cr_ratio",
    "total_alloying",
    "cr_temp_interaction",
    "ni_temp_interaction"
]

X = df[feature_columns]

y = df["Mass Change"]

# ============================================================
# OPTIMIZED CATBOOST MODEL
# ============================================================

model = CatBoostRegressor(

    depth=8,

    learning_rate=0.1,

    iterations=1000,

    loss_function="RMSE",

    task_type="GPU",

    devices="0",

    verbose=0
)

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "="*60)
print("RUNNING FEATURE-ENGINEERED CV")
print("="*60 + "\n")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# OUT-OF-FOLD PREDICTIONS

y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=kf,
    n_jobs=1
)

# ============================================================
# METRICS
# ============================================================

r2 = r2_score(y, y_pred)

rmse = np.sqrt(mean_squared_error(y, y_pred))

mae = mean_absolute_error(y, y_pred)

print("\n" + "="*60)
print("FEATURE ENGINEERING RESULTS")
print("="*60 + "\n")

print(f"CV R²   : {r2:.4f}")
print(f"CV RMSE : {rmse:.4f}")
print(f"CV MAE  : {mae:.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame({

    "Measured": y,
    "Predicted": y_pred,
    "Residual": y - y_pred
})

results_df.to_excel(
    "outputs/tables/feature_engineered_predictions.xlsx",
    index=False
)

print("\nSaved: outputs/tables/feature_engineered_predictions.xlsx")

# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": ["R²", "RMSE", "MAE"],

    "Value": [r2, rmse, mae]
})

metrics_df.to_excel(
    "outputs/tables/feature_engineered_metrics.xlsx",
    index=False
)

print("Saved: outputs/tables/feature_engineered_metrics.xlsx")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*60)
print("TRAINING FINAL MODEL FOR FEATURE IMPORTANCE")
print("="*60 + "\n")

model.fit(X, y)

importance_df = pd.DataFrame({

    "Feature": feature_columns,

    "Importance": model.get_feature_importance()
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

importance_df.to_excel(
    "outputs/tables/engineered_feature_importance.xlsx",
    index=False
)

print("\nSaved: outputs/tables/engineered_feature_importance.xlsx")

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING ANALYSIS COMPLETED")
print("="*60)