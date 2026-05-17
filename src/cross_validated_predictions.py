import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# FEATURES / TARGET
# ============================================================

X = df[["Fe", "Ni", "Cr", "Temperature", "Time"]]

y = df["Mass Change"]

# ============================================================
# OPTIMIZED CATBOOST MODEL
# ============================================================

model = CatBoostRegressor(

    iterations=327,
    learning_rate=0.13950258050858852,
    depth=6,
    l2_leaf_reg=2.490445971811424,
    bagging_temperature=1.0499269447570991,
    random_strength=1.1339773723159667,

    loss_function="RMSE",

    task_type="GPU",
    devices="0",

    verbose=0
)

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "="*60)
print("RUNNING 5-FOLD CROSS VALIDATION")
print("="*60 + "\n")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# TRUE OUT-OF-FOLD PREDICTIONS

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
print("CROSS-VALIDATED METRICS")
print("="*60 + "\n")

print(f"CV R²   : {r2:.4f}")
print(f"CV RMSE : {rmse:.4f}")
print(f"CV MAE  : {mae:.4f}")

# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": ["R²", "RMSE", "MAE"],

    "Value": [r2, rmse, mae]
})

metrics_df.to_excel(
    "outputs/tables/cv_metrics.xlsx",
    index=False
)

print("\nSaved: outputs/tables/cv_metrics.xlsx")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

pred_df = pd.DataFrame({

    "Measured": y,
    "Predicted": y_pred,
    "Residual": y - y_pred
})

pred_df.to_excel(
    "outputs/tables/cv_predictions.xlsx",
    index=False
)

print("Saved: outputs/tables/cv_predictions.xlsx")

# ============================================================
# MEASURED VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y,
    y_pred,
    s=80
)

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())

# PERFECT FIT

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linewidth=2
)

# +5%

plt.plot(
    [min_val, max_val],
    [min_val * 1.05, max_val * 1.05],
    linestyle="--",
    linewidth=2
)

# -5%

plt.plot(
    [min_val, max_val],
    [min_val * 0.95, max_val * 0.95],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Measured Mass Change")

plt.ylabel("Predicted Mass Change")

plt.title("Cross-Validated Measured vs Predicted")

plt.tight_layout()

plt.savefig(
    "outputs/figures/cv_measured_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: outputs/figures/cv_measured_vs_predicted.png")

plt.close()

# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = y - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    s=80
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Predicted Mass Change")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    "outputs/figures/residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: outputs/figures/residual_plot.png")

plt.close()

# ============================================================
# ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=20
)

plt.xlabel("Residual Error")

plt.ylabel("Frequency")

plt.title("Residual Error Distribution")

plt.tight_layout()

plt.savefig(
    "outputs/figures/error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: outputs/figures/error_distribution.png")

plt.close()

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "="*60)
print("CROSS-VALIDATED ANALYSIS COMPLETED")
print("="*60)