import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# MODELS
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    # PAPER MODELS

    "CatBoost Regressor": CatBoostRegressor(
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
    ),

    "Extreme Gradient Boosting": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        random_state=42
    ),

    "Gradient Boosting Regressor":
        GradientBoostingRegressor(random_state=42),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=42
        ),

    "AdaBoost Regressor":
        AdaBoostRegressor(random_state=42),

    "Decision Tree Regressor":
        DecisionTreeRegressor(random_state=42),

    "Lasso Regression":
        Lasso(),

    "Linear Regression":
        LinearRegression(),

    # MODERN MODELS

    "Extra Trees Regressor":
        ExtraTreesRegressor(
            n_estimators=500,
            n_jobs=-1,
            random_state=42
        ),

    "LightGBM":
        LGBMRegressor(
            n_estimators=300,
            device="gpu",
            random_state=42
        ),

    "HistGradientBoosting":
        HistGradientBoostingRegressor(random_state=42),

    "ElasticNet":
        ElasticNet(),

    "Bayesian Ridge":
        BayesianRidge(),

    "Huber Regressor":
        HuberRegressor(),

    "KNN Regressor":
        KNeighborsRegressor(),

    "SVR (RBF)":
        SVR(),

    "MLP Regressor":
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42
        )
}

# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60 + "\n")

for name, model in models.items():

    print(f"Running: {name}")

    try:

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mae = mean_absolute_error(y_test, preds)

        results.append({

            "Model": name,
            "R²": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })

    except Exception as e:

        print(f"FAILED: {name}")
        print(e)

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R²",
    ascending=False
).reset_index(drop=True)

results_df.index = results_df.index + 1

print("\n" + "="*60)
print("FINAL MODEL BENCHMARK")
print("="*60 + "\n")

print(results_df)

# ============================================================
# SAVE TABLE
# ============================================================

results_df.to_excel(
    "outputs/tables/model_benchmark.xlsx",
    index=True
)

print("\nSaved: outputs/tables/model_benchmark.xlsx")

# ============================================================
# BARPLOT
# ============================================================

plt.figure(figsize=(12, 8))

plt.barh(
    results_df["Model"],
    results_df["R²"]
)

plt.xlabel("R² Score")

plt.ylabel("Models")

plt.title("Model Benchmark Comparison")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/figures/model_benchmark_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: outputs/figures/model_benchmark_comparison.png")

plt.close()

print("\n" + "="*60)
print("BENCHMARKING COMPLETED")
print("="*60)