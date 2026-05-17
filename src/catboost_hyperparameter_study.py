# catboost_hyperparameter_study.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

print("\n============================================================")
print("LOADING DATASET")
print("============================================================\n")

df = pd.read_csv("data/Oxi_Cycle.csv")

print(df.head())

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[["Fe", "Ni", "Cr", "Temperature", "Time"]]
y = df["Mass Change"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

depth_list = [4, 6, 8, 10]
learning_rate_list = [0.01, 0.03, 0.05, 0.1, 0.2]

results = []

print("\n============================================================")
print("RUNNING HYPERPARAMETER STUDY")
print("============================================================\n")

# ============================================================
# GRID SEARCH
# ============================================================

for depth in depth_list:
    for lr in learning_rate_list:

        print(f"Depth={depth}, Learning Rate={lr}")

        model = CatBoostRegressor(
            depth=depth,
            learning_rate=lr,
            iterations=1000,
            loss_function="RMSE",
            verbose=False
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)

        results.append({
            "Depth": depth,
            "Learning Rate": lr,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Test R²": test_r2
        })

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test RMSE",
    ascending=True
)

print("\n============================================================")
print("FINAL RESULTS")
print("============================================================\n")

print(results_df)

# Save table
results_df.to_excel(
    "outputs/tables/catboost_hyperparameter_study.xlsx",
    index=False
)

print("\nSaved: outputs/tables/catboost_hyperparameter_study.xlsx")

# ============================================================
# BEST MODEL
# ============================================================

best_row = results_df.iloc[0]

best_depth = int(best_row["Depth"])
best_lr = float(best_row["Learning Rate"])

print("\n============================================================")
print("BEST PARAMETERS")
print("============================================================\n")

print(f"Best Depth         : {best_depth}")
print(f"Best Learning Rate : {best_lr}")
print(f"Best Test RMSE     : {best_row['Test RMSE']:.4f}")
print(f"Best Test R²       : {best_row['Test R²']:.4f}")

# ============================================================
# TRAIN FINAL OPTIMIZED MODEL
# ============================================================

final_model = CatBoostRegressor(
    depth=best_depth,
    learning_rate=best_lr,
    iterations=1000,
    loss_function="RMSE",
    verbose=False
)

final_model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ============================================================
# RMSE VS ITERATIONS
# ============================================================

eval_results = final_model.get_evals_result()

train_rmse_curve = eval_results["learn"]["RMSE"]
val_rmse_curve = eval_results["validation"]["RMSE"]

plt.figure(figsize=(10, 6))

plt.plot(
    train_rmse_curve,
    label="Training RMSE"
)

plt.plot(
    val_rmse_curve,
    label="Validation RMSE"
)

plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.title("CatBoost RMSE vs Iterations")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/figures/catboost_rmse_vs_iterations.png",
    dpi=300
)

print("Saved: outputs/figures/catboost_rmse_vs_iterations.png")

# ============================================================
# HEATMAP STYLE TABLE PLOT
# ============================================================

pivot_table = results_df.pivot(
    index="Depth",
    columns="Learning Rate",
    values="Test RMSE"
)

plt.figure(figsize=(8, 6))

plt.imshow(pivot_table, aspect='auto')

plt.colorbar(label="Test RMSE")

plt.xticks(
    range(len(pivot_table.columns)),
    pivot_table.columns
)

plt.yticks(
    range(len(pivot_table.index)),
    pivot_table.index
)

plt.xlabel("Learning Rate")
plt.ylabel("Depth")

plt.title("CatBoost Hyperparameter RMSE Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/figures/catboost_hyperparameter_heatmap.png",
    dpi=300
)

print("Saved: outputs/figures/catboost_hyperparameter_heatmap.png")

# ============================================================
# COMPLETED
# ============================================================

print("\n============================================================")
print("HYPERPARAMETER STUDY COMPLETED")
print("============================================================")