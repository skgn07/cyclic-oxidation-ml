import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from catboost import CatBoostRegressor

# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("\n" + "="*60)
print("LOADING ORIGINAL DATASET")
print("="*60 + "\n")

df = pd.read_csv("data/Oxi_Cycle.csv")

print(f"Original dataset size: {len(df)}")

# ============================================================
# DUPLICATE DATASET
# ============================================================

print("\n" + "="*60)
print("DUPLICATING DATASET")
print("="*60 + "\n")

df_doubled = pd.concat(
    [df, df],
    ignore_index=True
)

print(f"Doubled dataset size: {len(df_doubled)}")

# SAVE DOUBLED DATASET

df_doubled.to_csv(
    "data/Oxi_Cycle_doubled.csv",
    index=False
)

print("\nSaved: data/Oxi_Cycle_doubled.csv")

# ============================================================
# FEATURES / TARGET
# ============================================================

X = df_doubled[[
    "Fe",
    "Ni",
    "Cr",
    "Temperature",
    "Time"
]]

y = df_doubled["Mass Change"]

# ============================================================
# RANDOM TRAIN TEST SPLIT
# ============================================================

print("\n" + "="*60)
print("PERFORMING RANDOM TRAIN-TEST SPLIT")
print("="*60 + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(f"Train size: {len(X_train)}")
print(f"Test size : {len(X_test)}")

# ============================================================
# CATBOOST MODEL
# ============================================================

print("\n" + "="*60)
print("TRAINING CATBOOST MODEL")
print("="*60 + "\n")

model = CatBoostRegressor(

    depth=8,

    learning_rate=0.1,

    iterations=1000,

    loss_function="RMSE",

    task_type="GPU",

    devices="0",

    verbose=100
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*60)
print("RESULTS ON DOUBLED DATASET")
print("="*60 + "\n")

print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = pd.DataFrame({

    "Measured": y_test.values,

    "Predicted": y_pred,

    "Residual": y_test.values - y_pred
})

results_df.to_excel(
    "outputs/tables/doubled_dataset_predictions.xlsx",
    index=False
)

print("\nSaved: outputs/tables/doubled_dataset_predictions.xlsx")

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "="*60)
print("DUPLICATED DATASET EXPERIMENT COMPLETED")
print("="*60)