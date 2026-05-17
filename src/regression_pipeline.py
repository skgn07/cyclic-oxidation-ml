# ============================================================
# REGRESSION PIPELINE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.regression import *

from catboost import CatBoostRegressor

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from utils import *

from config import *

# ============================================================
# INITIALIZATION
# ============================================================

create_directories()

apply_plot_style()

print_section("LOADING DATASET")

df = load_dataset()

print(df.head())

# ============================================================
# CORRELATION HEATMAP
# ============================================================

print_section("PEARSON CORRELATION MATRIX")

corr = df.corr(method='pearson')

plt.figure(figsize=(8, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    square=True,
    linewidths=0.5
)

plt.title("Pearson Correlation Matrix")

save_figure("correlation_heatmap.png")

plt.show()

# ============================================================
# CORRELATION BARPLOT
# ============================================================

target_corr = corr["Mass Change"].drop("Mass Change")

plt.figure(figsize=(8, 5))

plt.bar(
    target_corr.index,
    target_corr.values
)

plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")

plt.title("Correlation with Mass Change")

save_figure("correlation_barplot.png")

plt.show()

# ============================================================
# PYCARET MODEL COMPARISON
# ============================================================

print_section("PYCARET MODEL COMPARISON")

setup_obj = setup(
    data=df,
    target='Mass Change',
    session_id=RANDOM_STATE,
    numeric_features=[
        'Fe',
        'Ni',
        'Cr',
        'Temperature',
        'Time'
    ],
    train_size=(1 - TEST_SIZE),
    use_gpu=False,
    verbose=False
)

best_model = compare_models(
    fold=CV_FOLDS
)

print("\nBEST MODEL:\n")

print(best_model)

# ============================================================
# SPLIT DATA
# ============================================================

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# ============================================================
# CATBOOST REGRESSOR
# ============================================================

print_section("TRAINING CATBOOST REGRESSOR")

model = CatBoostRegressor(

    iterations=327,

    learning_rate=0.13950258050858852,

    depth=6,

    l2_leaf_reg=2.490445971811424,

    bagging_temperature=1.0499269447570991,

    random_strength=1.1339773723159667,

    loss_function='RMSE',

    eval_metric='R2',

    task_type='GPU',

    devices='0',

    verbose=100
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    verbose=100
)

# ============================================================
# PREDICTIONS
# ============================================================

preds = model.predict(X_test)

# ============================================================
# REGRESSION METRICS
# ============================================================

mae, mse, rmse, r2 = print_regression_metrics(
    y_test,
    preds
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = X_test.copy()

prediction_df['Measured'] = y_test.values

prediction_df['Predicted'] = preds

save_dataframe(
    prediction_df,
    "regression_predictions.xlsx"
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print_section("FEATURE IMPORTANCE")

plt.figure(figsize=(10, 5))

plt.barh(
    model.feature_names_,
    model.feature_importances_
)

plt.xlabel("Importance")

plt.ylabel("Features")

plt.title("CatBoost Feature Importance")

save_figure(
    "feature_importance.png"
)

plt.show()

# ============================================================
# RMSE VS ITERATIONS
# ============================================================

print_section("RMSE VS ITERATIONS")

evals = model.get_evals_result()

train_rmse = evals['learn']['RMSE']

test_rmse = evals['validation']['RMSE']

plt.figure(figsize=(8, 5))

plt.plot(
    train_rmse,
    label='Training RMSE'
)

plt.plot(
    test_rmse,
    label='Validation RMSE'
)

plt.xlabel("Iterations")

plt.ylabel("RMSE")

plt.title("RMSE Reduction During Training")

plt.legend()

save_figure(
    "rmse_vs_iterations.png"
)

plt.show()

# ============================================================
# MEASURED VS PREDICTED
# ============================================================

print_section("MEASURED VS PREDICTED")

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    preds,
    marker='x'
)

min_val = min(
    min(y_test),
    min(preds)
)

max_val = max(
    max(y_test),
    max(preds)
)

x_line = np.linspace(
    min_val,
    max_val,
    100
)

# Perfect prediction line
plt.plot(
    x_line,
    x_line
)

# +5%
plt.plot(
    x_line,
    1.05 * x_line,
    linestyle='--'
)

# -5%
plt.plot(
    x_line,
    0.95 * x_line,
    linestyle='--'
)

plt.xlabel("Measured (mg.cm$^{-2}$)")

plt.ylabel("Predicted (mg.cm$^{-2}$)")

plt.title("Measured vs Predicted")

plt.text(
    max_val * 0.60,
    max_val * 0.80,
    "+5%"
)

plt.text(
    max_val * 0.75,
    max_val * 0.60,
    "-5%"
)

plt.axis('equal')

save_figure(
    "measured_vs_predicted.png"
)

plt.show()

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print_section("5-FOLD CROSS VALIDATION")

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_model = CatBoostRegressor(
    **CATBOOST_PARAMS
)

cv_scores = cross_val_score(
    cv_model,
    X,
    y,
    cv=kf,
    scoring='r2'
)

print("\n5-Fold CV R² Scores:")
print(cv_scores)

print(f"\nAverage CV R²: {cv_scores.mean():.4f}")
# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    MODEL_DIR + "catboost_regressor.cbm"
)

print("\nSaved trained CatBoost model.")

# ============================================================
# FINISHED
# ============================================================

print_section("REGRESSION PIPELINE COMPLETED")