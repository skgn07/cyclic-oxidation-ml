# ============================================================
# HYPERPARAMETER OPTIMIZATION PIPELINE
# ============================================================

import numpy as np
import pandas as pd
import optuna

from catboost import CatBoostRegressor

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)

from sklearn.metrics import (
    mean_squared_error
)

from utils import *

from config import *

# ============================================================
# INITIALIZATION
# ============================================================

create_directories()

print_section("LOADING DATASET")

df = load_dataset()

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# ============================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================

def objective(trial):

    params = {

        "iterations": trial.suggest_int(
            "iterations",
            200,
            1200
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.005,
            0.15,
            log=True
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            4,
            8
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1,
            15
        ),

        "bagging_temperature": trial.suggest_float(
            "bagging_temperature",
            0,
            5
        ),

        "random_strength": trial.suggest_float(
            "random_strength",
            0,
            5
        ),

        "loss_function": "RMSE",

        "task_type": "GPU",

        "devices": GPU_DEVICE,

        "verbose": 0
    }

    model = CatBoostRegressor(
        **params
    )

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring='r2'
    )

    return np.mean(scores)

# ============================================================
# START OPTIMIZATION
# ============================================================

print_section("STARTING OPTUNA OPTIMIZATION")

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

# ============================================================
# BEST PARAMETERS
# ============================================================

print_section("BEST PARAMETERS")

print(study.best_params)

print(f"\nBest CV R²: {study.best_value:.4f}")

# ============================================================
# SAVE BEST PARAMETERS
# ============================================================

best_params_df = pd.DataFrame(
    [study.best_params]
)

best_params_df['Best_CV_R2'] = study.best_value

save_dataframe(
    best_params_df,
    "best_hyperparameters.xlsx"
)

# ============================================================
# TRAIN FINAL OPTIMIZED MODEL
# ============================================================

print_section("TRAINING FINAL OPTIMIZED MODEL")

best_params = study.best_params

best_params.update({

    "loss_function": "RMSE",

    "task_type": "GPU",

    "devices": GPU_DEVICE,

    "verbose": 100
})

final_model = CatBoostRegressor(
    **best_params
)

final_model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test)
)

# ============================================================
# PREDICTIONS
# ============================================================

preds = final_model.predict(X_test)

# ============================================================
# FINAL METRICS
# ============================================================

print_regression_metrics(
    y_test,
    preds
)

# ============================================================
# CROSS VALIDATION
# ============================================================

# ============================================================
# CROSS VALIDATION
# ============================================================

print_section("CROSS VALIDATION")

cv_final_model = CatBoostRegressor(

    iterations=best_params["iterations"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    l2_leaf_reg=best_params["l2_leaf_reg"],
    bagging_temperature=best_params["bagging_temperature"],
    random_strength=best_params["random_strength"],

    loss_function="RMSE",

    task_type="GPU",

    devices=GPU_DEVICE,

    verbose=0
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    cv_final_model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.4f}")

print(f"\nAverage CV R²: {np.mean(cv_scores):.4f}")

# ============================================================
# SAVE OPTIMIZED MODEL
# ============================================================

final_model.save_model(
    MODEL_DIR + "optimized_catboost_regressor.cbm"
)

print("\nSaved optimized CatBoost model.")

# ============================================================
# FINISHED
# ============================================================

print_section("HYPERPARAMETER OPTIMIZATION COMPLETED")