# ============================================================
# SHAP ANALYSIS PIPELINE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

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
# TRAIN CATBOOST MODEL
# ============================================================

print_section("TRAINING CATBOOST MODEL")

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
    eval_set=(X_test, y_test)
)

# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print_section("CREATING SHAP EXPLAINER")

explainer = shap.TreeExplainer(model)

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

print_section("COMPUTING SHAP VALUES")

shap_values = explainer.shap_values(X_test)

# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print_section("SHAP SUMMARY PLOT")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

save_figure(
    "shap_summary_plot.png"
)

plt.show()

# ============================================================
# SHAP BAR PLOT
# ============================================================

print_section("SHAP FEATURE IMPORTANCE")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

save_figure(
    "shap_bar_plot.png"
)

plt.show()

# ============================================================
# SHAP DEPENDENCE PLOTS
# ============================================================

features = X.columns.tolist()

for feature in features:

    print(f"\nGenerating SHAP dependence plot for: {feature}")

    plt.figure()

    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        show=False
    )

    filename = f"shap_dependence_{feature}.png"

    save_figure(filename)

    plt.show()

# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=X.columns
)

save_dataframe(
    shap_df,
    "shap_values.xlsx"
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    MODEL_DIR + "shap_catboost_model.cbm"
)

print("\nSaved SHAP analysis model.")

# ============================================================
# FINISHED
# ============================================================

print_section("SHAP ANALYSIS COMPLETED")