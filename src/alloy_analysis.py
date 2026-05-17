# ============================================================
# ALLOY-WISE OXIDATION CURVE PIPELINE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# TRAIN MODEL
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
# PREDICT ENTIRE DATASET
# ============================================================

print_section("PREDICTING COMPLETE DATASET")

df['Predicted_Mass_Change'] = model.predict(X)

# ============================================================
# SAVE COMPLETE PREDICTIONS
# ============================================================

save_dataframe(
    df,
    "complete_dataset_predictions.xlsx"
)

# ============================================================
# UNIQUE TEMPERATURES
# ============================================================

temperatures = sorted(
    df['Temperature'].unique()
)

# ============================================================
# ALLOY-WISE PLOTTING
# ============================================================

print_section("GENERATING ALLOY-WISE CURVES")

group_columns = [
    'Fe',
    'Ni',
    'Cr',
    'Temperature'
]

alloy_groups = df.groupby(group_columns)

for group_key, group_df in alloy_groups:

    fe, ni, cr, temp = group_key

    print(
        f"\nGenerating plot for "
        f"Fe={fe}, Ni={ni}, Cr={cr}, Temp={temp}"
    )

    group_df = group_df.sort_values('Time')

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(8, 6))

    # Experimental data
    plt.plot(
        group_df['Time'],
        group_df['Mass Change'],
        marker='o',
        label='Experimental'
    )

    # Predicted data
    plt.plot(
        group_df['Time'],
        group_df['Predicted_Mass_Change'],
        marker='x',
        linestyle='--',
        label='Predicted'
    )

    plt.xlabel("Time (h)")

    plt.ylabel("Mass Change (mg.cm$^{-2}$)")

    plt.title(
        f"Fe={fe}, Ni={ni}, Cr={cr}, Temp={temp}°C"
    )

    plt.legend()

    filename = (
        f"alloy_Fe{fe}_Ni{ni}_Cr{cr}_T{temp}.png"
    )

    save_figure(filename)

    plt.show()

# ============================================================
# BINARY Fe-Cr ALLOY PLOTS
# ============================================================

print_section("GENERATING BINARY Fe-Cr PLOTS")

binary_df = df[df['Ni'] == 0]

binary_groups = binary_df.groupby(
    ['Cr', 'Temperature']
)

for group_key, group_df in binary_groups:

    cr, temp = group_key

    print(
        f"\nGenerating binary alloy plot "
        f"Cr={cr}, Temp={temp}"
    )

    group_df = group_df.sort_values('Time')

    plt.figure(figsize=(8, 6))

    plt.plot(
        group_df['Time'],
        group_df['Mass Change'],
        marker='o',
        label='Experimental'
    )

    plt.plot(
        group_df['Time'],
        group_df['Predicted_Mass_Change'],
        marker='x',
        linestyle='--',
        label='Predicted'
    )

    plt.xlabel("Time (h)")

    plt.ylabel("Mass Change (mg.cm$^{-2}$)")

    plt.title(
        f"Binary Fe-Cr Alloy | Cr={cr}, Temp={temp}°C"
    )

    plt.legend()

    filename = (
        f"binary_FeCr_Cr{cr}_T{temp}.png"
    )

    save_figure(filename)

    plt.show()

# ============================================================
# Fe-Cr-10Ni PLOTS
# ============================================================

print_section("GENERATING Fe-Cr-10Ni PLOTS")

ni10_df = df[df['Ni'] == 10]

ni10_groups = ni10_df.groupby(
    ['Cr', 'Temperature']
)

for group_key, group_df in ni10_groups:

    cr, temp = group_key

    print(
        f"\nGenerating Fe-Cr-10Ni plot "
        f"Cr={cr}, Temp={temp}"
    )

    group_df = group_df.sort_values('Time')

    plt.figure(figsize=(8, 6))

    plt.plot(
        group_df['Time'],
        group_df['Mass Change'],
        marker='o',
        label='Experimental'
    )

    plt.plot(
        group_df['Time'],
        group_df['Predicted_Mass_Change'],
        marker='x',
        linestyle='--',
        label='Predicted'
    )

    plt.xlabel("Time (h)")

    plt.ylabel("Mass Change (mg.cm$^{-2}$)")

    plt.title(
        f"Fe-Cr-10Ni Alloy | Cr={cr}, Temp={temp}°C"
    )

    plt.legend()

    filename = (
        f"FeCr10Ni_Cr{cr}_T{temp}.png"
    )

    save_figure(filename)

    plt.show()

# ============================================================
# FINISHED
# ============================================================

print_section("ALLOY-WISE OXIDATION ANALYSIS COMPLETED")