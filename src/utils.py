# ============================================================
# UTILITY FUNCTIONS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import *

# ============================================================
# FONT + PLOT CONFIGURATION
# ============================================================

matplotlib.rcParams['font.family'] = FONT_FAMILY
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df

# ============================================================
# PRINT REGRESSION METRICS
# ============================================================

def print_regression_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    print("\n================================================")
    print("REGRESSION METRICS")
    print("================================================\n")

    print(f"MAE   : {mae:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")

    return mae, mse, rmse, r2

# ============================================================
# PRINT CLASSIFICATION METRICS
# ============================================================

def print_classification_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_true,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='weighted'
    )

    print("\n================================================")
    print("CLASSIFICATION METRICS")
    print("================================================\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return accuracy, precision, recall, f1

# ============================================================
# SAVE DATAFRAME
# ============================================================

def save_dataframe(df, filename):

    save_path = TABLE_DIR + filename

    df.to_excel(
        save_path,
        index=False
    )

    print(f"\nSaved: {save_path}")

# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(filename):

    save_path = FIGURE_DIR + filename

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=FIG_DPI
    )

    print(f"\nSaved: {save_path}")

# ============================================================
# PLOT STYLE
# ============================================================

def apply_plot_style():

    plt.rcParams['figure.figsize'] = FIG_SIZE

    plt.rcParams['axes.grid'] = True

# ============================================================
# PRINT SECTION TITLE
# ============================================================

def print_section(title):

    print("\n================================================")
    print(title)
    print("================================================\n")