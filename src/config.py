# ============================================================
# CONFIGURATION FILE
# ============================================================

# -----------------------------
# DATASET PATHS
# -----------------------------

DATASET_PATH = "data/Oxi_Cycle.csv"

# -----------------------------
# OUTPUT DIRECTORIES
# -----------------------------

FIGURE_DIR = "outputs/figures/"
TABLE_DIR = "outputs/tables/"
MODEL_DIR = "outputs/models/"

# -----------------------------
# RANDOM SEED
# -----------------------------

RANDOM_STATE = 42

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

TEST_SIZE = 0.20

# -----------------------------
# GPU SETTINGS
# -----------------------------

USE_GPU = True

GPU_DEVICE = '0'

# -----------------------------
# CATBOOST PARAMETERS
# -----------------------------

CATBOOST_PARAMS = {

    "iterations": 327,

    "learning_rate": 0.13950258050858852,

    "depth": 6,

    "l2_leaf_reg": 2.490445971811424,

    "bagging_temperature": 1.0499269447570991,

    "random_strength": 1.1339773723159667,

    "loss_function": "RMSE",

    "eval_metric": "R2",

    "task_type": "GPU",

    "devices": "0",

    "verbose": 100
}

# -----------------------------
# CROSS VALIDATION
# -----------------------------

CV_FOLDS = 5

# -----------------------------
# CLASSIFICATION PARAMETERS
# -----------------------------

CLASSIFIER_PARAMS = {

    "loss_function": "MultiClass",

    "task_type": "GPU",

    "devices": GPU_DEVICE,

    "verbose": 100
}

# -----------------------------
# PLOT SETTINGS
# -----------------------------

FIG_DPI = 300

FIG_SIZE = (8, 6)

# -----------------------------
# FONT FIX
# -----------------------------

FONT_FAMILY = "DejaVu Sans"