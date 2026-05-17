# Cyclic Oxidation Prediction using Machine Learning

Machine learning framework for predicting cyclic oxidation behavior in Fe–Cr and Fe–Cr–Ni alloys using ensemble regression models, cross-validation, SHAP analysis, and materials-science interpretation.

---

# Overview

Cyclic oxidation (CO) behavior of high-temperature alloys depends on several interacting factors including:

- alloy composition
- temperature
- oxidation time
- oxide scale stability
- spallation behavior

This project reproduces and extends published work on ML-based cyclic oxidation prediction using experimental datasets of Fe–Cr and Fe–Cr–Ni alloys exposed to high-temperature oxidizing environments.

The repository includes:

- regression modeling
- model benchmarking
- cross-validation
- SHAP explainability analysis
- alloy-wise oxidation trend analysis
- CatBoost hyperparameter optimization
- validation methodology studies

---

# Objectives

- Predict mass change during cyclic oxidation using machine learning
- Compare multiple regression algorithms on oxidation datasets
- Study the influence of alloy composition and temperature
- Analyze model interpretability using SHAP
- Reproduce published cyclic oxidation ML results
- Investigate effects of validation methodology and dataset duplication on R² performance

---

# Dataset

The dataset contains experimental cyclic oxidation measurements for:

- Binary Fe–Cr alloys
- Ternary Fe–Cr–Ni alloys

## Input Features

- Fe composition
- Cr composition
- Ni composition
- Temperature
- Exposure time

## Target

- Mass change during cyclic oxidation

---

# Machine Learning Models Used

The project benchmarks several regression models including:

- CatBoost Regressor
- Extra Trees Regressor
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- AdaBoost Regressor
- Linear Regression
- ElasticNet
- KNN Regressor
- SVR
- MLP Regressor

---

# Key Results

## Clean Cross-Validation Workflow

- Cross-validated CatBoost R² ≈ 0.88
- Strong generalization across oxidation datasets
- Temperature and chromium concentration identified as dominant factors

## Hyperparameter Optimization

Best CatBoost performance observed near:

- depth ≈ 8
- moderate learning rates

## Validation Methodology Analysis

Dataset duplication combined with random train-test splitting produced artificially inflated R² values approaching ~0.99, demonstrating the importance of proper validation methodology in small scientific datasets.

---

# Repository Structure

```text
cyclic-oxidation-ml/
│
├── data/
├── src/
├── notebooks/
├── outputs/
├── docs/
├── archive_experiments/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/skgn07/cyclic-oxidation-ml.git
cd cyclic-oxidation-ml
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

## Linux / WSL

```bash
source venv/bin/activate
```

## Windows

```bash
venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the main regression pipeline:

```bash
python src/regression_pipeline.py
```

Run model benchmarking:

```bash
python src/advanced_model_benchmark.py
```

Run cross-validation workflow:

```bash
python src/cross_validated_predictions.py
```

Run SHAP analysis:

```bash
python src/shap_analysis.py
```

Run alloy-wise analysis:

```bash
python src/alloy_analysis.py
```

Run hyperparameter study:

```bash
python src/catboost_hyperparameter_study.py
```

---

# Example Outputs

The repository includes:

- measured vs predicted plots
- residual plots
- SHAP summary plots
- alloy-wise oxidation trend plots
- model benchmark comparisons
- CatBoost RMSE convergence plots

---

# Scientific Findings

- Temperature was found to be the dominant oxidation driver.
- Chromium strongly improved oxidation resistance.
- Oxidation behavior showed strong nonlinear dependence on composition and temperature.
- Ensemble tree-based methods significantly outperformed linear regression models.
- Temperature-time interaction effects were important for predicting oxidation kinetics.
- Validation strategy strongly affected apparent model accuracy.

---

# Educational Purpose

This repository is intended for:

- materials informatics learning
- machine learning for materials science
- oxidation modeling studies
- computational metallurgy education
- reproducible ML workflows

---

# Future Work

Potential future directions include:

- larger oxidation datasets
- CALPHAD-integrated descriptors
- DFT-informed feature engineering
- physics-informed machine learning
- oxidation regime classification
- quantum-inspired materials informatics workflows

---

# Reference

This repository reproduces and extends parts of the methodology presented in the following work:

```text
M. K. Anirudh,
M. Sreenidhi Iyengar,
P. H. Anantha Desik,
M. P. Phaniraj

"Artificial Intelligence Approach to Predict Elevated Temperature Cyclic Oxidation of Fe–Cr and Fe–Cr–Ni Alloys"

Oxidation of Metals (2022)

DOI: 10.1007/s11085-022-10123-5
```

Paper Link:

https://link.springer.com/article/10.1007/s11085-022-10123-5

The present repository further includes:
- model benchmarking
- cross-validation analysis
- SHAP explainability
- validation methodology studies
- dataset duplication experiments
- educational notebook workflows

---


# Acknowledgements

This work was developed as part of a materials science and machine learning research initiative focused on cyclic oxidation modeling of Fe–Cr and Fe–Cr–Ni alloys.