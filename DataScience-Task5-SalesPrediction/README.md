# OIBSIP/DataScience-Task5-SalesPrediction

**Track:** Data Science | **Task:** 5 — Sales Prediction Using Python

## Objective
Predict product sales from advertising spend across TV, Radio, and Newspaper
channels.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## ⚠️ Data note
Built without internet access, so the classic Kaggle "Advertising.csv" dataset
could not be downloaded. The notebook generates a synthetic dataset with the same
columns (`TV, Radio, Newspaper, Sales`) and the same well-known underlying
relationship (TV strongest predictor, Newspaper weakest). **Before submitting,
download the real `Advertising.csv` from Kaggle and replace the generated dataframe
with `pd.read_csv("Advertising.csv")`** — no other code needs to change.

## Approach
- EDA: nulls, descriptive stats, pairplot, per-channel scatter plots, correlation
  heatmap.
- Train/test split.
- Trained Linear Regression (baseline) and Random Forest Regressor.
- Evaluated with MAE, RMSE, R².
- Residual plot and coefficient-based channel-impact interpretation.

## Files
- `Sales_Prediction.ipynb` — full executed notebook with outputs.
