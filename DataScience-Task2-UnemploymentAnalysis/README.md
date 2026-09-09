# OIBSIP/DataScience-Task2-UnemploymentAnalysis

**Track:** Data Science | **Task:** 2 — Unemployment Analysis with Python

## Objective
Explore regional and temporal unemployment trends in India, focusing on the impact
of COVID-19 on unemployment rates.

## Tech Stack
Python, pandas, matplotlib, seaborn, Jupyter Notebook

## ⚠️ Data note
Built without internet access, so the real Kaggle "Unemployment in India" dataset
could not be downloaded. The notebook generates a synthetic monthly, state-wise
dataset with a realistic COVID-19 lockdown spike, using the same column names as the
real dataset. **Before submitting, download the real dataset from Kaggle
("Unemployment in India") and replace the generated dataframe with
`pd.read_csv("Unemployment_Rate.csv")`** — no other code needs to change, since
downstream cells reference the same column names.

## Approach
- Data loading, shape/null/dtype inspection.
- Region-wise and month-wise average unemployment.
- Time-series line chart for the top 3 states.
- Bar chart of top 10 states by unemployment.
- Correlation heatmap (unemployment, employment, labour participation).
- Pre-COVID vs during-COVID vs post-COVID comparison.
- Written observations after each chart.

## Files
- `Unemployment_Analysis.ipynb` — full executed notebook with outputs.
