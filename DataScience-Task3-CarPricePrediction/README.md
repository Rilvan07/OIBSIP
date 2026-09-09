# OIBSIP/DataScience-Task3-CarPricePrediction

**Track:** Data Science | **Task:** 3 — Car Price Prediction with Machine Learning

## Objective
Build a regression model predicting used-car selling price from brand, age,
mileage, fuel type and transmission.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## ⚠️ Data note
Built without internet access, so the real "Vehicle dataset from cardekho" (Kaggle)
could not be downloaded. The notebook generates a synthetic dataset with the same
columns (`car_name, year, selling_price, present_price, kms_driven, fuel_type,
transmission`), deliberately including messy values (inconsistent casing,
duplicates, missing values) to mirror a real scraped dataset. **Before submitting,
download the real dataset from Kaggle and replace the generated dataframe with
`pd.read_csv("car data.csv")`** — no other code needs to change.

## Approach
- Data cleaning: nulls, duplicates, inconsistent categorical casing.
- Feature engineering: `car_age` from year, `brand` from car name.
- EDA: price distribution, price vs fuel type, price vs car age.
- One-hot encoding + correlation heatmap.
- Trained Linear Regression and Random Forest Regressor.
- Evaluated with MAE, RMSE, R².
- Feature importance chart for the best model.

## Files
- `Car_Price_Prediction.ipynb` — full executed notebook with outputs.

- ### Project Visualizations & Outputs

![car price Analysis](car%20price%20Analysis.png)

![Feature correlation heatmap](Feature%20correlation%20heatmap.png)

![Random Forest Evaluation](Random%20Forest.png)
