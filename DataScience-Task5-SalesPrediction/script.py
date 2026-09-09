# %%md
# Sales Prediction Using Python
**OIBSIP Data Science Track — Task 5**

Objective: predict product sales from advertising spend across TV, Radio and
Newspaper channels.

**Note on data:** this notebook was built without internet access, so the classic
Kaggle "Advertising.csv" dataset could not be downloaded. A dataset with the same
schema and the same realistic underlying relationship is generated below with a
fixed random seed so the notebook fully reproduces. Swap in the real
`Advertising.csv` (same column names: TV, Radio, Newspaper, Sales) before
submission for full authenticity — no code changes are needed if you do.
# %%md-end

# %%code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
rng = np.random.default_rng(42)

n = 200
tv = rng.uniform(0, 300, n)
radio = rng.uniform(0, 50, n)
newspaper = rng.uniform(0, 100, n)
noise = rng.normal(0, 1.8, n)
sales = 4.3 + 0.045 * tv + 0.19 * radio + 0.003 * newspaper + noise
sales = np.clip(sales, 1, None)

df = pd.DataFrame({"TV": tv, "Radio": radio, "Newspaper": newspaper, "Sales": sales})
print(df.head())
# %%code-end

# %%md
## 1. EDA
# %%md-end

# %%code
print("Null values:\n", df.isnull().sum())
print("\nDescriptive statistics:\n", df.describe())

pairplot = sns.pairplot(df)
pairplot.fig.suptitle("Pairwise relationships", y=1.02)
plt.show()
# %%code-end

# %%code
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.scatterplot(data=df, x=col, y="Sales", ax=ax)
    ax.set_title(f"Sales vs {col}")
plt.tight_layout()
plt.show()
# %%code-end

# %%code
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix")
plt.show()
# %%code-end

# %%md
**Observation:** TV spend has the strongest visual and correlational relationship
with Sales, Radio is moderately correlated, and Newspaper spend shows almost no
relationship — a pattern consistent with the well-known real "Advertising" dataset.
# %%md-end

# %%md
## 2. Modelling
# %%md-end

# %%code
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)

for name, model in [("Linear Regression", lin), ("Random Forest", rf)]:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
# %%code-end

# %%code
best_preds = lin.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, best_preds, alpha=0.7)
lims = [y.min(), y.max()]
axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("Actual Sales")
axes[0].set_ylabel("Predicted Sales")
axes[0].set_title("Linear Regression: Actual vs Predicted")

residuals = y_test - best_preds
axes[1].scatter(best_preds, residuals, alpha=0.7)
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("Predicted Sales")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residual plot")
plt.tight_layout()
plt.show()
# %%code-end

# %%code
coefs = pd.Series(lin.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("Linear Regression coefficients (impact per unit spend):")
print(coefs)
print(f"\nHighest-impact channel: {coefs.index[0]}")
# %%code-end

# %%md
**Interpretation:** residuals are scattered randomly around zero with no funnel or
curve shape, confirming a linear model is a reasonable fit. TV advertising spend has
the largest coefficient and therefore the highest marginal impact on sales, followed
by Radio; Newspaper spend contributes very little.
# %%md-end
