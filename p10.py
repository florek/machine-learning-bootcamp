import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
np.set_printoptions(
    precision=6,
    suppress=True,
    edgeitems=30,
    linewidth=120,
    formatter=dict(float=lambda x: f'{x:.2f}')
)
sns.set(font_scale=1.3)
print(sklearn.__version__)

df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
print(df_raw.head())

df = df_raw.copy()
print(df.info())
print(df[df.duplicated()])
print(df[df['charges'] == 1639.5631])
df = df.drop_duplicates()
data = df.copy()
target = data.pop('charges')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

import statsmodels.api as sm

X_train_ols = X_train.copy()
X_train_ols = pd.get_dummies(X_train_ols, drop_first=True)
predictors = ['const'] + list(X_train_ols.columns)
X_train_ols = X_train_ols.values.astype(float)
X_train_ols = sm.add_constant(X_train_ols)
print(X_train_ols)

ols = sm.OLS(endog=y_train.values, exog=X_train_ols).fit()
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 6, 7, 8]]
predictors.remove('sex_male')
ols = sm.OLS(endog=y_train.values, exog=X_selected).fit()
print(ols.summary(xname=predictors))

X_selected = X_selected[:, [0, 1, 2, 3, 4, 6, 7]]
predictors.remove('region_northwest')
ols = sm.OLS(endog=y_train.values, exog=X_selected).fit()
print(ols.summary(xname=predictors))

X_selected = X_selected[:, [0, 1, 2, 3, 4]]
predictors.remove('region_southeast')
predictors.remove('region_southwest')
ols = sm.OLS(endog=y_train.values, exog=X_selected).fit()
print(ols.summary(xname=predictors))

