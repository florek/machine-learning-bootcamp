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
print(df.info())

cat_cols = [column for column in df.columns if df[column].dtype == 'O']
print(cat_cols)

for cat in cat_cols:
    df[cat] = df[cat].astype('category')

print(df.info())
print(df.describe().T)
print(df.describe(include='category').T)
print(df.isnull().sum())
print(df.sex.value_counts())
df.sex.value_counts().plot(kind='pie')
plt.show()
print(df.smoker.value_counts())
print(df.region.value_counts())
df.charges.plot(kind='hist')
plt.show()

import plotly.express as px
fig = px.histogram(df, x='charges', width=800, height=400, facet_col='smoker', title='Rozkład kosztów medycznych', facet_row='sex')
fig.show()
fig = px.histogram(df, x='smoker', facet_col='sex')
fig.show()

df_dummies = pd.get_dummies(df, drop_first=True)
print(df_dummies)

corr = df_dummies.corr()

sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

print(df_dummies.corr()['charges'].sort_values(ascending=False))
sns.set()
df_dummies.corr()['charges'].sort_values()[:-1].plot(kind='barh')
plt.show()

data = df_dummies.copy()
target = data.pop('charges')
print(data.head())
print(target.head())

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))

y_pred = regressor.predict(X_test)
print(y_pred[:10])

y_true = y_test.copy()
predictions = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
print(predictions.head())

predictions['error'].plot(kind='hist', bins=50, figsize=(8, 6))
plt.show()

mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')

print(regressor.intercept_)
print(regressor.coef_)
print(data.columns)
