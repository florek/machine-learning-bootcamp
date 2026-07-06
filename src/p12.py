import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
import seaborn as sns
import plotly.express as px
import sklearn

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
sns.set(font_scale=1.3)
print(sklearn.__version__)

X = np.arange(-10, 10, 0.5)
noise = 80 * np.random.randn(40)
y = -X**3 + 10*X**2 - 2*X + 3 + noise
X = X.reshape(40, 1)

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)
y_pred_lin = regressor.predict(X)

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_lin, color='red', label='regresja liniowa')
plt.legend()
plt.show()

from sklearn.metrics import r2_score

print(r2_score(y, y_pred_lin))

df = pd.DataFrame(data={'X': X.ravel()})
print(df.head(10))

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(X_poly)
print(X_poly.shape)
df = pd.DataFrame(X_poly)
df.columns = ['1', 'x', 'x^2']
print(df.head(10))

regressor_poly = LinearRegression()
regressor_poly.fit(X_poly, y)
y_pred_poly = regressor_poly.predict(X_poly)

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_poly, color='red', label='regresja wielomianowa')
plt.plot(X, y_pred_lin, color='blue', label='regresja liniowa')
plt.legend()
plt.show()
print(r2_score(y, y_pred_poly))

X_poly_3 = PolynomialFeatures(degree=3).fit_transform(X)
regressor_poly_3 = LinearRegression()
regressor_poly_3.fit(X_poly_3, y)
y_pred_poly_3 = regressor_poly_3.predict(X_poly_3)
print(r2_score(y, y_pred_poly_3))

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_poly_3, color='red', label='regresja wielomianowa 3 stopnia')
plt.plot(X, y_pred_poly, color='green', label='regresja wielomianowa 2 stopnia')
plt.plot(X, y_pred_lin, color='blue', label='regresja liniowa')
plt.legend()
plt.show()  

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

results = pd.DataFrame(data={
    'name': ['regresja liniowa', 'regresja wielomianowa 2 stopnia', 'regresja wielomianowa 3 stopnia'],
    'r2_score': [r2_score(y, y_pred_lin), r2_score(y, y_pred_poly), r2_score(y, y_pred_poly_3)],
    'mse': [mse(y, y_pred_lin), mse(y, y_pred_poly), mse(y, y_pred_poly_3)],
    'mae': [mae(y, y_pred_lin), mae(y, y_pred_poly), mae(y, y_pred_poly_3)],
    'rmse': [np.sqrt(mse(y, y_pred_lin)), np.sqrt(mse(y, y_pred_poly)), np.sqrt(mse(y, y_pred_poly_3))],
})
print(results)

px.bar(results, x='name', y='r2_score', color='name').show()
px.bar(results, x='name', y='mse', color='name').show()
px.bar(results, x='name', y='mae', color='name').show()
px.bar(results, x='name', y='rmse', color='name').show()