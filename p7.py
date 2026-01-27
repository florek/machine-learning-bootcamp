import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

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

from sklearn.datasets import make_regression

data, target = make_regression(
    n_samples=100, 
    n_features=1, 
    n_informative=1, 
    noise=10, 
    random_state=42
)

print(f'data shape: {data.shape}')
print(f'target shape: {target.shape}')
print(data[:5])
print(target[:5])


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(data, target)
print(regressor.score(data, target))
y_pred = regressor.predict(data)
print(y_pred)


plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, y_pred, color='red', label='model')
plt.legend()
plt.plot()
plt.show()

print(item for item in dir(regressor) if not item.startswith('_'))
print(regressor.coef_)
print(regressor.intercept_)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, regressor.intercept_ + regressor.coef_[0] * data, color='red', label='model')
plt.legend()
plt.plot()
plt.show()
