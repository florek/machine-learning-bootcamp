import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

data, target = make_regression(n_samples=1000, n_features=1, n_informative=1, noise=15.0, random_state=42)

print('Data shape:', data.shape)
print('Target shape:', target.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='dane treningowe')
plt.scatter(X_test, y_test, label='dane testowe')
plt.legend()
plt.show()

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa zbiór treningowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='dane treningowe')
plt.plot(X_train, regressor.predict(X_train), color='red', label='model')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa zbiór testowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_test, y_test, label='dane testowe')
plt.plot(X_test, regressor.predict(X_test), color='green', label='model')
plt.legend()
plt.show()

y_pred = regressor.predict(X_test)

predictions_df = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
print(predictions_df.head())
predictions_df['error'] = predictions_df['y_test'] - predictions_df['y_pred']
print(predictions_df.head())
_ = predictions_df['error'].plot(kind='hist', bins=50, figsize=(8, 6))
plt.show()  
