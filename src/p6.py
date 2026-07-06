import numpy as np
import pandas as pd
import plotly.express as px


np.random.seed(42)
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)
print(f'Lata pracy: {X1}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}')
X1 = X1.reshape(m, 1)
Y = Y.reshape(-1, 1)
print(X1)
print(X1.shape)
print(Y)
bias = np.ones((m, 1))
print(bias)
print(bias.shape)
X = np.append(bias, X1, axis=1)
print(X)
print(X.shape)
eta = 0.01
weights = np.random.randn(2, 1)
print(X)
print(weights)
intercept = []
coef = []
for i in range(3000):
    gradient = (2 / m) * X.T.dot(X.dot(weights) - Y)
    weights = weights - eta * gradient
    intercept.append(weights[0][0])
    coef.append(weights[1][0])
print(weights)
df = pd.DataFrame(data={'intercept': intercept, 'coef': coef})
print(df.head())
fig = px.line(
    df,
    y='intercept',
    width=800,
    title='Dopasowanie: intercept (gradient descent)'
)
fig.show()
fig = px.line(
    df,
    y='coef',
    width=800,
    title='Dopasowanie: coef (gradient descent)'
)
fig.show()
