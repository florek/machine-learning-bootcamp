import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.datasets import make_regression
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
sns.set(font_scale=1.3)
print(sklearn.__version__)


y_true = 100 + 20 * np.random.randn(50)
print(y_true)
y_pred = y_true + 10 * np.random.randn(50)
print(y_pred)

results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
print(results.head())

results['error'] = results['y_true'] - results['y_pred']
results['squared_error'] = results['error'] ** 2
print(results.head())

print(f"Mean Absolute Error: {abs(results['error']).sum() / len(results)}")
print(f"Mean Squared Error: {results['squared_error'].mean()}")
print(f"Root Mean Squared Error: {np.sqrt(results['squared_error'].mean())}")


def plot_regression_results(y_true, y_pred):
    results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    min = results[['y_true', 'y_pred']].min().min()
    max = results[['y_true', 'y_pred']].max().max()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=results['y_true'],
                y=results['y_pred'],
                mode='markers',
                name='Data'
            ),
            go.Scatter(
                x=[min, max],
                y=[min, max]
            )
        ],
        layout=go.Layout(
            title='Regression Results',
            xaxis=dict(title='True Values'),
            yaxis=dict(title='Predicted Values')
        )
    )
    fig.show()

plot_regression_results(y_true, y_pred)


y_true = 100 + 20 * np.random.randn(1000)
y_pred = y_true + 10 * np.random.randn(1000)

results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
results['error'] = results['y_true'] - results['y_pred']
px.histogram(results, x='error', nbins=50, title='Error Distribution').show()

print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))
print(max_error(y_true, y_pred))
print(r2_score(y_true, y_pred))