import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from matplotlib.axes import Axes
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
sns.set(font_scale=1.3)


class PlotDecisionBoundaries:
    def __init__(
        self,
        data: np.ndarray,
        target: np.ndarray,
        n_neighbors: int = 5,
    ) -> None:
        self._data = data
        self._n_neighbors = n_neighbors
        self._target = target
    def plot(self, ax: Axes | None = None) -> None:
        classifier = KNeighborsClassifier(n_neighbors=self._n_neighbors)
        classifier.fit(
            self._data,
            self._target,
        )
        x_min, x_max = self._data[:, 0].min() - 0.5, self._data[:, 0].max() + 0.5
        y_min, y_max = self._data[:, 1].min() - 0.5, self._data[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = classifier.predict(grid).reshape(xx.shape)
        plot_ax = ax if ax is not None else plt.subplots(figsize=(8, 6))[1]
        plot_ax.contourf(
            xx,
            yy,
            predictions,
            alpha=0.3,
            cmap='viridis',
        )
        plot_ax.contour(
            xx,
            yy,
            predictions,
            colors='k',
            linewidths=0.5,
        )
        plot_ax.scatter(
            self._data[:, 0],
            self._data[:, 1],
            c=self._target,
            cmap='viridis',
            edgecolors='k',
        )
        plot_ax.set_title(f'Granice decyzyjne KNN (k={self._n_neighbors})')
        plot_ax.set_xlabel('cecha_1: sepal_length')
        plot_ax.set_ylabel('cecha_2: sepal_width')
        if ax is None:
            plt.show()


def plot_decision_boundaries_comparison(
    data: np.ndarray,
    target: np.ndarray,
    n_neighbors_range: range = range(1, 8),
) -> None:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
    )
    flat_axes = axes.ravel()
    for index, n_neighbors in enumerate(n_neighbors_range):
        PlotDecisionBoundaries(
            data=data,
            target=target,
            n_neighbors=n_neighbors,
        ).plot(ax=flat_axes[index])
    for index in range(len(n_neighbors_range), len(flat_axes)):
        flat_axes[index].axis('off')
    fig.suptitle('Porównanie granic decyzyjnych KNN (k=1..7)')
    fig.tight_layout()
    plt.show()


raw_data = load_iris()
print(raw_data.keys())
all_data = raw_data.copy()
data = all_data['data']
target = all_data['target']
print(f'{data[:5]}\n')
print(f'{target[:5]}\n')
print(all_data['target_names'])
df = pd.DataFrame(
    data=np.c_[data, target],
    columns=all_data['feature_names'] + ['class']
)
print(df.head())
print(df.info())
print(df.describe().T)
print(df['class'].value_counts())
print(df.columns)
_ = sns.pairplot(df, vars=all_data['feature_names'], hue='class')
plt.show()
print(df.corr())
data = data[:, :2]
print('data shape:', data.shape)
print('target shape:', target.shape)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis')
plt.title('Wykres punktowy')
plt.xlabel('cecha_1: sepal_length')
plt.ylabel('ceche_2: sepal_width')
plt.show()
df = pd.DataFrame(
    data=np.c_[data, target],
    columns=['sepal_length', 'sepal_width', 'class']
)
fig = px.scatter(
    df,
    x='sepal_length',
    y='sepal_width',
    color='class',
    width=800,
)
fig.show()
plot_decision_boundaries_comparison(
    data=data,
    target=target,
)
