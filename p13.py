import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
import seaborn as sns
import plotly.express as px
import sklearn
from sklearn.datasets import make_regression

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
sns.set(font_scale=1.3)
print(sklearn.__version__)

data, target = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)
target = target ** 2
print(f'{data[:5]}\n')
print(f'{target[:5]}\n')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='dane')
plt.legend()
plt.show()


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(data, target)
plot_data = np.arange(-3, 3, 0.01).reshape(-1, 1)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='dane')
plt.plot(plot_data, regressor.predict(plot_data), color='red', label='model')
plt.legend()
plt.show()

from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(max_depth=2)
regressor_tree.fit(data, target)

plt.figure(figsize=(8, 6))
plt.title('Regresja drzewa decyzyjnego')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='dane')
plt.plot(plot_data, regressor_tree.predict(plot_data), color='green', label='model')
plt.legend()
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(regressor_tree, filled=True, rounded=True, feature_names=['cecha x'])
plt.title('Struktura drzewa decyzyjnego')
plt.tight_layout()
plt.savefig('tree.png')
plt.show()

def make_dt_regression(max_depth=2):
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(data, target)
    plt.figure(figsize=(8, 6))
    plt.plot(plot_data, regressor.predict(plot_data), color='green', label='model')
    plt.scatter(data, target, label='dane')
    plt.legend()
    plt.title(f'Regresja drzewa decyzyjnego (max_depth={max_depth})')
    plt.show()
    plt.figure(figsize=(12, 8))
    plot_tree(regressor, filled=True, rounded=True, feature_names=['cecha x'])
    plt.title(f'Struktura drzewa (max_depth={max_depth})')
    plt.tight_layout()
    plt.savefig(f'tree_depth_{max_depth}.png')
    plt.show()

make_dt_regression(max_depth=2)
make_dt_regression(max_depth=3)
make_dt_regression(max_depth=4)
make_dt_regression(max_depth=5)
