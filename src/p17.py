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

print('Wskaźnik Gini')
print(1 - (50 / 150) ** 2 - (50 / 150)**2 - (50/150)**2)
print(1 - (45 / 52)**2 - (6/52)**2 - (1/52)**2)
print(1-(5/98)**2 - (44/98)**2 - (49/98)**2)

print('Wskaźnik Entropii')
print(-(50 / 150) * np.log2(50 / 150) - (50 / 150) * np.log2(50 / 150) - (50 / 150) * np.log2(50 / 150))
print(-(45 / 52) * np.log2(45 / 52) - (6/52) * np.log2(6/52) - (1/52) * np.log2(1/52))
print(-(3/91) * np.log2(3/91) - (44/98) * np.log2(44/98) - (49/98) * np.log2(49/98))


from scipy.stats import entropy

print(entropy([0.5, 0.5], base=2))
print(entropy([0.8, 0.2], base=2))
print(entropy([0.95, 0.05], base=2))

def entropy(x):
    return -np.sum(x * np.log2(x))

print(entropy([0.5, 0.5]))
print(entropy([0.8, 0.2]))
print(entropy([0.95, 0.05]))

p = np.arange(0.01, 1.0, 0.01)
q = 1 - p
pq = np.c_[p, q]
print(pq[:10])

entropies = [entropy(pair) for pair in pq]
print(entropies[:10])
plt.plot(p, entropies)
plt.show()

print('Zysk informacyjny jest tam gdzie entropia jest mniejsza przy podziale')