import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris


np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: f'{x:.2f}'))
sns.set()
print(sklearn.__version__)


# Załadowanie danych
print('--- Załadowanie danych ---\n\n')
raw_data = load_iris()
raw_data_copy = raw_data.copy()
print(raw_data_copy.keys())
print(raw_data_copy['DESCR'])

# Przygotowanie danych
print('--- Przygotowanie danych ---\n\n')
data = raw_data_copy['data']
target = raw_data_copy['target']
print(f'{data[:5]}\n')
print(target[:5])

# Połączenie atrybutów ze zmienną docelową
all_data = np.c_[data, target]
print(all_data[:5])

# Budowa obiektu DataFrame
df = pd.DataFrame(data=all_data, columns=raw_data.feature_names + ['target'])
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.describe().T.apply(lambda x: round(x, 2)))
print(df.target.value_counts())
df.target.value_counts().plot(kind='pie')
# plt.show()

data = df.copy()
target = data.pop('target')
print(data.head())
print(target.head())

# Podział danych na zbiór treningowy i testowy - iris data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42, stratify=target)
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')


# Podział danych na zbiór treningowy i testowy - breast cancer data
from sklearn.datasets import load_breast_cancer

raw_data = load_breast_cancer()
raw_data_copy = raw_data.copy()
print(raw_data_copy.keys())
print(raw_data_copy['DESCR'])

data = raw_data_copy['data']
target = raw_data_copy['target']
print(f'{data[:5]}\n')
print(target[:5])

all_data = np.c_[data, target]
print(all_data[:5])

df = pd.DataFrame(data=all_data, columns=list(raw_data['feature_names']) + ['target'])
print(df.head())
print(df.target.value_counts())

data = df.copy()
target = data.pop('target')
print(data.head())
print(target.head())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, stratify=target, random_state=42)

print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ntarget:\n{target.value_counts() / len(target)}')
print(f'\ny_train:\n{y_train.value_counts() / len(y_train)}')
print(f'\ny_test:\n{y_test.value_counts() / len(y_test)}')



