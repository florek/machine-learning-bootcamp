# Cheat Sheet – szybkie przypomnienie kluczowych konceptów

Szybkie przypomnienie najważniejszych konceptów z bootcampu ML. Szczegółowe wyjaśnienia w summary_p6.md – summary_p12.md.

---

## 📚 Importy (standardowe)

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
```

**Konfiguracja:**
```python
np.random.seed(42)
sns.set(font_scale=1.3)
```

---

## 📊 Przygotowanie danych

### Wczytanie danych
```python
df = pd.read_csv('path/to/file.csv')
df = df_raw.copy()
```

### Eksploracja danych (EDA)
```python
df.info()
df.describe()
df.describe(include='category')
df.isnull().sum()
df.duplicated()
df.drop_duplicates()
```

### Podział na cechy i target
```python
target = data.pop('charges')
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
```

---

## 🔢 Kodowanie zmiennych kategorycznych

### One-Hot Encoding
```python
df_dummies = pd.get_dummies(df, drop_first=True)
```

**Dlaczego `drop_first=True`:**
- Unika pułapki zmiennych fikcyjnych (dummy variable trap)
- Redukuje kolinearność
- Jedna kategoria jest referencyjna (domyślna)

**Przykład:**
- `sex`: ['male', 'female'] → `sex_male` (0 lub 1)
- `region`: 4 kategorie → 3 kolumny (jedna usunięta)

### Konwersja na kategorie
```python
cat_cols = [col for col in df.columns if df[col].dtype == 'O']
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
```

---

## 🤖 Regresja liniowa (scikit-learn)

### Podstawowy model
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
```

### Parametry modelu
```python
regressor.coef_
regressor.intercept_
```

**Równanie:** `y = intercept_ + coef_[0] * x1 + coef_[1] * x2 + ...`

### Ocena modelu
```python
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)
```

**`score()` zwraca R²:**
- 1.0 = idealne dopasowanie
- 0.0 = model nie lepszy niż średnia
- < 0 = model gorszy niż średnia

### Metryki błędów
```python
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
max_err = max_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## 📈 Regresja OLS (statsmodels)

### Przygotowanie danych
```python
X_train_ols = pd.get_dummies(X_train, drop_first=True)
predictors = ['const'] + list(X_train_ols.columns)
X_train_ols = X_train_ols.values.astype(float)
X_train_ols = sm.add_constant(X_train_ols)
```

### Model OLS
```python
ols = sm.OLS(endog=y_train.values, exog=X_train_ols).fit()
print(ols.summary(xname=predictors))
```

**Interpretacja p-value:**
- **p < 0.05** → zmienna istotna statystycznie
- **p ≥ 0.05** → zmienna nieistotna (można usunąć)

---

## 📐 Regresja wielomianowa

Gdy zależność jest nieliniowa (np. wielomianowa), rozszerz cechy i użyj zwykłej regresji liniowej.

### Kształt danych (jedna cecha)
```python
X = X.reshape(n, 1)
```
scikit-learn oczekuje macierzy 2D (próbki × cechy).

### Rozszerzenie cech (PolynomialFeatures)
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)
y_pred = regressor.predict(X_poly)
```

**Uwaga:** przy wysokim stopniu i małej liczbie obserwacji ryzyko przeuczenia; rozważ regularyzację (Ridge/Lasso) lub niższy stopień.

### Ocena
```python
from sklearn.metrics import r2_score
r2_score(y, y_pred)
```

---

## 📉 Wizualizacje

### Podstawowe (matplotlib)
```python
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='dane')
plt.plot(X, y_pred, color='red', label='model')
plt.legend()
plt.show()
```

### Histogram (pandas)
```python
df['column'].plot(kind='hist', bins=50, figsize=(8, 6))
```

### Wykresy pandas
```python
df.plot(kind='hist')    # histogram
df.plot(kind='line')    # liniowy
df.plot(kind='bar')     # słupkowy
df.plot(kind='barh')    # poziomy słupkowy
df.plot(kind='box')     # pudełkowy
df.plot(kind='scatter') # punktowy
df.plot(kind='pie')     # kołowy
```

### Heatmap korelacji (seaborn)
```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

### Plotly Express (interaktywne)
```python
import plotly.express as px

fig = px.histogram(df, x='charges', facet_col='smoker')
fig.show()
```

---

## 🔍 Analiza danych

### Statystyki opisowe
```python
df.describe().T
df.value_counts()
df['column'].value_counts()
```

### Korelacja
```python
df.corr()
df.corr()['target'].sort_values(ascending=False)
```

### Analiza błędów
```python
predictions = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred
})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
predictions['error'].plot(kind='hist', bins=50)
```

---

## 🎯 Gradient Descent (ręczna implementacja)

```python
eta = 0.01
weights = np.random.randn(2, 1)

for i in range(3000):
    gradient = (2 / m) * X.T.dot(X.dot(weights) - Y)
    weights = weights - eta * gradient
```

**Co się dzieje:**
1. Predykcja: `X.dot(weights)`
2. Błąd: `X.dot(weights) - Y`
3. Gradient: `(2/m) * X.T.dot(błąd)`
4. Aktualizacja: `weights = weights - eta * gradient`

---

## 🔄 Selekcja zmiennych

### Automatyczna Backward Elimination
```python
sl = 0.05
while True:
    ols = sm.OLS(endog=y_train.values, exog=X_train_numpy).fit()
    max_pval = max(ols.pvalues.astype('float'))
    if max_pval > sl:
        max_idx = np.argmax(ols.pvalues.astype('float'))
        X_train_numpy = np.delete(X_train_numpy, max_idx, axis=1)
        predictors.remove(predictors[max_idx])
    else:
        break
```

**Funkcje NumPy:**
- `np.argmax(array)` → indeks elementu z najwyższą wartością
- `np.delete(array, idx, axis=1)` → usuwa kolumnę o indeksie idx

### Ręczne usuwanie kolumn (numpy)
```python
X_selected = X_train_ols[:, [0, 1, 2, 3, 5]]
predictors.remove('column_name')
```

### Zapis modelu do pliku
```python
ols.save('model.pickle')

import pickle
with open('model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)
```

---

## ⚠️ Najczęstsze błędy i rozwiązania

### Problem: TypeError z stringami w statsmodels
**Rozwiązanie:** `X_train_ols.values.astype(float)`

### Problem: Pandas Series w statsmodels
**Rozwiązanie:** `y_train.values` (konwersja na numpy array)

### Problem: Boolean w numpy array
**Rozwiązanie:** `.astype(float)` po `get_dummies()`

### Problem: Kolinearność w one-hot encoding
**Rozwiązanie:** `drop_first=True` w `get_dummies()`

---

## 📝 Dobre praktyki

1. **Zawsze używaj `random_state`** dla powtarzalności
2. **Zachowaj oryginalne dane:** `df = df_raw.copy()`
3. **Trenuj na train, oceniaj na test** – nigdy odwrotnie!
4. **Sprawdzaj overfitting:** porównaj score na train vs test
5. **EDA przed modelowaniem** – zrozum dane najpierw
6. **Usuwaj duplikaty** przed trenowaniem
7. **Koduj kategorie** przed użyciem w modelach ML

---

## 🎓 Kluczowe koncepty

### Overfitting vs Underfitting
- **Overfitting:** score(train) >> score(test) → model zapamiętał dane
- **Underfitting:** oba niskie → model za prosty
- **Idealnie:** podobne wyniki na train i test

### R² vs MAE
- **R²:** procent wyjaśnionej wariancji (0-1)
- **MAE:** średni błąd w jednostkach targetu (łatwiejsze do zrozumienia)

### Train/Test Split
- **80/20** lub **75/25** to standard
- **random_state** dla powtarzalności
- **Nigdy nie oceniaj na danych treningowych!**

---

## 📚 Mapowanie konceptów do lekcji

- **P6** → Gradient Descent (ręczna implementacja)
- **P7** → Regresja liniowa scikit-learn (syntetyczne dane)
- **P8** → Train/test split, ocena modelu
- **P9** → Rzeczywiste dane, EDA, feature engineering
- **P10** → OLS statsmodels, selekcja zmiennych
- **P11** → Automatyczna backward elimination, zapis modelu
- **P12** → Regresja wielomianowa (PolynomialFeatures)
- **P13** → Regresja drzewa decyzyjnego (DecisionTreeRegressor, plot_tree)
- **P14** → Metryki regresji (MAE, MSE, RMSE, max_error, R²), wizualizacja

---

## 🌳 Regresja drzewa decyzyjnego

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree

regressor_tree = DecisionTreeRegressor(max_depth=2)
regressor_tree.fit(data, target)
y_pred = regressor_tree.predict(plot_data)

plt.figure(figsize=(12, 8))
plot_tree(regressor_tree, filled=True, rounded=True, feature_names=['cecha x'])
plt.tight_layout()
plt.show()
```

**max_depth:** ogranicza głębokość; małe = prostszy model, duże = lepsze dopasowanie, ryzyko przeuczenia. Krzywa predykcji ma postać „schodków”.

---

> **Tip:** Używaj tego cheat sheet jako szybkiego przypomnienia. Szczegółowe wyjaśnienia w summary_p6.md – summary_p14.md.
