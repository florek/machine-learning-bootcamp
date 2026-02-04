# Szybka powtórka przed lekcją

Ultra-skondensowane przypomnienie najważniejszych rzeczy z każdej lekcji. Użyj przed każdą lekcją, żeby szybko odświeżyć wiedzę.

---

## 📌 P6: Gradient Descent

**Co robi:** Ręczna implementacja regresji liniowej metodą gradient descent

**Kluczowe elementy:**
- `bias = np.ones((m, 1))` → dodanie wyrazu wolnego
- `gradient = (2/m) * X.T.dot(X.dot(weights) - Y)` → obliczenie gradientu
- `weights = weights - eta * gradient` → aktualizacja wag
- Pętla 3000 iteracji → uczenie modelu

**Równanie:** `Y = w0 * 1 + w1 * X`

---

## 📌 P7: Regresja liniowa scikit-learn

**Co robi:** Regresja liniowa na syntetycznych danych

**Kluczowe elementy:**
- `make_regression()` → generowanie danych testowych
- `LinearRegression().fit()` → trenowanie modelu
- `coef_`, `intercept_` → parametry modelu
- `score()` → R² (współczynnik determinacji)

**Równanie:** `y = intercept_ + coef_[0] * x`

---

## 📌 P8: Train/Test Split

**Co robi:** Podział danych i ocena modelu na osobnych zbiorach

**Kluczowe elementy:**
- `train_test_split(test_size=0.25)` → podział 75/25
- `score(X_train)` vs `score(X_test)` → wykrycie overfittingu
- Analiza błędów: `error = y_test - y_pred`
- Histogram błędów → rozkład powinien być normalny wokół zera

**Złota zasada:** Model trenuje na train, ocenia na test!

---

## 📌 P9: Rzeczywiste dane + EDA

**Co robi:** Pełny pipeline: eksploracja → feature engineering → modelowanie

**Kluczowe elementy:**
- `read_csv()` → wczytanie danych
- `drop_duplicates()` → usuwanie duplikatów
- `get_dummies(drop_first=True)` → one-hot encoding
- `corr()` → macierz korelacji
- `mean_absolute_error()` → metryka MAE

**Pipeline:**
1. EDA (`info()`, `describe()`, `value_counts()`)
2. Czyszczenie (duplikaty, braki)
3. Feature engineering (one-hot encoding)
4. Analiza korelacji
5. Train/test split
6. Trenowanie i ocena

---

## 📌 P10: OLS statsmodels + selekcja zmiennych

**Co robi:** Model OLS z analizą statystyczną i backward elimination

**Kluczowe elementy:**
- `pd.get_dummies().values.astype(float)` → przygotowanie danych
- `sm.add_constant()` → dodanie intercept
- `sm.OLS().fit()` → model OLS
- `ols.summary()` → statystyki (p-value, R²)
- Backward elimination → usuwanie nieistotnych zmiennych (p ≥ 0.05)

**Interpretacja p-value:**
- **p < 0.05** → istotna statystycznie ✅
- **p ≥ 0.05** → nieistotna (usuń) ❌

**Proces selekcji:**
1. Pełny model → sprawdź p-value
2. Usuń zmienną z najwyższym p ≥ 0.05
3. Powtórz dla nowego modelu

---

## 🔄 Powtarzające się koncepty (wszystkie lekcje)

### Importy (standardowe)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### Konfiguracja
```python
np.random.seed(42)
sns.set(font_scale=1.3)
```

### One-Hot Encoding
```python
df_dummies = pd.get_dummies(df, drop_first=True)
```

### Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
```

### Model regresji
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)
```

---

## ⚡ Szybkie przypomnienie przed lekcją

**Przed P6:** Pamiętaj o reshape danych i dodaniu biasu

**Przed P7:** `make_regression()` do generowania danych, `score()` zwraca R²

**Przed P8:** Zawsze dziel dane na train/test przed trenowaniem!

**Przed P9:** EDA → czyszczenie → encoding → modelowanie

**Przed P10:** `drop_first=True` w get_dummies, `.astype(float)` przed statsmodels, p-value < 0.05 = istotna

---

## 🎯 Najważniejsze zasady

1. **random_state=42** → powtarzalność
2. **train/test split** → zawsze przed trenowaniem
3. **drop_first=True** → unika kolinearności
4. **EDA przed modelowaniem** → zrozum dane
5. **score(train) vs score(test)** → wykrycie overfittingu
6. **p-value < 0.05** → zmienna istotna

---

> **Użycie:** Przeczytaj sekcję dla danej lekcji przed zajęciami. Pełne wyjaśnienia w summary_p*.md, szczegóły techniczne w cheat_sheet.md.
