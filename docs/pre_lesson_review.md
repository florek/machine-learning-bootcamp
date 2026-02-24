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

**Co robi:** Model OLS z analizą statystyczną i ręczną backward elimination

**Kluczowe elementy:**
- `pd.get_dummies().values.astype(float)` → przygotowanie danych
- `sm.add_constant()` → dodanie intercept
- `sm.OLS().fit()` → model OLS
- `ols.summary()` → statystyki (p-value, R²)
- Ręczna backward elimination → krok po kroku usuwanie nieistotnych zmiennych

**Interpretacja p-value:**
- **p < 0.05** → istotna statystycznie ✅
- **p ≥ 0.05** → nieistotna (usuń) ❌

**Proces selekcji:**
1. Pełny model → sprawdź p-value
2. Usuń zmienną z najwyższym p ≥ 0.05 (ręcznie)
3. Powtórz dla nowego modelu

---

## 📌 P11: Automatyczna backward elimination

**Co robi:** Automatyczna selekcja zmiennych w pętli while

**Kluczowe elementy:**
- `while True:` → automatyczna pętla eliminacji
- `max(ols.pvalues)` → najwyższe p-value
- `np.argmax()` → indeks zmiennej z najwyższym p-value
- `np.delete(array, idx, axis=1)` → usunięcie kolumny
- `ols.save('model.pickle')` → zapis modelu do pliku

**Proces automatyczny:**
1. Dopasuj model → znajdź max p-value
2. Jeśli max p-value > 0.05 → usuń zmienną
3. Powtórz, dopóki wszystkie zmienne są istotne (p ≤ 0.05)

**Różnica od P10:**
- P10: ręczne usuwanie (3 kroki)
- P11: automatyczna pętla (działa dla dowolnej liczby zmiennych)

---

## 📌 P12: Regresja wielomianowa

**Co robi:** Uchwycenie nieliniowej zależności (wielomianowej) przez rozszerzenie cech (X, X², X³) i zwykłą regresję liniową

**Kluczowe elementy:**
- `np.random.seed(42)` → powtarzalność danych i szumu
- `X.reshape(n, 1)` → kształt 2D (próbki × cechy) dla scikit-learn
- Regresja liniowa na jednej cesze → słabe R² przy zależności wielomianowej
- `PolynomialFeatures(degree=k)` → tworzy cechy 1, X, X², …; potem `LinearRegression`
- `r2_score(y, y_pred)` / `score()` → ocena dopasowania

**Zasada:** Model pozostaje liniowy względem parametrów; nieliniowość wynika z transformacji cech. Przy wysokim stopniu i małej liczbie danych – ryzyko przeuczenia (regularyzacja lub niższy stopień).

---

## 📌 P13: Regresja drzewa decyzyjnego

**Co robi:** Model regresji, który dzieli oś cechy na przedziały i w każdym przewiduje stałą (średnią); przy nieliniowej zależności daje „schodkową” krzywą zamiast prostej.

**Kluczowe elementy:**
- `DecisionTreeRegressor(max_depth=k)` → drzewo o ograniczonej głębokości
- `fit(data, target)`, `predict(plot_data)` → API jak w LinearRegression
- `plot_tree(regressor, filled=True, rounded=True, feature_names=[...])` → wizualizacja struktury drzewa
- Większe max_depth → więcej schodków, lepsze dopasowanie, większe ryzyko przeuczenia

**Porównanie:** Regresja liniowa = jedna prosta; drzewo = schodki dopasowane do krzywej. Do wizualizacji krzywej używa się gęstej siatki punktów (np. np.arange().reshape(-1, 1)).

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

**Przed P11:** `while True` z `break`, `np.argmax()` do znajdowania indeksu, `np.delete()` do usuwania kolumn

**Przed P12:** `reshape(n, 1)` dla jednej cechy, regresja wielomianowa = rozszerzenie cech + LinearRegression, R² przy nieliniowej zależności

**Przed P13:** DecisionTreeRegressor(max_depth=k), plot_tree do wizualizacji struktury, krzywa predykcji = schodki; max_depth kontroluje złożoność i przeuczenie

---

## 🎯 Najważniejsze zasady

1. **random_state=42** → powtarzalność
2. **train/test split** → zawsze przed trenowaniem
3. **drop_first=True** → unika kolinearności
4. **EDA przed modelowaniem** → zrozum dane
5. **score(train) vs score(test)** → wykrycie overfittingu
6. **p-value < 0.05** → zmienna istotna

---

> **Użycie:** Przeczytaj sekcję dla danej lekcji przed zajęciami. Pełne wyjaśnienia w summary_p6.md – summary_p12.md, szczegóły techniczne w cheat_sheet.md.
