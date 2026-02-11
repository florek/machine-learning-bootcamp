# Regresja wielomianowa

Ten plik opisuje **regresję wielomianową**: sytuację, gdy zależność między cechą a zmienną docelową jest nieliniowa (np. wielomianowa), a do jej uchwycenia używa się rozszerzenia cech i zwykłej regresji liniowej.

---

## 1. Kiedy regresja liniowa nie wystarcza

Gdy prawdziwa zależność jest np. wielomianowa (kwadratowa, sześcienna), prosta regresja liniowa (jedna cecha X) dopasowuje prostą i **słabo opisuje dane**. Metryka R² jest wtedy niska, a wykres pokazuje systematyczne odchylenia (reszty nie są losowe).

**Rozwiązanie:** zamiast jednej cechy X użyć cech pochodnych: X, X², X³, … (oraz ewentualnie stała). Model pozostaje **liniowy względem parametrów**, ale **nieliniowy względem oryginalnej cechy**.

---

## 2. Konfiguracja i dane

Standardowa konfiguracja środowiska:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
sns.set(font_scale=1.3)
```

**Ziarno (seed)** zapewnia powtarzalność generowanych danych i szumu.

Dane syntetyczne z zależnością wielomianową i szumem: wektor cechy X oraz wektor y obliczony np. jako wielomian plus szum. **Kształt X** musi być 2D (próbki × cechy), np. `X.reshape(n, 1)`, bo scikit-learn oczekuje macierzy.

---

## 3. Regresja liniowa na oryginalnej cesze

Dopasowanie `LinearRegression` do jednej cechy X daje prostą. Przy nieliniowej zależności **R² jest niskie**, a wykres (punkty vs linia) pokazuje wyraźne niedopasowanie (np. krzywą zamiast prostej).

---

## 4. Regresja wielomianowa w scikit-learn

**Idee:**

- Stworzyć dodatkowe cechy: 1, X, X², X³ (do wybranego stopnia).
- Użyć **tego samego** `LinearRegression` na rozszerzonym zbiorze cech.
- Model jest wciąż liniowy względem współczynników; nieliniowość wynika z transformacji cech.

**Składnia (koncepcyjnie):**

- `PolynomialFeatures(degree=k)` – tworzy potęgi cech do stopnia k (z opcją kolumny jedynek).
- `fit_transform(X)` na wejściu 2D (np. jedna kolumna) daje macierz z kolumnami: stała (opcjonalnie), X, X², …, X^k.
- Następnie `LinearRegression().fit(X_poly, y)` i `predict(X_poly)`.

**Uwaga:** przy wysokim stopniu wielomianu i małej liczbie obserwacji rośnie ryzyko **przeuczenia**. Wtedy rozważ regularyzację (Ridge, Lasso) lub niższy stopień.

---

## 5. Ocena i wizualizacja

- **R²** (`r2_score(y, y_pred)` lub `regressor.score(X, y)`) – im wyższe, tym lepsze dopasowanie (na tych samych danych; do generalizacji potrzebny zbiór testowy).
- **Wykres:** najpierw punkty (scatter), potem krzywa/linia modelu (plot), żeby widać było zarówno dane, jak i dopasowanie.

---

## 6. Podsumowanie

- Regresja wielomianowa = rozszerzenie cech (X, X², …) + zwykła regresja liniowa.
- Stosuj gdy zależność jest nieliniowa; unikaj zbyt wysokiego stopnia przy małej liczbie danych.
- Ziarno losowe i kształt macierzy (reshape) są istotne dla powtarzalności i zgodności ze scikit-learn.
