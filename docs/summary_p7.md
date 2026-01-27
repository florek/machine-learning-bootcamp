# Regresja liniowa z scikit-learn – syntetyczne dane i wizualizacja

Ten plik dokumentuje użycie **scikit-learn** do regresji liniowej na syntetycznych danych z wizualizacją wyników.

---

## 1. Konfiguracja środowiska

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set(font_scale=1.3)
```

**Dlaczego `random_state=42`?**

* zapewnia **powtarzalność** eksperymentów
* te same dane = te same wyniki
* kluczowe dla debugowania i porównań

**Seaborn `font_scale`:**

* zwiększa czytelność wykresów
* lepsze prezentacje i dokumentacja

---

## 2. Generowanie danych syntetycznych

```python
from sklearn.datasets import make_regression

data, target = make_regression(
    n_samples=100, 
    n_features=1, 
    n_informative=1, 
    noise=10, 
    random_state=42
)
```

**Parametry `make_regression`:**

* `n_samples=100` → liczba obserwacji
* `n_features=1` → jedna cecha (prosta regresja)
* `n_informative=1` → wszystkie cechy są informatywne
* `noise=10` → poziom szumu (jak bardzo dane odbiegają od prostej)
* `random_state=42` → powtarzalność

**Zwracane wartości:**

* `data` → macierz cech (X)
* `target` → wektor wartości docelowych (y)

**Zastosowanie:**

* szybkie testowanie modeli
* kontrola nad złożonością problemu
* idealne do nauki i demonstracji

---

## 3. Trenowanie modelu regresji liniowej

```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(data, target)
```

**Co się dzieje:**

* model dopasowuje prostą do danych
* minimalizuje błąd średniokwadratowy (MSE)
* oblicza współczynniki optymalne

**API scikit-learn:**

* `fit()` → trenowanie
* `predict()` → predykcje
* `score()` → ocena jakości

---

## 4. Ocena modelu

```python
regressor.score(data, target)
```

**Co zwraca `score()`:**

* **R² (współczynnik determinacji)**
* zakres: od -∞ do 1
* 1.0 = idealne dopasowanie
* 0.0 = model nie lepszy niż średnia
* < 0 = model gorszy niż średnia

**Uwaga:**

* tutaj oceniamy na danych treningowych
* w praktyce zawsze oceniaj na danych testowych!

---

## 5. Predykcje

```python
y_pred = regressor.predict(data)
```

**Co to robi:**

* dla każdej obserwacji zwraca przewidywaną wartość
* używa dopasowanej prostej: `y = intercept + coef * x`

---

## 6. Wizualizacja wyników (matplotlib)

```python
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, y_pred, color='red', label='model')
plt.legend()
plt.show()
```

**Elementy wykresu:**

* `scatter()` → punkty danych (rzeczywiste wartości)
* `plot()` → linia regresji (przewidywania modelu)
* `legend()` → opis elementów wykresu

**Interpretacja:**

* im bliżej punktów jest linia, tym lepszy model
* punkty rozrzucone wokół linii = szum w danych

---

## 7. Współczynniki modelu

```python
regressor.coef_
regressor.intercept_
```

**Co to znaczy:**

* `coef_` → współczynnik kierunkowy (nachylenie prostej)
* `intercept_` → wyraz wolny (punkt przecięcia z osią Y)

**Równanie modelu:**

```
y = intercept_ + coef_[0] * x
```

---

## 8. Ręczne rysowanie linii regresji

```python
plt.plot(
    data, 
    regressor.intercept_ + regressor.coef_[0] * data, 
    color='red', 
    label='model'
)
```

**Dlaczego to działa:**

* to jest **dokładnie to samo** co `predict()`
* pokazuje matematyczną naturę modelu
* potwierdza, że model to po prostu prosta

**Równanie matematyczne:**

```
y_pred = intercept + coef * x
```

---

## 9. Atrybuty modelu LinearRegression

```python
dir(regressor)
```

**Najważniejsze atrybuty:**

* `coef_` → współczynniki (dla każdej cechy)
* `intercept_` → wyraz wolny
* `fit()` → trenowanie
* `predict()` → predykcje
* `score()` → ocena jakości

**Konwencja scikit-learn:**

* atrybuty zakończone `_` są obliczane podczas `fit()`
* metody bez `_` są wywoływalne

---

## 10. Porównanie: scikit-learn vs implementacja ręczna

**scikit-learn (p7.py):**

* gotowy model
* walidacja danych
* stabilność numeryczna
* spójne API

**Implementacja ręczna (p6.py):**

* pełna kontrola nad algorytmem
* zrozumienie matematyki
* gradient descent krok po kroku

**Kiedy używać:**

* **scikit-learn** → produkcja, szybkie prototypy
* **ręczna implementacja** → nauka, zrozumienie, custom algorytmy

---

## 11. TL;DR

* `make_regression()` → szybkie generowanie danych testowych
* `LinearRegression().fit()` → trenowanie w jednej linii
* `coef_` i `intercept_` → parametry dopasowanej prostej
* `predict()` = `intercept + coef * x`
* wizualizacja → kluczowa dla zrozumienia modelu
* scikit-learn = wygoda + stabilność + spójność

> Regresja liniowa to fundament ML – prosta, szybka, interpretowalna. Zaczynaj od niej, zanim sięgniesz po bardziej złożone modele.
