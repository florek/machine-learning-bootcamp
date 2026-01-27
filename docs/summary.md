# Data Science Handbook – przygotowanie danych (Pandas + scikit-learn)

---

## 1. Braki danych (NaN)

### Reprezentacja braków

```python
np.nan
```

* standardowa reprezentacja braków danych w NumPy i Pandas
* większość algorytmów ML **nie obsługuje wartości NaN**
* **braki muszą zostać obsłużone przed trenowaniem modelu** (inaczej błąd lub zniekształcone wyniki)

**Złota zasada:** brak danych to też informacja – najpierw zrozum *dlaczego* brakuje wartości, a dopiero potem je uzupełniaj.

---

## 2. Imputacja danych (`SimpleImputer`)

```python
from sklearn.impute import SimpleImputer
```

Imputacja polega na **zastąpieniu braków danych wartościami zastępczymi**, w sposób kontrolowany i powtarzalny.

---

### 2.1 Imputacja średnią (cechy numeryczne)

```python
imputer = SimpleImputer(strategy="mean")
df['weight'] = imputer.fit_transform(df[['weight']])
```

**Stosuj gdy:**

* cecha jest ciągła (np. waga, wzrost, cena)
* braki są losowe
* nie ma sensownej wartości domyślnej

**Uwaga:**

* średnia może być wrażliwa na outliery
* w takich przypadkach rozważ `median`

---

### 2.2 Imputacja stałą wartością (numeryczne)

```python
imputer = SimpleImputer(strategy="constant", fill_value=99.0)
df['price'] = imputer.fit_transform(df[['price']])
```

**Stosuj gdy:**

* brak danych sam w sobie niesie informację
* chcesz jawnie zaznaczyć „brak” (placeholder)
* planujesz później stworzyć flagę typu `is_missing`

---

### 2.3 Imputacja stałą wartością (kategorie)

```python
imputer = SimpleImputer(strategy="constant", fill_value='L')
df['size'] = imputer.fit_transform(df[['size']]).ravel()
```

**Dlaczego `ravel()`?**

* `SimpleImputer` zwraca tablicę 2D
* kolumna Pandas musi być 1D

**Dobra praktyka:**

* używaj jawnych etykiet typu `"UNKNOWN"`, `"MISSING"`

---

## 3. Braki danych w szeregach czasowych

### Dane czasowe

```python
pd.date_range(start, end, periods)
```

Szeregi czasowe rządzą się innymi prawami niż dane statyczne – **kolejność obserwacji ma znaczenie**.

### Metody uzupełniania

```python
fillna(0)
fillna(mean)
interpolate()
fillna(method='bfill')
fillna(method='ffill')
```

**Zasady praktyczne:**

* dane czasowe → interpolacja / `ffill` / `bfill`
* dane statyczne → `mean` / `median` / `constant`
* nigdy nie mieszaj przyszłości z przeszłością (data leakage!)

---

## 4. Feature engineering – generowanie nowych cech

Feature engineering to **jeden z najważniejszych etapów ML** – często ważniejszy niż sam wybór algorytmu.

---

### 4.1 Cechy czasowe

```python
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
```

**Cel:**

* wydobycie informacji ukrytej w dacie
* umożliwienie modelowi uczenia się sezonowości i trendów

**Tip:**

* często warto dodać też `dayofweek`, `is_weekend`

---

## 5. Dyskretyzacja zmiennych ciągłych

```python
pd.cut(df.height, bins=3)
pd.cut(df.height, bins=(160, 175, 180, 195))
pd.cut(df.height, bins=(...), labels=[...])
```

**Zastosowanie:**

* zamiana wartości ciągłych na kategorie
* uproszczenie problemu decyzyjnego
* poprawa interpretowalności modelu

---

### One-hot encoding

```python
pd.get_dummies(df, drop_first=True)
```

**Dlaczego `drop_first=True`?**

* unikasz pułapki zmiennych fikcyjnych
* redukujesz nadmiarowość cech

---

## 6. Ekstrakcja cech

### 6.1 Listy → cechy liczbowe

```python
df['lang_number'] = df['lang'].apply(len)
```

* zamiana złożonej struktury na prostą cechę liczbową
* często bardzo skuteczna technika

---

### 6.2 Flagi binarne

```python
df['PL_flag'] = df['lang'].apply(lambda x: 1 if 'PL' in x else 0)
```

* sygnał typu TAK/NIE
* bardzo dobrze współpracuje z modelami liniowymi

---

### 6.3 Tekst → cechy

```python
df.website.str.split('.', expand=True)
```

* ekstrakcja domeny
* ekstrakcja rozszerzenia
* pierwszy krok do feature engineeringu tekstu

---

## 7. Dane wbudowane w scikit-learn

```python
from sklearn.datasets import load_iris, load_breast_cancer
```

* gotowe datasety do nauki i testów
* struktura: `data`, `target`, `DESCR`
* idealne do eksperymentów i porównań modeli

---

## 8. Rozdzielenie cech i targetu

```python
target = df.pop('target')
```

**Zasada:**

* cechy (X) i target (y) zawsze trzymane osobno
* zapobiega przypadkowemu „podglądaniu” odpowiedzi przez model

---

## 9. Podział na zbiór treningowy i testowy

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

**Dlaczego to ważne:**

* `test_size` → kontrola ilości danych testowych
* `random_state` → powtarzalność eksperymentów
* `stratify=y` → zachowanie proporcji klas

Bez poprawnego podziału **nie da się rzetelnie ocenić modelu**.

---

## 10. Regresja liniowa od zera (NumPy)

Ten przykład pokazuje **matematyczne podstawy regresji liniowej** – bez użycia gotowych modeli ML.

### Dane wejściowe

```python
import numpy as np

X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)
```

* `X1` – cecha (np. lata pracy)
* `Y` – target (wynagrodzenie)
* `m` – liczba próbek

---

### Przygotowanie macierzy cech

```python
X1 = X1.reshape(m, 1)
bias = np.ones((m, 1))
X = np.append(bias, X1, axis=1)
```

* `reshape(m, 1)` → kolumna
* `bias` → wyraz wolny (intercept)
* `X` → pełna macierz projektu

---

### Równanie normalne (Normal Equation)

```python
L = np.linalg.inv(np.dot(X.T, X))
P = np.dot(X.T, Y)
theta = np.dot(L, P)
```

To implementacja wzoru:

**θ = (XᵀX)⁻¹ Xᵀy**

* `theta[0]` → intercept
* `theta[1]` → współczynnik kierunkowy

---

## 11. Regresja liniowa w scikit-learn

```python
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X1, Y)

regression.intercept_
regression.coef_[0]
```

**Co się dzieje pod spodem:**

* scikit-learn robi dokładnie to samo
* ale dodaje stabilność numeryczną i walidację
* API jest spójne z całym ekosystemem ML

---

## 11.1 Generowanie danych syntetycznych

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

**Zastosowanie:**

* szybkie testowanie modeli bez zbierania danych
* kontrola nad złożonością problemu (szum, liczba cech)
* idealne do nauki i demonstracji

**Parametry:**

* `n_samples` → liczba obserwacji
* `n_features` → liczba cech
* `noise` → poziom szumu w danych
* `random_state` → powtarzalność

---

## 11.2 Ocena i wizualizacja modelu

```python
regressor.score(data, target)
y_pred = regressor.predict(data)
```

**`score()` zwraca R²:**

* 1.0 = idealne dopasowanie
* 0.0 = model nie lepszy niż średnia
* < 0 = model gorszy niż średnia

**Wizualizacja:**

* `scatter()` → punkty danych
* `plot()` → linia regresji
* porównanie pokazuje jakość dopasowania

**Równanie modelu:**

```
y = intercept_ + coef_[0] * x
```

---

## 12. TL;DR – przygotowanie danych

* NaN → zawsze świadoma imputacja lub interpolacja
* Dane czasowe ≠ dane statyczne
* Feature engineering często decyduje o jakości modelu
* `pd.cut` → dyskretyzacja
* `get_dummies` → encoding kategorii
* Regresja liniowa = algebra liniowa + dane
* scikit-learn = wygodne, bezpieczne API

> Dobry model zaczyna się od dobrych danych – algorytm jest dopiero na końcu.
