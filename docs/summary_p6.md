# Gradient Descent – prosta regresja liniowa (krowie na rowie)

Ten plik tłumaczy **linijka po linijce**, co dokładnie dzieje się w poniższym kodzie. To jest **ręczna implementacja regresji liniowej** uczonej metodą **gradient descent**.

---

## 1. Import bibliotek

```python
import numpy as np
import pandas as pd
import plotly.express as px
```

* **NumPy** – liczenie wektorów, macierzy, pochodnych
* **Pandas** – trzymanie historii uczenia w tabeli
* **Plotly** – rysowanie wykresów (jak parametry się uczą)

---

## 2. Dane wejściowe

```python
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
```

Interpretacja:

* **X1** → lata pracy
* **Y** → wynagrodzenie

Zakładamy model:

> im więcej lat pracy, tym większa pensja

---

## 3. Liczba próbek

```python
m = len(X1)
```

* `m = 6`
* tyle mamy obserwacji (punktów danych)

---

## 4. Zmiana kształtu danych (reshape)

```python
X1 = X1.reshape(m, 1)
Y = Y.reshape(-1, 1)
```

Dlaczego?

* gradient descent **operuje na macierzach**
* chcemy mieć kolumny, a nie listy

Efekt:

* `X1.shape == (6, 1)`
* `Y.shape == (6, 1)`

---

## 5. Dodanie biasu (wyraz wolny)

```python
bias = np.ones((m, 1))
X = np.append(bias, X1, axis=1)
```

Bias = kolumna jedynek:

```
[1, 1]
[1, 2]
[1, 3]
[1, 4]
[1, 5]
[1, 6]
```

Dlaczego?

Model matematyczny:

```
Y = w0 * 1 + w1 * X
```

* `w0` → intercept (punkt startowy)
* `w1` → współczynnik (nachylenie prostej)

---

## 6. Parametry uczenia

```python
eta = 0.01
weights = np.random.randn(2, 1)
```

* `eta` → learning rate (jak duży krok robimy)
* `weights` → losowy start:

  * `weights[0]` → intercept
  * `weights[1]` → współczynnik przy X

---

## 7. Gradient Descent – serce algorytmu

```python
for i in range(3000):
    gradient = (2 / m) * X.T.dot(X.dot(weights) - Y)
    weights = weights - eta * gradient
```

Co tu się dzieje:

### a) Predykcja

```
X.dot(weights)
```

→ aktualne przewidywane pensje

### b) Błąd

```
X.dot(weights) - Y
```

→ o ile się mylimy dla każdego punktu

### c) Gradient

```
(2 / m) * X.T.dot(błąd)
```

→ kierunek, w którym trzeba **zmniejszyć błąd MSE**

### d) Aktualizacja wag

```
weights = weights - eta * gradient
```

→ robimy mały krok w dół zbocza błędu

---

## 8. Zapisywanie historii uczenia

```python
intercept.append(weights[0][0])
coef.append(weights[1][0])
```

Po co?

* żeby **zobaczyć, jak model się uczy**
* jak stabilizują się parametry

---

## 9. Wynik końcowy

```python
print(weights)
```

To jest gotowy model:

```
Y = intercept + coef * X
```

---

## 10. DataFrame z historią

```python
df = pd.DataFrame({
    'intercept': intercept,
    'coef': coef
})
```

Każdy wiersz = jeden krok gradient descent

---

## 11. Wizualizacja uczenia

### Intercept

```python
px.line(df, y='intercept')
```

→ pokazuje, jak stabilizuje się punkt przecięcia z osią Y

### Współczynnik

```python
px.line(df, y='coef')
```

→ pokazuje, jak zmienia się nachylenie prostej

---

## 12. TL;DR (mega skrót)

* masz dane: lata pracy → pensja
* zgadujesz losową prostą
* liczysz, jak bardzo się myli
* poprawiasz prostą **3000 razy**
* na końcu dostajesz sensowny model

To jest **dokładnie to**, co robi sklearn – tylko bez magii 🎯
