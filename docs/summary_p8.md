# Podział danych na zbiór treningowy i testowy – ocena modelu

Ten plik dokumentuje **podział danych na zbiór treningowy i testowy** oraz **ocenę modelu regresji liniowej** na obu zbiorach.

---

## 1. Konfiguracja środowiska

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)
sns.set(font_scale=1.3)
```

**Dlaczego to ważne:**

* `random_state=42` → powtarzalność podziału danych
* bez tego każdy uruchomienie da inny podział
* niemożliwe porównanie wyników między eksperymentami

---

## 2. Generowanie danych syntetycznych

```python
data, target = make_regression(
    n_samples=100, 
    n_features=1, 
    n_informative=1, 
    noise=15.0, 
    random_state=42
)
```

**Parametry:**

* `n_samples=100` → 100 obserwacji
* `n_features=1` → jedna cecha (prosta regresja)
* `noise=15.0` → poziom szumu (np. wyższy niż w prostym przykładzie regresji)
* `random_state=42` → powtarzalność

**Zwracane wartości:**

* `data` → macierz cech (X)
* `target` → wektor wartości docelowych (y)

---

## 3. Podział danych na zbiór treningowy i testowy

```python
X_train, X_test, y_train, y_test = train_test_split(
    data, 
    target, 
    test_size=0.25
)
```

**Co się dzieje:**

* dane są losowo podzielone na dwa zbiory
* `test_size=0.25` → 25% danych do testów, 75% do treningu
* domyślnie `random_state=None` → różny podział przy każdym uruchomieniu

**Wynik:**

* `X_train` → cechy treningowe (75 próbek)
* `X_test` → cechy testowe (25 próbek)
* `y_train` → wartości docelowe treningowe
* `y_test` → wartości docelowe testowe

**Złota zasada:**

* model **trenuje** na danych treningowych
* model **ocenia** na danych testowych
* nigdy nie oceniaj modelu na danych, na których trenował!

---

## 4. Wizualizacja podziału danych

```python
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='dane treningowe')
plt.scatter(X_test, y_test, label='dane testowe')
plt.legend()
plt.show()
```

**Co pokazuje:**

* rozkład danych treningowych i testowych
* czy podział jest reprezentatywny
* czy nie ma oczywistych różnic między zbiorami

**Interpretacja:**

* punkty powinny być wymieszane
* jeśli są wyraźnie oddzielone → problem z podziałem

---

## 5. Trenowanie modelu

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

**Co się dzieje:**

* model uczy się **tylko na danych treningowych**
* dopasowuje prostą do `X_train` i `y_train`
* nie widzi danych testowych!

**Dlaczego to ważne:**

* symuluje sytuację produkcyjną
* model musi działać na nowych, niewidzianych danych
* testowanie na danych treningowych = oszukiwanie

---

## 6. Ocena modelu na obu zbiorach

```python
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))
```

**Co zwraca `score()`:**

* **R² (współczynnik determinacji)**
* zakres: od -∞ do 1
* 1.0 = idealne dopasowanie
* 0.0 = model nie lepszy niż średnia

**Interpretacja wyników:**

* **score treningowy** → jak dobrze model pasuje do danych treningowych
* **score testowy** → jak dobrze model generalizuje na nowe dane

**Oczekiwane zachowanie:**

* score treningowy ≥ score testowy (zazwyczaj)
* jeśli różnica duża → **overfitting** (przeuczenie)
* jeśli oba niskie → **underfitting** (niedouczenie)

---

## 7. Wizualizacja modelu na danych treningowych

```python
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa zbiór treningowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='dane treningowe')
plt.plot(X_train, regressor.predict(X_train), color='red', label='model')
plt.legend()
plt.show()
```

**Co pokazuje:**

* punkty danych treningowych
* linię regresji dopasowaną do danych treningowych
* jak dobrze model pasuje do danych, na których trenował

**Interpretacja:**

* im bliżej punktów jest linia, tym lepsze dopasowanie
* punkty rozrzucone wokół linii = szum w danych

---

## 8. Wizualizacja modelu na danych testowych

```python
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa zbiór testowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_test, y_test, label='dane testowe')
plt.plot(X_test, regressor.predict(X_test), color='green', label='model')
plt.legend()
plt.show()
```

**Co pokazuje:**

* punkty danych testowych (niewidzianych przez model)
* predykcje modelu na danych testowych
* jak dobrze model generalizuje na nowe dane

**Interpretacja:**

* jeśli linia dobrze pasuje → model generalizuje dobrze
* jeśli linia źle pasuje → model nie nauczył się ogólnych wzorców

**Porównanie z wykresem treningowym:**

* podobne dopasowanie → dobry model
* dużo gorsze dopasowanie → overfitting

---

## 9. Analiza błędów predykcji

```python
y_pred = regressor.predict(X_test)

predictions_df = pd.DataFrame(data={
    'y_test': y_test, 
    'y_pred': y_pred
})
predictions_df['error'] = predictions_df['y_test'] - predictions_df['y_pred']
```

**Co to robi:**

* tworzy DataFrame z rzeczywistymi i przewidywanymi wartościami
* oblicza błąd jako różnicę: `y_test - y_pred`

**Interpretacja błędów:**

* dodatnie → model przewiduje za nisko
* ujemne → model przewiduje za wysoko
* blisko zera → dobre predykcje

---

## 10. Histogram błędów

```python
predictions_df['error'].plot(kind='hist', bins=50, figsize=(8, 6))
plt.show()
```

**Metoda `plot()` w pandas:**

* `Series.plot()` → wbudowana metoda pandas do tworzenia wykresów
* automatycznie używa matplotlib pod spodem
* wygodniejsza niż bezpośrednie użycie matplotlib dla danych z DataFrame/Series

**Parametr `kind`:**

* określa **typ wykresu** do narysowania
* dostępne opcje:
  * `'hist'` → histogram (rozkład wartości)
  * `'line'` → wykres liniowy
  * `'bar'` → wykres słupkowy
  * `'box'` → wykres pudełkowy (boxplot)
  * `'scatter'` → wykres punktowy
  * `'pie'` → wykres kołowy
  * `'area'` → wykres powierzchniowy

**Inne parametry:**

* `bins=50` → liczba przedziałów w histogramie
  * więcej bins → bardziej szczegółowy wykres
  * mniej bins → bardziej ogólny wykres
* `figsize=(8, 6)` → rozmiar wykresu (szerokość, wysokość w calach)
* `title='Tytuł'` → tytuł wykresu
* `xlabel='Etykieta X'` → etykieta osi X
* `ylabel='Etykieta Y'` → etykieta osi Y

**Co pokazuje:**

* rozkład błędów predykcji
* jak często występują różne wielkości błędów

**Interpretacja:**

* **rozkład normalny wokół zera** → dobry model
* **przesunięcie w prawo/lewo** → systematyczny błąd
* **duże ogony** → duże błędy (outliery)

**Idealny rozkład:**

* symetryczny wokół zera
* wąski (małe błędy)
* dzwonowy kształt (rozkład normalny)

**Dlaczego `_ =` przed `plot()`:**

* `plot()` zwraca obiekt Axes
* `_ =` oznacza, że ignorujemy zwracaną wartość
* nie jest konieczne, ale dobry styl (unika nieużywanych zmiennych)

---

## 11. Różnice: bez podziału vs z podziałem train/test

**Regresja bez podziału (szybkie prototypy):**

* trenowanie i ocena na tych samych danych
* nierealistyczna ocena jakości
* nie pokazuje, jak model działa na nowych danych

**Regresja z podziałem train/test:**

* trenowanie na danych treningowych
* ocena na danych testowych
* realistyczna ocena generalizacji
* analiza błędów predykcji

**Kiedy używać:**

* Bez podziału → szybkie prototypy, zrozumienie podstaw
* Z podziałem → właściwa ocena modelu, przygotowanie do produkcji

---

## 12. Metryki oceny modelu

**R² (współczynnik determinacji):**

* `score()` zwraca R²
* interpretacja: jaka część wariancji jest wyjaśniona przez model
* 1.0 = 100% wariancji wyjaśnionej

**Inne metryki (nie omawiane w tej lekcji, ale ważne):**

* **MSE (Mean Squared Error)** → średni błąd kwadratowy
* **MAE (Mean Absolute Error)** → średni błąd bezwzględny
* **RMSE (Root Mean Squared Error)** → pierwiastek z MSE

**Kiedy używać:**

* R² → ogólna ocena jakości
* MSE/MAE → interpretacja w jednostkach targetu
* RMSE → podobne do MSE, ale w jednostkach targetu

---

## 13. TL;DR

* `train_test_split()` → podział danych na train/test
* `test_size=0.25` → 25% danych do testów
* model trenuje na `X_train`, ocenia na `X_test`
* `score()` na train vs test → wykrycie overfittingu
* wizualizacja na obu zbiorach → zrozumienie generalizacji
* analiza błędów → szczegółowa ocena jakości predykcji
* histogram błędów → rozkład błędów (powinien być normalny wokół zera)

> Podział danych to fundament rzetelnej oceny modelu. Bez niego nie wiesz, czy model działa, czy tylko zapamiętał dane treningowe.
