# SVM w Pythonie na zbiorze Iris — praktyczne użycie `SVC`

## 1. Cel lekcji

W tej części wykorzystujemy **Support Vector Machine** w praktyce, używając języka Python i biblioteki **scikit-learn**.

Będziemy:

1. ładować dane Iris,
2. wybierać konkretne klasy i cechy,
3. dzielić dane na zbiór treningowy i testowy,
4. standaryzować cechy,
5. trenować klasyfikator `SVC`,
6. sprawdzać accuracy,
7. wizualizować granice decyzyjne,
8. porównywać kernel liniowy i RBF,
9. sprawdzać działanie dla dwóch oraz trzech klas.

---

## 2. Załadowanie danych Iris

Na początku importujemy standardowy stack data science'owy oraz zbiór Iris:

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

Następnie pobieramy dane, zmienną docelową, nazwy cech i nazwy klas.

```python
data = iris.data
target = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
```

---

## 3. Ograniczenie problemu do dwóch klas

Na początku zostawiamy tylko:

```text
klasa 0
klasa 1
```

Dzięki temu mamy łącznie około:

```text
100 próbek
```

To upraszcza problem i pozwala dobrze zobaczyć działanie liniowej maszyny wektorów nośnych.

---

## 4. Wybór dwóch cech

Z całego zbioru wybieramy tylko dwie kolumny:

```text
petal length
sepal width
```

W materiale są to kolumny o indeksach:

```text
2 i 1
```

Dwie cechy wybieramy głównie po to, aby móc później narysować granice decyzyjne na wykresie 2D.

---

## 5. Podział na zbiór treningowy i testowy

Następnie używamy:

```python
from sklearn.model_selection import train_test_split
```

Przykład:

```python
X_train, X_test, y_train, y_test = train_test_split(
    data,
    target
)
```

Model uczy się na danych treningowych, a jego działanie sprawdzamy na danych testowych.

---

## 6. Standaryzacja cech

Przy SVM ważne jest przeskalowanie danych.

Do tego używamy:

```python
from sklearn.preprocessing import StandardScaler
```

Tworzymy scaler:

```python
scaler = StandardScaler()
```

Najpierw dopasowujemy go tylko do danych treningowych:

```python
scaler.fit(X_train)
```

Następnie transformujemy oba zbiory:

```python
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Najważniejsza zasada:

> **Scaler uczymy tylko na danych treningowych.**

Dane testowe tylko transformujemy tym samym scalerem.

---

## 7. Dlaczego nie dopasowujemy scalera do testu?

Gdybyśmy wykonali `fit()` również na danych testowych, model pośrednio uzyskałby informacje o zbiorze testowym.

To byłby przykład:

## data leakage

Poprawny schemat:

```text
X_train
   ↓
scaler.fit()
   ↓
scaler.transform(X_train)

X_test
   ↓
scaler.transform(X_test)
```

---

## 8. Import klasyfikatora SVM

Do klasyfikacji używamy klasy:

```python
from sklearn.svm import SVC
```

`SVC` oznacza:

## Support Vector Classifier

---

## 9. Model liniowy

Na początku budujemy klasyfikator liniowy:

```python
classifier = SVC(
    C=1,
    kernel="linear"
)
```

Parametr `C` pozostaje na wartości domyślnej, a kernel ustawiamy na:

```text
linear
```

---

## 10. Trenowanie modelu

Model dopasowujemy do danych treningowych:

```python
classifier.fit(
    X_train,
    y_train
)
```

---

## 11. Ocena modelu

Następnie sprawdzamy wynik na danych testowych:

```python
classifier.score(
    X_test,
    y_test
)
```

W przykładzie z dwiema klasami model uzyskał:

```text
100% accuracy
```

Nie jest to zaskakujące, ponieważ wybrane klasy i cechy były bardzo łatwo liniowo separowalne.

---

## 12. Granice decyzyjne

Po treningu możemy narysować granice decyzyjne.

Do kolorowych regionów klas w przestrzeni 2D często używamy `plot_decision_regions` (np. z biblioteki mlxtend).

Dla kernela liniowego granica jest prostą linią.

Schematycznie:

```text
● ● ● ●

-------------

○ ○ ○ ○
```

Dzięki temu możemy zobaczyć, jak model podzielił przestrzeń cech.

---

## 13. Wizualizacja danych treningowych

Najpierw oglądamy granice na zbiorze treningowym.

Widać, że model bardzo dobrze oddziela obie klasy.

---

## 14. Wizualizacja danych testowych

Następnie rysujemy dane testowe i sprawdzamy, jak model radzi sobie z próbkami, których wcześniej nie widział.

W przykładzie również nie popełnia błędów.

---

## 15. Kernel RBF

Następnie budujemy model nieliniowy:

```python
classifier = SVC(
    C=1,
    kernel="rbf"
)
```

RBF pozwala tworzyć bardziej złożone granice decyzyjne.

---

## 16. Linear vs RBF

Dla dwóch łatwo separowalnych klas oba modele mogą osiągnąć taki sam wynik.

Różnica pojawia się głównie w kształcie granicy:

```text
linear
→ prosta granica

RBF
→ bardziej złożona, zakrzywiona granica
```

---

## 17. Powrót do trzech klas

W dalszej części usuwamy ograniczenie do dwóch klas.

Znowu wykorzystujemy:

```text
klasa 0
klasa 1
klasa 2
```

czyli pełny zbiór Iris.

---

## 18. Model liniowy dla trzech klas

Po ponownym uruchomieniu notebooka model liniowy osiąga około:

```text
94,6% accuracy
```

Problem staje się trudniejszy, ponieważ część klas nakłada się na siebie.

---

## 19. Wynik na zbiorze testowym

W przykładzie wynik na danych testowych był nieco lepszy niż na treningowych.

To może się zdarzyć.

Często mamy:

```text
train accuracy > test accuracy
```

ale nie jest to reguła, szczególnie przy bardzo małych zbiorach.

---

## 20. Dlaczego test może być lepszy niż train?

Przy małej liczbie danych wszystko zależy od konkretnego losowego podziału.

Może się zdarzyć, że:

- trudniejsze próbki trafią do train,
- łatwiejsze próbki trafią do test.

Wtedy accuracy na teście może wyjść wyższe.

---

## 21. RBF dla trzech klas

Następnie używamy kernela:

```python
kernel="rbf"
```

W przykładzie wynik na danych treningowych wyniósł około:

```text
93,75%
```

Granice decyzyjne są nieliniowe.

---

## 22. RBF na zbiorze testowym

W przykładzie accuracy na danych testowych wyniosło około:

```text
97,37%
```

Ponownie wynik testowy okazał się lepszy niż treningowy.

Najważniejszy powód:

> zbiór danych jest mały.

---

## 23. Mała liczba danych zniekształca ocenę

Iris zawiera tylko:

```text
150 próbek
```

Po podziale zbiór testowy jest jeszcze mniejszy.

Dlatego pojedyncze próbki mogą mocno zmieniać wynik procentowy.

---

## 24. Losowość podziału danych

Jeżeli nie ustawimy:

```python
random_state
```

w `train_test_split`, przy każdym uruchomieniu możemy dostać inny podział danych.

Ta sama próbka może raz trafić do:

```text
train
```

a innym razem do:

```text
test
```

---

## 25. `random_state`

Aby wynik był powtarzalny, możemy ustawić:

```python
random_state=42
```

Przykład:

```python
X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    random_state=42
)
```

Dzięki temu przy każdym uruchomieniu dostaniemy ten sam podział.

---

## 26. Po co używać `random_state`?

To ważne przy:

- porównywaniu modeli,
- debugowaniu,
- eksperymentach,
- powtarzalności wyników.

Bez ustalonego ziarna losowego różnice w accuracy mogą wynikać po prostu z innego podziału danych.

---

## 27. Cały workflow

```text
load_iris()
   ↓
wybór klas
   ↓
wybór 2 cech
   ↓
train_test_split()
   ↓
StandardScaler.fit(X_train)
   ↓
transform(X_train)
transform(X_test)
   ↓
SVC(kernel="linear")
   ↓
fit()
   ↓
score()
   ↓
granice decyzyjne
   ↓
SVC(kernel="rbf")
   ↓
porównanie wyników
```

---

## 28. Najważniejsze elementy kodu

### Załadowanie danych

```python
from sklearn.datasets import load_iris
```

### Podział danych

```python
from sklearn.model_selection import train_test_split
```

### Standaryzacja

```python
from sklearn.preprocessing import StandardScaler
```

### Model

```python
from sklearn.svm import SVC
```

### Model liniowy

```python
SVC(
    C=1,
    kernel="linear"
)
```

### Model RBF

```python
SVC(
    C=1,
    kernel="rbf"
)
```

---

## 29. Co warto zapamiętać?

1. **SVM często wymaga standaryzacji cech.**
2. **Scaler dopasowujemy tylko do danych treningowych.**
3. **Dane testowe jedynie transformujemy.**
4. **`SVC` służy do klasyfikacji metodą SVM.**
5. **Kernel `linear` buduje liniowe granice decyzyjne.**
6. **Kernel `rbf` pozwala budować granice nieliniowe.**
7. **Dwie klasy Iris są w tym przykładzie bardzo łatwo separowalne.**
8. **Przy trzech klasach problem staje się trudniejszy.**
9. **Mały zbiór danych może dawać zmienne wyniki accuracy.**
10. **Brak `random_state` oznacza inny podział danych przy kolejnych uruchomieniach.**
11. **Wynik na teście może czasem być lepszy niż na treningu.**
12. **Przy małych zbiorach trzeba ostrożnie interpretować pojedynczy wynik accuracy.**

---

## 30. Najważniejsze zdanie

> **Praktyczne użycie SVM to nie tylko wybór kernela, ale też poprawne przygotowanie danych, standaryzacja, uczciwy podział train/test i ostrożna interpretacja accuracy.**
