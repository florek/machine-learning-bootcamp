# Gaussian Naive Bayes w Pythonie — praktyczny przykład ze spacerem

## 1. Cel lekcji

W tej części implementujemy w Pythonie **naiwny klasyfikator Bayesa**, dokładniej **Gaussian Naive Bayes**.

Korzystamy z tego samego przykładu co wcześniej:

```text
pogoda
temperatura
spacer
```

Chcemy zbudować model, który odpowie:

> Czy wyjść na spacer?

---

## 2. Dane wejściowe

Dane są zapisane w Pythonie jako trzy listy, a następnie tworzymy z nich obiekt `DataFrame`.

Po utworzeniu tabeli mamy trzy kolumny:

```text
pogoda
temperatura
spacer
```

Kolumna `spacer` jest zmienną docelową, a `pogoda` i `temperatura` są cechami.

---

## 3. Kodowanie zmiennej docelowej

Zmienna `spacer` zawiera wartości tekstowe:

```text
tak
nie
```

Do jej zakodowania używamy:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
```

W przykładzie otrzymujemy mapowanie:

```text
nie → 0
tak → 1
```

---

## 4. Kodowanie cech kategorycznych

Cechy `pogoda` i `temperatura` również są tekstowe.

Dlatego używamy:

```python
pd.get_dummies()
```

W materiale przekazujemy te dwie kolumny i stosujemy:

```python
drop_first=True
```

czyli usuwamy pierwszą kategorię z każdej grupy.

Przykład:

```python
pd.get_dummies(
    df,
    columns=["pogoda", "temperatura"],
    drop_first=True
)
```

---

## 5. Oddzielenie targetu od cech

Następnie oddzielamy zmienną docelową.

W materiale używana jest metoda:

```python
pop()
```

Przykład:

```python
target = df.pop("spacer")
```

Po tej operacji:

```text
X = zakodowana pogoda + temperatura
y = spacer
```

---

## 6. Import Gaussian Naive Bayes

Model importujemy z:

```python
from sklearn.naive_bayes import GaussianNB
```

Tworzymy instancję:

```python
model = GaussianNB()
```

---

## 7. Trenowanie modelu

Model dopasowujemy:

```python
model.fit(X, y)
```

Po wykonaniu `fit()` model jest dopasowany do danych.

---

## 8. Ocena modelu

W przykładzie otrzymujemy około:

```text
77,7% accuracy
```

Czyli model poprawnie klasyfikuje około 7 z 9 obserwacji w pokazanym zbiorze.

W materiale celem jest przede wszystkim pokazanie mechaniki działania modelu krok po kroku.

---

## 9. Predykcja dla pierwszego wiersza

Pierwszy wiersz odpowiada przypadkowi:

```text
pogoda = słonecznie
temperatura = ciepło
```

czyli temu samemu przykładowi, który wcześniej był liczony ręcznie.

Do predykcji używamy:

```python
model.predict(...)
```

Model zwraca:

```text
1
```

---

## 10. Co oznacza `1`?

Wcześniej `LabelEncoder` zakodował klasy jako:

```text
nie → 0
tak → 1
```

Dlatego:

```text
1
```

oznacza:

```text
tak
```

Czyli model przewiduje:

> wyjść na spacer.

---

## 11. Powrót z kodu liczbowego do etykiety

Możemy wykorzystać:

```python
encoder.classes_
```

aby odtworzyć tekstową nazwę klasy i zamiast `1` dostać:

```text
tak
```

---

## 12. `predict_proba()`

Jeżeli chcemy sprawdzić nie tylko klasę, ale również prawdopodobieństwa, używamy:

```python
model.predict_proba(...)
```

Model zwraca prawdopodobieństwa dla obu klas:

```text
P(nie)
P(tak)
```

W przykładzie prawdopodobieństwo klasy `tak` jest bardzo wysokie.

---

## 13. Dlaczego wynik różni się od ręcznego przykładu?

We wcześniejszym przykładzie ręcznie liczyliśmy Naive Bayes dla danych dyskretnych.

Tutaj używamy:

```text
Gaussian Naive Bayes
```

czyli wariantu z założeniem rozkładu normalnego.

Dlatego dokładne prawdopodobieństwa mogą się różnić.

---

## 14. Cały workflow

```text
dane w listach
↓
DataFrame
↓
LabelEncoder dla targetu
↓
get_dummies dla cech
↓
oddzielenie X i y
↓
GaussianNB()
↓
fit(X, y)
↓
score()
↓
predict()
↓
predict_proba()
```

---

## 15. Najważniejsze elementy kodu

### LabelEncoder

```python
from sklearn.preprocessing import LabelEncoder
```

### Kodowanie cech

```python
pd.get_dummies(
    df,
    columns=["pogoda", "temperatura"],
    drop_first=True
)
```

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
```

### Model

```python
model = GaussianNB()
```

### Trenowanie

```python
model.fit(X, y)
```

### Predykcja

```python
model.predict(X_sample)
```

### Prawdopodobieństwa

```python
model.predict_proba(X_sample)
```

---

## 16. Co warto zapamiętać?

1. **Target tekstowy można zakodować przez `LabelEncoder`.**
2. **Cechy kategoryczne można zakodować przez `get_dummies`.**
3. **`drop_first=True` usuwa pierwszą kategorię z kodowania dummy.**
4. **`pop()` może posłużyć do oddzielenia targetu od cech.**
5. **`GaussianNB` implementuje Gaussian Naive Bayes w scikit-learn.**
6. **Model trenujemy przez `fit()`.**
7. **`predict()` zwraca przewidywaną klasę.**
8. **`predict_proba()` zwraca prawdopodobieństwa klas.**
9. **Dla „słonecznie + ciepło” model przewiduje `tak`.**
10. **Wynik może różnić się od wcześniejszego ręcznego przykładu, ponieważ tutaj używany jest wariant Gaussian Naive Bayes.**

---

## 17. Najważniejsze zdanie

> **W praktycznej implementacji Naive Bayes najpierw zamieniamy dane kategoryczne na liczby, następnie trenujemy `GaussianNB`, a potem możemy przewidzieć klasę przez `predict()` i sprawdzić jej prawdopodobieństwo przez `predict_proba()`.**
