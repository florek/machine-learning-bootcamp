# Naiwny klasyfikator Bayesa — uporządkowane notatki z wykładu

## 1. Czym jest Naive Bayes?

**Naiwny klasyfikator Bayesa** to statystyczna metoda klasyfikacji oparta na:

## twierdzeniu Bayesa

Nazwa „Bayes” pochodzi od nazwiska:

**Thomas Bayes** — angielskiego matematyka.

Algorytm jest:

- prosty,
- szybki,
- dobrze radzi sobie z dużą liczbą danych,
- często działa skutecznie mimo bardzo uproszczonego założenia o niezależności cech.

---

## 2. Twierdzenie Bayesa

Dla dwóch zdarzeń:

```text
A
B
```

przy założeniu, że:

```text
P(B) > 0
```

twierdzenie Bayesa ma postać:

```text
P(A|B) = P(B|A) * P(A) / P(B)
```

gdzie:

### `P(A|B)`

prawdopodobieństwo zajścia zdarzenia A pod warunkiem, że zaszło B

### `P(B|A)`

prawdopodobieństwo zajścia zdarzenia B pod warunkiem, że zaszło A

### `P(A)`

prawdopodobieństwo zdarzenia A

### `P(B)`

prawdopodobieństwo zdarzenia B

---

# 3. Przykład: grypa i gorączka

Załóżmy:

```text
A = pacjent ma grypę
B = pacjent ma wysoką gorączkę
```

Mamy dane:

```text
P(A) = 15%
P(B) = 20%
P(B|A) = 80%
```

Czyli:

- 15% pacjentów ma grypę,
- 20% pacjentów ma wysoką gorączkę,
- 80% osób chorych na grypę ma wysoką gorączkę.

Chcemy obliczyć:

```text
P(A|B)
```

czyli:

> Jakie jest prawdopodobieństwo, że pacjent ma grypę, jeżeli ma wysoką gorączkę?

---

## 4. Obliczenie

Korzystamy ze wzoru:

```text
P(A|B) = P(B|A) * P(A) / P(B)
```

Podstawiamy:

```text
P(A|B) = 0.8 * 0.15 / 0.20
```

Otrzymujemy:

```text
P(A|B) = 0.6
```

czyli:

```text
60%
```

Wniosek:

> Wśród osób z wysoką gorączką około 60% ma grypę.

---

# 5. Niezależność zdarzeń

W dalszej części wykładu potrzebne jest jeszcze pojęcie niezależności.

Jeżeli zdarzenia są niezależne, to:

```text
P(A ∩ B) = P(A) * P(B)
```

Czyli prawdopodobieństwo jednoczesnego zajścia dwóch niezależnych zdarzeń jest iloczynem ich prawdopodobieństw.

Ta sama idea może być rozszerzona również na zmienne losowe.

---

# 6. Skąd nazwa „naiwny”?

Część:

```text
Bayesowski
```

pochodzi oczywiście od twierdzenia Bayesa.

Ale dlaczego:

```text
naiwny?
```

Ponieważ algorytm przyjmuje bardzo silne założenie:

> **poszczególne cechy są od siebie niezależne.**

---

# 7. Dlaczego to założenie jest „naiwne”?

W rzeczywistych danych cechy często są ze sobą powiązane.

Przykład:

```text
wzrost
masa ciała
BMI
```

Nie są to zmienne całkowicie niezależne.

Mimo to Naive Bayes zakłada niezależność, ponieważ pozwala to bardzo mocno uprościć obliczenia.

---

# 8. Czy naruszenie tego założenia niszczy model?

Nie zawsze.

W praktyce:

> mimo że założenie niezależności często nie jest dokładnie spełnione, Naive Bayes potrafi działać zaskakująco dobrze.

Dlatego model jest nadal szeroko stosowany.

---

# 9. Typowe zastosowania

W wykładzie wymieniono m.in.:

- klasyfikację dokumentów,
- filtrowanie spamu.

Naive Bayes dobrze sprawdza się szczególnie tam, gdzie mamy dużo cech i duże zbiory danych.

---

# 10. Przejście do zapisu modelu

Oznaczmy:

```text
Y
```

jako zmienną docelową.

Natomiast:

```text
X
```

jako wektor cech.

Czyli:

```text
X = (X1, X2, ..., Xn)
```

Przykładowo dla czterech cech:

```text
X1
X2
X3
X4
```

---

# 11. Twierdzenie Bayesa dla klasyfikacji

Chcemy obliczyć:

```text
P(Y | X1, X2, ..., Xn)
```

czyli:

> prawdopodobieństwo klasy Y przy znanych wartościach cech X1, X2, ..., Xn

Twierdzenie Bayesa daje nam:

```text
P(Y | X1, ..., Xn)
=
P(X1, ..., Xn | Y) * P(Y)
/
P(X1, ..., Xn)
```

---

# 12. Problem: wspólne prawdopodobieństwo wielu cech

Najtrudniejszym elementem jest:

```text
P(X1, X2, ..., Xn | Y)
```

czyli wspólne prawdopodobieństwo wszystkich cech przy znanej klasie Y.

Bez dodatkowego założenia obliczenie tego może być trudne.

---

# 13. Naiwne założenie niezależności

Naive Bayes zakłada, że cechy są warunkowo niezależne względem klasy Y.

Dzięki temu:

```text
P(X1, X2, ..., Xn | Y)
```

możemy uprościć do:

```text
P(X1|Y)
*
P(X2|Y)
*
...
*
P(Xn|Y)
```

---

# 14. Dlaczego to ogromne uproszczenie?

Zamiast modelować jedno skomplikowane wspólne prawdopodobieństwo:

```text
P(X1, X2, ..., Xn | Y)
```

liczymy osobno:

```text
P(X1|Y)
P(X2|Y)
...
P(Xn|Y)
```

a następnie je mnożymy.

To znacznie upraszcza obliczenia.

---

# 15. Pełna postać modelu

Po zastosowaniu założenia niezależności otrzymujemy:

```text
P(Y | X1, ..., Xn)
=
P(Y)
*
P(X1|Y)
*
P(X2|Y)
*
...
*
P(Xn|Y)
/
P(X1, ..., Xn)
```

Czyli w skrócie:

```text
posterior
=
prior
*
likelihoods
/
evidence
```

---

# 16. Co oznaczają te elementy?

### `P(Y)`

Prawdopodobieństwo klasy Y przed uwzględnieniem cech.

To tzw.:

```text
prior
```

---

### `P(Xi|Y)`

Prawdopodobieństwo danej cechy przy założeniu klasy Y.

To element:

```text
likelihood
```

---

### `P(X1, ..., Xn)`

Prawdopodobieństwo obserwowanych danych.

To:

```text
evidence
```

---

### `P(Y|X1, ..., Xn)`

Prawdopodobieństwo klasy po uwzględnieniu cech.

To:

```text
posterior
```

---

# 17. Jak klasyfikujemy?

Dla każdej możliwej klasy Y obliczamy wartość:

```text
P(Y | X1, ..., Xn)
```

Następnie wybieramy klasę o największym prawdopodobieństwie.

Schemat:

```text
klasa A → 0.10
klasa B → 0.75
klasa C → 0.15
```

Wybieramy:

```text
klasa B
```

---

# 18. Dlaczego mianownik często nie ma znaczenia przy porównaniu klas?

Dla tej samej próbki:

```text
P(X1, ..., Xn)
```

jest takie samo dla każdej klasy.

Dlatego przy samym porównywaniu klas możemy skupić się głównie na liczniku:

```text
P(Y)
*
P(X1|Y)
*
...
*
P(Xn|Y)
```

---

# 19. Różne odmiany Naive Bayes

W wykładzie wspomniano, że istnieją różne wersje klasyfikatora.

Przykładowo:

- Gaussian Naive Bayes,
- Multinomial Naive Bayes,
- inne warianty.

Różnica polega głównie na tym:

> jakie założenie przyjmujemy co do rozkładu cech.

---

# 20. Gaussian Naive Bayes

## Gaussian Naive Bayes

To odmiana Naive Bayes dla cech ciągłych.

Zakładamy, że wartości cechy w obrębie danej klasy mają **rozkład normalny** (Gaussa).

Model sprawdza, jak prawdopodobna jest konkretna wartość cechy w każdej klasie, a potem wybiera klasę z najwyższym wynikiem.

W scikit-learn używamy klasy:

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
model.predict(X_test)
```

Dla cech dyskretnych (np. kategorie pogody) liczymy częstości. Dla cech ciągłych (np. temperatura w °C) potrzebujemy założenia o rozkładzie — stąd właśnie wariant gaussowski.

---

# 21. Najważniejsza intuicja

Naive Bayes działa tak:

```text
znam cechy próbki
       ↓
dla każdej klasy liczę:
P(klasa) * P(cecha1|klasa) * ... * P(cechan|klasa)
       ↓
porównuję wyniki
       ↓
wybieram klasę z najwyższą wartością
```

---

# 22. Dlaczego model jest szybki?

Dzięki założeniu niezależności nie trzeba modelować skomplikowanych zależności pomiędzy wszystkimi cechami.

Każda cecha jest rozpatrywana osobno.

To znacząco upraszcza obliczenia.

---

# 23. Co warto zapamiętać?

1. **Naive Bayes opiera się na twierdzeniu Bayesa.**
2. **Twierdzenie Bayesa pozwala odwrócić prawdopodobieństwo warunkowe.**
3. **„Naiwność” modelu wynika z założenia niezależności cech.**
4. **Założenie to często nie jest dokładnie spełnione.**
5. **Mimo to model może działać bardzo dobrze.**
6. **Naive Bayes jest szybki i dobrze skaluje się do dużych zbiorów.**
7. **Typowe zastosowania to klasyfikacja dokumentów i spam filtering.**
8. **Model liczy prawdopodobieństwo klasy na podstawie cech.**
9. **Dzięki niezależności wspólne prawdopodobieństwo można rozpisać jako iloczyn prostszych prawdopodobieństw.**
10. **Istnieją różne wersje Naive Bayes zależnie od założonego rozkładu cech.**

---

# 24. Całość w jednym schemacie

```text
CECHY X1, X2, ..., Xn
        ↓
dla każdej klasy Y
        ↓
P(Y)
*
P(X1|Y)
*
P(X2|Y)
*
...
*
P(Xn|Y)
        ↓
porównanie wyników
        ↓
WYBÓR NAJBARDZIEJ PRAWDOPODOBNEJ KLASY
```

---

# 25. Najważniejsze zdanie

> **Naiwny klasyfikator Bayesa wykorzystuje twierdzenie Bayesa i upraszcza obliczenia, zakładając niezależność cech, dzięki czemu jest bardzo szybki i często zaskakująco skuteczny.**
