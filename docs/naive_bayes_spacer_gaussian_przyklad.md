# Naive Bayes krok po kroku — przykład ze spacerem i Gaussian Naive Bayes

## 1. Cel przykładu

Chcemy lepiej zrozumieć działanie **naiwnego klasyfikatora Bayesa** na prostym przykładzie.

Mamy trzy kolumny:

```text
pogoda
temperatura
spacer
```

Zmienna docelowa `spacer` przyjmuje dwie wartości:

```text
tak
nie
```

Czyli mamy problem **klasyfikacji binarnej**.

---

## 2. Dostępne wartości cech

### Pogoda

```text
słonecznie
pochmurno
deszczowo
```

### Temperatura

```text
ciepło
umiarkowanie
zimno
```

### Spacer

```text
tak
nie
```

---

## 3. Rozkład zmiennej docelowej

W danych:

```text
spacer = tak → 5 przypadków
spacer = nie → 4 przypadki
```

Łącznie mamy 9 obserwacji, więc:

```text
P(spacer = tak) = 5/9
P(spacer = nie) = 4/9
```

To są prawdopodobieństwa klas przed uwzględnieniem cech, czyli tzw. **priory**.

---

## 4. Prawdopodobieństwa dla cechy „pogoda”

Interesuje nas wartość:

```text
słonecznie
```

W danych:

```text
słonecznie + spacer = tak → 3 razy
słonecznie + spacer = nie → 1 raz
```

Dlatego:

```text
P(słonecznie | spacer = tak) = 3/5
P(słonecznie | spacer = nie) = 1/4
```

---

## 5. Prawdopodobieństwa dla temperatury

Dla:

```text
temperatura = ciepło
```

mamy:

```text
ciepło + spacer = tak → 4 razy
ciepło + spacer = nie → 1 raz
```

Czyli:

```text
P(ciepło | spacer = tak) = 4/5
P(ciepło | spacer = nie) = 1/4
```

---

## 6. Pytanie klasyfikacyjne

Chcemy odpowiedzieć na pytanie:

> Czy wyjść na spacer, jeżeli jest słonecznie i ciepło?

Czyli znamy:

```text
pogoda = słonecznie
temperatura = ciepło
```

i chcemy przewidzieć:

```text
spacer = tak
```

lub:

```text
spacer = nie
```

---

## 7. Wzór Naive Bayes dla dwóch cech

Dla dwóch cech `X1` i `X2`:

```text
P(Y | X1, X2)
=
P(X1 | Y)
*
P(X2 | Y)
*
P(Y)
/
P(X1, X2)
```

W naszym przypadku:

```text
X1 = pogoda
X2 = temperatura
Y = spacer
```

---

## 8. Prawdopodobieństwo wyjścia na spacer

Chcemy policzyć:

```text
P(spacer = tak | słonecznie, ciepło)
```

czyli:

```text
P(tak | słonecznie, ciepło)
=
P(słonecznie | tak)
*
P(ciepło | tak)
*
P(tak)
/
P(słonecznie, ciepło)
```

---

## 9. Prawdopodobieństwo pozostania w domu

Analogicznie:

```text
P(nie | słonecznie, ciepło)
=
P(słonecznie | nie)
*
P(ciepło | nie)
*
P(nie)
/
P(słonecznie, ciepło)
```

---

## 10. Najważniejsza obserwacja: mianownik jest taki sam

W obu wzorach występuje:

```text
P(słonecznie, ciepło)
```

Czyli wspólny mianownik.

Jeżeli chcemy tylko wybrać klasę z większym prawdopodobieństwem, nie musimy go liczyć.

> **Przy porównywaniu klas wystarczy porównać liczniki.**

---

## 11. Wynik dla klasy „tak”

Liczymy:

```text
P(słonecznie | tak)
*
P(ciepło | tak)
*
P(tak)
```

Podstawiamy:

```text
3/5 * 4/5 * 5/9
```

Otrzymujemy:

```text
12/45 ≈ 0,2667
```

---

## 12. Wynik dla klasy „nie”

Liczymy:

```text
P(słonecznie | nie)
*
P(ciepło | nie)
*
P(nie)
```

Podstawiamy:

```text
1/4 * 1/4 * 4/9
```

Otrzymujemy:

```text
1/36 ≈ 0,0278
```

---

## 13. Samo porównanie już wystarcza

Mamy:

```text
tak → 0,2667
nie → 0,0278
```

Ponieważ:

```text
0,2667 > 0,0278
```

model wybiera:

```text
spacer = tak
```

---

## 14. Normalizacja wyników

Jeżeli chcemy uzyskać wartości sumujące się do 1, normalizujemy oba wyniki.

Suma:

```text
0,2667 + 0,0278 = 0,2945
```

Zatem:

```text
P(tak | słonecznie, ciepło)
≈ 0,2667 / 0,2945
≈ 90,5%
```

oraz:

```text
P(nie | słonecznie, ciepło)
≈ 0,0278 / 0,2945
≈ 9,5%
```

---

## 15. Ostateczna decyzja

Model przewiduje:

```text
spacer = tak
```

z prawdopodobieństwem około:

```text
90,5%
```

a:

```text
spacer = nie
```

z prawdopodobieństwem około:

```text
9,5%
```

Czyli dla danych:

```text
słonecznie + ciepło
```

model wybiera wyjście na spacer.

---

## 16. Cały przykład w jednym schemacie

```text
Pogoda = słonecznie
Temperatura = ciepło
        ↓

Dla klasy TAK:

P(słonecznie|tak)
*
P(ciepło|tak)
*
P(tak)

= 3/5 * 4/5 * 5/9
≈ 0,2667


Dla klasy NIE:

P(słonecznie|nie)
*
P(ciepło|nie)
*
P(nie)

= 1/4 * 1/4 * 4/9
≈ 0,0278

        ↓

porównanie

0,2667 > 0,0278

        ↓

SPACER = TAK
```

---

## 17. Gdzie pojawia się „naiwność”?

Model zakłada, że po ustaleniu klasy cechy są niezależne.

Czyli:

```text
P(słonecznie, ciepło | spacer = tak)
```

upraszczamy do:

```text
P(słonecznie | tak)
*
P(ciepło | tak)
```

To właśnie jest **naiwne założenie o niezależności cech**.

---

## 18. Dane dyskretne vs ciągłe

W tym przykładzie mamy dane dyskretne:

```text
słonecznie
deszczowo
pochmurno
```

oraz:

```text
ciepło
umiarkowanie
zimno
```

Ale co w przypadku danych ciągłych?

Na przykład:

```text
temperatura = 21,7°C
wiek = 37
wzrost = 182,4 cm
```

Wtedy potrzebujemy dodatkowego założenia dotyczącego rozkładu cechy.

---

## 19. Różne warianty Naive Bayes

Różne wersje Naive Bayes różnią się głównie tym:

> jaki rozkład zakładają dla `P(Xi | Y)`.

Czyli dla prawdopodobieństwa wartości danej cechy przy znanej klasie.

---

## 20. Gaussian Naive Bayes

W **Gaussian Naive Bayes** zakładamy, że wartości cechy w obrębie klasy mają:

## rozkład normalny

czyli rozkład Gaussa.

Stąd nazwa:

```text
Gaussian Naive Bayes
```

---

## 21. Intuicja Gaussian Naive Bayes

Załóżmy cechę:

```text
temperatura ciała
```

Dla klasy:

```text
zdrowy
```

możemy mieć jeden rozkład normalny.

Dla klasy:

```text
chory
```

inny.

Model sprawdza:

> Jak prawdopodobna jest konkretna wartość cechy w każdej klasie?

---

## 22. Gaussian Naive Bayes w scikit-learn

Import:

```python
from sklearn.naive_bayes import GaussianNB
```

Tworzymy model:

```python
model = GaussianNB()
```

Trenujemy:

```python
model.fit(X_train, y_train)
```

A potem możemy wykonywać predykcje:

```python
model.predict(X_test)
```

---

## 23. Typowy workflow

```text
dane
 ↓
podział train/test
 ↓
GaussianNB()
 ↓
fit()
 ↓
predict()
 ↓
ocena modelu
```

---

## 24. Najważniejsza intuicja

Naive Bayes dla każdej klasy oblicza:

```text
P(klasa)
*
P(cecha1 | klasa)
*
P(cecha2 | klasa)
*
...
```

Następnie wybiera klasę z największym wynikiem.

Wspólny mianownik nie jest potrzebny, jeśli interesuje nas tylko wybór klasy.

---

## 25. Co warto zapamiętać?

1. **Naive Bayes porównuje prawdopodobieństwa poszczególnych klas.**
2. **Najpierw liczymy prior dla każdej klasy.**
3. **Następnie liczymy prawdopodobieństwa cech przy danej klasie.**
4. **Wyniki mnożymy.**
5. **Wspólny mianownik można pominąć przy samym porównywaniu klas.**
6. **W przykładzie „słonecznie + ciepło” model wybiera spacer.**
7. **Po normalizacji otrzymujemy około 90,5% dla „tak” i 9,5% dla „nie”.**
8. **Naive Bayes zakłada warunkową niezależność cech.**
9. **Dla danych ciągłych potrzebujemy założeń dotyczących rozkładu.**
10. **Gaussian Naive Bayes zakłada rozkład normalny cech w obrębie klas.**
11. **W scikit-learn korzystamy z klasy `GaussianNB`.**

---

## 26. Najważniejsze zdanie

> **Naive Bayes wybiera klasę, dla której iloczyn prioru i prawdopodobieństw cech przy tej klasie jest największy; wspólny mianownik nie jest potrzebny do samego porównania klas.**
