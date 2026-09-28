# Support Vector Machine (SVM) — uporządkowane notatki z wykładu

## 1. Czym jest SVM?

**SVM — Support Vector Machine**, czyli **maszyna wektorów nośnych**, to algorytm uczenia maszynowego używany m.in. do:

- klasyfikacji liniowej,
- klasyfikacji nieliniowej,
- regresji.

Najłatwiej zrozumieć go wizualnie.

---

## 2. Główna intuicja

Wyobraźmy sobie dwie klasy danych, np. psy i koty, rozmieszczone na płaszczyźnie.

Jeżeli dane są liniowo separowalne, możemy narysować linię rozdzielającą obie klasy.

SVM nie szuka jednak dowolnej linii.

> Szuka takiej granicy decyzyjnej, która pozostawia **najszerszy możliwy pas między klasami**.

Ten pas nazywamy **marginesem**.

---

## 3. Support vectors — wektory nośne

Na krańcach marginesu znajdują się próbki położone najbliżej granicy decyzyjnej.

To właśnie:

## support vectors — wektory nośne

To one w największym stopniu wyznaczają położenie granicy decyzyjnej.

Schematycznie:

```text
klasa A

●      ●
   ●
-------------------  granica marginesu
===================  granica decyzyjna
-------------------  granica marginesu
      ○
○          ○

klasa B
```

---

## 4. Klasyfikacja nowej próbki

Gdy pojawia się nowa obserwacja, model sprawdza, po której stronie granicy decyzyjnej się znajduje.

Na tej podstawie przypisuje próbkę do odpowiedniej klasy.

---

## 5. Problem obserwacji odstających

SVM może być wrażliwy na **outliery**, czyli dane odstające.

Pojedyncza nietypowa próbka może wpłynąć na położenie granicy i szerokość marginesu.

Dlatego rozróżniamy dwa podejścia:

- hard margin,
- soft margin.

---

## 6. Hard margin — twardy margines

Hard margin zakłada, że dane muszą być rozdzielone bez błędów klasyfikacji.

Model nie pozwala na naruszenie marginesu.

To podejście działa najlepiej, gdy:

- dane są dobrze rozdzielone,
- nie ma istotnych obserwacji odstających,
- klasy są liniowo separowalne.

---

## 7. Soft margin — miękki margines

Soft margin jest bardziej elastyczny.

Pozwala:

- naruszyć margines,
- zaakceptować część błędnych klasyfikacji,
- lepiej radzić sobie z danymi odstającymi.

Model nie musi więc poprawnie sklasyfikować każdej próbki za wszelką cenę.

---

## 8. Parametr `C`

W scikit-learn kompromis ten kontroluje hiperparametr:

```python
C
```

Domyślnie:

```python
C = 1
```

Intuicyjnie:

```text
mniejsze C
→ szerszy margines
→ większa tolerancja na błędy

większe C
→ węższy margines
→ mniejsza tolerancja na błędy
```

Czyli `C` reguluje, jak mocno model ma karać błędne klasyfikacje.

---

## 9. Przykład na zbiorze Iris

W wykładzie użyto danych Iris i dwóch cech, m.in.:

- `petal length`,
- `sepal width`.

Dane ograniczono do dwóch klas:

- Setosa,
- Versicolor.

Przy odpowiednim wyborze cech dane można rozdzielić liniowo i zobaczyć granicę decyzyjną SVM.

---

## 10. Kernel — jądro

SVM posiada bardzo ważny parametr:

## `kernel`

Jądro określa sposób budowania granicy decyzyjnej.

Dzięki kernelom SVM może rozwiązywać zarówno problemy liniowe, jak i nieliniowe.

---

## 11. Kernel liniowy

Przykład:

```python
kernel="linear"
```

Model szuka liniowej granicy decyzyjnej.

W dwóch wymiarach jest to linia, a w większej liczbie wymiarów — hiperpłaszczyzna.

---

## 12. RBF kernel

Przykład:

```python
kernel="rbf"
```

RBF pozwala tworzyć bardziej złożone, nieliniowe granice.

To przydaje się wtedy, gdy danych nie da się sensownie oddzielić prostą linią.

---

## 13. Kernel wielomianowy

Możemy też użyć:

```python
kernel="poly"
```

Na przykład:

```python
kernel="poly"
degree=3
```

Takie jądro pozwala budować wielomianowe granice decyzyjne.

---

## 14. Dlaczego kernel działa?

Najważniejsza intuicja jest taka:

> Dane, których nie da się rozdzielić liniowo w jednej przestrzeni, mogą stać się liniowo separowalne po odpowiednim przekształceniu.

Przykład:

```text
problem nieliniowy
      ↓
transformacja danych
      ↓
inna przestrzeń
      ↓
problem liniowo separowalny
```

---

## 15. Przykład w jednym wymiarze

Załóżmy, że punkty leżą na jednej osi:

```text
●   ○   ○   ●
```

Nie da się ich rozdzielić jednym prostym progiem.

Po przekształceniu np. funkcją kwadratową może się jednak okazać, że w nowej przestrzeni rozdzielenie jest możliwe.

---

## 16. Przejście z R² do R³

Podobnie w dwóch wymiarach dane mogą być nieliniowo separowalne.

Po przekształceniu do trzech wymiarów:

```text
R² → R³
```

może się okazać, że klasy da się rozdzielić zwykłą płaszczyzną.

To właśnie daje intuicję działania jąder w SVM.

---

## 17. SVM w scikit-learn

Do klasyfikacji używamy klasy:

```python
from sklearn.svm import SVC
```

Następnie tworzymy model:

```python
model = SVC()
```

i trenujemy go:

```python
model.fit(X_train, y_train)
```

---

## 18. Najważniejsze parametry `SVC`

### `C`

Reguluje kompromis między szerokością marginesu a tolerancją błędów.

```python
SVC(C=1)
```

### `kernel`

Określa rodzaj jądra:

```python
SVC(kernel="linear")
```

```python
SVC(kernel="rbf")
```

```python
SVC(kernel="poly")
```

Domyślnym kernelem w `SVC` jest `rbf`.

---

## 19. Przykład modelu

```python
from sklearn.svm import SVC

model = SVC(
    C=1,
    kernel="rbf"
)

model.fit(X_train, y_train)
```

---

## 20. Najważniejsze typy kerneli

```text
linear
→ granica liniowa

rbf
→ bardziej złożone granice nieliniowe

poly
→ granice wielomianowe
```

---

## 21. Całość w jednym schemacie

```text
DANE
  ↓
czy są liniowo separowalne?
  │
  ├── TAK
  │    ↓
  │  kernel="linear"
  │    ↓
  │  maksymalny margines
  │
  └── NIE
       ↓
   RBF / poly
       ↓
   transformacja przestrzeni
       ↓
   łatwiejszy podział klas
```

---

## 22. Co warto zapamiętać?

1. **SVM szuka granicy decyzyjnej z możliwie szerokim marginesem.**
2. **Punkty najbliżej granicy to support vectors.**
3. **Hard margin nie dopuszcza błędów klasyfikacji.**
4. **Soft margin pozwala na pewne naruszenia marginesu.**
5. **Parametr `C` reguluje tolerancję na błędy i szerokość marginesu.**
6. **Kernel pozwala rozwiązywać problemy nieliniowe.**
7. **Popularne kernele to `linear`, `rbf` i `poly`.**
8. **Transformacja przestrzeni może zamienić problem nieliniowy w liniowo separowalny.**
9. **W scikit-learn do klasyfikacji używamy klasy `SVC`.**

---

## 23. Najważniejsze zdanie

> **SVM szuka granicy decyzyjnej z możliwie dużym marginesem, a dzięki kernelom potrafi poradzić sobie również z problemami nieliniowymi.**
