# Random Forest na zbiorze Iris — uporządkowane notatki z wykładu

## 1. Cel ćwiczenia

W tej części sprawdzamy, jak **Random Forest**, czyli **las losowy**, poradzi sobie z klasycznym zbiorem danych **Iris**.

Chcemy zrobić kilka rzeczy:

1. załadować dane Iris,
2. zbudować model lasu losowego,
3. najpierw użyć tylko dwóch cech, aby dało się narysować granice decyzyjne,
4. później zbudować model na wszystkich czterech cechach,
5. podzielić dane na zbiór treningowy i testowy,
6. sprawdzić dokładność modelu,
7. zobaczyć, które cechy są dla modelu najważniejsze.

---

# 2. Załadowanie danych Iris

Na początku importujemy potrzebne biblioteki i ładujemy zbiór Iris.

W scikit-learn możemy zrobić to za pomocą:

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

Zbiór Iris zawiera:

- dane wejściowe,
- klasy docelowe,
- nazwy cech,
- nazwy klas.

Możemy przypisać je do wygodnych zmiennych:

```python
data = iris.data
target = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
```

---

# 3. Jakie cechy znajdują się w Iris?

Zbiór Iris posiada cztery cechy:

1. **sepal length** — długość działki kielicha,
2. **sepal width** — szerokość działki kielicha,
3. **petal length** — długość płatka,
4. **petal width** — szerokość płatka.

Zmienna docelowa `target` określa gatunek irysa.

Mamy trzy klasy:

- Iris Setosa,
- Iris Versicolor,
- Iris Virginica.

---

# 4. DataFrame dla lepszego podglądu danych

Dla wygody możemy utworzyć obiekt `DataFrame`.

```python
import pandas as pd

df = pd.DataFrame(data, columns=feature_names)
df["target"] = target
```

Dzięki temu dane są łatwiejsze do przeglądania i analizowania.

---

# 5. Dlaczego najpierw używamy tylko dwóch cech?

Na początku wykorzystujemy tylko:

- `sepal length`,
- `sepal width`.

Czyli dwie pierwsze cechy:

```python
X = data[:, :2]
y = target
```

Robimy tak celowo.

## Powód

Chcemy później narysować **granice decyzyjne modelu**.

Dwie cechy można łatwo przedstawić na płaszczyźnie:

```text
cecha 2
  ↑
  |
  |
  |
  +----------→ cecha 1
```

Gdybyśmy użyli wszystkich czterech cech, przestrzeń byłaby czterowymiarowa i nie dałoby się jej wygodnie narysować na zwykłym wykresie 2D.

---

# 6. Import modelu Random Forest

Las losowy jest modelem należącym do grupy:

## Ensemble Learning

czyli **uczenia zespołowego**.

Dlatego `RandomForestClassifier` importujemy z pakietu `ensemble`:

```python
from sklearn.ensemble import RandomForestClassifier
```

---

# 7. Tworzenie modelu

Tworzymy instancję modelu:

```python
model = RandomForestClassifier(
    n_estimators=100
)
```

Najważniejszy parametr w tym miejscu to:

## `n_estimators`

Określa on liczbę drzew decyzyjnych w lesie.

Czyli:

```python
n_estimators=100
```

oznacza:

> Model będzie składał się ze 100 drzew decyzyjnych.

---

# 8. Jak działa Random Forest?

W uproszczeniu:

```text
dane
 ↓
losowanie próbek
 ↓
budowa wielu drzew
 ↓
każde drzewo dokonuje predykcji
 ↓
głosowanie
 ↓
wynik końcowy
```

Przykład:

```text
Drzewo 1 → Setosa
Drzewo 2 → Setosa
Drzewo 3 → Versicolor
Drzewo 4 → Setosa
Drzewo 5 → Versicolor
```

Większość drzew wskazała:

```text
Setosa
```

więc cały las przewiduje klasę:

```text
Setosa
```

---

# 9. Trenowanie modelu

Model dopasowujemy do danych:

```python
model.fit(X, y)
```

Po treningu możemy sprawdzić wynik:

```python
model.score(X, y)
```

W przykładzie z wykładu wynik wynosi około:

```text
92,6%
```

---

# 10. Ważna uwaga dotycząca `score()`

Jeżeli sprawdzamy:

```python
model.score(X, y)
```

na tych samych danych, na których model był trenowany, to nie jest jeszcze rzetelna ocena jakości modelu.

Model widział już te dane podczas treningu.

Dlatego później będziemy używać osobnego zbioru testowego.

---

# 11. Granice decyzyjne

Do wizualizacji można wykorzystać funkcję:

```python
plot_decision_regions
```

np. z biblioteki `mlxtend`.

Dzięki temu możemy zobaczyć, jak model dzieli przestrzeń cech na poszczególne klasy.

---

# 12. Random Forest a pojedyncze drzewo

Pojedyncze drzewo decyzyjne może tworzyć dość sztywne granice.

Random Forest łączy wiele różnych drzew.

Dzięki temu granice decyzyjne mogą być:

- bardziej stabilne,
- lepiej dopasowane,
- mniej zależne od pojedynczego drzewa.

To jest jedna z głównych zalet uczenia zespołowego.

---

# 13. Budowanie modelu na wszystkich czterech cechach

Następnie używamy już wszystkich cech:

```python
X = iris.data
y = iris.target
```

Czyli:

```text
sepal length
sepal width
petal length
petal width
```

Dzięki temu model otrzymuje znacznie więcej informacji.

Minusem jest to, że nie możemy już tak łatwo narysować granic decyzyjnych w 2D.

---

# 14. Podział na dane treningowe i testowe

Przy prawidłowej ocenie modelu dane należy podzielić.

Używamy:

```python
from sklearn.model_selection import train_test_split
```

Przykład:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

W przykładzie z wykładu otrzymano około:

```text
112 próbek treningowych
38 próbek testowych
```

---

# 15. Dlaczego dzielimy dane?

Model uczy się wyłącznie na:

```text
X_train
y_train
```

A następnie sprawdzamy jego działanie na:

```text
X_test
y_test
```

Dane testowe są dla niego nowe.

Model ich wcześniej nie widział.

To pozwala sprawdzić, czy nauczył się ogólnych zależności, czy tylko zapamiętał dane treningowe.

---

# 16. Trenowanie Random Forest na danych treningowych

Tworzymy nowy model:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Następnie:

```python
model.fit(X_train, y_train)
```

Model uczy się wyłącznie na danych treningowych.

---

# 17. Predykcja na danych testowych

Po treningu wykonujemy predykcję:

```python
y_pred = model.predict(X_test)
```

Teraz możemy porównać:

```text
y_test
```

czyli prawdziwe klasy

z:

```text
y_pred
```

czyli klasami przewidzianymi przez model.

---

# 18. Accuracy Score

Do oceny wykorzystujemy:

```python
from sklearn.metrics import accuracy_score
```

Następnie:

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)
```

W przykładzie z wykładu uzyskano wynik:

```text
accuracy = 1.0
```

czyli:

```text
100%
```

Model nie popełnił żadnego błędu na tym konkretnym zbiorze testowym.

---

# 19. Czy `accuracy = 1.0` oznacza model idealny?

Nie.

To bardzo ważne.

Zbiór Iris jest:

- mały,
- stosunkowo prosty,
- dobrze rozdzielalny.

Cały zbiór ma tylko:

```text
150 próbek
```

a zbiór testowy w przykładzie:

```text
38 próbek
```

Dlatego wynik:

```text
100%
```

nie oznacza, że model zawsze będzie działał idealnie.

Przy kilku tysiącach nowych próbek prawdopodobieństwo popełnienia jakiegoś błędu byłoby większe.

---

# 20. Więcej cech może ułatwić problem

Kiedy model korzystał tylko z:

```text
sepal length
sepal width
```

problem był trudniejszy.

Po dodaniu:

```text
petal length
petal width
```

model otrzymał dwie bardzo wartościowe cechy.

Dlatego klasy można rozdzielić znacznie łatwiej.

---

# 21. Feature Importance

Random Forest pozwala sprawdzić, które cechy miały największy wpływ na predykcje.

Służy do tego atrybut:

```python
model.feature_importances_
```

Przykładowy wynik może wyglądać tak:

```text
sepal length    0.10
sepal width     0.03
petal length    0.44
petal width     0.43
```

Wartości sumują się mniej więcej do:

```text
1.0
```

---

# 22. Jak interpretować Feature Importance?

Im większa wartość, tym większe znaczenie danej cechy w decyzjach podejmowanych przez drzewa.

W zbiorze Iris zwykle najważniejsze są:

```text
petal length
petal width
```

czyli:

- długość płatka,
- szerokość płatka.

Jest to zgodne z wcześniejszą obserwacją, że właśnie te cechy dobrze rozdzielają poszczególne gatunki.

---

# 23. Wykres ważności cech

Same liczby można przedstawić na wykresie.

Schematycznie:

```text
petal length  ███████████████
petal width   ██████████████
sepal length  ████
sepal width   ██
```

W wykładzie wykorzystano do tego bibliotekę Plotly.

---

# 24. Po co analizować ważność cech?

Przy czterech cechach łatwo zobaczyć zależności samodzielnie.

Ale w prawdziwych projektach możemy mieć:

- 20 cech,
- 100 cech,
- 1000 cech.

Wtedy trudno intuicyjnie określić:

> Które zmienne rzeczywiście mają największe znaczenie dla modelu?

Feature Importance daje szybki sposób na znalezienie najważniejszych cech.

---

# 25. Dlaczego wcześniej używaliśmy słabszych cech?

W pierwszej części ćwiczenia używaliśmy tylko:

```text
sepal length
sepal width
```

mimo że:

```text
petal length
petal width
```

są znacznie lepszymi predyktorami.

Zrobiono to celowo.

Dzięki temu:

- problem nie był zbyt łatwy,
- można było zobaczyć różnice między modelami,
- granice decyzyjne były ciekawsze,
- można było porównać drzewo decyzyjne z lasem losowym.

---

# 26. Random Forest — najważniejsza intuicja

Najprościej:

> Random Forest to wiele drzew decyzyjnych pracujących razem.

Każde drzewo:

1. dostaje trochę inne dane,
2. może podejmować trochę inne decyzje,
3. zwraca swoją predykcję.

Na końcu drzewa głosują.

```text
            ┌─ Drzewo 1 ─→ klasa A
            │
            ├─ Drzewo 2 ─→ klasa A
Dane ───────┼─ Drzewo 3 ─→ klasa B
            │
            ├─ Drzewo 4 ─→ klasa A
            │
            └─ Drzewo 5 ─→ klasa B

                  ↓

              większość

                  ↓

               klasa A
```

---

# 27. Najważniejsze pojęcia z lekcji

## `RandomForestClassifier`

Klasa ze scikit-learn służąca do budowania modelu lasu losowego dla problemów klasyfikacji.

```python
from sklearn.ensemble import RandomForestClassifier
```

---

## `n_estimators`

Liczba drzew w lesie.

```python
RandomForestClassifier(
    n_estimators=100
)
```

---

## `fit()`

Trenuje model.

```python
model.fit(
    X_train,
    y_train
)
```

---

## `predict()`

Przewiduje klasy dla nowych danych.

```python
y_pred = model.predict(X_test)
```

---

## `accuracy_score()`

Sprawdza procent poprawnych odpowiedzi.

```python
accuracy_score(
    y_test,
    y_pred
)
```

---

## `feature_importances_`

Pokazuje względną ważność poszczególnych cech.

```python
model.feature_importances_
```

---

# 28. Cały proces w skrócie

```text
Załaduj Iris
     ↓
Wybierz cechy
     ↓
Podziel dane
     ↓
X_train / X_test
     ↓
Utwórz Random Forest
     ↓
model.fit()
     ↓
model.predict()
     ↓
accuracy_score()
     ↓
feature_importances_
```

---

# 29. Najważniejsze wnioski

1. **Random Forest jest modelem zespołowym.**

   Zamiast jednego drzewa wykorzystuje wiele drzew.

2. **`n_estimators` określa liczbę drzew.**

3. **Końcowa klasyfikacja powstaje na podstawie głosowania drzew.**

4. **Model powinien być oceniany na danych, których wcześniej nie widział.**

5. **Accuracy = 1.0 na małym zbiorze nie oznacza modelu idealnego.**

6. **W Iris cechy `petal length` i `petal width` są szczególnie użyteczne.**

7. **`feature_importances_` pozwala sprawdzić, które cechy są najważniejsze dla modelu.**

---

# 30. Jednozdaniowe podsumowanie

> **Random Forest buduje wiele różnych drzew decyzyjnych i łączy ich odpowiedzi, dzięki czemu zazwyczaj otrzymujemy model bardziej stabilny i skuteczny niż pojedyncze drzewo.**
