# Klasyfikacyjne drzewo decyzyjne (DecisionTreeClassifier)

Ten plik opisuje **klasyfikacyjne drzewo decyzyjne**: budowę modelu na zbiorze Iris, wizualizację granic decyzyjnych w 2D, eksport grafu drzewa oraz wpływ parametru `max_depth` na złożoność i dopasowanie.

---

## 1. DecisionTreeClassifier vs DecisionTreeRegressor

**DecisionTreeClassifier** przewiduje **etykietę klasy** (np. gatunek kosaćca). W liściu dominuje klasa większościowa wśród obserwacji przypisanych do tego regionu.

**DecisionTreeRegressor** (wcześniejsza lekcja) przewiduje **wartość ciągłą** – typowo średnią targetu w liściu.

Oba modele dzielą przestrzeń cech **podziałami prostopadłymi do osi** (axis-aligned). W 2D granice tworzą regiony o kształtach prostokątnych / „schodkowych”.

---

## 2. Dane: Iris i redukcja do dwóch cech

`load_iris()` daje 150 próbek, 4 cechy i 3 klasy (`setosa`, `versicolor`, `virginica`). Klasy są **zrównoważone** (po 50 próbek) – accuracy jest wtedy łatwiejsza do interpretacji niż przy silnym niezbalansowaniu.

Do wizualizacji granic na płaszczyźnie wybiera się **dwie cechy** (np. `sepal_length`, `sepal_width`):

- granice decyzyjne da się narysować jako mapę kolorów na wykresie 2D,
- pełne 4 wymiary nie mieszczą się na jednym wykresie 2D,
- redukcja cech upraszcza obraz, ale **tracimy informację** z pozostałych atrybutów.

Przy samej parze cech działka (`sepal_*`) klasa `setosa` zwykle oddziela się wyraźniej, a `versicolor` i `virginica` mocno się **nakładają** – płytkie drzewo ma więc ograniczoną accuracy, mimo zrównoważonych klas.

Target konwertuje się do liczb całkowitych (np. `astype('int16')`) przed trenowaniem.

---

## 3. Budowa modelu

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=1, random_state=42)
classifier.fit(data, target)
acc = classifier.score(data, target)
```

- **`max_depth`** – maksymalna głębokość drzewa (liczba poziomów podziałów).
- **`random_state`** – powtarzalność przy ewentualnej losowości.
- Domyślne kryterium podziału: **`criterion='gini'`** (alternatywa: `'entropy'`).
- **`score()`** dla klasyfikatora zwraca **accuracy** (odsetek poprawnych etykiet).

Przy `max_depth=1` drzewo wykonuje **jeden podział** (jeden próg na jednej cesze) – granica to zwykle jedna prosta linia prostopadła do osi cechy.

---

## 4. Wizualizacja granic decyzyjnych (mlxtend)

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(data, target, classifier, legend=2)
```

**`plot_decision_regions`** rysuje kolorowe regiony klas w przestrzeni cech oraz punkty treningowe. Pokazuje, jak drzewo dzieli płaszczyznę na prostokątne obszary.

Porównanie z innymi modelami:

| Model | Kształt granic w 2D |
|-------|---------------------|
| Regresja logistyczna | linia (hiperpłaszczyzna) |
| KNN | nieregularne, zależne od k |
| Drzewo decyzyjne | prostopadłe do osi (schodki / prostokąty) |

---

## 5. Eksport grafu drzewa (export_graphviz)

Strukturę drzewa (węzły, progi, rozkład klas, przewidywaną klasę) można wyeksportować do grafu:

- **`export_graphviz`** – generuje opis drzewa w formacie DOT,
- **`feature_names`** – nazwy cech przy progach podziału,
- **`class_names`** – nazwy klas w liściach,
- **`filled=True`, `rounded=True`** – czytelniejsze węzły (kolor, zaokrąglenia),
- typowy pipeline renderu: opis DOT trafia do bufora tekstowego → narzędzie zewnętrzne (np. **pydotplus**) buduje graf i zapisuje / wyświetla **PNG**.

To uzupełnienie względem `plot_tree` używanego przy regresji drzewa – tu nacisk na **graf z nazwami klas** i porównanie głębokości.

`plot_decision_regions` i eksport grafu odpowiadają na inne pytania: pierwszy pokazuje **jak model dzieli płaszczyznę cech**, drugi – **jaką strukturę podziałów** zbudował algorytm.

---

## 6. Wpływ max_depth

Porównanie `max_depth` = 1, 2, 3, 4, 5, …:

- **mała głębokość** – proste granice, mniejsza accuracy na danych treningowych, mniejsze ryzyko przeuczenia,
- **większa głębokość** – więcej prostokątnych regionów, wyższa accuracy na train, granice coraz bardziej dopasowane do punktów,
- **bardzo duża głębokość** (np. 15) na małym zbiorze – drzewo może niemal idealnie dopasować train → silne ryzyko **overfittingu**.

**Pułapka:** ocena wyłącznie przez `score(data, target)` na tych samych danych, na których trenowano, **nie wykrywa** przeuczenia. Do diagnozy potrzeba podziału train/test (lub walidacji) i porównania accuracy na obu zbiorach.

Praktyczny wzorzec: funkcja pomocnicza, która dla zadanego `max_depth` trenuje model, rysuje granice i eksportuje graf – ułatwia porównanie głębokości.

---

## 7. Porównanie z KNN i regresją logistyczną

- **Drzewo** buduje jawną strukturę podziałów podczas `fit()`; predykcja to przejście ścieżką od korzenia do liścia.
- **KNN** to lazy learning – `fit()` głównie zapamiętuje dane; decyzja przy `predict()`.
- **Regresja logistyczna** daje gładką (liniową w przestrzeni cech) granicę; drzewo – ostre, osiowo wyrównane regiony.

Wybór modelu zależy od geometrii problemu: czy klasy da się oddzielić prostymi progami na cechach, czy potrzeba gładkich / lokalnych granic.

---

## 8. Podsumowanie

- `DecisionTreeClassifier` – klasyfikacja wieloklasowa z podziałami prostopadłymi do osi.
- Redukcja do 2 cech umożliwia wizualizację granic; `plot_decision_regions` pokazuje regiony klas.
- `export_graphviz` (+ renderowanie) wizualizuje strukturę drzewa z nazwami cech i klas.
- `max_depth` kontroluje złożoność: wyższa wartość → lepsze dopasowanie do train, większe ryzyko przeuczenia.
- `score()` klasyfikatora = accuracy; ocena tylko na train nie wystarczy do oceny generalizacji.
