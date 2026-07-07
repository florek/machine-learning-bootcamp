# K-Nearest Neighbors (KNN) – klasyfikacja i granice decyzyjne

Ten plik opisuje algorytm **K-Nearest Neighbors** w wersji klasyfikacyjnej: zasadę działania, parametr liczby sąsiadów, wizualizację granic decyzyjnych oraz eksplorację wieloklasowego zbioru Iris.

---

## 1. Idea algorytmu KNN

KNN to algorytm **instance-based** (oparty na instancjach) i **lazy learning** (leniwe uczenie):

- Podczas `fit()` model **nie uczy się parametrów** w sensie regresji liniowej – głównie **zapamiętuje** dane treningowe.
- Podczas `predict()` dla nowej obserwacji wyszukuje **k najbliższych sąsiadów** w przestrzeni cech (np. odległość euklidesowa).
- Etykieta klasy wynika z **głosowania większościowego** sąsiadów (w klasyfikacji) lub uśrednienia (w regresji).

Brak jawnej fazy „uczenia wag” – obliczenia przypadają głównie na moment predykcji.

---

## 2. KNeighborsClassifier w scikit-learn

**KNeighborsClassifier** służy do **klasyfikacji** (nie regresji ciągłej).

Kluczowy parametr:

- **`n_neighbors`** – liczba najbliższych sąsiadów branych pod uwagę przy głosowaniu (typowo mała liczba całkowita, np. 1, 3, 5, 7).

API zgodne z innymi estymatorami sklearn:

- `fit(X, y)` – zapamiętanie danych i etykiet.
- `predict(X)` – przypisanie klasy na podstawie k sąsiadów.

---

## 3. Wpływ parametru k (n_neighbors)

**Małe k (np. k = 1):**

- granice decyzyjne są **poszarpane**, wrażliwe na pojedyncze punkty,
- wysokie dopasowanie do train, ryzyko **overfittingu**.

**Duże k (np. k = 7):**

- granice są **wygładzone**, decyzja opiera się na większej grupie sąsiadów,
- mniejsza wrażliwość na szum, ale możliwe **underfitting** przy zbyt dużym k.

Porównanie granic dla k = 1…7 pokazuje ewolucję od bardzo złożonych do coraz prostszych podziałów przestrzeni cech.

---

## 4. Zbiór Iris (load_iris)

**load_iris()** zwraca słownik z kluczami m.in. `data`, `target`, `feature_names`, `target_names`.

Charakterystyka:

- **4 cechy numeryczne** (np. długość i szerokość działki kielicha i płatka),
- **3 klasy** docelowe – przykład **klasyfikacji wieloklasowej** (nie binarnej),
- równomierny rozkład klas (po 50 obserwacji na klasę).

Typowy pipeline eksploracji:

- konwersja do DataFrame z kolumną `class`,
- `info()`, `describe()`, `value_counts()` – podstawowa EDA,
- `corr()` – macierz korelacji między cechami,
- `pairplot` z `hue='class'` – wizualna ocena separacji klas w parach cech.

---

## 5. Redukcja do dwóch cech (wizualizacja 2D)

Do rysowania granic decyzyjnych na płaszczyźnie wybiera się **dwie cechy** (np. pierwsze dwie kolumny macierzy `data`).

Powody:

- granice decyzyjne w 2D można narysować jako mapę kolorów na siatce punktów,
- pełne 4 wymiary nie dają się bezpośrednio przedstawić na jednym wykresie 2D.

Wykres punktowy `scatter(x, y, c=target)` pokazuje rozkład klas w wybranej parze cech.

---

## 6. Wizualizacja granic decyzyjnych

Typowy schemat rysowania granic:

1. Wytrenować `KNeighborsClassifier(n_neighbors=k)` na danych 2D.
2. Zbudować **siatkę** punktów (`meshgrid`) pokrywającą obszar wykresu z małym krokiem (np. 0,02).
3. Dla każdego punktu siatki wywołać `predict()` – otrzymujemy przypisaną klasę w całej przestrzeni.
4. Narysować **`contourf`** (wypełnione regiony klas) i **`contour`** (linie graniczne).
5. Na wierzchu **`scatter`** rzeczywistych punktów treningowych z kolorami klas.

Efekt: kolorowe „regiony” decyzji modelu oraz czarne obwódki granic między klasami.

Porównanie wielu wykresów (np. subplot 2×4 dla k = 1…7) ilustruje wpływ k na kształt granic.

---

## 7. Narzędzia wizualizacyjne

W bootcampie wykorzystywane są:

- **matplotlib** – scatter, contourf, contour, subplots,
- **seaborn** – pairplot z hue dla wielu par cech,
- **plotly express** – interaktywny scatter z kolorem = klasa.

Wspólna konfiguracja: `np.random.seed(42)`, `sns.set(font_scale=1.3)` – powtarzalność i czytelność wykresów.

---

## 8. KNN vs inne algorytmy z bootcampu

| Aspekt | KNN | Regresja logistyczna | Drzewo decyzyjne |
|--------|-----|----------------------|------------------|
| Typ uczenia | leniwe (lazy) | parametryczne | struktura drzewa |
| fit() | zapamiętuje dane | uczy współczynników | buduje drzewo |
| Granice decyzyjne | nieregularne, zależne od k | liniowe (w przestrzeni cech) | prostopadłe podziały (schodki) |
| Klasy | wieloklasowa i binarna | binarna (domyślnie) | regresja i klasyfikacja |

---

## 9. Pułapki i dobre praktyki

- **Skalowanie cech:** odległość euklidesowa jest wrażliwa na skalę – cechy o dużych wartościach dominują odległość. Przed KNN warto rozważyć `StandardScaler` (fit na train).
- **Wybór k:** zbyt małe k → overfitting; zbyt duże k → underfitting. Dobór przez walidację (np. cross-validation) lub analizę granic.
- **Koszt predykcji:** przy dużym zbiorze treningowym każda predykcja wymaga przeszukania sąsiadów – wolniejsze niż modele parametryczne.
- **Wymiarowość:** w wielu wymiarach odległości stają się mniej znaczące (curse of dimensionality) – KNN działa najlepiej przy niewielkiej liczbie istotnych cech.

---

## 10. Podsumowanie

- KNN klasyfikuje obserwację na podstawie **k najbliższych sąsiadów** w przestrzeni cech.
- Parametr **`n_neighbors`** kontroluje kompromis między dopasowaniem a generalizacją.
- **Granice decyzyjne** można wizualizować przez siatkę punktów i `predict()` na całym obszarze wykresu 2D.
- Zbiór **Iris** to standardowy przykład **klasyfikacji wieloklasowej** z czterema cechami numerycznymi.
