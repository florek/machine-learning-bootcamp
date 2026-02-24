# Regresja drzewa decyzyjnego

Ten plik opisuje **regresję drzewa decyzyjnego** (Decision Tree Regressor): model, który dzieli przestrzeń cech na regiony i przewiduje wartość docelową jako średnią w liściu. Przy nieliniowej zależności (np. kwadratowej) drzewo lepiej dopasowuje dane niż prosta regresja liniowa.

---

## 1. Kiedy używać drzewa decyzyjnego do regresji

Gdy zależność między cechą a targetem jest nieliniowa (np. wielomianowa), regresja liniowa daje prostą i słabe R². **DecisionTreeRegressor** dzieli oś cechy na przedziały i w każdym przedziale przewiduje stałą wartość (np. średnią targetu), co daje schodkową krzywą – lepsze dopasowanie do nieliniowości bez ręcznego rozszerzania cech (PolynomialFeatures).

**Parametr max_depth:** ogranicza głębokość drzewa; małe drzewo = prostszy model (mniej przeuczenia), duże = bardziej dopasowany do danych (ryzyko przeuczenia).

---

## 2. Konfiguracja i dane

Dane syntetyczne jak w regresji liniowej: `make_regression` z jedną cechą, ewentualna transformacja targetu (np. `target = target ** 2`) dla zależności nieliniowej. Cecha w kształcie 2D: `(n_samples, 1)` – scikit-learn wymaga macierzy.

---

## 3. Regresja liniowa vs drzewo

**LinearRegression** na jednej cesze daje prostą; przy zależności kwadratowej R² jest niskie. **DecisionTreeRegressor(max_depth=k)** dzieli oś X na kawałki i w każdym kawałku przewiduje stałą; wykres to „schodki”, lepsze dopasowanie do krzywej.

---

## 4. DecisionTreeRegressor w scikit-learn

```python
from sklearn.tree import DecisionTreeRegressor

regressor_tree = DecisionTreeRegressor(max_depth=2)
regressor_tree.fit(data, target)
y_pred = regressor_tree.predict(plot_data)
```

**max_depth:** liczba poziomów podziałów; im wyższa, tym więcej „schodków” i większa złożoność. Do wizualizacji krzywej predykcji używa się gęstej siatki punktów (np. `np.arange(-3, 3, 0.01).reshape(-1, 1)`), żeby zobaczyć kształt funkcji.

---

## 5. Wizualizacja struktury drzewa

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(regressor_tree, filled=True, rounded=True, feature_names=['cecha x'])
plt.title('Struktura drzewa decyzyjnego')
plt.tight_layout()
plt.savefig('tree.png')
plt.show()
```

**plot_tree** rysuje węzły, progi podziału i liście. `filled=True` koloruje węzły, `feature_names` nadaje nazwy cech w etykietach. Zapis do PNG pozwala zachować wykres poza notebookiem.

---

## 6. Porównanie głębokości

Przy zwiększaniu **max_depth** (2, 3, 4, …) krzywa predykcji ma więcej schodków i lepiej dopasowuje się do danych treningowych. Zbyt duże max_depth przy małej liczbie obserwacji prowadzi do przeuczenia – model „zapamiętuje” szum. Należy porównywać score na zbiorze treningowym i testowym (train_test_split).

---

## 7. Podsumowanie

- Regresja drzewa decyzyjnego dzieli cechę na przedziały i przewiduje stałą w każdym przedziale.
- DecisionTreeRegressor(max_depth=k), fit, predict – API jak w LinearRegression.
- plot_tree() służy do wizualizacji struktury drzewa (progi, liście).
- max_depth kontroluje złożoność; wyższa wartość = lepsze dopasowanie, większe ryzyko przeuczenia.
