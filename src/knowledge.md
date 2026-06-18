# Machine Learning Bootcamp — notatki z kursu

## Przygotowanie danych

### Imputacja brakujących wartości

`SimpleImputer` uzupełnia braki według wybranej strategii:
- `mean` — średnia kolumny numerycznej,
- `constant` — stała wartość (`fill_value`),
- `most_frequent` — najczęstsza wartość.

Poprawny workflow: `fit` na kolumnie (obliczenie statystyki), potem `transform`. Osobny imputer dla każdej kolumny, gdy strategie różnią się (np. waga → mean, cena → constant 99, rozmiar → constant „L”).

### Szeregi czasowe — uzupełnianie braków

Metody `fillna`:
- stała (np. 0),
- średnia kolumny,
- `interpolate()` — interpolacja liniowa między sąsiednimi punktami,
- `method='bfill'` — wypełnienie wstecz (backward fill),
- `method='ffill'` — wypełnienie w przód (forward fill).

`dropna()` usuwa wiersze z brakami. Wizualizacja szeregów czasowych ujawnia wpływ metody na kształt krzywej.

### Inżynieria cech

**Ekstrakcja z dat:** atrybuty `day`, `month`, `year` z indeksu datetime.

**Dyskretyzacja:** `pd.cut(x, bins=...)` dzieli zmienną ciągłą na przedziały; opcjonalnie `labels` nadają nazwy kategoriom.

**Kodowanie kategorii:** `pd.get_dummies(..., drop_first=True)` tworzy zmienne zero-jedynkowe; `drop_first=True` usuwa jedną kategorię referencyjną i ogranicza współliniowość (dummy variable trap).

**Ekstrakcja z list i tekstu:** `apply(len)` liczy elementy; `apply(lambda x: 1 if 'PL' in x else 0)` tworzy flagi; `str.split('.', expand=True)` rozbija stringi na kolumny.

---

## Podział danych

`train_test_split(X, y, test_size=..., random_state=..., stratify=y)`:
- `test_size` — udział zbioru testowego (np. 0.2 = 20%, 0.3 = 30%),
- `random_state` — powtarzalność losowego podziału,
- `strategia=stratify` — zachowuje proporcje klas w train i test (kluczowe przy niezbalansowanych danych).

Zbiór treningowy służy do uczenia parametrów modelu; testowy — do oceny generalizacji. Nigdy nie trenujemy ani nie dopasowujemy preprocessingu na danych testowych.

---

## Regresja liniowa

### Równanie modelu

y = intercept_ + coef_[0]·x₁ + coef_[1]·x₂ + …

W scikit-learn: `LinearRegression().fit(X, y)` → `intercept_`, `coef_`, `predict(X)`, `score(X, y)` zwraca R².

### Normal equation (rozwiązanie analityczne)

Macierz projektowa X z kolumną jedynek (bias). Wagi: β = (XᵀX)⁻¹ XᵀY. Kolumna jedynek odpowiada za intercept.

### Gradient descent

Aktualizacja wag: gradient = (2/m) · Xᵀ(X·w − Y); w = w − η · gradient, gdzie η to learning rate. Iteracyjne dopasowanie — wagi zbiegają do rozwiązania regresji liniowej.

### Regresja wielomianowa

`PolynomialFeatures(degree=n)` rozszerza X o potęgi cech: [1, x, x², …, xⁿ]. Na rozszerzonej macierzy stosuje się zwykłą `LinearRegression`. Wyższy stopień lepiej dopasowuje nieliniowości, ale grozi przeuczeniem.

### OLS (statsmodels)

`sm.add_constant(X)` dodaje kolumnę stałej. `sm.OLS(endog=y, exog=X).fit()` → `summary()` z R², p-values, współczynnikami.

**Selekcja zmiennych (backward elimination):** pętla usuwa zmienną o najwyższym p-value, dopóki wszystkie p-value ≤ poziom istotności (np. sl = 0.05). `np.argmax(pvalues)` wskazuje kandydat do usunięcia.

---

## Regresja — drzewa decyzyjne

`DecisionTreeRegressor(max_depth=...)` dzieli przestrzeń cech na regiony i przewiduje stałą wartość w liściu. Krzywa predykcji ma kształt „schodków”. `max_depth` ogranicza głębokość — mniejsza głębokość = prostszy model, mniejsze ryzyko przeuczenia. `plot_tree()` wizualizuje strukturę podziałów.

---

## Metryki regresji

- **Błąd:** error = y_true − y_pred
- **MAE** (Mean Absolute Error) — średnia z |error|; w tych samych jednostkach co target
- **MSE** (Mean Squared Error) — średnia z error²; kary za duże błędy
- **RMSE** — √MSE
- **max_error** — największy pojedynczy błąd
- **R²** — współczynnik determinacji; 1 = idealne dopasowanie, 0 = model nie lepszy od średniej, ujemne = gorszy od średniej

Wykres y_true vs y_pred z linią diagonalną (y=x) ocenia jakość predykcji. Histogram błędów pokazuje rozkład reszt.

---

## Regresja logistyczna i klasyfikacja binarna

### Funkcja sigmoid

σ(x) = 1 / (1 + e^(−x)). Dla dużego dodatniego x → σ(x) ≈ 1; dla dużego ujemnego x → σ(x) ≈ 0.

### Funkcja straty (binary cross-entropy)

Dla y=1: l = −log(y_pred). Dla y=0: l = −log(1 − y_pred). Postać zwarta: l = −y·log(y_pred) − (1−y)·log(1−y_pred).

### Funkcja kosztu

L = −(1/m) · Σ[y·log(y_pred) + (1−y)·log(1−y_pred)] — średnia strata; minimalizowana w treningu.

### Prog decyzyjny

Domyślnie: y_pred ≥ 0,5 → klasa 1, w przeciwnym razie klasa 0.

### Pipeline klasyfikacji

1. `train_test_split`
2. `StandardScaler`: `fit` tylko na X_train, `transform` na X_train i X_test (te same μ i σ — unikanie data leakage)
3. `LogisticRegression().fit(X_train, y_train)`
4. `predict(X_test)` — etykiety; `predict_proba(X_test)` — prawdopodobieństwa klas

### Metryki klasyfikacji

- **Confusion matrix** — diagonala = poprawne klasyfikacje; poza diagonalą = błędy (FP, FN)
- **Accuracy** — odsetek poprawnych predykcji
- **classification_report** — precision, recall, F1 per klasa

---

## K-nearest neighbors (KNN)

### Algorytm

KNN to klasyfikator instancyjny (lazy learning): nie buduje jawnego modelu w treningu, tylko zapamiętuje dane. Dla nowego punktu znajduje k najbliższych sąsiadów (domyślnie metryka euklidesowa) i przypisuje klasę większości (głosowanie).

Parametr `n_neighbors` (k) kontroluje liczbę sąsiadów:
- **k=1** — granice bardzo złożone, wysokie ryzyko przeuczenia (overfitting),
- **wyższe k** — gładsze granice decyzyjne, lepsza generalizacja, ale możliwe niedouczenie przy zbyt dużym k.

### Wizualizacja granic decyzyjnych

Workflow:
1. Wytrenuj `KNeighborsClassifier(n_neighbors=k)` na danych 2D.
2. Utwórz siatkę punktów (`np.meshgrid` + `np.arange` z krokiem np. 0.02) obejmującą zakres cech ± margines.
3. Spłaszcz siatkę (`ravel`), połącz w macierz (`np.c_`), wywołaj `predict` na każdym punkcie.
4. `contourf` — wypełnione regiony klas (kolor tła); `contour` — linie granic.
5. `scatter` — rzeczywiste punkty treningowe z `c=target` i `cmap`.

Porównanie granic dla k=1..7 pokazuje ewolucję od poszarpanych do gładkich powierzchni.

### Ograniczenie wymiarów

Granice decyzyjne wizualizuje się w 2D — wymaga dwóch cech. Przy wielu cechach (np. Iris ma 4) redukuje się wymiar do pary cech (sepal_length, sepal_width) kosztem informacji z pozostałych atrybutów.

---

## Eksploracja danych — Iris

`load_iris()` zwraca obiekt Bunch z kluczami: `data` (macierz cech), `target` (etykiety 0/1/2), `target_names`, `feature_names`, `DESCR`.

**Analiza:**
- `DataFrame` z cechami i kolumną `class`,
- `describe()`, `value_counts()`, `corr()` — statystyki i korelacje,
- `sns.pairplot(..., hue='class')` — macierz wykresów par cech z kolorowaniem klas,
- wykres punktowy 2D (matplotlib, plotly) dla wybranych cech.

Korelacja między cechami pomaga ocenić redundancję. Pairplot ujawnia separowalność klas w poszczególnych parach cech.

---

## Zbiory danych ze scikit-learn

- **load_iris** — 4 cechy, 3 klasy, klasyfikacja wieloklasowa
- **load_breast_cancer** — cechy numeryczne, target binarny (0/1), klasyfikacja binarna
- **make_regression** — syntetyczne dane regresji (`n_samples`, `n_features`, `noise`, `random_state`)

Obiekt Bunch: dostęp przez atrybuty (`.data`, `.target`) lub jak słownik (`['data']`).

---

## Przeuczenie i niedouczenie

- **Przeuczenie:** wysoki score na train, niski na test; model zbyt złożony (np. k=1 w KNN, wysoki stopień wielomianu, głębokie drzewo).
- **Niedouczenie:** niski score na train i test; model zbyt prosty (np. regresja liniowa na nieliniowych danych, zbyt duże k w KNN).

Porównanie `score(X_train)` vs `score(X_test)` diagnozuje problem.

---

## Wizualizacja

- **matplotlib** — `scatter`, `plot`, `contourf`, `contour`, `subplots`
- **seaborn** — `pairplot`, `heatmap` (mapa korelacji z maską górnego trójkąta)
- **plotly** — interaktywne wykresy (`px.scatter`, `px.histogram`, `px.bar`, `ff.create_annotated_heatmap`)

`np.random.seed(42)` zapewnia powtarzalność losowych operacji.
