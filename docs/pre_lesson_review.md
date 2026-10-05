# Szybka powtórka przed lekcją

Ultra-skondensowane przypomnienie najważniejszych rzeczy z każdej lekcji. Użyj przed każdą lekcją, żeby szybko odświeżyć wiedzę.

---

## 📌 P6: Gradient Descent

**Co robi:** Ręczna implementacja regresji liniowej metodą gradient descent

**Kluczowe elementy:**
- `bias = np.ones((m, 1))` → dodanie wyrazu wolnego
- `gradient = (2/m) * X.T.dot(X.dot(weights) - Y)` → obliczenie gradientu
- `weights = weights - eta * gradient` → aktualizacja wag
- Pętla 3000 iteracji → uczenie modelu

**Równanie:** `Y = w0 * 1 + w1 * X`

---

## 📌 P7: Regresja liniowa scikit-learn

**Co robi:** Regresja liniowa na syntetycznych danych

**Kluczowe elementy:**
- `make_regression()` → generowanie danych testowych
- `LinearRegression().fit()` → trenowanie modelu
- `coef_`, `intercept_` → parametry modelu
- `score()` → R² (współczynnik determinacji)

**Równanie:** `y = intercept_ + coef_[0] * x`

---

## 📌 P8: Train/Test Split

**Co robi:** Podział danych i ocena modelu na osobnych zbiorach

**Kluczowe elementy:**
- `train_test_split(test_size=0.25)` → podział 75/25
- `score(X_train)` vs `score(X_test)` → wykrycie overfittingu
- Analiza błędów: `error = y_test - y_pred`
- Histogram błędów → rozkład powinien być normalny wokół zera

**Złota zasada:** Model trenuje na train, ocenia na test!

---

## 📌 P9: Rzeczywiste dane + EDA

**Co robi:** Pełny pipeline: eksploracja → feature engineering → modelowanie

**Kluczowe elementy:**
- `read_csv()` → wczytanie danych
- `drop_duplicates()` → usuwanie duplikatów
- `get_dummies(drop_first=True)` → one-hot encoding
- `corr()` → macierz korelacji
- `mean_absolute_error()` → metryka MAE

**Pipeline:**
1. EDA (`info()`, `describe()`, `value_counts()`)
2. Czyszczenie (duplikaty, braki)
3. Feature engineering (one-hot encoding)
4. Analiza korelacji
5. Train/test split
6. Trenowanie i ocena

---

## 📌 P10: OLS statsmodels + selekcja zmiennych

**Co robi:** Model OLS z analizą statystyczną i ręczną backward elimination

**Kluczowe elementy:**
- `pd.get_dummies().values.astype(float)` → przygotowanie danych
- `sm.add_constant()` → dodanie intercept
- `sm.OLS().fit()` → model OLS
- `ols.summary()` → statystyki (p-value, R²)
- Ręczna backward elimination → krok po kroku usuwanie nieistotnych zmiennych

**Interpretacja p-value:**
- **p < 0.05** → istotna statystycznie ✅
- **p ≥ 0.05** → nieistotna (usuń) ❌

**Proces selekcji:**
1. Pełny model → sprawdź p-value
2. Usuń zmienną z najwyższym p ≥ 0.05 (ręcznie)
3. Powtórz dla nowego modelu

---

## 📌 P11: Automatyczna backward elimination

**Co robi:** Automatyczna selekcja zmiennych w pętli while

**Kluczowe elementy:**
- `while True:` → automatyczna pętla eliminacji
- `max(ols.pvalues)` → najwyższe p-value
- `np.argmax()` → indeks zmiennej z najwyższym p-value
- `np.delete(array, idx, axis=1)` → usunięcie kolumny
- `ols.save('model.pickle')` → zapis modelu do pliku

**Proces automatyczny:**
1. Dopasuj model → znajdź max p-value
2. Jeśli max p-value > 0.05 → usuń zmienną
3. Powtórz, dopóki wszystkie zmienne są istotne (p ≤ 0.05)

**Różnica od P10:**
- P10: ręczne usuwanie (3 kroki)
- P11: automatyczna pętla (działa dla dowolnej liczby zmiennych)

---

## 📌 P12: Regresja wielomianowa

**Co robi:** Uchwycenie nieliniowej zależności (wielomianowej) przez rozszerzenie cech (X, X², X³) i zwykłą regresję liniową

**Kluczowe elementy:**
- `np.random.seed(42)` → powtarzalność danych i szumu
- `X.reshape(n, 1)` → kształt 2D (próbki × cechy) dla scikit-learn
- Regresja liniowa na jednej cesze → słabe R² przy zależności wielomianowej
- `PolynomialFeatures(degree=k)` → tworzy cechy 1, X, X², …; potem `LinearRegression`
- `r2_score(y, y_pred)` / `score()` → ocena dopasowania

**Zasada:** Model pozostaje liniowy względem parametrów; nieliniowość wynika z transformacji cech. Przy wysokim stopniu i małej liczbie danych – ryzyko przeuczenia (regularyzacja lub niższy stopień).

---

## 📌 P13: Regresja drzewa decyzyjnego

**Co robi:** Model regresji, który dzieli oś cechy na przedziały i w każdym przewiduje stałą (średnią); przy nieliniowej zależności daje „schodkową” krzywą zamiast prostej.

**Kluczowe elementy:**
- `DecisionTreeRegressor(max_depth=k)` → drzewo o ograniczonej głębokości
- `fit(data, target)`, `predict(plot_data)` → API jak w LinearRegression
- `plot_tree(regressor, filled=True, rounded=True, feature_names=[...])` → wizualizacja struktury drzewa
- Większe max_depth → więcej schodków, lepsze dopasowanie, większe ryzyko przeuczenia

**Porównanie:** Regresja liniowa = jedna prosta; drzewo = schodki dopasowane do krzywej. Do wizualizacji krzywej używa się gęstej siatki punktów (np. np.arange().reshape(-1, 1)).

---

## 📌 P14: Metryki regresji i wizualizacja

**Co robi:** Ocena modelu regresji za pomocą MAE, MSE, RMSE, max_error, R² oraz wizualizacja: wykres y_true vs y_pred (z linią y=x) i histogram błędów.

**Kluczowe elementy:**
- `mean_absolute_error`, `mean_squared_error`, `r2_score`, `max_error` z sklearn.metrics
- RMSE: `mean_squared_error(y_true, y_pred, squared=False)`
- Wykres punktowy true vs pred + linia y=x; histogram kolumny error

---

## 📌 P15: Regresja logistyczna – teoria i klasyfikacja w sklearn

**Co robi:** Klasyfikacja binarna – teoria straty (binary cross-entropy), sigmoida, pipeline z LogisticRegression i metrykami.

**Kluczowe elementy (teoria):**
- Strata: y=1 → −log(y_pred); y=0 → −log(1−y_pred); postać zwarta binary cross-entropy
- Funkcja kosztu = średnia strat; minimalizowana w treningu
- Sigmoida σ(x)=1/(1+e^(−x)); próg 0,5 → klasa 0 vs 1

**Kluczowe elementy (praktyka):**
- `load_breast_cancer()` → data + target (klasy 0/1)
- `train_test_split` → train / test
- `StandardScaler`: fit na train, transform na train i test (bez leakage)
- `LogisticRegression`: fit, predict (etykiety), predict_proba (prawdopodobieństwa)
- `accuracy_score`, `confusion_matrix`, `classification_report`
- Confusion matrix: wiersze = prawdziwe etykiety, kolumny = predykcje
- Accuracy przy niezbalansowanych klasach może być myląco wysoka – patrz na F1 i recall

---

## 📌 P16: K-Nearest Neighbors (KNN)

**Co robi:** Klasyfikacja wieloklasowa na zbiorze Iris z wizualizacją granic decyzyjnych dla różnych k.

**Kluczowe elementy:**
- `KNeighborsClassifier(n_neighbors=k)` → klasyfikacja przez głosowanie k najbliższych sąsiadów
- `load_iris()` → 4 cechy, 3 klasy (klasyfikacja wieloklasowa)
- `fit()` zapamiętuje dane (lazy learning); `predict()` wyszukuje sąsiadów
- Małe k (np. 1) → poszarpane granice, ryzyko overfittingu; duże k → wygładzone granice
- Wizualizacja granic: `meshgrid` + `predict` na siatce + `contourf` / `scatter`
- EDA: `pairplot` z `hue='class'`, `corr()`, redukcja do 2 cech do wykresu 2D

**Porównanie:** LogisticRegression = granice liniowe; KNN = nieregularne granice zależne od k i rozkładu punktów.

---

## 📌 P17: Wskaźnik Gini, entropia, zysk informacyjny

**Co robi:** Miary nieczystości węzła drzewa klasyfikacyjnego – obliczenia Gini, entropii i zysku informacyjnego przy wyborze podziału.

**Kluczowe elementy:**
- Gini = 1 − Σp_i²; Gini = 0 → węzeł czysty (jedna klasa)
- Klasyfikacja binarna: Gini ∈ [0, 0,5]; maksimum 0,5 przy rozkładzie 50/50
- Entropia H = −Σ p_i·log₂(p_i); H = 0 → czysty węzeł; binarna maksimum 1 bit przy 50/50
- `scipy.stats.entropy(..., base=2)` → wynik w bitach; bez `base=2` → logarytm naturalny (inna skala)
- IG = H_rodzic − ważona średnia H_dzieci; drzewo wybiera podział z **największym IG**
- `criterion='gini'` (domyślnie) lub `criterion='entropy'` w DecisionTreeClassifier
- `max_depth`, `min_samples_split`, `min_samples_leaf` – ograniczanie przeuczenia

**Porównanie:** Gini szybsze (bez logarytmów); drzewa z Gini i entropii często dają bardzo podobne wyniki.

---

## 📌 P18: Klasyfikacyjne drzewo decyzyjne

**Co robi:** `DecisionTreeClassifier` na Iris (2 cechy), wizualizacja granic i grafu drzewa, porównanie `max_depth`.

**Kluczowe elementy:**
- `DecisionTreeClassifier(max_depth=k)` → klasyfikacja; w liściu klasa większościowa
- Redukcja Iris do 2 cech → wizualizacja granic na płaszczyźnie (tracimy pozostałe atrybuty)
- Iris: klasy zrównoważone (po 50); przy cechach działka `versicolor`/`virginica` często się nakładają
- `plot_decision_regions` → kolorowe regiony klas (podziały prostopadłe do osi)
- `export_graphviz` → DOT; render PNG (np. pydotplus) → graf z `feature_names` i `class_names`
- `score()` klasyfikatora → accuracy
- Większe `max_depth` → bardziej złożone regiony, wyższa accuracy na train, ryzyko overfittingu
- Ocena tylko na danych treningowych **nie** wykrywa przeuczenia
- Funkcja pomocnicza (train + granice + graf) ułatwia porównanie różnych `max_depth`

**Porównanie:** LogisticRegression = granica liniowa; KNN = nieregularne; drzewo = prostokątne / schodkowe regiony.

---

## 📌 Random Forest (las losowy)

**Co robi:** Ensemble wielu drzew decyzyjnych; klasyfikacja (Iris), ważne cechy, ocena na zbiorze testowym.

**Kluczowe elementy:**
- `RandomForestClassifier` z `sklearn.ensemble` → uczenie zespołowe
- `n_estimators` → liczba drzew (np. 100)
- końcowa klasa = głosowanie większościowe drzew
- 2 cechy → granice na wykresie 2D; 4 cechy → więcej informacji, bez wygodnej wizualizacji 2D
- `train_test_split` + `accuracy_score` na teście (nie oceniaj wyłącznie na train)
- `feature_importances_` → względna ważność cech (w Iris zwykle petal length / petal width)
- accuracy = 1.0 na małym teście ≠ gwarancja idealnego modelu na nowych danych

---

## 📌 SVM (Support Vector Machine)

**Co robi:** Szuka granicy decyzyjnej z maksymalnym marginesem; kernele umożliwiają problemy nieliniowe.

**Kluczowe elementy:**
- margines = pas między klasami; SVM maksymalizuje jego szerokość
- support vectors = punkty najbliżej granicy
- hard margin vs soft margin (elastyczność przy outlierach)
- `C` → kompromis: małe C = szerszy margines / większa tolerancja błędów; duże C = węższy margines
- `SVC` z `sklearn.svm`; domyślny `kernel="rbf"`
- kernele: `linear`, `rbf`, `poly` (np. `degree=3`)
- kernel = przekształcenie przestrzeni → problem nieliniowy może stać się liniowo separowalny
- praktyka Iris: często 2 cechy (np. petal length, sepal width) → granice 2D
- SVM zwykle wymaga `StandardScaler` (`fit` tylko na train, `transform` na train i test)
- `kernel="linear"` vs `kernel="rbf"` → inny kształt granicy; na łatwych 2 klasach wynik może być podobny
- 2 klasy Iris bywają łatwo liniowo separowalne (~100% accuracy); 3 klasy trudniejsze
- mały zbiór (150 próbek) + brak `random_state` → zmienne wyniki; test czasem > train

---

## 📌 Naive Bayes (naiwny klasyfikator Bayesa)

**Co robi:** Liczy prawdopodobieństwo klasy na podstawie cech, korzystając z twierdzenia Bayesa i silnego uproszczenia o niezależności cech.

**Kluczowe elementy:**
- twierdzenie Bayesa: `P(A|B) = P(B|A) * P(A) / P(B)`
- „naiwny” = założenie, że cechy są (warunkowo) niezależne względem klasy
- założenie często nie jest dokładnie spełnione, a mimo to model bywa skuteczny
- `prior` = `P(Y)`; `likelihood` = `P(Xi|Y)`; `evidence` = `P(X)`; `posterior` = `P(Y|X)`
- wspólne `P(X1,...,Xn|Y)` upraszcza się do iloczynu `P(X1|Y) * ... * P(Xn|Y)`
- przy porównaniu klas mianownik `P(X)` jest ten sam → wystarczy porównać liczniki
- klasyfikacja: wybór klasy z największym posterior
- odmiany (Gaussian, Multinomial, …) różnią się założeniem o rozkładzie cech
- cechy dyskretne → częstości; cechy ciągłe → potrzebne założenie o rozkładzie (np. Gauss)
- Gaussian Naive Bayes: rozkład normalny cechy w obrębie klasy; w sklearn: `GaussianNB`
- przy samym wyborze klasy wystarczą liczniki; normalizacja (dzielenie przez sumę) daje % sumujące się do 1
- typowe zastosowania: klasyfikacja dokumentów, filtrowanie spamu
- praktyka Python (spacer): `LabelEncoder` dla targetu (`nie→0`, `tak→1`); `get_dummies(..., drop_first=True)` dla cech; `pop()` oddziela target
- `GaussianNB().fit` → `score` ≈ 77,7% (ok. 7/9); `predict` zwraca klasę (np. `1` = tak); `predict_proba` zwraca P(nie), P(tak); `encoder.classes_` odtwarza etykietę tekstową
- prawdopodobieństwa z `GaussianNB` mogą różnić się od ręcznego przykładu dyskretnego — inny wariant modelu

---

## 🔄 Powtarzające się koncepty (wszystkie lekcje)

### Importy (standardowe)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### Konfiguracja
```python
np.random.seed(42)
sns.set(font_scale=1.3)
```

### One-Hot Encoding
```python
df_dummies = pd.get_dummies(df, drop_first=True)
```

### Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
```

### Model regresji
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)
```

---

## ⚡ Szybkie przypomnienie przed lekcją

**Przed P6:** Pamiętaj o reshape danych i dodaniu biasu

**Przed P7:** `make_regression()` do generowania danych, `score()` zwraca R²

**Przed P8:** Zawsze dziel dane na train/test przed trenowaniem!

**Przed P9:** EDA → czyszczenie → encoding → modelowanie

**Przed P10:** `drop_first=True` w get_dummies, `.astype(float)` przed statsmodels, p-value < 0.05 = istotna

**Przed P11:** `while True` z `break`, `np.argmax()` do znajdowania indeksu, `np.delete()` do usuwania kolumn

**Przed P12:** `reshape(n, 1)` dla jednej cechy, regresja wielomianowa = rozszerzenie cech + LinearRegression, R² przy nieliniowej zależności

**Przed P13:** DecisionTreeRegressor(max_depth=k), plot_tree do wizualizacji struktury, krzywa predykcji = schodki; max_depth kontroluje złożoność i przeuczenie

**Przed P14:** MAE, MSE, RMSE (squared=False), max_error, r2_score; wykres true vs pred z linią y=x, histogram błędów

**Przed P15:** Binary cross-entropy (y=1 → −log(y_pred), y=0 → −log(1−y_pred)), sigmoida, próg 0,5; StandardScaler (fit train); LogisticRegression; classification_report; confusion matrix (wiersze=prawda, kolumny=pred); accuracy przy imbalance

**Przed P16:** KNeighborsClassifier, n_neighbors, lazy learning; granice decyzyjne (meshgrid + predict); Iris = klasyfikacja wieloklasowa; małe k vs duże k

**Przed P17:** Gini = 1 − Σp_i²; entropia = −Σ p_i·log₂(p_i); IG = spadek ważonej entropii po podziale; criterion='gini' vs 'entropy'; scipy.stats.entropy(..., base=2)

**Przed P18:** DecisionTreeClassifier(max_depth=k), score=accuracy; 2 cechy do granic 2D; plot_decision_regions; export_graphviz; duże max_depth + ocena tylko na train = ryzyko ukrytego overfittingu

**Przed Random Forest:** Ensemble = wiele drzew + głosowanie; n_estimators; oceniaj na teście; feature_importances_; accuracy 1.0 na małym Iris ≠ model idealny

**Przed SVM:** maksymalny margines; support vectors; hard vs soft margin; C; kernel linear/rbf/poly; SVC; transformacja przestrzeni; StandardScaler (fit tylko train); 2 cechy do granic 2D; linear vs rbf; mały Iris → ostrożna interpretacja accuracy; `random_state`

**Przed Naive Bayes:** twierdzenie Bayesa; „naiwność” = niezależność cech; prior × likelihoods / evidence = posterior; mianownik często niepotrzebny przy porównaniu klas; wybór max posterior; Gaussian vs Multinomial (różny rozkład cech); dyskretne vs ciągłe; `GaussianNB`; opcjonalna normalizacja do %; w Pythonie: LabelEncoder + get_dummies + pop → fit/score/predict/predict_proba; wynik Gauss ≠ zawsze wynik ręcznych częstości

---

## 🎯 Najważniejsze zasady

1. **random_state=42** → powtarzalność
2. **train/test split** → zawsze przed trenowaniem
3. **drop_first=True** → unika kolinearności
4. **EDA przed modelowaniem** → zrozum dane
5. **score(train) vs score(test)** → wykrycie overfittingu
6. **p-value < 0.05** → zmienna istotna

---

> **Użycie:** Przeczytaj sekcję dla danej lekcji przed zajęciami. Pełne wyjaśnienia w summary_p6.md – summary_p18.md, szczegóły techniczne w cheat_sheet.md.
