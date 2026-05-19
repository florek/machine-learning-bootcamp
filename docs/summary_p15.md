# Regresja logistyczna – funkcja straty i klasyfikacja binarna

Ten plik opisuje **funkcję straty w regresji logistycznej** (binary cross-entropy), jej postać kawałkową i zwartą, **funkcję kosztu** oraz **klasyfikację binarną** z krzywą sigmoidalną i progiem decyzyjnym.

---

## 1. Regresja logistyczna – funkcja straty (dla jednej obserwacji)

Dla pojedynczej obserwacji i (gdzie i = 1, …, m) funkcja straty zależy od prawdziwej etykiety y_i (0 lub 1) oraz przewidywanej prawdopodobieństwa y_pred_i (wynik sigmoidy).

### 1.1 Postać kawałkowa

**Gdy y_i = 1:**  
l(y_pred_i) = −log(y_pred_i)

**Gdy y_i = 0:**  
l(y_pred_i) = −log(1 − y_pred_i)

- Gdy **y_i = 1**: strata to −log(y_pred_i). Im bliżej y_pred_i do 1 (dobra predykcja), tym strata bliżej 0; im y_pred_i bliżej 0 (zła predykcja), tym strata rośnie do +∞.
- Gdy **y_i = 0**: strata to −log(1 − y_pred_i). Im bliżej y_pred_i do 0 (dobra predykcja), tym strata bliżej 0; im y_pred_i bliżej 1 (zła predykcja), tym strata rośnie do +∞.

Zmienna y_pred_i to przewidywane prawdopodobieństwo przynależności do klasy 1 (wyjście funkcji sigmoidalnej) dla i-tej obserwacji.

---

## 2. Postać zwarta (binary cross-entropy dla jednej obserwacji)

Obie gałęzie można zapisać jednym wzorem:

**l(y_pred_i) = −y_i · log(y_pred_i) − (1 − y_i) · log(1 − y_pred_i)**

- Dla y_i = 1: drugi składnik znika (1 − y_i = 0), zostaje −log(y_pred_i).
- Dla y_i = 0: pierwszy składnik znika (y_i = 0), zostaje −log(1 − y_pred_i).

To **binary cross-entropy** (entropia krzyżowa) dla jednej próbki.

---

## 3. Funkcja kosztu (średnia strata na całym zbiorze)

Funkcja kosztu L(y_pred) to średnia ze strat po wszystkich m obserwacjach:

**L(y_pred) = −(1/m) · Σ od i=1 do m [ y_i · log(y_pred_i) + (1 − y_i) · log(1 − y_pred_i) ]**

- Suma po i=1…m agreguje straty dla każdej obserwacji.
- Czynnik 1/m daje średnią stratę.
- Minus przed sumą sprawia, że strata jest nieujemna (logarytmy prawdopodobieństw z przedziału (0, 1) są ujemne).  
W treningu regresji logistycznej **minimalizuje się** tę funkcję kosztu.

---

## 4. Wykresy funkcji straty

- **Wykres dla y = 1**: oś X = y_pred_i (0–1), oś Y = l(y_pred_i). Gdy y_pred_i → 1, strata → 0; gdy y_pred_i → 0, strata rośnie (nawet do +∞). Krzywa opada z lewej do prawej.
- **Wykres dla y = 0**: oś X = y_pred_i (0–1), oś Y = l(y_pred_i). Gdy y_pred_i → 0, strata → 0; gdy y_pred_i → 1, strata rośnie. Krzywa rośnie z lewej do prawej.

W obu przypadkach strata jest duża, gdy model się myli (przewiduje przeciw do prawdziwej etykiety), i mała, gdy przewiduje zgodnie z prawdą.

---

## 5. Regresja logistyczna – klasyfikacja binarna (wykres)

Na wykresie **Scoring (oś X)** vs **Decyzja (oś Y)**:

- **Oś X (Scoring)**: zmienna wejściowa (cecha lub wynik liniowy np. wᵀx).
- **Oś Y (Decyzja)**: przewidywane prawdopodobieństwo przynależności do klasy 1 (zakres 0–1).

**Krzywa sigmoidalna (S):**
- Czerwona krzywa w kształcie S to **funkcja logistyczna (sigmoida)**. Dla niskiego Scoringu prawdopodobieństwo jest bliskie 0, dla wysokiego – bliskie 1.
- Model zwraca **prawdopodobieństwo** klasy 1 w zależności od wartości Scoringu.

**Próg decyzyjny 0,5:**
- Pozioma linia na wysokości y = 0,5 to **próg decyzyjny**. Zazwyczaj: jeśli y_pred ≥ 0,5 → klasa 1, jeśli y_pred < 0,5 → klasa 0.
- Punkt przecięcia sigmoidy z linią y = 0,5 wyznacza **granicę decyzyjną w przestrzeni Scoringu**: obserwacje z Scoringiem większym od tej wartości są klasyfikowane jako 1, mniejszym – jako 0.

**Punkty danych:**
- Niebieskie punkty przy y = 0 lub y = 1 to prawdziwe etykiety. Strefa nakładania się klas (np. Scoring 20–30) to obszar niepewności, gdzie klasy nie są idealnie rozdzielone.

---

## 6. Funkcja sigmoidalna

Sigmoida mapuje dowolną wartość rzeczywistą na przedział (0, 1):

**σ(x) = 1 / (1 + e^(−x))**

- Dla ujemnego x wynik zbliża się do 0; dla dodatniego x – do 1.
- Wykres ma kształt litery S; punkt przecięcia z y = 0,5 odpowiada x = 0.
- W regresji logistycznej sigmoida przekształca wynik liniowy (scoring) na prawdopodobieństwo klasy 1.

---

## 7. Zbiór danych do klasyfikacji binarnej

Gotowy zbiór z scikit-learn (np. dane o nowotworze piersi) zwraca słownik z kluczami m.in. `data` (macierz cech), `target` (etykiety 0/1) oraz opisem `DESCR`. Cechy są numeryczne; target to klasy binarne – typowy punkt startowy do nauki klasyfikacji.

---

## 8. Podział i skalowanie cech

**Podział train/test:** `train_test_split` oddziela cechy i etykiety na zbiór treningowy i testowy. Model uczy się wyłącznie na train; ocena na test symuluje nowe dane.

**StandardScaler:** standaryzacja (średnia 0, odchylenie 1). Kolejność:
1. `fit` **tylko** na `X_train` – parametry (średnia, std) liczone z danych treningowych.
2. `transform` na `X_train` i `X_test` – oba zbiory skalowane tymi samymi parametrami.

**Pułapka:** `fit` na połączonych train+test lub osobny `fit` na test → **data leakage** (model pośrednio „widzi” test). Poprawnie: fit na train, transform na obu.

---

## 9. LogisticRegression w scikit-learn

**LogisticRegression** służy do **klasyfikacji** (nie regresji ciągłej). API jak w innych estymatorach sklearn:
- `fit(X_train, y_train)` – uczenie (minimalizacja funkcji kosztu / binary cross-entropy).
- `predict(X_test)` – przewidywane **etykiety** klas (0 lub 1) po progu domyślnym 0,5.
- `predict_proba(X_test)` – **prawdopodobieństwa** dla każdej klasy (np. kolumna dla klasy 0 i dla klasy 1); suma w wierszu = 1.

Regresja logistyczna zakłada zależność liniową między cechami a logitem szansy; skalowanie cech często poprawia zbieżność i stabilność.

---

## 10. Metryki klasyfikacji

**Accuracy (dokładność):** odsetek poprawnych predykcji – `accuracy_score(y_test, y_pred)`. Prosta metryka; przy niezbalansowanych klasach może być myląca.

**Macierz pomyłek (confusion matrix):** tabela 2×2 (dla dwóch klas):
- wiersze → prawdziwe etykiety,
- kolumny → predykcje modelu,
- elementy na diagonali → poprawne klasyfikacje,
- poza diagonalą → błędy (fałszywe pozytywy / fałszywe negatywy).

**ConfusionMatrixDisplay** – wizualizacja macierzy w matplotlib.

**classification_report** – zestawienie precision, recall, F1-score i support per klasa; uzupełnia samą accuracy o jakość na poziomie klas.

---

## 11. Wizualizacja macierzy pomyłek (Plotly)

Macierz można przedstawić jako heatmapę z adnotacjami (np. Plotly `create_annotated_heatmap`). Czasem odwraca się kolejność wierszy (`cm[::-1]`), aby oś Y odpowiadała intuicyjnej kolejności klas (np. true_1 u góry). Etykiety osi: kolumny = predykcje, wiersze = prawdziwe klasy.

---

## 12. Podsumowanie

- **Funkcja straty (obserwacja)**: postać kawałkowa dla y=1 i y=0; postać zwarta = binary cross-entropy dla jednej próbki.
- **Funkcja kosztu**: średnia z binary cross-entropy po wszystkich obserwacjach; jest minimalizowana w trakcie uczenia.
- **Wykresy straty**: dla y=1 strata maleje, gdy y_pred → 1; dla y=0 strata maleje, gdy y_pred → 0.
- **Klasyfikacja binarna**: sigmoida mapuje scoring na prawdopodobieństwo; próg 0,5 wyznacza decyzję (klasa 0 vs 1) i granicę w przestrzeni cech.
- **Pipeline praktyczny**: podział train/test → StandardScaler (fit train) → LogisticRegression → predict / predict_proba → accuracy, confusion matrix, classification_report.
