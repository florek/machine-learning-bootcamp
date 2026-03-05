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

## 6. Podsumowanie

- **Funkcja straty (obserwacja)**: postać kawałkowa dla y=1 i y=0; postać zwarta = binary cross-entropy dla jednej próbki.
- **Funkcja kosztu**: średnia z binary cross-entropy po wszystkich obserwacjach; jest minimalizowana w trakcie uczenia.
- **Wykresy straty**: dla y=1 strata maleje, gdy y_pred → 1; dla y=0 strata maleje, gdy y_pred → 0.
- **Klasyfikacja binarna**: sigmoida mapuje Scoring na prawdopodobieństwo; próg 0,5 wyznacza decyzję (klasa 0 vs 1) i granicę w przestrzeni cech.
