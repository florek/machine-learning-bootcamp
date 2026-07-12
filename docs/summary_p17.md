# Wskaźnik Gini, entropia i zysk informacyjny – kryteria podziału drzew decyzyjnych

Ten plik opisuje **miary nieczystości (impurity)** węzła drzewa klasyfikacyjnego: wskaźnik Gini, entropię Shannona oraz ideę **zysku informacyjnego (information gain)**. Te miary decydują, który podział danych jest „najlepszy” przy budowie drzewa decyzyjnego.

---

## 1. Po co miary nieczystości?

Drzewo decyzyjne rekurencyjnie dzieli dane według wartości cech. W każdym węźle trzeba wybrać **najlepszy podział** – taki, który maksymalnie **zwiększa jednorodność klas** w węzłach potomnych.

Do oceny „mieszaniny klas” w węźle stosuje się miary nieczystości:

- **wskaźnik Gini**,
- **entropię Shannona**.

Im **niższa** wartość miary, tym **czystszy** węzeł (dominuje jedna klasa). Im **wyższa** – tym większy chaos (klasy wymieszane).

---

## 2. Wskaźnik Gini

Wzór dla węzła z \(K\) klasami, gdzie \(p_i\) to udział (proporcja) klasy \(i\):

\[
\text{Gini} = 1 - \sum_{i=1}^{K} p_i^2
\]

**Interpretacja:**

- **Gini = 0** → węzeł **czysty** (100% jednej klasy),
- **Gini bliskie maksimum** → klasy **równomiernie wymieszane**.

**Przykład – trzy klasy po równo (50/150, 50/150, 50/150):**

\[
1 - \left(\frac{50}{150}\right)^2 - \left(\frac{50}{150}\right)^2 - \left(\frac{50}{150}\right)^2 = 1 - 3 \cdot \left(\frac{1}{3}\right)^2 = \frac{2}{3} \approx 0{,}667
\]

**Przykład – dominacja jednej klasy (45/52, 6/52, 1/52):**

Gini jest **niższe** niż przy równomiernym rozkładzie – węzeł jest **czystszy**.

**Właściwości:**

- wartość zawsze w przedziale **[0, 1)** dla klasyfikacji wieloklasowej,
- dla **klasyfikacji binarnej** Gini ∈ **[0, 0,5]**; maksimum 0,5 przy rozkładzie 50/50,
- szybka obliczeniowo (brak logarytmów),
- domyślne kryterium w `DecisionTreeClassifier` w scikit-learn (`criterion='gini'`).

---

## 3. Entropia Shannona

Wzór (logarytm o podstawie 2 – wynik w **bitach informacji**):

\[
H = -\sum_{i=1}^{K} p_i \cdot \log_2(p_i)
\]

**Konwencja:** gdy \(p_i = 0\), składnik \(p_i \log_2(p_i)\) traktuje się jako **0** (granica matematyczna).

**Interpretacja:**

- **H = 0** → węzeł czysty (jedna klasa),
- **H rośnie** wraz z mieszaniem klas,
- dla **dwóch klas** z \(p = q = 0{,}5\) entropia osiąga **maksimum = 1 bit**.

**Przykład – trzy klasy po równo:**

\[
-\frac{1}{3}\log_2\frac{1}{3} - \frac{1}{3}\log_2\frac{1}{3} - \frac{1}{3}\log_2\frac{1}{3} \approx 1{,}585 \text{ bitów}
\]

**Przykład – rozkład skrajnie nierównomierny (0,95 / 0,05):**

Entropia jest **bliska zeru** – prawie czysty węzeł.

**Alternatywne kryterium w sklearn:** `criterion='entropy'` w `DecisionTreeClassifier`.

---

## 4. Porównanie Gini vs entropia

| Aspekt | Wskaźnik Gini | Entropia |
|--------|---------------|----------|
| Wzór | \(1 - \sum p_i^2\) | \(-\sum p_i \log_2 p_i\) |
| Czysty węzeł | 0 | 0 |
| Obliczenia | bez logarytmów | wymaga logarytmów |
| Maksimum (2 klasy, 50/50) | 0,5 | 1,0 (bit) |
| Kierunek optymalizacji | minimalizacja | minimalizacja |

Obie miary **maleją**, gdy węzeł staje się bardziej jednorodny, i **rosną**, gdy klasy są wymieszane. W praktyce drzewa zbudowane na Gini i entropii często dają **bardzo podobne** wyniki.

---

## 5. Entropia w Pythonie

### Ręczna implementacja (NumPy)

```python
def entropy(probabilities: np.ndarray) -> float:
    return -np.sum(probabilities * np.log2(probabilities))
```

Tablica `probabilities` musi zawierać **nieujemne udziały klas sumujące się do 1** (np. `[0.5, 0.5]`).

### scipy.stats.entropy

```python
from scipy.stats import entropy

entropy([0.5, 0.5], base=2)   # maksimum dla dwóch klas: 1.0
entropy([0.8, 0.2], base=2)   # niższa wartość – bardziej nierównomierny rozkład
entropy([0.95, 0.05], base=2) # blisko zera – prawie czysty węzeł
```

Parametr **`base=2`** daje wynik w bitach, zgodnie ze wzorem Shannona w kontekście drzew decyzyjnych. Bez `base=2` funkcja `scipy.stats.entropy` domyślnie stosuje **logarytm naturalny** – wynik ma inną skalę niż wzór z log₂.

---

## 6. Krzywa entropii dla rozkładu binarnego

Dla dwóch klas z prawdopodobieństwem \(p\) i \(q = 1 - p\):

- entropia jest **symetryczna** względem \(p = 0{,}5\),
- **maksimum** przy \(p = 0{,}5\) (maksymalna niepewność),
- **minimum** przy \(p \to 0\) lub \(p \to 1\) (pewność co do klasy).

Wizualizacja: wykres entropii w funkcji \(p\) ma kształt **„dzwonu”** – rośnie od 0 do 1 bit, potem maleje z powrotem do 0. Do narysowania krzywej można wygenerować serię wartości \(p\) (np. od 0,01 do 0,99) z komplementarnym \(q = 1 - p\) i obliczyć entropię dla każdej pary \([p, q]\).

---

## 7. Zysk informacyjny (Information Gain)

**Zysk informacyjny** mierzy, o ile **spada nieczystość** węzła po wykonaniu podziału:

\[
\text{IG} = H_{\text{rodzic}} - \sum_{\text{dzieci}} \frac{n_{\text{dziecko}}}{n_{\text{rodzic}}} \cdot H_{\text{dziecko}}
\]

Gdzie:

- \(H_{\text{rodzic}}\) – entropia (lub Gini) węzła przed podziałem,
- drugi składnik – **ważona średnia** entropii węzłów potomnych (waga = liczność dziecka / liczność rodzica).

**Reguła wyboru podziału:**

- drzewo wybiera podział z **największym zyskiem informacyjnym**,
- czyli tam, gdzie **entropia (lub Gini) po podziale jest najmniejsza** w sensie ważonej średniej.

**Intuicja z ćwiczenia:** „Zysk informacyjny jest tam, gdzie entropia jest mniejsza przy podziale” – dobry podział **redukuje chaos** w węzłach potomnych.

**Przykład liczbowy:**

- entropia rodzica = 1,0 bit,
- ważona entropia dzieci po podziale = 0,3 bit,
- zysk informacyjny = 1,0 − 0,3 = **0,7 bit**.

---

## 8. Powiązanie z drzewami decyzyjnymi w scikit-learn

`DecisionTreeClassifier` domyślnie buduje drzewo, wybierając w każdym węźle podział minimalizujący **Gini** (`criterion='gini'`).

Alternatywa: `criterion='entropy'` – wybór podziału maksymalizujący zysk informacyjny w sensie entropii.

Parametry ograniczające złożoność (znane z wcześniejszych lekcji):

- **`max_depth`** – maksymalna głębokość drzewa,
- **`min_samples_split`** – minimalna liczba próbek do podziału węzła,
- **`min_samples_leaf`** – minimalna liczność liścia.

Bez ograniczeń drzewo może dopasować się idealnie do danych treningowych → **overfitting**.

---

## 9. Pułapki i dobre praktyki

- **Proporcje klas:** udziały \(p_i\) liczy się jako **liczba obserwacji klasy i / łączna liczba w węźle**, nie jako globalny rozkład całego zbioru.
- **Logarytm zerowy:** przy implementacji ręcznej upewnij się, że składniki z \(p_i = 0\) nie dają NaN (konwencja 0·log(0) = 0).
- **Gini vs entropia:** wybór kryterium rzadko zmienia jakość modelu drastycznie; ważniejsze są **ograniczenia głębokości** i **walidacja**.
- **Overfitting:** czyste liście (Gini = 0, H = 0) na danych treningowych nie gwarantują dobrej generalizacji – stosuj podział train/test i parametry regularyzacji.
- **Porównanie z KNN:** drzewo **uczy strukturę** podczas `fit()` (parametryczne w sensie struktury); KNN **zapamiętuje** dane (lazy learning).

---

## 10. Podsumowanie

- **Wskaźnik Gini:** \(1 - \sum p_i^2\); 0 = czysty węzeł, wyższe = większe mieszanie klas.
- **Entropia:** \(-\sum p_i \log_2 p_i\); 0 = czysty węzeł, maksimum przy równomiernym rozkładzie.
- **Zysk informacyjny:** spadek ważonej entropii (lub Gini) po podziale; drzewo wybiera podział z **największym IG**.
- **scipy.stats.entropy(..., base=2)** i ręczna funkcja NumPy dają ten sam wynik co wzór Shannona w bitach.
- Miary nieczystości to **fundament klasyfikacyjnych drzew decyzyjnych** – łączą statystykę rozkładu klas z algorytmem uczenia struktury modelu.
