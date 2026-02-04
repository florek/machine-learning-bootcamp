# Automatyczna selekcja zmiennych – Backward Elimination (statsmodels)

Ten plik opisuje **automatyczną implementację backward elimination** do selekcji zmiennych w modelu OLS. W przeciwieństwie do p10.py, gdzie zmienne były usuwane ręcznie, tutaj proces jest zautomatyzowany w pętli.

---

## 1. Przygotowanie danych

### 1.1 Wczytanie i wstępna analiza

```python
df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
df = df.drop_duplicates()
```

Identyczny proces jak w p9.py i p10.py – wczytanie danych insurance.csv i usunięcie duplikatów.

### 1.2 Podział na zbiory treningowy i testowy

```python
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
```

Podział 80/20 z ustalonym ziarnem losowym.

---

## 2. Przygotowanie danych do modelu OLS

### 2.1 Kodowanie zmiennych kategorycznych

```python
X_train_numpy = pd.get_dummies(X_train, drop_first=True)
predictors = ['const'] + list(X_train_numpy.columns)
X_train_numpy = X_train_numpy.values.astype(float)
X_train_numpy = sm.add_constant(X_train_numpy)
```

**Proces:**
1. `get_dummies(drop_first=True)` → one-hot encoding z usunięciem pierwszej kategorii
2. `predictors` → lista nazw zmiennych (dla późniejszego wyświetlania)
3. `.values.astype(float)` → konwersja na numpy array typu float
4. `add_constant()` → dodanie kolumny stałej (intercept)

**Dlaczego to ważne:**
- `statsmodels` wymaga numpy array typu float
- Boolean z `get_dummies()` muszą być przekonwertowane na 1.0/0.0
- Kolumna stałej jest wymagana dla intercept w modelu OLS

---

## 3. Automatyczna Backward Elimination

### 3.1 Pętla eliminacji

```python
sl = 0.05
while True:
    ols = sm.OLS(endog=y_train.values, exog=X_train_numpy).fit()
    max_pval = max(ols.pvalues.astype('float'))
    if max_pval > sl:
        max_idx = np.argmax(ols.pvalues.astype('float'))
        X_train_numpy = np.delete(X_train_numpy, max_idx, axis=1)
        predictors.remove(predictors[max_idx])
    else:
        break
```

**Parametry:**
- `sl = 0.05` → significance level (poziom istotności)
- `while True` → nieskończona pętla (kończy się przez `break`)

**Proces w każdej iteracji:**

1. **Dopasowanie modelu:**
   ```python
   ols = sm.OLS(endog=y_train.values, exog=X_train_numpy).fit()
   ```
   - Tworzy model OLS z aktualnymi zmiennymi
   - Oblicza p-values dla wszystkich zmiennych

2. **Znalezienie najwyższego p-value:**
   ```python
   max_pval = max(ols.pvalues.astype('float'))
   ```
   - `ols.pvalues` → Series z p-values dla każdej zmiennej
   - `.astype('float')` → konwersja na float (bezpieczeństwo typów)
   - `max()` → najwyższa wartość p-value

3. **Sprawdzenie warunku:**
   ```python
   if max_pval > sl:
   ```
   - Jeśli najwyższe p-value > 0.05 → zmienna nieistotna, usuń
   - Jeśli wszystkie p-values ≤ 0.05 → zakończ pętlę

4. **Znalezienie indeksu zmiennej do usunięcia:**
   ```python
   max_idx = np.argmax(ols.pvalues.astype('float'))
   ```
   - `np.argmax()` → zwraca indeks elementu z najwyższą wartością
   - To jest indeks kolumny w `X_train_numpy`, którą trzeba usunąć

5. **Usunięcie zmiennej:**
   ```python
   X_train_numpy = np.delete(X_train_numpy, max_idx, axis=1)
   predictors.remove(predictors[max_idx])
   ```
   - `np.delete(array, index, axis=1)` → usuwa kolumnę o indeksie `max_idx`
   - `axis=1` → usuwanie kolumny (nie wiersza)
   - `predictors.remove()` → synchronizacja listy nazw zmiennych

6. **Zakończenie pętli:**
   ```python
   else:
       break
   ```
   - Gdy wszystkie zmienne mają p-value ≤ 0.05 → przerwij pętlę

---

## 4. Różnice między p10.py a p11.py

### p10.py (ręczna selekcja)
- Zmienne usuwane ręcznie, krok po kroku
- Trzy iteracje z ręcznym wyborem zmiennych
- Kod powtarzalny dla każdego kroku
- Wymaga ręcznej analizy p-values

### p11.py (automatyczna selekcja)
- Pętla `while True` automatycznie usuwa zmienne
- Proces kontynuuje się, dopóki wszystkie zmienne są istotne
- Kod bardziej zwięzły i elastyczny
- Automatyczna analiza p-values

**Zalety p11.py:**
- Mniej kodu
- Automatyczny proces
- Łatwiejsze do utrzymania
- Działa dla dowolnej liczby zmiennych

---

## 5. Funkcje NumPy używane w selekcji

### 5.1 `np.argmax()`

```python
max_idx = np.argmax(ols.pvalues.astype('float'))
```

**Co robi:**
- Zwraca indeks elementu z najwyższą wartością
- Dla p-values: zwraca indeks zmiennej z najwyższym p-value

**Przykład:**
```python
pvalues = [0.001, 0.787, 0.467, 0.125]
np.argmax(pvalues)  # zwraca 1 (indeks wartości 0.787)
```

### 5.2 `np.delete()`

```python
X_train_numpy = np.delete(X_train_numpy, max_idx, axis=1)
```

**Parametry:**
- `array` → tablica do modyfikacji
- `max_idx` → indeks elementu do usunięcia
- `axis=1` → usuwanie kolumny (axis=0 to wiersz)

**Co robi:**
- Tworzy nową tablicę bez wskazanej kolumny
- Nie modyfikuje oryginalnej tablicy (immutable)

**Przykład:**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.delete(arr, 1, axis=1)  # usuwa kolumnę o indeksie 1
# wynik: [[1, 3], [4, 6]]
```

---

## 6. Synchronizacja listy predyktorów

```python
predictors.remove(predictors[max_idx])
```

**Dlaczego to ważne:**
- `X_train_numpy` i `predictors` muszą być zsynchronizowane
- Po usunięciu kolumny z tablicy, trzeba usunąć odpowiadającą nazwę z listy
- `ols.summary(xname=predictors)` używa tej listy do wyświetlania nazw zmiennych

**Uwaga:**
- `predictors[max_idx]` → nazwa zmiennej do usunięcia
- `predictors.remove()` → usuwa pierwsze wystąpienie tej nazwy

---

## 7. Zapis modelu do pliku

```python
ols.save('model.pickle')
```

**Co robi:**
- Zapisuje dopasowany model OLS do pliku pickle
- Można później wczytać model bez ponownego trenowania

**Użycie:**
```python
import pickle
with open('model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)
```

**Zastosowanie:**
- Zapisywanie modeli do produkcji
- Wczytywanie modeli bez ponownego trenowania
- Deployment modeli

---

## 8. Interpretacja wyników

### 8.1 Proces selekcji

Model automatycznie usuwa zmienne w kolejności od najmniej istotnych:
1. Zmienna z najwyższym p-value > 0.05 → usuń
2. Dopasuj nowy model bez tej zmiennej
3. Powtórz, dopóki wszystkie zmienne mają p-value ≤ 0.05

### 8.2 Finalny model

Po zakończeniu pętli, model zawiera tylko statystycznie istotne zmienne (p ≤ 0.05).

**Przykładowy wynik:**
- `const`, `age`, `bmi`, `children`, `smoker_yes` → wszystkie istotne
- `sex_male`, `region_*` → usunięte (nieistotne)

---

## 9. Uwagi techniczne

### 9.1 Konwersja typów

```python
ols.pvalues.astype('float')
```

**Dlaczego:**
- `ols.pvalues` może być Series z różnymi typami
- Konwersja na float zapewnia poprawne porównania
- `max()` i `np.argmax()` wymagają numerycznych typów

### 9.2 Bezpieczeństwo pętli

```python
while True:
    # ...
    if max_pval > sl:
        # usuń zmienną
    else:
        break  # zawsze kończy pętlę
```

**Zabezpieczenie:**
- Pętla zawsze kończy się przez `break`
- Nie ma ryzyka nieskończonej pętli (zmienne są usuwane, więc w końcu wszystkie będą istotne)
- Jeśli wszystkie zmienne są istotne od początku → pętla wykona się raz i zakończy

### 9.3 Indeksowanie po usunięciu

Po `np.delete()`:
- Indeksy kolumn się zmieniają
- Kolejne iteracje używają nowych indeksów
- `np.argmax()` zawsze zwraca poprawny indeks dla aktualnej tablicy

---

## 10. Porównanie z innymi metodami selekcji

### Backward Elimination (p11.py)
- Start: wszystkie zmienne
- Proces: usuwa najmniej istotne
- Zakończenie: gdy wszystkie istotne

### Forward Selection
- Start: puste
- Proces: dodaje najbardziej istotne
- Zakończenie: gdy nie ma więcej istotnych do dodania

### Stepwise Selection
- Kombinacja forward i backward
- Może dodawać i usuwać zmienne

### Lasso Regression
- Automatyczna selekcja przez regularyzację L1
- Współczynniki mogą być zerowane

---

## 11. Podsumowanie

**Proces:**
1. Przygotowanie danych (one-hot encoding, konwersja typów)
2. Automatyczna pętla backward elimination
3. Usuwanie zmiennych z p-value > 0.05
4. Finalny model z tylko istotnymi zmiennymi
5. Zapis modelu do pliku

**Zalety automatycznej selekcji:**
- Mniej kodu
- Elastyczność (działa dla dowolnej liczby zmiennych)
- Automatyczny proces
- Łatwiejsze utrzymanie

**Wynik:**
Model zawierający tylko statystycznie istotne predyktory, wybrany automatycznie przez algorytm backward elimination.
