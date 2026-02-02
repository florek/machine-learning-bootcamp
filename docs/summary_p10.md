# Regresja liniowa OLS z selekcją zmiennych (statsmodels)

Ten plik opisuje budowę modelu regresji liniowej metodą najmniejszych kwadratów (OLS) z wykorzystaniem biblioteki `statsmodels` oraz proces selekcji zmiennych poprzez usuwanie nieistotnych statystycznie predyktorów.

---

## 1. Przygotowanie danych

### 1.1 Wczytanie i wstępna analiza

```python
df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
```

Dane dotyczą ubezpieczeń zdrowotnych (insurance.csv):
- **cechy**: wiek, płeć, BMI, liczba dzieci, palenie, region
- **target**: koszty ubezpieczenia (`charges`)

### 1.2 Usuwanie duplikatów

```python
df = df.drop_duplicates()
```

Usunięto zduplikowane wiersze (np. wiersze 195 i 581 z identycznymi wartościami).

### 1.3 Podział na zbiory treningowy i testowy

```python
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
```

Podział 80/20 z ustalonym ziarnem losowym dla powtarzalności.

---

## 2. Przygotowanie danych do modelu OLS

### 2.1 Kodowanie zmiennych kategorycznych

```python
X_train_ols = pd.get_dummies(X_train_ols, drop_first=True)
```

**One-Hot Encoding** z parametrem `drop_first=True`:
- Tworzy kolumny binarne dla każdej kategorii
- Usuwa pierwszą kategorię z każdej zmiennej (aby uniknąć kolinearności)
- Przykład: `sex` → `sex_male` (female jest referencyjna, usunięta)
- Przykład: `region` → `region_northwest`, `region_southeast`, `region_southwest` (northeast jest referencyjna)

### 2.2 Konwersja typów danych

```python
X_train_ols = X_train_ols.values.astype(float)
```

Konwersja DataFrame na numpy array typu `float`:
- Wymagane przez `statsmodels`
- Konwertuje wartości boolean (True/False) z `get_dummies()` na 1.0/0.0

### 2.3 Dodanie stałej (intercept)

```python
X_train_ols = sm.add_constant(X_train_ols)
```

Dodaje kolumnę jedynek na początku macierzy (dla wyrazu wolnego w modelu).

---

## 3. Model OLS – pełny model

### 3.1 Dopasowanie modelu

```python
ols = sm.OLS(endog=y_train.values, exog=X_train_ols).fit()
```

**Parametry:**
- `endog` – zmienna zależna (target) jako numpy array
- `exog` – zmienne niezależne (features) z kolumną stałej

**Wyniki pełnego modelu:**
- R-squared: 0.730 (73% wariancji wyjaśnione)
- Wszystkie zmienne: `const`, `age`, `bmi`, `children`, `sex_male`, `smoker_yes`, `region_northwest`, `region_southeast`, `region_southwest`

### 3.2 Interpretacja istotności statystycznej

Sprawdzamy wartość **p-value (P>|t|)**:
- **p < 0.05** → zmienna istotna statystycznie
- **p ≥ 0.05** → zmienna nieistotna (można usunąć)

**Pełny model:**
- `sex_male`: p=0.787 → **nieistotna**
- `region_northwest`: p=0.467 → **nieistotna**
- `region_southeast`: p=0.125 → **nieistotna**
- `region_southwest`: p=0.222 → **nieistotna**

---

## 4. Selekcja zmiennych – krok po kroku

### 4.1 Krok 1: Usunięcie `sex_male`

```python
X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 6, 7, 8]]
predictors.remove('sex_male')
```

Usunięto kolumnę 4 (`sex_male`) – zmienna nieistotna (p=0.787).

**Pozostałe zmienne:**
- `const`, `age`, `bmi`, `children`, `smoker_yes`, `region_northwest`, `region_southeast`, `region_southwest`

### 4.2 Krok 2: Usunięcie `region_northwest`

```python
X_selected = X_selected[:, [0, 1, 2, 3, 4, 6, 7]]
predictors.remove('region_northwest')
```

Usunięto kolumnę 5 (`region_northwest`) – zmienna nieistotna (p=0.467).

**Pozostałe zmienne:**
- `const`, `age`, `bmi`, `children`, `smoker_yes`, `region_southeast`, `region_southwest`

### 4.3 Krok 3: Usunięcie wszystkich zmiennych region

```python
X_selected = X_selected[:, [0, 1, 2, 3, 4]]
predictors.remove('region_southeast')
predictors.remove('region_southwest')
```

Usunięto wszystkie zmienne region – wszystkie były nieistotne statystycznie.

**Finalny model zawiera:**
- `const` (intercept)
- `age` (wiek)
- `bmi` (wskaźnik masy ciała)
- `children` (liczba dzieci)
- `smoker_yes` (czy pali)

---

## 5. Interpretacja wyników selekcji

### 5.1 Dlaczego usuwamy nieistotne zmienne?

1. **Uproszczenie modelu** – mniej parametrów do interpretacji
2. **Redukcja overfittingu** – model lepiej generalizuje
3. **Zwiększenie czytelności** – skupiamy się na istotnych predyktorach

### 5.2 Zmienne istotne w finalnym modelu

- **`age`** – wiek (p < 0.001) – im starsza osoba, tym wyższe koszty
- **`bmi`** – BMI (p < 0.001) – wyższe BMI → wyższe koszty
- **`children`** – liczba dzieci (p < 0.001) – więcej dzieci → wyższe koszty
- **`smoker_yes`** – palenie (p < 0.001) – **najsilniejszy predyktor** (współczynnik ~23,000)

### 5.3 Zmienne usunięte

- **`sex_male`** – płeć nie ma istotnego wpływu na koszty
- **Wszystkie zmienne `region`** – region zamieszkania nie ma istotnego wpływu

---

## 6. Metodologia selekcji zmiennych

### 6.1 Backward Elimination (eliminacja wsteczna)

Stosowana metoda to **backward elimination**:
1. Startujemy z pełnym modelem
2. Sprawdzamy p-value dla każdej zmiennej
3. Usuwamy zmienną z najwyższym p-value (jeśli ≥ 0.05)
4. Powtarzamy proces dla nowego modelu

### 6.2 Alternatywne metody

- **Forward Selection** – zaczynamy od pustego modelu, dodajemy zmienne
- **Stepwise Selection** – kombinacja forward i backward
- **Lasso Regression** – automatyczna selekcja przez regularyzację L1

---

## 7. Uwagi techniczne

### 7.1 Konwersja danych

- `y_train.values` – konwersja pandas Series na numpy array
- `.astype(float)` – wymagane dla statsmodels (nie obsługuje boolean w numpy array)

### 7.2 Indeksowanie kolumn

Przy usuwaniu kolumn używamy indeksowania numpy:
- `X_train_ols[:, [0, 1, 2, 3, 5, 6, 7, 8]]` – wybiera określone kolumny
- Indeks 0 to zawsze `const` (dodana przez `add_constant()`)

### 7.3 Synchronizacja listy predyktorów

```python
predictors.remove('sex_male')
```

Lista `predictors` musi być zsynchronizowana z kolumnami w `X_selected`, aby `summary()` wyświetlał poprawne nazwy zmiennych.

---

## 8. Podsumowanie

**Proces:**
1. Przygotowanie danych (one-hot encoding, konwersja typów)
2. Budowa pełnego modelu OLS
3. Analiza istotności statystycznej (p-value)
4. Iteracyjne usuwanie nieistotnych zmiennych
5. Finalny model z 5 zmiennymi (const + 4 features)

**Wynik:**
Model zawierający tylko statystycznie istotne predyktory: wiek, BMI, liczba dzieci i palenie. Region i płeć nie mają istotnego wpływu na koszty ubezpieczenia zdrowotnego.
