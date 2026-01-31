# Praca z rzeczywistymi danymi – eksploracja i regresja liniowa

Ten plik dokumentuje **pracę z rzeczywistymi danymi** (insurance.csv), **eksplorację danych**, **feature engineering** i **ocenę modelu regresji liniowej**.

---

## 1. Konfiguracja środowiska

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

**Nowe importy:**

* `mean_absolute_error` → metryka oceny modelu (MAE)
* `plotly.express` → zaawansowane wizualizacje interaktywne

---

## 2. Wczytanie danych z pliku CSV

```python
df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
print(df_raw.head())
```

**Co to robi:**

* wczytuje dane z URL (lub lokalnego pliku)
* tworzy DataFrame z danymi
* `head()` → pokazuje pierwsze 5 wierszy

**Dane insurance.csv:**

* rzeczywiste dane o kosztach ubezpieczenia medycznego
* zawiera cechy: wiek, płeć, BMI, dzieci, palacz, region, koszty (charges)

**Dobra praktyka:**

* zawsze zaczynaj od `df_raw.copy()` → zachowaj oryginalne dane
* nigdy nie modyfikuj danych źródłowych bezpośrednio

---

## 3. Eksploracja danych (EDA - Exploratory Data Analysis)

```python
df = df_raw.copy()
print(df.info())
print(df[df.duplicated()])
print(df[df['charges'] == 1639.5631])
df = df.drop_duplicates()
```

**`df.info()`:**

* typy danych w każdej kolumnie
* liczba nie-null wartości
* zużycie pamięci

**`df.duplicated()`:**

* znajduje zduplikowane wiersze
* zwraca DataFrame z duplikatami

**`drop_duplicates()`:**

* usuwa zduplikowane wiersze
* domyślnie sprawdza wszystkie kolumny
* można użyć `subset=['col1', 'col2']` dla wybranych kolumn

**Dlaczego usuwać duplikaty:**

* mogą zniekształcić wyniki modelu
* sztucznie zwiększają wagę niektórych obserwacji
* w rzeczywistych danych często są błędami

---

## 4. Identyfikacja kolumn kategorycznych

```python
cat_cols = [column for column in df.columns if df[column].dtype == 'O']
print(cat_cols)

for cat in cat_cols:
    df[cat] = df[cat].astype('category')
```

**`dtype == 'O'`:**

* `'O'` → object (string) w pandas
* kolumny z tekstem są typu object

**List comprehension:**

* `[column for column in df.columns if warunek]` → lista kolumn spełniających warunek
* pythoniczny sposób filtrowania

**`astype('category')`:**

* konwersja na typ kategoryczny
* oszczędza pamięć (przechowuje unikalne wartości raz)
* przyspiesza operacje na kategoriach
* lepsze dla modeli ML (one-hot encoding)

**Dlaczego to ważne:**

* pandas automatycznie wykrywa typy, ale nie zawsze poprawnie
* kategorie wymagają innego traktowania niż liczby
* przygotowanie do one-hot encoding

---

## 5. Statystyki opisowe

```python
print(df.describe().T)
print(df.describe(include='category').T)
print(df.isnull().sum())
```

**`describe()`:**

* statystyki dla kolumn numerycznych
* count, mean, std, min, max, percentyle
* `.T` → transpozycja (kolumny jako wiersze)

**`describe(include='category')`:**

* statystyki dla kolumn kategorycznych
* count, unique, top (najczęstsza wartość), freq (częstotliwość)

**`isnull().sum()`:**

* liczba brakujących wartości w każdej kolumnie
* kluczowe przed trenowaniem modelu

**Dlaczego to ważne:**

* zrozumienie rozkładu danych
* wykrycie problemów (braki, outliery)
* decyzje o feature engineering

---

## 6. Analiza rozkładu zmiennych kategorycznych

```python
print(df.sex.value_counts())
df.sex.value_counts().plot(kind='pie')
plt.show()

print(df.smoker.value_counts())
print(df.region.value_counts())
```

**`value_counts()`:**

* liczy wystąpienia każdej wartości
* sortuje od najczęstszych
* zwraca Series

**`plot(kind='pie')`:**

* wykres kołowy
* pokazuje proporcje kategorii
* wizualizacja rozkładu

**Interpretacja:**

* czy kategorie są zbalansowane?
* czy nie ma dominującej kategorii?
* czy są rzadkie kategorie?

---

## 7. Analiza zmiennej docelowej (target)

```python
df.charges.plot(kind='hist')
plt.show()
```

**Histogram zmiennej docelowej:**

* rozkład wartości charges (koszty)
* czy rozkład jest normalny?
* czy są outliery?
* czy jest przesunięcie?

**Interpretacja:**

* prawoskośny rozkład → większość niskich wartości, kilka bardzo wysokich
* może wymagać transformacji (log, sqrt)
* outliery mogą wpływać na model

---

## 8. Zaawansowane wizualizacje z Plotly

```python
import plotly.express as px

fig = px.histogram(
    df, 
    x='charges', 
    width=800, 
    height=400, 
    facet_col='smoker', 
    title='Rozkład kosztów medycznych', 
    facet_row='sex'
)
fig.show()
```

**Plotly Express:**

* interaktywne wykresy (zoom, pan, hover)
* łatwiejsze w użyciu niż matplotlib dla złożonych wykresów
* automatyczna legenda i tooltips

**Parametry:**

* `x='charges'` → zmienna na osi X
* `facet_col='smoker'` → osobne wykresy dla każdej wartości smoker (kolumny)
* `facet_row='sex'` → osobne wykresy dla każdej wartości sex (wiersze)
* `width`, `height` → rozmiar wykresu
* `title` → tytuł

**Co pokazuje:**

* rozkład charges rozbity na kategorie smoker i sex
* porównanie rozkładów między grupami
* wykrycie różnic między grupami

**Dlaczego to ważne:**

* palacze mogą mieć wyższe koszty
* płeć może wpływać na koszty
* wizualizacja pomaga zrozumieć dane przed modelowaniem

---

## 9. One-Hot Encoding

```python
df_dummies = pd.get_dummies(df, drop_first=True)
print(df_dummies)
```

**`get_dummies()`:**

* konwersja kolumn kategorycznych na kolumny binarne (0/1)
* każda kategoria → osobna kolumna
* 1 = obserwacja należy do kategorii, 0 = nie należy

**`drop_first=True`:**

* usuwa pierwszą kategorię z każdej zmiennej
* unika pułapki zmiennych fikcyjnych (dummy variable trap)
* redukuje nadmiarowość (jedna kategoria jest domyślna)

**Przykład:**

* `sex`: ['male', 'female'] → `sex_male` (0 lub 1)
* `smoker`: ['yes', 'no'] → `smoker_yes` (0 lub 1)
* `region`: ['northeast', 'northwest', 'southeast', 'southwest'] → 3 kolumny (northeast usunięte)

**Dlaczego to konieczne:**

* większość modeli ML wymaga danych numerycznych
* regresja liniowa nie rozumie kategorii bezpośrednio
* one-hot encoding zachowuje informację o kategoriach

---

## 10. Macierz korelacji

```python
corr = df_dummies.corr()
```

**`corr()`:**

* oblicza współczynniki korelacji Pearsona między wszystkimi parami kolumn
* zakres: -1 do 1
* 1 = doskonała korelacja dodatnia
* -1 = doskonała korelacja ujemna
* 0 = brak korelacji

**Interpretacja:**

* wysoka korelacja między cechami → multikolinearność (problem)
* wysoka korelacja cecha-target → ważna cecha
* niska korelacja → cecha może być nieistotna

---

## 11. Heatmap korelacji (Seaborn)

```python
sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr, 
    mask=mask, 
    cmap=cmap, 
    vmax=.3, 
    center=0, 
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .5}
)
plt.show()
```

**`sns.heatmap()`:**

* wizualizacja macierzy korelacji jako mapy ciepła
* kolory reprezentują wartości korelacji

**Parametry:**

* `mask=mask` → ukrywa górny trójkąt (symetria macierzy)
* `cmap` → mapa kolorów (diverging = od ujemnych do dodatnich)
* `vmax=.3` → maksymalna wartość na skali (dla lepszej czytelności)
* `center=0` → centrum skali kolorów
* `square=True` → kwadratowe komórki
* `linewidths=.5` → grubość linii między komórkami
* `cbar_kws` → parametry paska kolorów

**`mask` - ukrywanie górnego trójkąta:**

* `np.zeros_like(corr)` → macierz zer o tym samym kształcie
* `np.triu_indices_from(mask)` → indeksy górnego trójkąta
* `mask[...] = True` → ustawia True w górnym trójkącie
* efekt: pokazuje tylko dolny trójkąt (unika duplikacji)

**Co pokazuje:**

* które cechy są ze sobą skorelowane
* które cechy są skorelowane z targetem (charges)
* wykrycie multikolinearności

---

## 12. Korelacja z zmienną docelową

```python
print(df_dummies.corr()['charges'].sort_values(ascending=False))
df_dummies.corr()['charges'].sort_values()[:-1].plot(kind='barh')
plt.show()
```

**`corr()['charges']`:**

* wybiera kolumnę 'charges' z macierzy korelacji
* pokazuje korelację każdej cechy z targetem

**`sort_values(ascending=False)`:**

* sortuje od najwyższej do najniższej korelacji
* `ascending=False` → malejąco

**`[:-1]`:**

* usuwa ostatni element (charges z samym sobą = 1.0)
* nie potrzebujemy tego w wykresie

**`plot(kind='barh')`:**

* poziomy wykres słupkowy (barh = bar horizontal)
* każdy słupek = jedna cecha
* wysokość słupka = wartość korelacji

**Interpretacja:**

* najwyższe słupki → najważniejsze cechy
* ujemne wartości → ujemna korelacja (wzrost cechy → spadek charges)
* dodatnie wartości → dodatnia korelacja (wzrost cechy → wzrost charges)

**Dlaczego to ważne:**

* feature selection → wybór najważniejszych cech
* zrozumienie, które cechy wpływają na target
* wykrycie nieistotnych cech (można usunąć)

---

## 13. Przygotowanie danych do modelu

```python
data = df_dummies.copy()
target = data.pop('charges')
print(data.head())
print(target.head())
```

**`data.pop('charges')`:**

* usuwa kolumnę 'charges' z DataFrame
* zwraca usuniętą kolumnę jako Series
* modyfikuje DataFrame in-place

**Dlaczego osobno:**

* scikit-learn wymaga osobnych obiektów:
  * `X` (data) → cechy (DataFrame/array)
  * `y` (target) → zmienna docelowa (Series/array)
* zapobiega przypadkowemu użyciu targetu jako cechy

---

## 14. Podział danych na zbiór treningowy i testowy

```python
X_train, X_test, y_train, y_test = train_test_split(
    data, 
    target, 
    test_size=0.2, 
    random_state=42
)
```

**Parametry:**

* `test_size=0.2` → 20% danych do testów, 80% do treningu
* `random_state=42` → powtarzalność podziału

**Różnica z p8.py:**

* p8.py: `test_size=0.25` (25% testowych)
* p9.py: `test_size=0.2` (20% testowych)
* więcej danych treningowych dla większego datasetu

---

## 15. Trenowanie i ocena modelu

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))
```

**Ocena modelu:**

* score treningowy → jak dobrze model pasuje do danych treningowych
* score testowy → jak dobrze model generalizuje

**Interpretacja:**

* podobne wyniki → dobry model (brak overfittingu)
* duża różnica → overfitting (model zapamiętał dane treningowe)
* niskie oba → underfitting (model za prosty)

---

## 16. Predykcje i analiza błędów

```python
y_pred = regressor.predict(X_test)
print(y_pred[:10])

y_true = y_test.copy()
predictions = pd.DataFrame(data={
    'y_true': y_true, 
    'y_pred': y_pred
})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
print(predictions.head())
```

**Predykcje:**

* `predict()` → przewidywane wartości dla danych testowych
* `y_pred[:10]` → pierwsze 10 predykcji

**DataFrame z błędami:**

* `y_true` → rzeczywiste wartości
* `y_pred` → przewidywane wartości
* `error` → różnica (błąd predykcji)

**Interpretacja błędów:**

* dodatnie → model przewiduje za nisko
* ujemne → model przewiduje za wysoko
* blisko zera → dobre predykcje

---

## 17. Histogram błędów

```python
predictions['error'].plot(kind='hist', bins=50, figsize=(8, 6))
plt.show()
```

**Co pokazuje:**

* rozkład błędów predykcji
* czy błędy są normalnie rozłożone wokół zera
* czy są systematyczne błędy (przesunięcie)

**Idealny rozkład:**

* symetryczny wokół zera
* wąski (małe błędy)
* kształt dzwonowy (rozkład normalny)

---

## 18. Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')
```

**Mean Absolute Error:**

* średni błąd bezwzględny
* średnia z wartości bezwzględnych błędów
* wzór: `MAE = mean(|y_true - y_pred|)`

**Interpretacja:**

* wartość w jednostkach targetu (tutaj: dolary)
* łatwiejsza do interpretacji niż R²
* np. MAE = 5000 → średnio błądzimy o 5000 dolarów

**Porównanie z R²:**

* R² → procent wyjaśnionej wariancji (0-1)
* MAE → średni błąd w jednostkach (łatwiejsze do zrozumienia)
* obie metryki są ważne

**Kiedy używać:**

* R² → ogólna ocena jakości modelu
* MAE → interpretacja w kontekście biznesowym

---

## 19. Współczynniki modelu

```python
print(regressor.intercept_)
print(regressor.coef_)
print(data.columns)
```

**`intercept_`:**

* wyraz wolny (punkt przecięcia z osią Y)
* wartość baseline (gdy wszystkie cechy = 0)

**`coef_`:**

* współczynniki dla każdej cechy
* pokazują wpływ każdej cechy na target
* dodatni → wzrost cechy → wzrost charges
* ujemny → wzrost cechy → spadek charges

**`data.columns`:**

* nazwy cech odpowiadające współczynnikom
* pozwala zinterpretować, które cechy są ważne

**Interpretacja:**

* najwyższe wartości bezwzględne → najważniejsze cechy
* znak → kierunek wpływu
* można porównać z korelacją (powinny być podobne)

---

## 20. Różnice między p8.py a p9.py

**p8.py (syntetyczne dane):**

* dane wygenerowane (`make_regression`)
* proste, kontrolowane dane
* jedna cecha numeryczna
* szybkie testy i nauka

**p9.py (rzeczywiste dane):**

* dane z pliku CSV
* rzeczywiste, złożone dane
* wiele cech (numeryczne i kategoryczne)
* pełny pipeline: EDA → feature engineering → modelowanie

**Nowe elementy w p9.py:**

* wczytanie danych z pliku
* eksploracja danych (EDA)
* obsługa kategorii (one-hot encoding)
* analiza korelacji
* zaawansowane wizualizacje (Plotly)
* metryka MAE
* interpretacja współczynników

---

## 21. TL;DR

* `read_csv()` → wczytanie danych z pliku/URL
* `info()`, `describe()`, `value_counts()` → eksploracja danych
* `drop_duplicates()` → usuwanie duplikatów
* `astype('category')` → konwersja na kategorie
* `get_dummies(drop_first=True)` → one-hot encoding
* `corr()` → macierz korelacji
* `sns.heatmap()` → wizualizacja korelacji
* `plotly.express` → interaktywne wykresy
* `mean_absolute_error()` → metryka MAE
* `coef_` i `intercept_` → interpretacja modelu

**Pipeline ML:**

1. Wczytanie i eksploracja danych
2. Czyszczenie danych (duplikaty, braki)
3. Feature engineering (one-hot encoding)
4. Analiza korelacji
5. Podział na train/test
6. Trenowanie modelu
7. Ocena modelu (R², MAE)
8. Analiza błędów
9. Interpretacja współczynników

> Rzeczywiste dane wymagają więcej pracy niż syntetyczne, ale dają prawdziwe wnioski i wartościowe modele.
