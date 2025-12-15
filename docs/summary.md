# Data Science Handbook – przygotowanie danych (Pandas + scikit-learn)
---

## 1. Braki danych (NaN)

### Reprezentacja braków
```python
np.nan
```

- standardowa reprezentacja braków danych w NumPy / Pandas
- większość algorytmów ML **nie obsługuje NaN** → trzeba je uzupełnić

---

## 2. Imputacja danych (`SimpleImputer`)

```python
from sklearn.impute import SimpleImputer
```

### 2.1 Imputacja średnią (cechy numeryczne)

```python
imputer = SimpleImputer(strategy="mean")
df['weight'] = imputer.fit_transform(df[['weight']])
```

**Stosuj gdy:**
- cecha ciągła
- brak wyraźnej wartości zastępczej

---

### 2.2 Imputacja stałą wartością (numeryczne)

```python
imputer = SimpleImputer(strategy="constant", fill_value=99.0)
df['price'] = imputer.fit_transform(df[['price']])
```

**Stosuj gdy:**
- brak danych sam w sobie niesie informację
- potrzebny placeholder

---

### 2.3 Imputacja stałą wartością (kategorie)

```python
imputer = SimpleImputer(strategy="constant", fill_value='L')
df['size'] = imputer.fit_transform(df[['size']]).ravel()
```

**Uwaga:**
- `ravel()` spłaszcza tablicę do 1D

---

## 3. Braki danych w szeregach czasowych

### Dane czasowe
```python
pd.date_range(start, end, periods)
```

### Metody uzupełniania
```python
fillna(0)
fillna(mean)
interpolate()
fillna(method='bfill')
fillna(method='ffill')
```

**Zasada:**
- szeregi czasowe → interpolacja / ffill / bfill
- dane statyczne → mean / constant

---

## 4. Feature engineering – generowanie nowych cech

### 4.1 Cechy czasowe

```python
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
```

**Cel:**
- wydobycie informacji z daty
- lepsza reprezentacja czasu dla modelu

---

## 5. Dyskretyzacja zmiennych ciągłych

```python
pd.cut(df.height, bins=3)
pd.cut(df.height, bins=(160, 175, 180, 195))
pd.cut(df.height, bins=(...), labels=[...])
```

**Zastosowanie:**
- zamiana wartości ciągłych na kategorie
- uproszczenie problemu decyzyjnego

---

### One-hot encoding

```python
pd.get_dummies(df, drop_first=True)
```

---

## 6. Ekstrakcja cech

### 6.1 Listy → cechy liczbowe

```python
df['lang_number'] = df['lang'].apply(len)
```

---

### 6.2 Flagi binarne

```python
df['PL_flag'] = df['lang'].apply(lambda x: 1 if 'PL' in x else 0)
```

---

### 6.3 Tekst → cechy

```python
df.website.str.split('.', expand=True)
```

- ekstrakcja domeny
- ekstrakcja rozszerzenia

---

## 7. Dane wbudowane w scikit-learn

```python
from sklearn.datasets import load_iris, load_breast_cancer
```

- gotowe datasety do nauki
- struktura: `data`, `target`, `DESCR`

---

## 8. Rozdzielenie cech i targetu

```python
target = df.pop('target')
```

**Zasada:**
- cechy i target przechowywane osobno

---

## 9. Podział na zbiór treningowy i testowy

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

**Dlaczego:**
- `stratify` → zachowanie proporcji klas
- `random_state` → powtarzalność wyników

---

## 10. TL;DR – przygotowanie danych

- NaN → imputacja lub interpolacja
- Dane czasowe ≠ dane statyczne
- Feature engineering to kluczowy etap ML
- `pd.cut` → dyskretyzacja
- `get_dummies` → encoding
- `train_test_split + stratify` → poprawny podział
