# Dokumentacja Bootcamp Machine Learning

Struktura dokumentacji i jak z niej korzystać.

---

## 📁 Struktura plików

### Szybkie przypomnienia (używaj przed lekcjami)
- **`pre_lesson_review.md`** → Ultra-skondensowane przypomnienie przed każdą lekcją
- **`cheat_sheet.md`** → Szybkie przypomnienie kluczowych konceptów i składni

### Quizy i wyniki
- **`docs/quiz/questions/*.md`** → Pytania quizowe z konkretnych dat
- **`docs/quiz/answers/*.md`** → Oficjalne odpowiedzi i wyjaśnienia
- **`docs/quiz/my_answers/*.md`** → Twoje odpowiedzi i auto-feedback
- **`docs/quiz/results/*.md`** + `docs/quiz/results/README.md` → Podsumowania wyników

### Szczegółowe wyjaśnienia (używaj do nauki)
- **`summary_p6.md`** → Gradient Descent (ręczna implementacja)
- **`summary_p7.md`** → Regresja liniowa scikit-learn
- **`summary_p8.md`** → Train/test split i ocena modelu
- **`summary_p9.md`** → Rzeczywiste dane, EDA, feature engineering
- **`summary_p10.md`** → OLS statsmodels, ręczna selekcja zmiennych
- **`summary_p11.md`** → Automatyczna backward elimination
- **`summary_p12.md`** → Regresja wielomianowa (rozszerzenie cech, PolynomialFeatures)
- **`summary_p13.md`** → Regresja drzewa decyzyjnego (DecisionTreeRegressor, plot_tree)
- **`summary_p14.md`** → Metryki regresji (MAE, MSE, RMSE, max_error, R²) i wizualizacja wyników

### Ogólne
- **`summary.md`** → Ogólne koncepty przygotowania danych

---

## 🎯 Jak korzystać z dokumentacji

### Przed lekcją (szybka powtórka - 5-10 min)
1. Otwórz **`pre_lesson_review.md`**
2. Przeczytaj sekcję dla danej lekcji (np. P6, P7, itd.)
3. Odśwież kluczowe koncepty i składnię

### Podczas nauki (szczegółowe wyjaśnienia)
1. Otwórz odpowiedni plik **`summary_p*.md`**
2. Przeczytaj szczegółowe wyjaśnienia linijka po linijce
3. Zrozum koncepty i metodologię

### Podczas kodowania (szybka referencja)
1. Otwórz **`cheat_sheet.md`**
2. Znajdź potrzebną składnię lub koncept
3. Skopiuj kod i dostosuj do swoich potrzeb

### Gdy masz pytanie (szukanie odpowiedzi)
1. Sprawdź **`cheat_sheet.md`** → sekcja "Najczęstsze błędy"
2. Sprawdź odpowiedni **`summary_p*.md`** → szczegółowe wyjaśnienia
3. Sprawdź **`pre_lesson_review.md`** → szybkie przypomnienie

---

## 📊 Mapowanie lekcji do plików

| Lekcja | Plik szczegółowy | Kluczowe koncepty |
|--------|-----------------|-------------------|
| P6 | `summary_p6.md` | Gradient Descent, ręczna implementacja |
| P7 | `summary_p7.md` | scikit-learn, syntetyczne dane |
| P8 | `summary_p8.md` | Train/test split, ocena modelu |
| P9 | `summary_p9.md` | EDA, feature engineering, rzeczywiste dane |
| P10 | `summary_p10.md` | OLS statsmodels, ręczna selekcja zmiennych |
| P11 | `summary_p11.md` | Automatyczna backward elimination, zapis modelu |
| P12 | `summary_p12.md` | Regresja wielomianowa, PolynomialFeatures |
| P13 | `summary_p13.md` | Regresja drzewa decyzyjnego, DecisionTreeRegressor, plot_tree |
| P14 | `summary_p14.md` | Metryki regresji (MAE, MSE, RMSE, max_error, R²), wizualizacja |

---

## 🔄 Eliminacja duplikatów

Dokumentacja została zoptymalizowana, aby uniknąć duplikatów:

- **Podstawowe koncepty** (importy, konfiguracja, podstawowa składnia) → `cheat_sheet.md`
- **Szybkie przypomnienia** → `pre_lesson_review.md`
- **Szczegółowe wyjaśnienia** → `summary_p*.md` (tylko unikalne treści dla danej lekcji)

**Przykład:**
- Koncept `train_test_split` szczegółowo opisany w `summary_p8.md`
- Szybkie przypomnienie w `pre_lesson_review.md` (P8)
- Składnia w `cheat_sheet.md`

---

## 💡 Strategia powtórek

### Przed każdą lekcją (5-10 min)
```
1. Otwórz pre_lesson_review.md
2. Przeczytaj sekcję dla danej lekcji
3. Sprawdź "Powtarzające się koncepty"
4. Gotowe!
```

### Po lekcji (głębsze zrozumienie)
```
1. Otwórz odpowiedni summary_p*.md
2. Przeczytaj szczegółowe wyjaśnienia
3. Porównaj z kodem ćwiczeń z danej lekcji
4. Zrozum koncepty i metodologię
```

### Podczas rozwiązywania zadań
```
1. Otwórz cheat_sheet.md
2. Znajdź potrzebną składnię
3. Skopiuj i dostosuj
4. Sprawdź "Najczęstsze błędy" jeśli coś nie działa
```

---

## 🎓 Progresja nauki

**P6 → P7 → P8 → P9 → P10 → P11 → P12 → P13 → P14**

Każda lekcja buduje na poprzedniej:
- **P6:** Zrozumienie matematyki (gradient descent)
- **P7:** Użycie gotowych narzędzi (scikit-learn)
- **P8:** Właściwa ocena modelu (train/test split)
- **P9:** Praca z rzeczywistymi danymi (EDA, feature engineering)
- **P10:** Zaawansowana analiza statystyczna (OLS, ręczna selekcja zmiennych)
- **P11:** Automatyzacja selekcji zmiennych (backward elimination w pętli)
- **P12:** Regresja wielomianowa (nieliniowa zależność, rozszerzenie cech)
- **P13:** Regresja drzewa decyzyjnego (DecisionTreeRegressor, wizualizacja drzewa)
- **P14:** Metryki regresji (MAE, MSE, RMSE, max_error, R²) i wizualizacja wyników

---

## ⚡ Szybkie linki

- [Szybka powtórka przed lekcją](pre_lesson_review.md)
- [Cheat Sheet](cheat_sheet.md)
- [P6: Gradient Descent](summary_p6.md)
- [P7: Regresja liniowa scikit-learn](summary_p7.md)
- [P8: Train/test split](summary_p8.md)
- [P9: Rzeczywiste dane + EDA](summary_p9.md)
- [P10: OLS + ręczna selekcja zmiennych](summary_p10.md)
- [P11: Automatyczna backward elimination](summary_p11.md)
- [P12: Regresja wielomianowa](summary_p12.md)
- [P13: Regresja drzewa decyzyjnego](summary_p13.md)
- [P14: Metryki regresji i wizualizacja](summary_p14.md)

---

> **Tip:** Zacznij od `pre_lesson_review.md` przed każdą lekcją. To zaoszczędzi Ci czas i pomoże szybko odświeżyć wiedzę!
