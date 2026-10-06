# Ocena modeli klasyfikacji — accuracy, confusion matrix, precision, recall i F1-score

## 1. Cel lekcji

Uczymy się, jak oceniać modele klasyfikacji na przykładzie wykrywania choroby zakaźnej.

```text
0 = osoba zdrowa
1 = osoba zakażona
```

Mamy:
- `y_true` — rzeczywista klasa,
- `y_pred` — predykcja modelu.

---

## 2. Accuracy

Accuracy mówi, jaki procent wszystkich predykcji był poprawny.

```text
accuracy =
liczba poprawnych predykcji
/
liczba wszystkich predykcji
```

W przykładzie:

```text
7 poprawnych predykcji / 10 próbek = 70%
```

Czyli:

```text
accuracy = 70%
```

---

## 3. Confusion matrix

Macierz konfuzji rozbija wyniki na cztery przypadki:

```text
                PRZEWIDZIANE
               0          1
RZECZYWISTE
0              TN         FP
1              FN         TP
```

W przykładzie:

```text
TN = 3
FP = 1
FN = 2
TP = 4
```

Suma:

```text
3 + 1 + 2 + 4 = 10
```

---

## 4. TN — True Negative

```text
rzeczywistość = 0
predykcja     = 0
```

Osoba jest zdrowa i model poprawnie przewiduje, że jest zdrowa.

```text
TN = 3
```

---

## 5. FP — False Positive

```text
rzeczywistość = 0
predykcja     = 1
```

Osoba jest zdrowa, ale model przewiduje chorobę.

```text
FP = 1
```

---

## 6. FN — False Negative

```text
rzeczywistość = 1
predykcja     = 0
```

Osoba jest chora, ale model uznaje ją za zdrową.

```text
FN = 2
```

---

## 7. TP — True Positive

```text
rzeczywistość = 1
predykcja     = 1
```

Osoba jest chora i model poprawnie wykrywa chorobę.

```text
TP = 4
```

---

## 8. False Positive Rate

Wzór:

```text
FPR = FP / (FP + TN)
```

Dla przykładu:

```text
FPR = 1 / (1 + 3)
    = 1/4
    = 25%
```

Interpretacja:

> 25% rzeczywiście zdrowych osób zostało błędnie uznanych za chore.

---

## 9. False Negative Rate

Wzór:

```text
FNR = FN / (FN + TP)
```

Dla przykładu:

```text
FNR = 2 / (2 + 4)
    = 2/6
    ≈ 33,3%
```

Interpretacja:

> Około 33,3% rzeczywiście chorych osób zostało błędnie uznanych za zdrowe.

---

## 10. Który błąd jest groźniejszy?

W przykładzie choroby zakaźnej groźniejszy jest:

```text
False Negative
```

Bo oznacza:

```text
osoba chora
↓
model: zdrowa
```

Taka osoba może dalej zakażać innych.

False Positive oznacza raczej dodatkowe badania zdrowej osoby.

---

## 11. Precision

Wzór:

```text
precision = TP / (TP + FP)
```

Dla przykładu:

```text
precision = 4 / (4 + 1)
          = 4/5
          = 80%
```

Precision odpowiada na pytanie:

> Jeżeli model powiedział „pozytywny”, jak często miał rację?

---

## 12. Recall

Wzór:

```text
recall = TP / (TP + FN)
```

Dla przykładu:

```text
recall = 4 / (4 + 2)
       = 4/6
       ≈ 66,7%
```

Recall odpowiada na pytanie:

> Spośród wszystkich naprawdę pozytywnych przypadków, ile model wykrył?

---

## 13. Precision vs recall

```text
Precision:
spośród przewidzianych pozytywów,
ile naprawdę było pozytywnych?

Recall:
spośród wszystkich prawdziwych pozytywów,
ile model wykrył?
```

To dwa różne pytania.

---

## 14. F1-score

F1-score to średnia harmoniczna precision i recall.

```text
F1 =
2 * precision * recall
/
(precision + recall)
```

W przykładzie:

```text
precision = 0,8
recall    ≈ 0,667
```

czyli:

```text
F1 ≈ 72,7%
```

F1 jest przydatne, gdy chcemy zachować balans między precision i recall.

---

## 15. Wszystkie wyniki z przykładu

```text
Accuracy  = 70%
FPR       = 25%
FNR       ≈ 33,3%
Precision = 80%
Recall    ≈ 66,7%
F1-score  ≈ 72,7%
```

---

## 16. Wszystkie wzory

### Accuracy

```text
Accuracy =
(TP + TN)
/
(TP + TN + FP + FN)
```

### FPR

```text
FPR =
FP
/
(FP + TN)
```

### FNR

```text
FNR =
FN
/
(FN + TP)
```

### Precision

```text
Precision =
TP
/
(TP + FP)
```

### Recall

```text
Recall =
TP
/
(TP + FN)
```

### F1-score

```text
F1 =
2 * Precision * Recall
/
(Precision + Recall)
```

---

## 17. Co warto zapamiętać?

1. **Accuracy pokazuje odsetek wszystkich poprawnych predykcji.**
2. **Confusion matrix pokazuje rodzaj popełnianych błędów.**
3. **TN = poprawnie rozpoznany negatyw.**
4. **FP = fałszywy alarm.**
5. **FN = pominięty przypadek pozytywny.**
6. **TP = poprawnie rozpoznany pozytyw.**
7. **FPR mierzy fałszywe alarmy wśród negatywów.**
8. **FNR mierzy pominięte pozytywy.**
9. **Precision ocenia jakość pozytywnych predykcji.**
10. **Recall mówi, jaki procent wszystkich pozytywów model wykrył.**
11. **F1-score łączy precision i recall.**
12. **Nie wszystkie błędy mają ten sam koszt.**
13. **W problemie choroby zakaźnej szczególnie groźny jest False Negative.**

---

## 18. Najważniejsze zdanie

> **Nie wystarczy patrzeć tylko na accuracy — trzeba rozumieć macierz konfuzji i wiedzieć, jaki rodzaj błędu jest najgroźniejszy w konkretnym problemie.**
