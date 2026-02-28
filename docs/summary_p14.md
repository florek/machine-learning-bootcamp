# Metryki regresji i wizualizacja wyników

Ten plik opisuje **metryki oceny modelu regresji** (MAE, MSE, RMSE, max_error, R²) oraz **wizualizację** wyników: wykres y_true vs y_pred i histogram błędów.

---

## 1. Błąd i błąd kwadratowy

Dla par (y_true, y_pred) definiuje się **błąd** jako różnicę: error = y_true − y_pred. **Błąd kwadratowy** to error². W DataFrame często dodaje się kolumny: error i squared_error, żeby potem liczyć metryki i rysować rozkład błędów.

---

## 2. MAE (Mean Absolute Error)

**Średni błąd bezwzględny:** średnia z |error|. Jednostki takie jak zmienna docelowa. W sklearn: `mean_absolute_error(y_true, y_pred)`. Ręcznie: `np.abs(errors).mean()` lub `abs(results['error']).sum() / len(results)`.

---

## 3. MSE i RMSE

**MSE (Mean Squared Error):** średnia z error². W sklearn: `mean_squared_error(y_true, y_pred)` (domyślnie zwraca MSE). **RMSE:** pierwiastek z MSE; też w tych samych jednostkach co target. W sklearn: `mean_squared_error(y_true, y_pred, squared=False)` zwraca RMSE.

---

## 4. max_error

**Maksymalny błąd:** największa wartość |y_true − y_pred|. Pokazuje najgorszą pojedynczą predykcję. W sklearn: `max_error(y_true, y_pred)`.

---

## 5. R² (współczynnik determinacji)

**r2_score(y_true, y_pred):** procent wariancji wyjaśnionej przez model. 1.0 = idealne dopasowanie, 0.0 = model nie lepszy niż średnia, może być ujemne. To samo zwraca `regressor.score(X, y)` dla LinearRegression.

---

## 6. Wizualizacja: y_true vs y_pred

Wykres punktowy: oś X = y_true, oś Y = y_pred. Idealny model dałby punkty na prostej y = x. Dodanie linii y = x (od min do max) ułatwia ocenę. W Plotly: go.Figure z go.Scatter (punkty) i drugim Scatter (linia [min,max], [min,max]).

---

## 7. Histogram błędów

Histogram kolumny error pokazuje rozkład błędów. Przy dobrym modelu rozkład jest w przybliżeniu symetryczny wokół zera. W Plotly: px.histogram(results, x='error', nbins=50).

---

## 8. Podsumowanie

- MAE, MSE, RMSE, max_error, r2_score – wszystkie z sklearn.metrics (odpowiednie importy).
- RMSE: mean_squared_error(..., squared=False).
- Wizualizacje: wykres true vs pred (z linią y=x) i histogram błędów pomagają w ocenie jakości regresji.
