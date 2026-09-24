# Quiz WWW (self-contained HTML)

Każdy plik `{DD.MM.RRRR}.html` to kompletny quiz na dany dzień: pytania, sprawdzanie i pobieranie markdownów.

## Użycie

1. Otwórz `docs/quiz/web/{data}.html` w przeglądarce.
2. Zaznacz odpowiedzi A–D **albo** „Nie mam pojęcia” (opcjonalnie pewność 1–5).
3. Kliknij **Sprawdź wynik**.
4. Przy **BŁĄD** i **NIE WIEM** przeczytaj mini-lekcję pod pytaniem.
5. Pobierz:
   - `{data}_my_answers.md` → `docs/quiz/my_answers/`
   - `{data}_results.md` → `docs/quiz/results/`

Nie trzeba serwera ani wklejania treści. Pliki generuje `/quizgenerator` (ETAP 3; mini-lekcje i opcja „Nie mam pojęcia” są wbudowane w HTML).
