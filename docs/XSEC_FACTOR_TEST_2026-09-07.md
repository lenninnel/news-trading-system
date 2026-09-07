# Querschnitts-Faktortest auf breitem US-Universum, 2026-09-07

Reine Offline-Analyse. Kein Zugriff auf den Live-Pfad, keine Änderung an
NTS-Code, keine DB-Zugriffe (auch nicht lesend: die Prod-DB enthält nur 20
Ticker und wurde nicht gebraucht). Verzeichnis `analysis/xsec_factor_test/`,
eigene venv, alle Polygon-Antworten unter `cache/` zwischengespeichert
(gitignored, ~1,6 GB); ein zweiter Lauf fragt Polygon nicht mehr.

Frage: Schlägt ein einfaches Querschnitts-Ranking auf den 500 liquidesten
US-Aktien die gleichgewichtete Universumsrendite nach Kosten?

## 0. Kurzantwort

**Keiner der vier Faktoren besteht.** Nach Kosten und nach Deflation der
Sharpe-Ratio (vier Tests) liegt kein Portfolio über der 95-%-Schwelle; die
höchste deflationierte Sharpe ist 0,61 (Short-Term-Reversal Long-Short),
0,50 für Reversal Long-only, 0,23 / 0,14 für Momentum. Low-Volatility,
52-Wochen-Hoch und die Kombination liegen **unter** der Benchmark, zum Teil
deutlich. „Nichts gefunden" ist das Ergebnis, mit einer Einschränkung und
einem Vorbehalt:

* **Einschränkung: Short-Term-Reversal ist der einzige Kandidat.** Oberstes
  Dezil (die 50 größten 5-Tage-Verlierer) +21,5 % p.a. über der Benchmark
  nach 10 bp, Sharpe 1,19, t = 2,36, in jedem vollen Jahr positiv
  (2023 +43,6 %, 2024 +6,1 %, 2025 +18,7 %, 2026 +19,9 %), Long-Short
  +22,1 % p.a., Sharpe 0,97. Aber: (a) der April 2026 allein liefert
  +25,7 % Überschuss (V-Erholung der KI-/Quantum-/Krypto-Namen BE, AAOI,
  IONQ, CRWV, NBIS, SNDK nach dem März-Einbruch); ohne die besten drei
  Monate bleiben +9,4 % p.a. bei Sharpe 0,89 (Long-Short +8,1 %, 0,56).
  (b) Es ist **nicht monoton**: D1 bis D9 liegen alle zwischen −0,5 % und
  +0,1 % je Monat, nur D10 bei +1,9 % (Spearman ρ = 0,42, p = 0,23). Alles
  kommt aus einem Extrem. (c) Das D10-Portfolio hat 53 % annualisierte
  Volatilität gegen 31 % im Universum, 27 % seiner Namen sind zugleich im
  höchsten Volatilitätsdezil, das Beta des Long-only-Überschusses zur
  Benchmark ist +0,44 (t = 2,9). Im Long-Short-Spread ist das Beta zwar
  ≈ 0 (+0,08), aber das Alpha hat t = 1,7. (d) Umschlag 10,5× (Long-only)
  bzw. 20,9× (Long-Short) p.a.; die 10 bp kosten 2,5 % bzw. 5,1 % p.a.
  Bei realen Kosten in 50-%-Vol-Namen (Spread, Impact, Fill-Risiko am Close)
  schrumpft das weiter. Schiefe 2,4, Kurtosis 13: die Serie lebt von
  wenigen großen Monaten.
* **Momentum 12-1** sieht brutto passabel aus (+14,1 % p.a. Long-only,
  +10,7 % Long-Short), ist aber das Gegenteil von robust: DSR 0,14 / 0,23,
  2023 negativ (−0,9 % Long-only, −26,8 % Long-Short), ohne die besten drei
  Monate +1,4 % p.a. (Long-only) bzw. −4,3 % (Long-Short); Beta des
  Long-only-Überschusses +0,62 (t = 3,0), Alpha-t = 0,4. Wieder nur D10,
  D1–D9 flach.
* **Low-Volatility** ist der einzige monotone Faktor, allerdings mit dem
  falschen Vorzeichen: ρ = −0,96 (p < 0,01), D1 (höchste Vol) +1,3 % je
  Monat, D10 (niedrigste Vol) −0,6 %. Long-only −9,1 % p.a., Long-Short
  −30,7 % p.a., MaxDD −86 %. Das ist das Regime: eine gleichgewichtete
  Benchmark aus den 500 liquidesten Namen machte +18,9 % p.a. (2023 +23 %,
  2024 +16 %, 2025 +20 %), und alles, was Beta hat, gewann.
* **52-Wochen-Hoch** −12,9 % Long-only, −23,9 % Long-Short; D10 (am
  dichtesten am Hoch) ist mit −0,96 % je Monat das schlechteste Dezil.
* **Kombination** −6,2 % / −10,1 %: das Rangmittel erbt die drei Verlierer.
* **Vorbehalt: 47 Monate.** Der Polygon-Plan reicht nur bis 2021-09-08
  zurück (rollierende 5 Jahre), die ersten 12 Monate gehen in die
  Faktorhistorie. Bei 47 Beobachtungen und vier Versuchen liegt die
  Deflationsschwelle (SR₀) bei 0,345 monatlich ≈ 1,2 annualisiert, genau
  dort, wo Reversal steht. Ein Effekt von 5–10 % p.a. Überschuss ist mit
  dieser Stichprobe weder nachweisbar noch ausschließbar. Der Zeitraum
  enthält keinen Bärenmarkt (Start Oktober 2022, am Tief).

**Für NTS heißt das:** Auf dem breiten Universum und mit diesem Zeitfenster
trägt keine der vier Absolut-Fragen („Momentum? Nähe zum Hoch?") einen
Überschuss nach Kosten, der die Deflation überlebt. Das bestätigt die
Diagnose aus dem Shadow-Buch von der anderen Seite: nicht die elf Ticker
waren das Problem, auch auf 500 Namen ist mit diesen Signalen im
Testzeitraum nichts Robustes drin. Der einzige Kandidat (Reversal) ist
ein Hoch-Vol-Bounce mit 10–20-fachem Umschlag, also genau das Gegenteil
dessen, was ein tägliches Long-only-System mit 1,5×ATR-Stops handeln kann.
Vor einem Bau: ein längeres Fenster (anderer Datenplan, ≥ 10 Jahre inkl.
2008/2020/2022) und derselbe Test, ohne neue Parameter.

## Daten und tatsächlich abgedeckter Zeitraum

**Polygon-Berechtigung.** Der Key (Stocks Starter) liefert ein rollierendes
5-Jahres-Fenster: 2021-09-07 wird mit `NOT_AUTHORIZED` abgelehnt, 2021-09-08
ist der erste lieferbare Tag. Der Auftrag verlangte "mindestens 2021-01";
das ist mit diesem Plan nicht möglich. Kursdaten: **2021-09-08 bis
2026-09-04** (1 254 Handelstage, ein Grouped-Daily-Call je Wochentag, roh
und split-adjustiert). Die ersten 12 Monate werden für die Faktorhistorie
verbraucht (12-1-Momentum, 52-Wochen-Hoch), daher **Monatsrenditen von
2022-10-03 bis 2026-09-01, 47 Monate**. Das ist kurz. Alle
Signifikanzaussagen unten stehen unter diesem Vorbehalt, und das rollierende
Fenster bedeutet: ohne den jetzt angelegten Cache wäre der September 2021 ab
morgen weg.

**Punkt-in-Zeit-Universum.** Für jedes Monatsende ein Polygon-Snapshot
`/v3/reference/tickers?date=…&type=CS` (Common Stock, Stand jenes Tages;
verifiziert: FB ist am 2022-03-01 aktiv, META nicht; TWTR am 2022-06-01
aktiv, am 2023-06-01 nicht). Ein Bar wird nur dann einer Entität zugeordnet,
wenn das Symbol im Snapshot des laufenden oder des vorigen Monatsendes als CS
geführt ist; alles andere (ETF, ADR, Preferred, Warrant, Unit, OTC) fällt
weg. Entitätsschlüssel ist die Composite-FIGI, die Umbenennungen überlebt
(FB→META, SQ→XYZ, FI→FISV bleiben eine Zeitreihe). Ein Rückgriff auf die
heutigen Referenztabellen wurde ausdrücklich verworfen, weil recycelte
Symbole sonst falsch zusammengeklebt werden (ACH war 2021 ein ADR und ist
2026 Owens & Minor).

**Korrekturen.** Splits aus Polygons eigenen adjustierten Bars
(Faktor = adjustierter/roher Schlusskurs je Zelle). Der Split-Feed
`/v3/reference/splits` wurde geprüft und verworfen: er enthält für
umbenannte Ticker keinen Eintrag (XSPA→XWEL 1:20, XL→SPRU 1:8 fehlen
komplett), 28 von 195 Splits im Stichprobenfenster wären falsch gewesen.
Bardividenden (Typen CD, SC) aus `/v3/reference/dividends` (987 838 Zeilen,
249 exakte Duplikate entfernt), am Ex-Tag in die Gesamtrendite. Dividenden
delisteter Namen sind enthalten (PXD, HES, DFS, MRO geprüft); bei recycelten
Symbolen fehlen sie (MIC-Sonderdividende 2021-10 fehlt, der Kurs fällt um
91 %). Übernahmen gegen Bar, die weder Split noch Dividende sind, erscheinen
als Kursrendite (BHVN 2022-10-04: −94,5 %, tatsächlich Cash + Spin-off).
Diese Fälle sind in `out/big_moves_in_holdings.csv` aufgelistet, siehe
Abschnitt "Datenqualität".

**Delistings.** Ein Name bleibt im Panel bis zu seinem letzten Bar; danach
wird die Position zum letzten Schlusskurs liquidiert (0 % Rendite bis zum
Monatsende). Das ist bei Übernahmen realistisch, bei Insolvenzen optimistisch
(der echte Delisting-Return fehlt). Nachweis, dass Delistings enthalten sind,
in der Tabelle unten.

## Festgelegte Regeln (einmal gewählt, nicht variiert)

| Regel | Wert |
|---|---|
| Rebalance | monatlich. S = letzter Handelstag des Vormonats (Signal), R = erster Handelstag (Einstieg zum Close), R′ = erster Handelstag des Folgemonats (Ausstieg zum Close). Ein Tag Umsetzungsverzögerung, kein Lookahead |
| Universum an S | CS im Snapshot, Name ohne "Acquisition" (SPAC), gelistete Börse (XNYS/XNAS/XASE/ARCX/BATS), Schlusskurs an S beobachtet und ≥ 5 USD, ≥ 253 Handelstage Historie (damit jeder Faktor definiert ist), 60-Tage-Median des Dollarvolumens (≥ 50 Beobachtungen), Top 500 |
| 12-1-Momentum | TR[S−21] / TR[S−252] − 1 |
| Short-Term-Reversal | −(TR[S] / TR[S−5] − 1) |
| Low-Volatility | −Std(tägliche TR-Renditen, 60 Tage bis S) |
| 52-Wochen-Hoch | adj. Close[S] / max(adj. High, 252 Tage bis S) |
| Kombination | Mittel der vier Perzentilränge, gleichgewichtet |
| Dezile | Rang innerhalb des Universums, 50 Namen je Dezil, D10 = Long-Seite |
| Long-Short | D10 − D1, je Bein gleichgewichtet, je 100 % Notional |
| Long-only | D10 |
| Benchmark | gleichgewichtetes Universum, monatlich rebalanciert, ohne Kosten |
| Kosten | 10 bp je Seite auf gehandelten Wert: Kosten = 0,001 × Σ\|w_ziel − w_gedriftet\| je Rebalance, Erstaufbau voll belastet |
| Umschlag | einseitig, 0,5 × Σ\|w_ziel − w_gedriftet\|, ×12 |
| Sharpe | Monats-Überschuss, ×√12 |
| DSR | Bailey & López de Prado 2014; N = 4 Versuche für die vier Faktoren, N = 5 für die Kombination (fünfter vorab festgelegter Test); Varianz der Sharpes über die Familie, Schiefe/Kurtosis der Serie |
| Überschuss | Long-only: Strategie − Benchmark je Monat. Long-Short: der Spread selbst (Benchmark hebt sich auf) |
| Tage ohne Bar | Forward-Fill (0 % Rendite) |

Reproduktion:

    cd analysis/xsec_factor_test
    python3 -m venv .venv && .venv/bin/pip install pandas numpy pyarrow requests scipy
    .venv/bin/python fetch_polygon.py --all      # ~90 min, resumierbar
    .venv/bin/python build_panel.py
    .venv/bin/python run_factors.py               # -> out/REPORT_TABLES.md, results.json


## Datenqualität

* 786 Zellen im Gesamtpanel mit Tages-Bruttorendite > 3× oder < 0,2× auf
  beobachteten Bars; **keine davon** fällt in ein Universumsmitglied während
  einer Haltephase (`out/big_moves_in_holdings.csv` ist leer). Die
  Liquiditätsschwelle des 500. Namens liegt bei 101–222 Mio. USD Tagesumsatz
  (Median 123 Mio.), das Universum ist im Wesentlichen S&P-500-Größe plus
  liquide Mid-Caps.
* Polygons adjustierte Bars korrigieren neben Splits auch Spin-offs,
  Aktiendividenden und einzelne Sonderdividenden (GE 2023-01/2024-04,
  HON, MMM, DD, FTV, CBSH 5 %-Aktiendividenden), reguläre Bardividenden
  nicht (AAPL, MSFT, COST: Faktor ≡ 1). 27 von 744 Faktor-Sprüngen auf
  ≥-5-USD-Namen fallen mit einer Bardividende im Feed zusammen (PHG, CBSH,
  SLG, RYN); dort wird eine Dividende von 1–5 % einmalig doppelt gezählt.
  Nicht korrigiert, dokumentiert.
* 2 078 von 2 143 Split-Feed-Einträgen auf Panel-Entitäten fallen mit einem
  Faktor-Sprung in den adjustierten Bars zusammen (97 %); die adjustierten
  Bars enthalten 2 849 Sprünge, also ~700 Korrekturen, die der Split-Feed
  nicht kennt.
* Entitäts-Zuordnung: 47,4 % aller Grouped-Bar-Zeilen sind an ihrem Datum
  Common Stock und werden übernommen, 0,3 % über den Vormonats-Snapshot,
  52,2 % (ETF, ADR, Preferred, Warrants, Units, Fonds) fallen weg.
* 1 254 Handelstage, 49 leere Wochentage, alle echte Feiertage (inkl.
  2025-01-09, Trauertag Carter). Eine Polygon-Zeitüberschreitung beim
  Dividenden-Feed, mit Backoff wiederholt, keine Lücke.

## Dateien

* `analysis/xsec_factor_test/` — `common.py`, `fetch_polygon.py`,
  `build_panel.py`, `run_factors.py`, `README.md`
* `analysis/xsec_factor_test/out/` — `REPORT_TABLES.md`, `results.json`
  (alle Kennzahlen inkl. Jahres- und LOYO-Reihen), `monthly_returns.csv`
  (47 Monate × alle Portfolios brutto/netto/Umschlag), `decile_returns.csv`,
  `holdings.parquet`, `big_moves_in_holdings.csv`
* Cache (gitignored, ~1,6 GB): 1 303 + 1 303 Grouped-Parquets roh/adjustiert,
  60 Snapshots, Referenztabellen, Splits, Dividenden

## Ergebnistabellen (aus `out/REPORT_TABLES.md`)

Zeitraum der Monatsrenditen: **2022-10-03 bis 2026-09-01** (47 Monate). Universum 500–500 Namen je Monat, 824 verschiedene Entitäten insgesamt.

Benchmark (gleichgewichtetes Universum, ohne Kosten): +18.88 % p.a., Sharpe absolut 1.17, Umschlag 1.11x p.a.


### Long-only, oberstes Dezil (Überschuss über Benchmark)

| Portfolio | Überschuss p.a. netto | brutto | Sharpe netto | DSR | PSR>0 | MaxDD | Trefferquote | Umschlag p.a. | Kosten p.a. | t |
|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | +14.06 % | +14.93 % | 0.66 | 0.14 | 0.90 | -25.55 % | 53 % | 3.8x | -0.87 % | 1.30 |
| st_rev | +21.49 % | +24.02 % | 1.19 | 0.50 | 1.00 | -8.64 % | 60 % | 10.5x | -2.53 % | 2.36 |
| low_vol | -9.14 % | -8.37 % | -0.59 | 0.00 | 0.12 | -36.51 % | 49 % | 4.2x | -0.77 % | -1.17 |
| hi52 | -12.90 % | -11.44 % | -1.29 | 0.00 | 0.00 | -42.06 % | 43 % | 8.2x | -1.46 % | -2.55 |
| combo | -6.23 % | -4.95 % | -0.49 | 0.00 | 0.15 | -24.32 % | 49 % | 6.7x | -1.27 % | -0.97 |

#### Diagnose: Konzentration und Beta (Überschuss netto)

| Portfolio | bester Monat | dessen Überschuss | p.a. ohne besten Monat | p.a. ohne beste 3 | Sharpe ohne beste 3 | Beta zur Benchmark (t) | Alpha p.a. (t) |
|---|---|---|---|---|---|---|---|
| mom_12_1 | 2026-04 | +19.97 % | +9.08 % | +1.39 % | 0.17 | +0.62 (3.0) | +4.65 % (0.4) |
| st_rev | 2026-04 | +25.71 % | +14.94 % | +9.43 % | 0.89 | +0.44 (2.9) | +13.62 % (1.5) |
| low_vol | 2025-02 | +9.30 % | -11.41 % | -14.41 % | -1.13 | -0.67 (-7.4) | +4.00 % (0.8) |
| hi52 | 2024-01 | +3.85 % | -14.01 % | -15.89 % | -1.71 | -0.45 (-6.5) | -4.75 % (-1.2) |
| combo | 2024-01 | +4.81 % | -7.50 % | -9.73 % | -0.85 | -0.51 (-6.5) | +3.77 % (0.8) |

#### Verteilung über die Jahre (Überschuss netto, Trefferquote)

| Portfolio | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| mom_12_1 | -2.72 % (33 %, n=3) | -0.88 % (50 %, n=12) | +31.65 % (58 %, n=12) | +28.07 % (58 %, n=12) | +2.97 % (50 %, n=8) |
| st_rev | -1.12 % (33 %, n=3) | +43.63 % (75 %, n=12) | +6.06 % (42 %, n=12) | +18.69 % (75 %, n=12) | +19.91 % (50 %, n=8) |
| low_vol | +3.16 % (67 %, n=3) | -17.15 % (50 %, n=12) | -2.36 % (50 %, n=12) | -13.55 % (33 %, n=12) | -4.79 % (62 %, n=8) |
| hi52 | -1.33 % (67 %, n=3) | -23.34 % (33 %, n=12) | -2.46 % (50 %, n=12) | -11.38 % (33 %, n=12) | -10.97 % (50 %, n=8) |
| combo | -2.10 % (33 %, n=3) | -9.99 % (50 %, n=12) | +1.44 % (50 %, n=12) | -12.41 % (42 %, n=12) | -0.70 % (62 %, n=8) |

#### Leave-one-year-out (Überschuss p.a. netto / Sharpe ohne das Jahr)

| Portfolio | ohne 2022 | ohne 2023 | ohne 2024 | ohne 2025 | ohne 2026 |
|---|---|---|---|---|---|
| mom_12_1 | +15.96 % / 0.72 | +19.68 % / 0.79 | +8.59 % / 0.45 | +9.62 % / 0.50 | +16.13 % / 0.81 |
| st_rev | +23.50 % / 1.27 | +14.72 % / 0.82 | +27.29 % / 1.33 | +22.47 % / 1.16 | +19.58 % / 1.47 |
| low_vol | -10.50 % / -0.69 | -6.22 % / -0.40 | -11.36 % / -0.69 | -7.58 % / -0.48 | -9.56 % / -0.66 |
| hi52 | -13.40 % / -1.36 | -9.01 % / -1.03 | -16.22 % / -1.52 | -13.42 % / -1.26 | -12.25 % / -1.24 |
| combo | -6.09 % / -0.47 | -4.90 % / -0.46 | -8.72 % / -0.65 | -4.00 % / -0.28 | -7.25 % / -0.60 |

### Long-Short, D10 minus D1

| Portfolio | Überschuss p.a. netto | brutto | Sharpe netto | DSR | PSR>0 | MaxDD | Trefferquote | Umschlag p.a. | Kosten p.a. | t |
|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | +10.71 % | +12.45 % | 0.47 | 0.23 | 0.81 | -34.05 % | 57 % | 7.8x | -1.74 % | 0.93 |
| st_rev | +22.08 % | +27.21 % | 0.97 | 0.61 | 0.99 | -15.34 % | 62 % | 20.9x | -5.13 % | 1.92 |
| low_vol | -30.73 % | -29.59 % | -0.56 | 0.00 | 0.13 | -85.95 % | 53 % | 7.9x | -1.15 % | -1.10 |
| hi52 | -23.85 % | -21.92 % | -0.71 | 0.00 | 0.06 | -69.46 % | 49 % | 12.2x | -1.92 % | -1.40 |
| combo | -10.11 % | -7.71 % | -0.24 | 0.01 | 0.32 | -43.60 % | 49 % | 13.1x | -2.41 % | -0.47 |

#### Diagnose: Konzentration und Beta (Überschuss netto)

| Portfolio | bester Monat | dessen Überschuss | p.a. ohne besten Monat | p.a. ohne beste 3 | Sharpe ohne beste 3 | Beta zur Benchmark (t) | Alpha p.a. (t) |
|---|---|---|---|---|---|---|---|
| mom_12_1 | 2026-04 | +22.65 % | +5.20 % | -4.25 % | 0.02 | +0.15 (0.5) | +14.28 % (0.7) |
| st_rev | 2026-04 | +30.84 % | +14.31 % | +8.11 % | 0.56 | +0.08 (0.4) | +23.29 % (1.7) |
| low_vol | 2025-02 | +22.77 % | -34.87 % | -41.22 % | -0.99 | -1.99 (-6.6) | +12.57 % (0.7) |
| hi52 | 2024-01 | +11.34 % | -26.39 % | -31.34 % | -1.09 | -1.35 (-6.6) | +3.37 % (0.3) |
| combo | 2026-06 | +12.29 % | -12.99 % | -18.50 % | -0.63 | -1.07 (-5.2) | +14.18 % (1.1) |

#### Verteilung über die Jahre (Überschuss netto, Trefferquote)

| Portfolio | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| mom_12_1 | +8.80 % (67 %, n=3) | -26.80 % (50 %, n=12) | +46.76 % (58 %, n=12) | +26.59 % (58 %, n=12) | +0.69 % (62 %, n=8) |
| st_rev | +1.33 % (67 %, n=3) | +43.21 % (83 %, n=12) | +6.39 % (50 %, n=12) | +2.32 % (50 %, n=12) | +38.30 % (62 %, n=8) |
| low_vol | +20.36 % (100 %, n=3) | -46.32 % (50 %, n=12) | -14.28 % (58 %, n=12) | -46.03 % (25 %, n=12) | -20.61 % (75 %, n=8) |
| hi52 | +12.64 % (67 %, n=3) | -50.92 % (50 %, n=12) | -6.30 % (42 %, n=12) | -23.99 % (50 %, n=12) | -12.63 % (50 %, n=8) |
| combo | +14.05 % (67 %, n=3) | -32.57 % (50 %, n=12) | +7.99 % (50 %, n=12) | -17.05 % (42 %, n=12) | -4.39 % (50 %, n=8) |

#### Leave-one-year-out (Überschuss p.a. netto / Sharpe ohne das Jahr)

| Portfolio | ohne 2022 | ohne 2023 | ohne 2024 | ohne 2025 | ohne 2026 |
|---|---|---|---|---|---|
| mom_12_1 | +8.95 % / 0.42 | +27.59 % / 0.87 | +0.52 % / 0.20 | +5.74 % / 0.34 | +12.81 % / 0.54 |
| st_rev | +23.31 % / 1.00 | +15.58 % / 0.70 | +27.98 % / 1.08 | +29.70 % / 1.18 | +15.10 % / 0.95 |
| low_vol | -35.78 % / -0.70 | -24.41 % / -0.37 | -35.62 % / -0.60 | -24.55 % / -0.36 | -31.03 % / -0.69 |
| hi52 | -27.64 % / -0.86 | -11.47 % / -0.37 | -29.07 % / -0.80 | -23.80 % / -0.62 | -24.93 % / -0.78 |
| combo | -13.91 % / -0.38 | -0.81 % / 0.08 | -15.60 % / -0.38 | -7.60 % / -0.10 | -10.84 % / -0.29 |

### Dezil-Monotonie (mittlerer Monats-Überschuss je Dezil, brutto)

| Faktor | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | Spearman ρ (p) | D10−D1 | D10−D9 | D2−D1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | -0.08 | -0.20 | -0.08 | -0.38 | -0.12 | -0.14 | -0.19 | -0.05 | -0.16 | +1.41 | 0.28 (0.43) | +1.49 | +1.57 | -0.12 |
| st_rev | -0.30 | -0.35 | +0.10 | -0.23 | -0.48 | -0.11 | -0.07 | -0.23 | -0.25 | +1.93 | 0.42 (0.23) | +2.23 | +2.18 | -0.05 |
| low_vol | +1.32 | +0.62 | +0.32 | +0.41 | -0.28 | -0.17 | -0.38 | -0.62 | -0.60 | -0.64 | -0.96 (0.00) | -1.96 | -0.04 | -0.71 |
| hi52 | +0.65 | -0.19 | +0.11 | -0.07 | -0.17 | +0.29 | +0.27 | -0.16 | +0.24 | -0.96 | -0.25 (0.49) | -1.61 | -1.20 | -0.84 |
| combo | -0.03 | +0.22 | -0.39 | +0.17 | +0.26 | +0.23 | -0.20 | +0.18 | -0.08 | -0.37 | -0.21 (0.56) | -0.33 | -0.29 | +0.25 |

### Universum: Delistings sind enthalten

| Jahr | Universumsmitglieder im Jahr | davon im Jahr aus dem Panel ausgeschieden | Polygon: delistete CS gesamtmarkt |
|---|---|---|---|
| 2022 | 529 | 6 | 519 |
| 2023 | 602 | 15 | 704 |
| 2024 | 605 | 11 | 569 |
| 2025 | 633 | 10 | 522 |
| 2026 | 614 | 14 | 448 |
