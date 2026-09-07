# Shadow-Buch je Strategie, 2026-09-07

Analyse, kein Deploy. Kein zusätzlicher Trade, keine Änderung am Live-Pfad.
Skript: `scripts/shadow_strategy_book.py` (read-only, `mode=ro`, stdlib).
Datenstand: Prod-DB `/home/trading/trading-data/news_trading.db` am
2026-09-07 ~15:10 UTC, `daily_ohlc` bis 2026-09-04. Der Lauf gegen den
Live-Pfad und gegen einen read-only Snapshot lieferten byte-identische
Ausgaben.

## 0. Kurzantwort

* **Positive Erwartung im Shadow-Buch:** nur NewsCatalyst zeigt eine
  (+0,95 % je Trade, N = 102). Das 95 %-Konfidenzintervall schließt null
  ein (−0,15 … +2,06 %), die einseitige PSR liegt bei 96 %. Die Nullreferenz
  (unbedingter Einstieg, gleiches Exit-Modell) liegt bei +0,19 %. Das ist
  ein Hinweis, kein Nachweis. Momentum (+0,42 %, N = 66) und Combined
  (+0,22 %, N = 150) liegen auf Höhe der Nullreferenz; Pullback ist negativ
  (−0,38 %, N = 53). Für Momentum, Pullback und Combined reicht die
  Stichprobe für keine Aussage.
* **Combined vs. Bestandteile:** Combined ist schlechter als NewsCatalyst
  und Momentum einzeln. 127 der 150 Combined-Trades standen auf einem
  einzigen Vote (−0,10 %); die 20 Zwei-Vote-Trades liegen bei +2,02 %.
  Das passt zur Richtung des Agreement-Gates, ist aber bei N = 20 keine
  Bestätigung.
* **Shadow vs. real:** Der Unterschied liegt in der **Signalauswahl**, nicht
  in der Ausführung. Die tatsächlich gefüllten Combined-Signale hatten im
  Shadow-Modell eine Erwartung von −0,91 % (alle Combined-Signale der
  Paper-Ära: +0,11 %). Real erzielten dieselben Signale −0,54 %. Auswahl
  −1,02 %, Ausführung +0,37 % je Trade.

## 1. Parametersatz (vorab festgelegt, ein einziger)

Aus dem Live-Modell (`agents/risk_agent.py`, `cluster_detector.py`,
`portfolio_manager.py`):

| Parameter | Wert |
|---|---|
| ATR | Wilder ATR(14), H/L/raw close (D3), 90 Kalendertage Fenster, ≥ 15 Bars |
| Stop / TP | 1,5 × ATR / 3,0 × ATR; Stop-Floor 1 % vom Einstieg; TP ≥ 2 × Stop |
| Eine Position je Ticker | wie das Live-„Already holding"-Gate |
| Nur Long | der Live-Pfad eröffnet keine Shorts; 15 WEAK-SELL-Combined-Zeilen ausgeschlossen |

Vom Live-Modell nicht vorgegeben, einmal gewählt und nicht variiert:

| Parameter | Wert | Begründung |
|---|---|---|
| Haltehorizont | 10 Handelstage inkl. Einstiegstag | Live hat keinen Zeitausstieg; 10 d ist der längste systemeigene Outcome-Horizont (`outcome_10d_pct`) |
| Einstieg | erste Tagesbar, deren Open (09:30 New York) strikt **nach** dem Signalzeitstempel liegt; Preis = raw open | „nächstes Open", kein Lookahead |
| Beide Levels in einer Bar | Stop zählt | konservativ, Tagesbars haben keinen Pfad |
| Gap über ein Level (ab Tag 2) | Fill am Open | realistisch, nicht am Level |
| Zeitausstieg | Close der 10. Bar | |
| Notional | 10 000 USD flach je Trade, nur für die $-Spalte | Live-Sizing ist Risiko/Stop mit 10 %-Cap, der Cap band fast immer; Flat-Notional ist der nähere Proxy |
| Sharpe | erst ab N ≥ 10; PSR und MinTRL bei 95 % | mit genau einem Regelsatz ist der DSR-Deflationsterm null, DSR ≡ PSR |

Verknüpfungsregeln (der einzige Ort, an dem etwas erschlossen wird):
Strategy-Zeilen gehören zur Combined-Zeile desselben Tickers und derselben
Session, die ≤ 20 min später geloggt wurde; ein `trade_history`-BUY gehört
zur jüngsten gerichteten Combined-Zeile des Tickers innerhalb von 26 h vor
dem Fill (Toleranz −5 min, wie in `scripts/cluster_gate_backtest.py`).

## 2. Datenlage, ehrlich

**Preisabdeckung.** `daily_ohlc` enthält genau 20 US-Ticker (2021-06 bis
2026-09-04, lückenlos, Polygon). Alles andere fällt weg: die EU-Ticker
(VIE.PA, COFA.PA, VNA.DE, FNTN.DE, LEG.DE, HOT.DE) und die Scanner-Namen
(MSTR, NBIS, NVDA, COST, UNH, BE). Ausschlüsse: Combined 124, Momentum 45,
Pullback 39, NewsCatalyst 68 gerichtete Zeilen. Das Shadow-Buch ist also
ein 20-Ticker-Buch.

**B3-Attribution.** Die Shadow-Bücher je Strategie brauchen sie **nicht**:
jede Strategie-Zeile in `signal_events` trägt seit 2026-03-28 ihren
Strategienamen (NewsCatalyst seit 2026-04-14). Erschlossen wird nur
(a) die Zuordnung Vote ↔ Combined-Zeile über das 20-min-Fenster und
(b) Fill ↔ Combined-Zeile. Beides ist im Bestand eindeutig: 926 von 980
gerichteten Combined-Zeilen haben genau drei Strategie-Zeilen im Fenster,
54 haben zwei (alle im April 2026, bevor NewsCatalyst existierte), kein
Fenster enthält eine Strategie doppelt. 162 von 165 realen Round-Trips
finden ihre Combined-Zeile; die 3 ohne stammen vom ersten Paper-Tag. Die
`signal_attribution`-Tabelle selbst hat in Prod 7 Zeilen (ab 2026-09-01),
sie wurde für diese Auswertung nicht gebraucht. Der frühe Teil ist damit
genauso belastbar wie der späte; die Ära-Trennung am 2026-08-28 ist im
Skript ausgewiesen, seit dem Stichtag konnte mit 10-Tage-Horizont noch kein
Trade schließen (N = 0).

**Reale Ausstiegsgründe** werden nicht persistiert; sie sind aus den
Levels des BUY rekonstruiert (Exit ≤ Stop × 1,003 → SL, ≥ TP × 0,997 → TP,
sonst OTHER = Trailing/Signal/manuell), wie im Cool-down-Gate des
PortfolioManagers.

**Einstiegszeitpunkt.** Signale der Sessions PEAD_OPEN (13:45 UTC), US_OPEN
und EOD steigen im Shadow am **Folgetag**-Open ein, US_PRE-Signale
(13:15 UTC) am selben Tag. Live füllt PEAD_OPEN-Signale sofort. Bei 81 von
162 verknüpften Trades liegt der Shadow-Einstieg deshalb einen Handelstag
nach dem realen Fill. Das ist eine Folge der Regel „nächstes Open", nicht
ein Fehler, und es begrenzt die Vergleichbarkeit von Zeile B und C in §5.

**Überlappung.** Die Bücher sind nicht unabhängig: bis zu 10
(NewsCatalyst), 11 (Momentum), 16 (Combined) Positionen gleichzeitig
offen, 17–20 Ticker, 18–20 Einstiegswochen. Die Monatssummen schwanken
stark (NewsCatalyst: April +34 %, Juni −5 %, Juli +52 %; Combined: Mai −32 %,
Juni −42 %, Juli +53 %). Das effektive N ist deutlich kleiner als die
Trade-Zahl; alle Konfidenzintervalle unten sind eher zu eng.

## 3. Population

| Buch | gerichtete Zeilen | ohne OHLC | SELL-Seite | ohne Einstiegsbar | auswertbar | in offene Position gefaltet | Trades | davon offen |
|---|---|---|---|---|---|---|---|---|
| Momentum | 282 | 45 | 0 | 1 | 236 | 165 | 71 | 5 |
| Pullback | 307 | 39 | 0 | 5 | 263 | 209 | 54 | 1 |
| NewsCatalyst | 606 | 68 | 0 | 4 | 534 | 423 | 111 | 9 |
| Combined | 995 | 124 | 15 | 9 | 847 | 690 | 157 | 7 |

„Gefaltet" heißt: dieselbe Strategie hat den Ticker erneut gemeldet,
während die Shadow-Position noch offen war (jede Session meldet jeden
Ticker neu). Offene Trades (Einstieg nach 2026-08-21) sind nicht in den
Kennzahlen; ihr unrealisierter Mittelwert liegt zwischen −1,1 % und +0,9 %.

## 4. Shadow-Bücher (geschlossene Trades)

| | Momentum | Pullback | NewsCatalyst | Combined | Nullreferenz |
|---|---|---|---|---|---|
| N | 66 | 53 | 102 | 150 | 1854 (überlappend) |
| Trefferquote | 45,5 % | 43,4 % | 52,0 % | 46,7 % | 45,8 % |
| Ø Gewinn | +5,64 % | +4,61 % | +5,61 % | +5,07 % | |
| Ø Verlust | −3,93 % | −4,20 % | −4,08 % | −4,02 % | |
| Erwartung je Trade | +0,42 % | −0,38 % | +0,95 % | +0,22 % | +0,19 % |
| 95 %-KI | −0,96 … +1,80 | −1,82 … +1,06 | −0,15 … +2,06 | −0,65 … +1,10 | (zu eng) |
| Median | −1,25 % | −2,42 % | +0,30 % | −0,77 % | −1,24 % |
| Summe | +27,7 % = +2 772 $ | −20,0 % = −2 002 $ | +97,1 % = +9 708 $ | +33,6 % = +3 356 $ | |
| Haltedauer p25 / Median / p75 (Bars) | 4 / 8 / 10 | 4 / 9 / 10 | 4 / 9 / 10 | 4 / 10 / 10 | |
| Exit SL / TP / Zeit | 44 / 20 / 36 % | 43 / 13 / 43 % | 41 / 22 / 37 % | 41 / 17 / 43 % | |
| Ø Ergebnis SL / TP / Zeit | −4,5 / +8,8 / +1,8 % | −4,9 / +9,7 / +1,1 % | −4,5 / +9,0 / +2,3 % | −4,7 / +8,9 / +1,5 % | |
| Sharpe je Trade (nicht annualisiert) | 0,074 | −0,071 | 0,167 | 0,041 | |
| PSR = DSR (1 Versuch), P(SR > 0) | 72,9 % | 31,1 % | 96,1 % | 69,4 % | |
| MinTRL @95 % vs. N | 476 vs. 66 | n/a (SR ≤ 0) | 89 vs. 102 | 1571 vs. 150 | |
| **reicht die Stichprobe?** | **nein** | **nein** | **knapp, einseitig** | **nein** | |

Nullreferenz: Einstieg an **jedem** Bar-Open der 20 Ticker im Zeitraum
2026-04-15 … 2026-09-04 mit demselben Exit-Modell. Sie ist keine Strategie,
sondern die Drift des Universums unter diesem Modell; ihre Pfade
überlappen täglich, das KI ist bedeutungslos.

Lesart:

* Alle vier Bücher haben dieselbe Form: ~40–45 % Stops um −4,5 %, ~15–20 %
  TPs um +9 %, der Rest Zeitausstiege leicht positiv. Der Unterschied
  zwischen den Strategien liegt fast nur in der Trefferquote.
* Momentum und Combined sind von der Nullreferenz nicht unterscheidbar.
  Pullback liegt darunter. NewsCatalyst liegt darüber; das ist die einzige
  Beobachtung, die eine Aussage trägt, und auch die nur einseitig auf
  95 %. Zwei Monate (April, Juli) tragen 89 % der NewsCatalyst-Summe.
* Die Sharpe-Werte sind je Trade, nicht annualisiert; das Buch ist keine
  fortlaufende Portfolioserie.

## 5. Combined vs. Bestandteile

Combined-Trade fällt (gleicher Ticker, gleicher Einstiegstag) mit einem
Trade von … zusammen:

| | N | Erwartung |
|---|---|---|
| Momentum | 37 | +1,01 % |
| Pullback | 36 | −1,50 % |
| NewsCatalyst | 72 | +1,29 % |
| keiner der drei | 22 | −0,09 % |

Nach Zahl der gerichteten Strategie-Votes im Run (ohne die 0,35-Schwelle):

| Votes | N | Erwartung | Trefferquote |
|---|---|---|---|
| 0 | 3 | +1,95 % | 33 % |
| 1 | 127 | −0,10 % | 46 % |
| 2 | 20 | +2,02 % | 55 % |

Combined ist die Vereinigung seiner Bestandteile plus das strengste
Ein-Vote-Verhalten: 85 % der Combined-Trades sind Solo-Votes und liegen
bei −0,10 %. Die Zwei-Vote-Trades sehen besser aus, aber N = 20 ist keine
Basis für eine Zahl mit Nachkommastellen. Was sich sagen lässt: Combined
verdünnt NewsCatalyst und Momentum mit Pullback-Solos (−1,50 %) und
verliert dabei gegenüber beiden.

## 6. Shadow vs. real: Auswahl oder Ausführung?

`trade_history`: 171 BUY / 173 SELL → 165 FIFO-Round-Trips, 8 verwaiste
SELLs verworfen (Teilfüllungs-Befund vom 2026-08-25), 6 BUYs offen.
Realisierter P&L laut Tabelle: **−24 498 USD**.

Reales Buch, Fill-zu-Fill: N = 165, Trefferquote 48,5 %, Ø Gewinn
+1,93 %, Ø Verlust −2,77 %, Erwartung **−0,49 %** (95 %-KI −0,96 …
−0,02 %), Median-Haltedauer 2 Kalendertage, Sharpe je Trade −0,16, PSR
2,3 %. Exit rekonstruiert: SL 84 (Ø −2,8 %), TP 23 (Ø +2,9 %), OTHER 58
(Ø +1,5 %).

Zerlegung auf denselben Signalen (Paper-Ära ab 2026-05-04):

| Linie | N | Erwartung | Trefferquote |
|---|---|---|---|
| A  Shadow, jedes gerichtete Combined-Signal (je Signal, überlappend) | 724 | +0,11 % | 47 % |
| A' Shadow, dasselbe mit einer Position je Ticker (Buch aus §4) | 132 | −0,02 % | 45 % |
| B  Shadow, nur die tatsächlich gefüllten Combined-Signale | 156 | **−0,91 %** | 39 % |
| C  Real, dieselben gefüllten Signale (Fill-zu-Fill) | 156 | −0,54 % | 47 % |

**Auswahleffekt B − A: −1,02 % je Trade. Ausführungseffekt C − B: +0,37 %.**

Die Signale, die das Live-System tatsächlich gefüllt hat, waren im
Shadow-Modell deutlich schlechter als der Durchschnitt aller
Combined-Signale desselben Zeitraums. Die Ausführung (Fill, Stops,
Trailing, PM-Exits) hat auf diesen Signalen nicht verloren, sondern
gegenüber dem Shadow-Modell leicht gewonnen. Details der Ausführung:

| | Wert |
|---|---|
| realer Fill vs. Shadow-Open, Mittel / Median | +0,22 % / +0,01 % |
| Shadow-Einstieg am selben Handelstag wie der Fill | 81 / 162 |
| Stop-Distanz real (Median) vs. Shadow (Median) | 2,35 % vs. 4,47 % |
| Haltedauer real (Kalendertage, Median) vs. Shadow (Bars, Median) | 1,5 vs. 8,0 |

Real ↔ Shadow Ausstieg (N): SL→SL 53, OTHER→TIME 23, SL→TIME 23,
OTHER→SL 22, TP→TIME 10, OTHER→TP 9, TP→SL 8, SL→TP 5, TP→TP 3, Rest offen.

Zwei Einschränkungen, die den Ausführungseffekt unsicher machen: Bei der
Hälfte der Trades steigt das Shadow-Modell einen Tag später ein als der
reale Fill, und das reale Buch lief mit halb so weiten Stops (Ära des
Close-to-Close-Proxys und des −2 %-Overrides, bis 2026-09-01) und ohne
Zeitausstieg. C − B vergleicht also zwei Exit-Modelle, nicht nur zwei
Fills. Der Auswahleffekt ist davon unberührt: A und B benutzen dasselbe
Modell.

Warum die gefüllten Signale schlechter waren, beantwortet der Bestand
nicht direkt. Was er zeigt: 53 der 162 gefüllten Signale endeten sowohl
real als auch im Shadow am Stop, und die gefüllten Signale konzentrieren
sich auf wenige Ticker (CASY 17, XOM 15, TRGP 14, MSFT 13 Fills) mit wiederholtem
Wiedereinstieg nach Stop. Die Kapazitätsgates füllen Slots, sobald sie
frei werden, also nach Stop-outs in fallenden Namen.

Reale Trades nach Vote im gefüllten Run (Mehrfachnennung möglich):

| Strategie hat BUY gevotet | N Round-Trips | real | Shadow derselben Signale |
|---|---|---|---|
| Momentum | 41 | −0,09 % | −0,75 % |
| Pullback | 53 | −0,64 % | −1,95 % |
| NewsCatalyst | 97 | −0,69 % | −0,31 % |

Auch hier: die NewsCatalyst-Signale, die gefüllt wurden, sind im Shadow
negativ (−0,31 %), während das volle NewsCatalyst-Shadow-Buch bei +0,95 %
liegt. Der Auswahleffekt trifft die einzige Strategie mit positiver
Shadow-Erwartung genauso.

## 7. Was sich daraus ergibt, und was nicht

* Eine Aussage „Strategie X hat positive Erwartung" trägt der Bestand für
  keine der drei. NewsCatalyst ist der einzige Kandidat, und er braucht
  mehr Trades in mehr als zwei tragenden Monaten.
* Die Divergenz zwischen Shadow und real ist ein Auswahlproblem. Jede
  Diskussion über Slippage oder Stop-Weite adressiert die kleinere
  Komponente.
* Das Shadow-Buch ist ein 20-Ticker-Buch. Für die EU- und Scanner-Ticker
  gibt es keine Preisbasis im Store; dort lässt sich nichts sagen.
* Nicht gemacht, absichtlich: keine Schwellen, keine Haltedauern, keine
  Fold-Varianten. Wer die Zwei-Vote-Zeile aus §5 als Beleg für das
  Agreement-Gate lesen will, hat N = 20.

## 8. Reproduktion

```bash
# auf dem VPS, read-only, keine Schreibzugriffe
python3 scripts/shadow_strategy_book.py
# lokal gegen einen Snapshot, mit Trade-CSVs
python3 scripts/shadow_strategy_book.py --db snapshot.db --csv /tmp/shadow
```

Stichprobenprüfung gegen die Rohbars (CASY 04-17 TP, VRT 04-17 Zeit,
VRT 05-08 SL, JPM 09-02 offen) und unabhängige ATR-Nachrechnung: identisch.
