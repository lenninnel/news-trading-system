# Q2 Earnings-Quellen-Eval — Verdikt (Stand 2026-09-23)

Reine Auswertung der seit 2026-08-24 laufenden Capture-Eval (Finnhub, FMP, Alpha Vantage, yfinance 1.6.0). VPS ausschließlich lesend (SQLite `mode=ro`, journalctl, systemctl status). Keine Writes, kein systemctl-Eingriff, kein Git auf dem VPS.

## 1. Verdikt

**Noch nicht entscheidbar auf dem Primärkriterium — warten bis zur Q3-Saison, Auswertung um den 2026-11-20 wiederholen.**

Begründung in einem Satz: Im Beobachtungsfenster 2026-08-24 bis 2026-09-22 hat genau **ein** Universum-Titel tatsächlich berichtet (CASY, 2026-09-08). Ein Verdikt „liefert Termine zuverlässig" braucht realisierte Termine, gegen die man die Vorab-Ansagen der Quellen prüfen kann. N=1 reicht nicht, egal wie gut die Zahl aussieht.

Was sich **jetzt schon** belastbar sagen lässt, unabhängig von der Saison:

| Quelle | Status heute | Grund |
|---|---|---|
| **FMP** | **ungetestet, N=0** | 60/60 Runs mit HTTP 403 auf `/api/v3/earning_calendar`. Nie eine Zeile geliefert. Der Key ist gesetzt (32 Zeichen); der Legacy-v3-Pfad ist für den Account offenbar gesperrt. Das ist ein Endpoint-/Plan-Problem, kein Datenqualitätsbefund. Ohne Code-Änderung (Umstellung auf den `/stable`-Pfad) bleibt FMP aus dem Rennen. |
| **Alpha Vantage** | **vorläufig disqualifiziert** | (a) 4 von 30 Kalender-Runs (3 von 29 Timer-Runs) lieferten HTTP 200 mit einer header-only-CSV — 0 Termine, kein Fehler, still. (b) Termine sind erkennbar Schätz-Platzhalter: 4 Titel auf identischem 10-28, 5 Titel auf 11-04/11-05; drei Termine sprangen während der Eval (BAC 10-14→10-21→10-14, MEDP 10-19→10-28, TSLA 10-21→10-28). (c) EPS-Pfad: Free-Budget 25/Tag < 26 Titel, PFE/TSLA nie abgefragt, in 21/29 Runs zusätzliche Rate-Limit-Fehler auf 1–5 Titel. (d) Keine EU-Namen, kein TOL, kein CASY-Folgetermin. |
| **yfinance 1.6.0** | **Kandidat mit Frische-Makel** | Kalender lieferte 26/26 Titel in jedem Run, 0 Helper-Fehler auf dem Kalender, Vorlauf ≥48 Tage. Aber: 59 von 728 Ticker-Tagen (8,1 %) tragen ein **abgelaufenes** Datum als „nächsten Termin": TOL zeigte 2026-08-18 bis zum letzten Capture (35 Tage danach), VIE.PA zeigte 2026-07-30 die gesamten 4 Wochen (54 Tage danach). Das ist genau das Fehlerbild von 0.2.58 in kleinerem Maßstab: die Quelle liefert still Altes. |
| **Finnhub** | **vorläufiger Favorit** | 60/60 Runs HTTP 200, 0 Fehler, Rate-Limit 60/min mit 1 Call belegt. Einziges realisiertes Event (CASY) vom ersten Capture an gelistet und mit korrektem Datum; EPS-Actual am Report-Tag um 23:00 UTC bereits vorhanden (`available_same_day=1`). Am 09-22 alle 8 US-Titel gelistet, die irgendeine Quelle ins +30-Fenster legt. Offen: EU-Namen (6 im Universum) sind auf dem Free-Tier nicht belegt — kein EU-Report fiel ins Fenster. Session-Flag nur bei 25/53 Zeilen. |

Vorläufige Rangfolge: Finnhub > yfinance 1.6.0 > Alpha Vantage; FMP ungetestet. Das Verdikt wird erst mit den realisierten Q3-Terminen gefällt (Kriterium bleibt unverändert, siehe §6).

## 2. Setup, wie vorgefunden

- Units: `~/.config/systemd/user/nts-earnings-eval-{cal,eps}.{service,timer}`, User `trading`, `Persistent=true`. Cal 12:00 UTC, EPS 23:00 UTC. Beide Timer aktiv, letzte Läufe 2026-09-22 mit Exit 0.
- Skript: `scripts/earnings_source_eval.py --mode cal|eps` (Prod-venv, yfinance 0.2.58 wird **nicht** importiert); yfinance-Helper `scripts/_yf_eval_fetch.py` läuft per Subprozess in `~/yfeval-venv` (yfinance 1.6.0, pandas 3.0.5, lxml 6.1.2). Skripte auf VPS und dev byte-identisch (md5 geprüft).
- Ziel-DB: `/home/trading/trading-data/earnings_source_eval.db`, Tabellen `cal_capture` (1.144 Zeilen), `eps_capture` (8 Zeilen), `run_log` (236 Zeilen). Kopie für die Auswertung im Scratchpad, nur `mode=ro`.
- Fenster: cal [T−2, T+30] als Abfrageparameter bei Finnhub/FMP; AV und yfinance liefern forward-only ohne Fensterparameter, deshalb stehen dort auch Termine >30 Tage in der Tabelle. Alle Fensterzahlen unten sind aus `days_ahead` nachgefiltert.
- Universum: 26 Titel = 11 US-Watchlist (META, JPM, VRT, AAPL, MSFT, AMZN, XOM, CVX, BAC, PFE, TSLA) + 15 PEAD (CASY, TXRH, DECK, TRGP, CACI, MEDP, UFPI, TOL, PBR, VNA.DE, HOT.DE, COFA.PA, VIE.PA, FNTN.DE, LEG.DE).
- Beobachtung: 30 Capture-Tage (2026-08-24 bis 2026-09-22). Finnhub/FMP ab 08-24, AV ab 08-25, yfinance ab 08-26. Journal: 29 Timer-Starts cal + 1 manueller Lauf, 30 Starts eps, 30/30 bzw. 29/29 „Finished", kein Traceback, kein Timeout-Abbruch des Prozesses.

## 3. Zahlen je Provider

### 3.1 Betrieb und Ausfälle

| Provider | Cal-Runs | Cal-Fehler | davon still (200 ohne Daten) | EPS-Runs | EPS-Fehler | Client |
|---|---|---|---|---|---|---|
| Finnhub | 30 | 0 | 0 | 30 | 0 | requests 2.33.0 |
| FMP | 30 | **30 (403)** | 0 | 30 | **30 (403)** | requests 2.33.0 |
| Alpha Vantage | 30 | 0 gemeldet | **4** (08-26 manuell, 08-27, 08-29, 09-01) | 29 | 21 Runs mit Ticker-Fehlern (Rate-Limit), 2 Read-Timeouts (Retry OK) | requests 2.33.0 |
| yfinance 1.6.0 | 29 | 0 | 0 | 28 | 1 Ticker-Fehler (DECK, KeyError 'Earnings Date', 09-02) | yfinance 1.6.0 |

AV-Rate-Limit-Header sind leer (`{}`), das Limit kommt als HTTP-200-JSON-Note; die 4 stillen Kalender-Runs haben keinen Fehlertext und wären ohne `rows_returned` unsichtbar.

### 3.2 Primärkriterium: Termin im Fenster [−2, +30]

| Provider | Titel mit Termin, gesamt (von 26) | Titel im Fenster am 2026-09-22 | Ticker-Tage im Fenster (Summe) | Realisierte Reports korrekt vorab gelistet | Max. beobachteter Vorlauf |
|---|---|---|---|---|---|
| Finnhub | 9 | **8** (JPM, BAC, MEDP, CACI, TSLA, VRT, DECK, TXRH) | 53 | **1/1** (CASY, ab erstem Capture, 15 Tage vorher) | 30 Tage (durch Abfragefenster begrenzt) |
| FMP | 0 | 0 | 0 | 0/1 | — |
| Alpha Vantage | 19 | 3 (JPM, BAC, DECK) | 32 | 1/1 (CASY, 14 Tage vorher) | 65 Tage |
| yfinance 1.6.0 | 26 | 7 (JPM, BAC, MEDP, CACI, TSLA, VRT, DECK) | 45 | 1/1 (CASY, 13 Tage vorher) | 90 Tage |

Lesehilfe: „Titel im Fenster" ist die Menge, die die Quelle selbst ins Fenster legt. Ob sie recht hat, entscheidet erst der realisierte Termin. Der einzige realisierte Termin (CASY 09-08) wurde von allen drei laufenden Quellen korrekt vorab gelistet; Vorlauf jeweils durch den Eval-Start begrenzt, nicht durch die Quelle.

### 3.3 Übereinstimmung der Termine (Capture 2026-09-22)

Referenz ist Finnhub, wo vorhanden; sonst AV gegen yfinance. Abweichung in Tagen relativ zur Referenz.

| Ticker | Finnhub | yfinance | AV | Abweichung |
|---|---|---|---|---|
| JPM | 10-13 | 10-13 | 10-13 | 0 / 0 |
| BAC | 10-14 bmo | 10-14 | 10-14 | 0 / 0 (AV zwischen 08-28 und 09-04 auf 10-21) |
| DECK | 10-22 | 10-22 | 10-22 | 0 / 0 (Finnhub am 09-21 noch 10-21) |
| CACI | 10-20 | 10-21 | 10-28 | yf +1, AV +8 |
| MEDP | 10-20 amc | 10-19 | 10-28 | yf −1, AV +8 |
| TSLA | 10-20 amc | 10-21 | 10-28 | yf +1, AV +8 |
| VRT | 10-20 | 10-21 | 10-28 | yf +1, AV +8 |
| TXRH | 10-22 amc | 11-05 | 11-05 | yf +14, AV +14 |
| AAPL, AMZN, CVX, XOM, PFE | — | = AV | = yf | 0 |
| META, MSFT | — | 10-28 | 11-04 | 7 |
| UFPI | — | 10-26 | 11-04 | 9 |
| PBR | — | 11-10 | 11-05 | 5 |
| TRGP | — | 11-05 | 11-04 | 1 |
| CASY (Folgetermin) | — | 12-08 | — | AV listet keinen |
| TOL | — | **08-18 (abgelaufen)** | — | AV listet keinen |
| 6 EU-Namen | — | 11-02 bis 11-12 | — | nur yfinance |

Muster: Finnhub und yfinance liegen bei den Oktober-Titeln bis auf ±1 Tag beieinander (vier Titel systematisch um genau einen Tag versetzt, plausibel ein amc/Zeitzonen-Effekt, aber nicht belegt). AV liegt bei denselben vier Titeln 8 Tage später auf einem gemeinsamen Datum, was auf Schätzwerte hindeutet. TXRH ist der einzige Fall mit >7 Tagen Differenz zwischen Finnhub und yfinance. Wer recht hat, zeigt sich zwischen dem 13.10. und 12.11.

### 3.4 Frische

- **yfinance Kalender:** 59/728 Ticker-Tage mit abgelaufenem Datum (TOL 29 Tage lang, VIE.PA 29 Tage lang, COFA.PA/HOT.DE/LEG.DE nur am 08-26). CASY dagegen am Tag nach dem Report korrekt auf 12-08 gerollt. Das Verhalten ist also titelabhängig und still. Für den PEAD-Anwendungsfall „hat in den letzten 2 Tagen berichtet" ist ein altes Datum unschädlich; für „wann kommt der nächste Report" ist es ein Fehlsignal.
- **Alpha Vantage:** keine abgelaufenen Termine, dafür fehlende (TOL, CASY-Folgetermin, alle EU) und die 4 stillen Leer-Antworten.
- **Finnhub:** keine Auffälligkeit; Datenmenge im Gesamtkalender bewegt sich mit der Saison (341 bis 909 Zeilen im 32-Tage-Fenster).

### 3.5 EPS (sekundär)

Einziger Datenpunkt: CASY, Report 2026-09-08 amc.

| Provider | Estimate | Actual | Surprise % | Actual verfügbar |
|---|---|---|---|---|
| Finnhub | 6.8777 | 7.37 | 7.16 | **am Report-Tag** (23:00 UTC) |
| Alpha Vantage | 6.60 | 7.37 | 11.67 | T+1 |
| yfinance 1.6.0 | 6.82 | 7.37 | 8.09 | T+1 |

Die Actuals stimmen überein, die Estimates nicht: Surprise schwankt zwischen 7,2 % und 11,7 % je Quelle. Bei einer PEAD-Schwelle von 5 % ist CASY in allen drei Fällen ein Beat, bei knapperen Fällen entscheidet die Quelle über das Signal. Das gehört in die November-Auswertung.

## 4. Saisonvorbehalt

August/September ist die ruhige Phase. Realisierte Reports im Universum während der Eval: **1** (CASY). TOL berichtete am 08-18, sechs Tage vor Eval-Start. Alle anderen 24 Titel berichten zwischen 2026-10-13 und ~2026-11-12.

| Provider | N realisierte Reports | N Titel mit Vorab-Termin (irgendwann) | N Titel im Fenster am Ende |
|---|---|---|---|
| Finnhub | 1 | 9 | 8 |
| FMP | 0 | 0 | 0 |
| Alpha Vantage | 1 | 19 | 3 |
| yfinance | 1 | 26 | 7 |

**Die Stichprobe reicht nicht.** Bis ~2026-11-15 werden 20 US-Titel und 6 EU-Titel realisiert sein; damit wird pro Quelle N≈26 auf dem Primärkriterium erreichbar, und die Divergenzen aus §3.3 lösen sich in richtig/falsch auf. Erst dann ist ein Verdikt belastbar.

## 5. Prod-Kontext (lesend erhoben)

- `PEAD_ENABLED=false` in der Prod-`.env`; der Scheduler überspringt PEAD_OPEN täglich („Skipping PEAD_OPEN — PEAD disabled"). Es gibt aktuell keinen Live-Konsumenten für Earnings-Termine.
- `pead_signal_log`: 975 Zeilen `yfinance_fallback` (2026-05-27 bis 2026-08-25) mit **0** announce_dates, 0 threshold_met. 378 Zeilen `benzinga` (recorded-only, 06-01 bis 07-28) mit announce_dates. Letzte Zeile 2026-08-25. Das bestätigt den Ausgangsbefund: der 0.2.58-Fallback hat nie einen Termin geliefert.
- Der primäre IBKR-Earnings-Cache (`ibkr_earnings_cache.json`) existiert auf dem VPS nicht (unter `~/trading-data` und im Repo gesucht).
- Prod-venv: yfinance 0.2.58, pandas 2.3.3, curl_cffi 0.14.0. yfinance 1.6.0 verlangt `curl_cffi>=0.15` und `pandas>=1.3`; der Prod-Pin `pandas<3.0` ist kompatibel, curl_cffi müsste mit.
- Alpha Vantage: `data/price_fallback.py` (Level 1 des Preis-Fallbacks) liest `ALPHA_VANTAGE_KEY`; die Eval liest `ALPHA_VANTAGE_API_KEY`. In der Prod-`.env` existiert nur der Eval-Name. Der Preis-Fallback Level 1 läuft in Prod also ohne Key (und wird übersprungen), eine Budget-Kollision mit der Eval gibt es nicht. Nebenbefund, kein Teil dieses Auftrags.

## 6. Empfehlungen

1. **Kein Verdikt jetzt.** Auswertung um den **2026-11-20** wiederholen, gleiche Kriterien, gleiche Queries (Anhang). Erwartetes N≈26 realisierte Reports je Quelle. Zusätzlich dann: EU-Abdeckung bei Finnhub (6 Titel), TXRH-Divergenz, ±1-Tag-Versatz Finnhub/yfinance, Estimate-Streuung.
2. **yfinance-Upgrade im Prod-Pfad: nein, nicht jetzt.** 19 Prod-Module importieren yfinance, darunter Position-Manager, Portfolio-Manager, Market-Data und der Preis-Fallback — ein Sprung 0.2.58 → 1.x berührt den Exit-Pfad. PEAD ist aus, es gibt keinen Handlungsdruck. Wenn yfinance Termine liefern soll, dann so wie in der Eval: isoliertes venv, Subprozess-Helper, Prod-Import unangetastet. Die Eval zeigt außerdem, dass 1.6.0 den `earnings_dates`-Endpoint repariert (26/26 Titel, 1 transienter Fehler in 728 Aufrufen), aber der Kalender in 8 % der Ticker-Tage Altes liefert. Ein Upgrade allein löst das Frische-Problem nicht.
3. **Eval-Timer: laufen lassen** bis zur November-Auswertung. Kosten: Finnhub 2 Calls/Tag, yfinance ~52 Ticker-Calls/Tag, AV 25/Tag (volles Free-Budget), FMP 6 nutzlose Calls/Tag. Danach abschalten.
4. **FMP:** entweder auf dev den Endpoint auf `/stable/earnings-calendar` umstellen, damit die November-Auswertung vier Quellen hat, oder FMP bewusst aus der Eval streichen. Weiterlaufen lassen wie jetzt liefert nichts. *Nachtrag 2026-09-23: FMP aus dem Capture gestrichen (Provider-Liste, Fetcher, Normalizer; historische run_log-Zeilen bleiben). Wirksam auf dem VPS erst nach Pull, der nächste Timer-Lauf nimmt es automatisch.*
5. **Alpha Vantage:** der EPS-Modus (24 Calls/Tag, chronischer Budget-Shortfall) trägt zum Primärkriterium nichts bei; für die November-Auswertung reicht der eine Kalender-Call/Tag. Abschalten des AV-EPS-Pfads wäre eine Code-Änderung auf dev, kein Muss.
6. Wenn nach November eine Quelle gewählt wird, gehört die Wahl in den PEAD-Pfad (`strategies/pead_strategy.py`, `_evaluate_from_yfinance`) und nicht in ein Pip-Upgrade. Das ist eine Code-Änderung mit eigenem Review.

## 7. Offene Punkte / nicht konstruiert

- Ob Finnhub Free-Tier EU-Titel (Xetra/Euronext) im Kalender führt: keine Evidenz, kein EU-Report im Fenster.
- Ob die ±1-Tag-Differenz Finnhub/yfinance bei CACI, MEDP, TSLA, VRT ein Zeitzonen-Artefakt ist: nicht belegt.
- Ursache des FMP-403 (Legacy-Pfad vs. Key): nicht verifiziert, nur konsistent mit dem Fehlerbild seit Run 1.
- Warum der Preis-Fallback `ALPHA_VANTAGE_KEY` erwartet, die `.env` aber nur `ALPHA_VANTAGE_API_KEY` führt (Level 1 des Fallbacks damit inaktiv): nicht untersucht, nur festgestellt.

## Anhang A — Reproduzierbare Queries (read-only)

```sql
-- Betrieb je Provider/Endpoint
SELECT provider, endpoint, COUNT(*), MIN(run_ts), MAX(run_ts),
       SUM(CASE WHEN error_text LIKE '%Error%' THEN 1 ELSE 0 END) AS err
FROM run_log GROUP BY 1,2;

-- Stille Leer-Antworten (AV Kalender)
SELECT run_ts, http_status, rows_returned FROM run_log
WHERE provider='alphavantage' AND endpoint LIKE '%CALENDAR%' AND rows_returned<=1;

-- Primärkriterium: Titel im Fenster je Provider am letzten Capture-Tag
SELECT provider, COUNT(DISTINCT ticker) FROM cal_capture
WHERE substr(capture_ts,1,10)=(SELECT MAX(substr(capture_ts,1,10)) FROM cal_capture)
  AND days_ahead BETWEEN -2 AND 30 GROUP BY 1;

-- Frische: abgelaufene Termine als "nächster Termin"
SELECT provider, ticker, report_date, COUNT(*), MIN(days_ahead), MAX(days_ahead)
FROM cal_capture WHERE days_ahead < -2 GROUP BY 1,2,3;

-- Termin-Timeline je Ticker/Provider (Sprünge sichtbar)
SELECT ticker, provider, report_date, MIN(substr(capture_ts,1,10)), MAX(substr(capture_ts,1,10))
FROM cal_capture GROUP BY 1,2,3 ORDER BY 1,2,4;

-- Realisierte Reports mit EPS
SELECT substr(capture_ts,1,10), provider, ticker, report_date, estimate_eps, actual_eps,
       surprise_pct, available_same_day FROM eps_capture ORDER BY 1,2;
```

Für die November-Auswertung zusätzlich: je realisiertem Report (aus `eps_capture` bzw. dem Finnhub-Kalender mit `days_ahead<=0`) prüfen, ob jede Quelle den Termin an mindestens einem Capture-Tag mit `days_ahead` in [−2, 30] und identischem `report_date` geführt hat. Das ist das Primärkriterium in einer Zahl pro Quelle.

## Anhang B — Was in der Eval-DB nicht steht

- Keine API-Keys, keine vollständigen URLs (`endpoint` ist Pfad ohne Query). Die journal-Zeilen enthalten die FMP-URL mit Query-String; für diesen Bericht wurden sie beim Lesen redigiert und nicht gespeichert.
- Kein Bezug zur Live-DB `news_trading.db` außer der lesenden `pead_signal_log`-Abfrage in §5.
