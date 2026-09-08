# Wiedereinstiegssperre nach Stop + feiertagsfähiger Stale-Guard, 2026-09-08

Zwei benannte Defekte, ein Deploy-Thema (Ausführungs- und Monitoring-Hygiene).
Kein Signalpfad, keine Strategie geändert. Arbeit auf `dev`, nicht deployed.

Datenstand: Prod-DB `/home/trading/trading-data/news_trading.db`, read-only
Snapshot (`sqlite3` Backup-API aus einer `mode=ro`-Verbindung) vom
2026-09-08 16:54 UTC; `trade_history` 2026-05-04 … 2026-09-04.
Skript: `scripts/reentry_after_stop_audit.py` (read-only, stdlib +
`data.market_calendar`). Shadow-Zuordnung über
`scripts/shadow_strategy_book.py --csv` gegen denselben Snapshot; dessen
Kennzahlen (A +0,11 %, B −0,91 %, C −0,54 %, Auswahl −1,02 %) reproduzieren
sich byte-identisch zum Report vom 2026-09-07.

## 0. Kurzantwort

* **Wiedereinstieg nach Stop kommt oft vor.** 171 BUYs, davon 78 in einen
  Ticker, dessen vorheriger Round-Trip ein Stop-Loss-Exit war (46 %). Das
  Universum hat 20 handelbare Ticker; nur 20 BUYs waren Ersteinstiege.
* **Die Verteilung hat eine Kante.** 48 der 78 (62 %) liegen innerhalb von
  4 Handelssessions nach dem Stop, Sessions 5 und 6 sind praktisch leer
  (1 / 0), danach ein diffuser Schwanz über Wochen. **Sperrdauer: 5
  Sessions (eine Handelswoche)** — sie deckt den dichten Block ab und endet
  an der empirischen Lücke. Nicht auf P&L gewählt; die P&L bestätigt sie
  nur (kurzfristige Wiedereinstiege −1,08 % je Trade, spätere +1,15 %).
* **Das bestehende 1-Session-Gate (Q-016, seit 2026-07-01) hat ein
  Feiertagsloch.** Sein Session-Zähler kennt nur Wochentage; Fr 2026-07-03
  (Independence Day observed) zählte als Session, TRGP und VRT wurden am
  Montag wieder gekauft. VRT: −2 902 USD (−10,6 %). Mit dem Kalender ist
  das Gap 1, nicht 2.
* **Der Auswahleffekt des Shadow-Buchs kommt NICHT von diesen
  Wiedereinstiegen.** Im Shadow-Modell haben die gefüllten Signale nach
  Stop eine Erwartung von −0,62 %, alle anderen gefüllten Signale −1,09 %,
  Ersteinstiege −1,93 %. Ohne die Nach-Stop-Wiedereinstiege wäre der
  Auswahleffekt größer (−1,20 % statt −0,98 %). Die Sperre trifft also
  realisierte Verluste (−14,1 k USD auf 49 Trades), sie repariert nicht
  die Signalauswahl. Das ist ein Befund, keine Rechtfertigung für mehr
  Gates.
* **Stale-Guard:** `PositionManager._is_market_hours()` (der Guard, der am
  Labor Day gefeuert hat) und `PriceMonitor` nutzen jetzt
  `data.market_calendar`. Zusätzlich: eine Meldung beim Auftreten, eine
  Erinnerung je 60 min, eine Erholungszeile. Am Labor Day wären das 0 statt
  mehrerer hundert Nachrichten gewesen.

## 1. Bestand: Wiedereinstieg nach Stop

### 1.1 Exit-Klassen aller 165 Round-Trips

Exit-Grund aus der `PositionManager`-Zeile in `signal_events` am SELL
(`stop_loss_triggered at $L` / `take_profit_triggered at $L`); Stop-Level
≤ Einstiegs-Stop × 1,003 → Entry-Stop, darüber → Trailing-Stop. Alle 165
Round-Trips haben eine solche Zeile.

| Exit | N | realisiert | Fill-zu-Fill | Q-016-Klassifikator sagt STOP |
|---|---|---|---|---|
| Entry-Stop (Stop-Loss) | 84 | −64 857 USD | Ø −2,79 %, Trefferquote 2 % | 84/84 |
| Trailing-Stop | 57 | +23 075 USD | Ø +1,50 %, Trefferquote 95 % | 0/57 |
| Take-Profit | 24 | +17 285 USD | Ø +2,81 %, Trefferquote 100 % | 0/24 |

Der Preis-vs-Stop-Klassifikator des Gates (`executed_price ≤ stop_loss ×
1,003`) trifft exakt die Entry-Stops und keinen Trailing-Exit. Er bleibt.

### 1.2 Einstiege nach dem Exit des vorherigen Round-Trips im selben Ticker

| vorheriger Exit | N Wiedereinstiege | geschlossen | realisiert | Fill-zu-Fill | Gap Sessions min / Median / p90 / max | Gap Stunden Median |
|---|---|---|---|---|---|---|
| Entry-Stop | 78 | 76 | −5 262 USD | Ø −0,25 %, Trefferquote 50 % | 0 / 3 / 23 / 64 | 118 h |
| Trailing-Stop | 50 | 47 | −13 477 USD | Ø −1,00 %, Trefferquote 38 % | 0 / 1 / 10 / 33 | 24 h |
| Take-Profit | 23 | 22 | +1 761 USD | Ø +0,35 %, Trefferquote 77 % | 0 / 2 / 9 / 44 | 96 h |

### 1.3 Gap-Verteilung nach Entry-Stop (Sessions, feiertagsbereinigt)

| Gap | N | realisiert | Ø Return | vor Gate (< 07-01) | mit 1-Session-Gate |
|---|---|---|---|---|---|
| 0 | 8 | −1 033 USD | −0,46 % | 8 | 0 |
| 1 | 17 | −1 879 USD | −0,39 % | 15 | 2 (beide über den 3. Juli) |
| 2 | 10 | −3 719 USD | −1,35 % | 5 | 5 |
| 3 | 8 | −5 670 USD | −2,68 % | 4 | 4 |
| 4 | 5 | −1 826 USD | −1,67 % | 1 | 4 |
| 5 | 1 | +28 USD | +0,11 % | 0 | 1 |
| 6 | 0 | | | | |
| 7 | 5 | +1 238 USD | +0,89 % | 3 | 2 |
| 8–10 | 9 | +5 244 USD | +2,3 % | 5 | 4 |
| 11–64 | 15 | +2 010 USD | Ø +0,4 % | 5 | 10 |

Kumuliert, was eine Sperre von N Sessions getroffen hätte:

| N | geblockt | Anteil | realisiert (geblockt) | Ø geblockt | Ø Rest nach Stop | davon in der Gate-Ära | deren P&L |
|---|---|---|---|---|---|---|---|
| 1 (bisher) | 25 | 32 % | −2 912 USD | −0,41 % | −0,18 % | 2 | −1 823 USD |
| 2 | 35 | 45 % | −6 631 USD | −0,68 % | +0,11 % | 7 | −4 815 USD |
| 3 | 43 | 55 % | −12 302 USD | −1,05 % | +0,78 % | 11 | −7 022 USD |
| 4 | 48 | 62 % | −14 128 USD | −1,10 % | +1,12 % | 15 | −9 122 USD |
| **5** | **49** | **63 %** | **−14 100 USD** | **−1,08 %** | **+1,15 %** | **16** | **−9 094 USD** |
| 7 | 54 | 69 % | −12 862 USD | −0,89 % | +1,21 % | 18 | −8 517 USD |
| 10 | 63 | 81 % | −7 618 USD | −0,46 % | +0,59 % | 22 | −5 903 USD |

Ticker-Konzentration: CASY 9 Nach-Stop-Wiedereinstiege (−3 921 USD), TRGP 7
(+1 563), MSFT 7 (−2 639), TOL 6 (+2 006), VRT 6 (−2 304), XOM 5 (+1 785).
Vollständige Liste aller 78 im Skript-Output (§6).

### 1.4 Auswahleffekt: erklären die Wiedereinstiege ihn?

Shadow-Erwartung (Linie B des Shadow-Buchs) je Teilmenge der gefüllten
Combined-Signale:

| Teilmenge | N | Shadow | real |
|---|---|---|---|
| alle gefüllten Signale mit Shadow-Return (= B) | 162 | −0,87 % | −0,50 % |
| Wiedereinstieg nach Entry-Stop | 76 | −0,62 % | −0,25 % |
| … davon Gap ≤ 1 | 25 | −0,58 % | −0,41 % |
| … davon Gap ≤ 3 | 43 | −1,25 % | −1,05 % |
| Wiedereinstieg nach Trailing-Stop | 47 | −1,10 % | −1,00 % |
| Wiedereinstieg nach TP | 22 | −0,44 % | +0,35 % |
| Ersteinstieg eines Tickers | 17 | −1,93 % | −1,29 % |
| alles außer Nach-Stop-Wiedereinstieg | 86 | −1,09 % | −0,71 % |

B − A = −0,98 % (auf den 162 zuordenbaren). Ohne Nach-Stop-Wiedereinstiege:
−1,20 %. Die Wiedereinstiege tragen −23 % des Effekts, das heißt, sie
verdünnen ihn. Nur die kurzen (Gap ≤ 3) sind im Shadow schlechter als der
Durchschnitt und tragen 14 %.

**Folgerung:** Die Signale, die das Live-System füllt, sind im Shadow
durchweg schlecht, unabhängig davon, ob der Ticker gerade gestoppt wurde.
Das Muster „Häufung auf wenige Ticker mit Wiedereinstieg nach Stop" aus dem
Shadow-Buch ist real (78 Fälle), aber es ist nicht der Mechanismus hinter
−1,02 %. Woher der Effekt kommt, beantwortet dieser Bestand nicht; die
naheliegende nächste Frage ist, warum Ersteinstiege (−1,93 %) und
Wiedereinstiege nach Trailing-Exits (−1,10 %) so schlecht sind — beides
sind Fälle, in denen ein Slot frei wird und das nächste verfügbare Signal
gefüllt wird.

## 2. Sperrdauer: Herleitung

Regel: die Sperre soll den dichten Block der beobachteten Wiedereinstiege
abdecken und an der empirischen Lücke enden, nicht an einem P&L-Optimum.

* Sessions 0–4 enthalten 48 von 78 (62 %), jede Session mit 5–17 Fällen.
* Session 5: 1 Fall, Session 6: 0. Ab Session 7 beginnt ein zweiter,
  flacher Modus (≤ 5 je Session), der sich bis Session 64 zieht und zu
  über 80 % im Plus liegt.
* **5 Sessions** = eine Handelswoche = die kleinste Dauer, die den Block
  vollständig einschließt und die Lücke als Ende nutzt. 4 wäre die letzte
  besetzte Session des Blocks, 7 der Beginn des Schwanzes.

Die P&L-Spalte ist eine Plausibilitätsprüfung, kein Selektor: der Ø-Return
der geblockten Trades ist bei jedem N ≤ 5 negativ (−0,4 … −1,1 %), der der
nicht geblockten Nach-Stop-Wiedereinstiege ab N ≥ 2 positiv. Ein längeres
N (7, 10) würde zunehmend Trades mit positivem Ausgang sperren.

Gerechnet wird in US-Handelssessions (`data.market_calendar`): Wochenenden
und ganztägige NYSE-Feiertage zählen nicht. Gap 0 = dieselbe Session,
Gap 1 = nächster Handelstag; gesperrt ist Gap ≤ 5, frei ab Gap 6. Stop am
Fr 2026-09-04 → frei ab Di 2026-09-15 (09-07 Labor Day zählt nicht).

## 3. Wo die Sperre greift, wie sie protokolliert wird

**Ort:** `PortfolioManager.can_add_position()`, Gate 1b, direkt nach dem
„Already holding"-Gate. Alle vier Einstiegspfade in
`orchestrator/coordinator.py` laufen bei einem BUY durch diese Methode:

| Pfad | Zeile (dev) |
|---|---|
| PEAD-Pfad (`_run_pead_ticker`, dormant) | ~625 |
| `run_combined` (Hauptpfad, XETRA/US_PRE/MIDDAY/EOD) | ~1770 |
| `run_combined_us_open` | ~2208 |
| Cached-US_OPEN-Executor (Forward-Signale) | ~2813 |

Kein Pfad ruft `track_trade("BUY")` an `can_add_position` vorbei. Exits
(PositionManager SELL, Stop, TP, Trailing) rufen die Methode nie auf und
sind von der Sperre unberührt.

**Klassifikation:** letzter FIFO-Round-Trip des Tickers aus
`trade_history`; Stop-Loss-Exit wenn `executed_price` (Fallback `price`)
≤ `stop_loss` des öffnenden BUY × 1,003. Trailing-Exits (PositionManager-
Zeile `stop_loss_triggered` mit Level über dem Einstiegs-Stop) sperren nur
mit `REENTRY_LOCK_INCLUDE_TRAILING = True` (Default False, siehe §5).

**Fail-open** bleibt: kein Round-Trip, kein Stop am BUY, Lese- oder
Parse-Fehler → kein Block.

**Protokoll** je abgelehntem BUY, zwei Zeilen:

1. `portfolio_violations` (`violation_type='cooldown_stop'`, wie bisher).
2. `signal_events` — neu:

| Spalte | Wert |
|---|---|
| `strategy` | `PortfolioManager` |
| `signal` | `HOLD` |
| `signal_path` | `REENTRY_LOCK` |
| `session` | die Scheduler-Session, aus der der BUY kam |
| `price_at_signal` | vorgeschlagener Einstiegspreis |
| `trade_executed` | 0 |
| `debate_outcome` | `BUY_REJECTED` |
| `bull_case` | `BUY requested by <strategy>` |
| `bear_case` | `reentry_lock: stop_loss exit at <ISO>; sessions_elapsed=1; sessions_remaining=4; lock_sessions=5; eligible_from=2026-09-15` |

Auswertung später: `SELECT … FROM signal_events WHERE signal_path =
'REENTRY_LOCK'`, dann per Ticker/Datum gegen `daily_ohlc` legen, wie das
Shadow-Buch es tut. Ohne diese Zeilen wäre der Effekt nicht messbar; die
`portfolio_violations`-Zeilen des alten Gates (21 Stück, 07-03 … 08-18)
tragen weder Restdauer noch Session.

**Konfiguration:** `config/settings.py` — `REENTRY_LOCK_SESSIONS = 5`,
`REENTRY_LOCK_INCLUDE_TRAILING = False`. Bewusst keine Env-Overrides:
Handelsregel, kein Deployment-Knopf.

## 4. Handelskalender und Stale-Guard

### 4.1 Wer den Kalender jetzt nutzt

| Stelle | vorher | jetzt |
|---|---|---|
| `monitoring/position_manager.py::_is_market_hours` (Alarm-Gate des Stale-Guards) | Wochentag + 09:30–16:00 ET | zusätzlich `is_us_trading_day` |
| `monitoring/price_monitor.py::_is_market_hours`, `_market_status`, `_seconds_until_open` | Wochentag + Fenster; Weckzeit übersprang nur Wochenenden | US-Fenster nur an Handelstagen; Status „US holiday"; Weckzeit = nächster Handelstag 09:30 ET (`next_us_rth_open`) |
| `execution/portfolio_manager.py` (Sperre) | `events_feed._trading_days_between` (Wochentage) | `us_sessions_between` (Handelssessions) |
| `scripts/ohlc_ingest` Freshness-Gate, `scripts/watchdog.py` OHLC-Check | Kalender seit 2026-09-01 | unverändert |

Neue Helfer in `data/market_calendar.py`: `us_sessions_between(a, b)`,
`next_us_trading_day(d)`, `is_us_rth(now)`, `next_us_rth_open(now)`,
Konstanten `NY_TZ`, `US_RTH_OPEN`, `US_RTH_CLOSE`.

`PriceMonitor` läuft auf dem VPS nicht als Dienst (nur `nts-trading` mit
dem `PositionManager`); er ist mitgezogen, weil er dieselbe Prämisse hatte.
Sein XETRA-Fenster bleibt wochentagsbasiert: es gibt keinen EU-Kalender im
Repo.

### 4.2 Alarmkadenz des Stale-Guards

Bisher: jede stale Abfrage (60 s) während „RTH" eine Telegram-Nachricht.
Jetzt (`_STALE_REALERT_MINUTES = 60`):

* erste stale Abfrage in RTH → **eine** Meldung `⚠️ STALE FEED: …`;
* solange stale → alle 60 min eine Erinnerung `⚠️ STALE FEED (still): …`
  mit der aktuellen Streak-Länge;
* erste frische Bar nach einer gemeldeten Störung → **eine** Zeile
  `✅ FEED RECOVERED: …`;
* jede stale Abfrage schreibt weiterhin die Journal-WARNING; die
  Evaluierbarkeits-Regel (kein Exit auf stale Preis) ist unverändert.

Zustand lebt im Prozess (`_stale_alerted_at`); nach einem Daemon-Neustart
gibt es bei fortbestehender Störung eine neue Erstmeldung, das ist gewollt.

### 4.3 Verhalten am nächsten Feiertag

Nächste ganztägige NYSE-Feiertage: Do 2026-11-26 (Thanksgiving),
Fr 2026-12-25. Erwartetes Verhalten des `PositionManager` an so einem Tag:

1. Der Daemon läuft, der PM pollt weiter (`_run_loop` prüft
   `_is_market_hours()` — das ist jetzt False → keine Positionsprüfung, wie
   am Wochenende).
2. Falls doch ein Preis geholt wird (z. B. `--check-now`, oder ein Aufruf
   außerhalb der Schleife): die Bar von Mittwoch ist > 5 min alt → Preis
   unevaluierbar, Journal-WARNING, **keine Telegram-Nachricht**, weil
   `_is_market_hours()` False ist.
3. Am Freitag 09:30 ET (Black Friday, Handelstag mit früherem Schluss
   13:00 ET) nimmt der Guard normal wieder auf. Halbtage sind als volle
   Sessions modelliert: zwischen 13:00 und 16:00 ET an einem Halbtag würde
   ein stale Feed einmal gemeldet und nach 60 min erinnert — bewusst
   hingenommen, siehe Docstring des Kalenders.

Getestet mit eingefrorener Uhr: Labor Day 10:00 ET → keine Eskalation,
Streak zählt; Dienstag danach 10:00 ET → Handelszeit.

## 5. Befunde ohne Änderung

* **Der Scheduler läuft an Feiertagen alle acht Sessions** (`scheduler/
  daily_runner.py`: `next_run_time`, `current_session`, `_run_for_time`
  prüfen nur `weekday() < 5`). Am 2026-09-07 haben US_PRE (44) und EOD (45)
  `signal_events`-Zeilen geschrieben, US_OPEN hat mangels Live-Preis nicht
  gehandelt. Das ist der Signalpfad; dazu gehört, dass EOD-Forward-Signale
  vom Feiertag am nächsten US_OPEN vom Cached-Executor konsumiert werden.
  Zudem sind XETRA-Sessions an US-Feiertagen regulär offen, und
  `scripts/watchdog.py::_due_sessions` erwartet die Sessions spiegelbildlich
  (er würde bei einem Feiertags-Skip „session missing" melden, wenn er
  nicht mitgezogen wird). Das ist eine gekoppelte Änderung an Scheduler +
  Wächter, die Sessions auf US-Feiertage aufteilt — außerhalb dieses
  Auftrags. Empfehlung: eigener Auftrag „US-Sessions an NYSE-Feiertagen
  überspringen, XETRA-Sessions weiterlaufen lassen, Wächter mitziehen".
* `data/events_feed.py::_trading_days_between` (Wochentage) speist
  `get_days_to_earnings` → Strategie-Input; unverändert, die Sperre nutzt
  ihn nicht mehr.
* `api/main.py::_add_business_days` spiegelt bewusst die Frontend-Logik
  (dokumentiert); unverändert.
* **Wiedereinstiege nach Trailing-Stop-Exits** sind im Bestand die
  schlechtere Gruppe (N = 47, −1,00 % je Trade, Trefferquote 38 %,
  Median-Gap 1 Session, realisiert −13 477 USD). Ein Trailing-Exit ist
  aber ein Gewinn-Exit nach > 2 % Anstieg, kein Stop-Loss; der Auftrag
  nennt Stop-Loss-Exits. Der Schalter `REENTRY_LOCK_INCLUDE_TRAILING`
  existiert, ist aus, und die Entscheidung liegt bei Lenni.
* Die Q-016-Tests haben die Ära-Kaskaden (MSFT/TSLA/CASY, Gap 0–1)
  weiterhin als Replay; sie bleiben mit N = 5 geblockt.

## 6. Reproduktion

```bash
# auf dem VPS, read-only, keine Schreibzugriffe
python3 scripts/reentry_after_stop_audit.py
# mit Shadow-Zuordnung (Linie A aus dem Shadow-Report)
python3 scripts/shadow_strategy_book.py --csv /tmp/shadow > /tmp/shadow/report.md
python3 scripts/reentry_after_stop_audit.py \
    --shadow-csv /tmp/shadow/real_round_trips.csv --line-a 0.0011
# Tests
python3 -m pytest tests/test_portfolio_cooldown.py tests/test_position_manager.py \
    tests/test_market_calendar.py tests/test_price_monitor_hours.py -q
```
