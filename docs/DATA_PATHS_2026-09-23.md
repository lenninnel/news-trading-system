# Datenpfade und ehrliches Logging — 2026-09-23

Jede Komponente nutzt die richtigen Daten zum richtigen Zeitpunkt und sagt
nachvollziehbar, welche. Ein Deploy-Thema, keine Strategieänderung: Schwellen,
Strategien und die D3-Aufteilung (Indikatoren aus dem `daily_ohlc`-Store,
Live-Preis von IBKR/Alpaca für Sizing und Preis-Guard) bleiben unverändert.

Stand: dev, aufgesetzt auf main `70a1ade`.

---

## 0. Das Preismodell in einem Satz

| Verwendung | Quelle | Wann | Sichtbar in |
|---|---|---|---|
| Indikatoren (RSI, SMA, ATR, Strategie-Votes) | `daily_ohlc` (Alpaca SIP, raw close), letzter **abgeschlossener** Tagesbar | T-1 in Intraday-Sessions, T in EOD (nach dem Ingest) | `signal_events.indicator_bar_date / _close / _source`, `price_at_signal` (= Bar-Close) |
| Sizing, Preis-Guard, Forward-Drift-Check | `MarketData.fetch()` → Alpaca latest trade, sonst Quote-Mid | Sekunden vor der Entscheidung | `signal_events.live_price / live_price_source`, `risk_calculations.current_price` |
| Outcomes 3d/5d/10d | `daily_ohlc` | EOD 22:45 UTC + Cron 23:00 UTC | `signal_events.price_Nd / outcome_Nd_pct / outcome_status / outcome_note` |

Ein externer Check hat diesen Aufbau als Fehler gelesen, weil aus den Zeilen
nicht hervorging, welcher Preis wofür verwendet wurde. Das ist jetzt aus jeder
Zeile ablesbar.

---

## 1. Outcome-Tracker — war er tot?

**Nein.** Read-only-Beleg auf der Prod-DB (2026-09-23, `mode=ro`):

| Kennzahl | Wert |
|---|---|
| Zeilen in `signal_events` | 20 410 (2026-03-28 … 2026-09-23) |
| davon `price_3d` gefüllt | 20 046 |
| davon `price_5d` gefüllt | 19 923 |
| letzter Fill (`price_5d`) | 2026-09-22 22:20 UTC (gestrige EOD) |
| Combined-Zeilen auf Store-Tickern ≥ 5 Tage alt mit `outcome_5d_pct` | 3 265 / 3 265 |
| Journal `EOD outcome backfill` in den letzten 30 Tagen | 20 Läufe |

Die Fills stimmen mit dem Store überein (z. B. TSLA 2026-09-17 → `price_5d`
378.90 = Store-Close 2026-09-22), nur über einen zweiten, ungeprüften
Preispfad (Alpaca-Bars, sonst yfinance).

Was der externe Check als „durchgehend pending“ gesehen hat, sind die
Zeilen, die **nie** auflösbar waren und es still blieben:

| Population | Zeilen | Grund |
|---|---|---|
| `PostSessionReviewer` (`ticker='SESSION'`) | 174 | Sentinel, kein Instrument; vom Cron mit `price_Nd = 0.0` „versiegelt“ |
| EU-Ticker (VNA.DE, HOT.DE, COFA.PA, VIE.PA, FNTN.DE, LEG.DE), 30.04./01.05. | 48 | nicht im Store, yfinance-Lücke am 1. Mai |
| Nicht-Store-Ticker mit offenem Horizont (Scanner-Tier-2, alte PEAD-Namen) | 199 | Ticker ohne `daily_ohlc`-Zeilen |

Der MCP-Server labelt `outcome_3d_pct IS NULL` als „pending“ — das trifft
auch alle Zeilen jünger als drei Tage und die o. g. Population. Die lokale
Repo-DB hat zudem nur 10 Zeilen vom 2026-03-29; ein Check darauf sieht
praktisch nichts.

### Neu (`analytics/outcome_tracker.py`)

- **Preisbasis nur noch `daily_ohlc`.** Kein Alpaca-/yfinance-Fetch pro Zeile mehr.
- **Horizont-Semantik unverändert:** `price_Nd` = Store-Close des letzten
  US-Handelstags ≤ `signal_date + N Kalendertage` (`data/market_calendar.py`).
  Gefüllt wird erst, wenn der Store den Bar hat; vorher bleibt die Zeile
  pending (kein Ersatzwert).
- **`outcome_correct`** wird wie bisher bei jedem Horizont-Durchlauf neu
  geschrieben (eine vollständig aufgelöste Zeile trägt also das Vorzeichen der
  10d-Bewegung). Bewusst identisch gelassen, weil `RiskAgent.calculate_kelly_position`
  `outcome_correct` zusammen mit `outcome_5d_pct` liest. → Befund, nicht geändert.
- **Neue Spalten** (additiv, idempotent): `outcome_status` (NULL = pending,
  `filled`, `unevaluable`), `outcome_note` (`sentinel`, `no_store_bars`,
  `no_entry_price`, `entry_backfilled_from_store`), `outcome_source`
  (`daily_ohlc`), `outcome_updated_at`.
- **Nicht auswertbar ist sichtbar:** Sentinel-Zeilen und Ticker ohne
  Store-Daten werden einmalig markiert statt ewig pending zu bleiben.
- **Fehlender Einstiegspreis** (PreMarketScanner) wird bei Store-Tickern aus
  dem Store nachgetragen: Close des letzten abgeschlossenen Bars zum
  Signalzeitpunkt (T-1 vor 22:00 UTC, T danach), Notiz
  `entry_backfilled_from_store`. Bisher machte das der Cron mit
  **adjustierten** Yahoo-Closes.
- **Journal:** jeder Lauf schreibt eine INFO-Summary (auch bei 0 Zeilen); ein
  Crash im EOD-Aufruf ist ERROR mit Traceback plus Telegram-Zeile (vorher
  WARNING ohne Traceback). Aufruf mit `--dry-run` möglich.
- `scripts/update_outcomes.py` (Cron 23:00 UTC, bleibt in der Crontab) ist
  jetzt ein dünner Wrapper um denselben Tracker; die drei yfinance-Fixups sind
  weg. Exit 1 bei Fehler.

### Trockenlauf auf einer Read-only-Kopie der Prod-DB

```
outcome tracker: rows_considered=756 filled=0 {'price_3d': 0, 'price_5d': 0, 'price_10d': 0}
  entry_backfilled=0 unevaluable=421 (no_store_bars=247, sentinel=174) pending=509
  store_tickers=11 source=daily_ohlc
```

Zweiter Lauf: `rows_considered=509 … unevaluable=0` → idempotent. Ergebnis
beim ersten Prod-Lauf: **0 Preiszellen nachzufüllen** (alles Erreichbare war
schon gefüllt), **421 Zeilen werden als `unevaluable` markiert**, 509 Zeilen
(3–10 Tage alt, Store-Ticker) warten regulär auf ihren 10d-Bar. Die 48
EU-Zeilen und die 174 SESSION-Zeilen verschwinden damit aus jeder
„pending“-Zählung.

**Konsequenz, bewusst:** Ticker außerhalb der US-20 (Scanner-Tier-2) bekommen
ab jetzt keine Outcomes mehr — sie haben keine Store-Daten und werden
`unevaluable:no_store_bars`. Kelly liest per Ticker und fällt sonst auf alle
Zeilen zurück; für die US-20 ändert sich nichts.

---

## 2. EOD rechnet auf dem Bar des Tages

| | vorher | nachher |
|---|---|---|
| `nts-ohlc-ingest.timer` | 22:30 UTC (~15 s Laufzeit, Alpaca) | unverändert |
| EOD-Session (`config/sessions.py`) | 22:15 UTC → bewertete **T-1** | **22:45 UTC** → bewertet **T** |
| Outcome-Tracker (in EOD) | 22:20, vor dem Ingest | 22:47, nach dem Ingest |
| Scheduler-Fenster `_WINDOW_END` | 22:30 | 23:00 |
| Wächter `_due_sessions` | liest `config.sessions.SCHEDULE` → EOD fällig ab 23:05 (22:45 + 20 min Grace); Fallback-Liste mitgezogen | kein Fehlalarm |
| Wächter `ohlc`-Check | erwartet Store ab 23:00 | unverändert |

**Sommerzeit:** US-Close 20:00 UTC (DST) / 21:00 UTC (ab Anfang November).
22:45 liegt in beiden Fällen nach dem Close, nach dem Same-Day-Cutoff des
Ingests (`SAME_DAY_CUTOFF_UTC = 22`, Fenster schließt „heute“ ein) und nach
dem Ingest-Timer. Der Timer bleibt bei 22:30 — kein Unit-File ändert sich.

**Wofür EOD-Signale verwendet werden:** EOD ist eine `signal`-Session. Sie
loggt Combined-/Strategie-Zeilen (September: 720 Zeilen) und schreibt
gerichtete Signale als `forward_signals` mit `target_session='US_OPEN'`
(September: 24 aus EOD, davon 13 bestätigt, 3 ausgeführt). Der US_OPEN-Executor
am nächsten Morgen validiert sie gegen den Live-Preis (Drift > 2 % invalidiert)
und handelt. EOD-BUYs werden nicht direkt ausgeführt (IBKR „held for next
session“). **Wirkung der Verschiebung:** die Forward-Signale für den nächsten
US_OPEN und die EOD-Zeilen selbst ruhen auf dem heute abgeschlossenen Bar
statt auf dem von gestern; `price_at_signal` der Forward-Zeile (Live-Preis,
sonst Bar-Close) und der Bar-Close fallen jetzt zusammen. US_PRE (13:15)
schreibt weiterhin eigene Forward-Signale auf demselben Bar T; der
Cached-Signal-Pfad des Executors (`get_cached_signal`, nur US_PRE/XETRA_PRE,
≤ 90 min) ist nicht betroffen. `expire_stale_forward_signals(24h)`: EOD-Forward
um 22:45 ist beim US_OPEN 14:30 knapp 16 h alt — innerhalb der Frist.

---

## 3. Jede Signalzeile sagt, auf welchem Bar sie beruht

Neue Spalten in `signal_events` (additiv, idempotent via `ALTER TABLE`,
Altzeilen NULL):

| Spalte | Inhalt |
|---|---|
| `indicator_bar_date` | Datum des letzten Bars, der in die Indikatoren einging |
| `indicator_bar_close` | dessen raw Close (identisch mit `price_at_signal`, das damit endlich erklärt ist) |
| `indicator_bar_source` | `daily_ohlc` \| `yfinance` (Fallback, laut geloggt) \| `eodhd` \| `binance` |
| `live_price` | Preis aus `MarketData.fetch()` zum Entscheidungszeitpunkt; NULL wenn degraded (dann kein Trade) |
| `live_price_source` | `alpaca` \| NULL |

Beispielzeile (US_PRE, Combined):

```
timestamp              2026-09-23T13:55:29+00:00
session                US_PRE
ticker                 AAPL
strategy               Combined
signal                 WEAK BUY          confidence 0.41
price_at_signal        339.75            ← Close des Indikator-Bars
indicator_bar_date     2026-09-22        ← T-1 in einer Intraday-Session
indicator_bar_close    339.75
indicator_bar_source   daily_ohlc
live_price             340.12            ← Alpaca latest trade, Sizing/Guard
live_price_source      alpaca
outcome_status         NULL              ← pending, Horizont noch nicht erreicht
```

Woher die Werte kommen: `TechnicalAgent._fetch_history` markiert den
Daily-Frame mit seiner Quelle (`DataFrame.attrs["bar_source"]`); `run()` legt
`bar_date/bar_close/bar_source` neben die Indikatoren. Der Coordinator
schreibt sie in die Combined-Zeile (`_log_signal_event`) und reicht denselben
Frame plus das `market`-Dict in die Strategie-Zeilen (`_gather_strategy_votes`
→ `_log_strategy_result`, NewsCatalyst inklusive). PEAD-, Scanner- und
Sentinel-Zeilen tragen keine Bars → NULL.

---

## 4. Keine Signale auf veralteten Bars ohne Alarm

`scheduler/bar_freshness.py`, Hook in `DailyScheduler._execute_run` **vor**
`run_batch`, nach der Ticker-Auflösung.

- **Gilt für** `signal`- und `execution`-Sessions (US_PRE, PEAD_OPEN, US_OPEN,
  XETRA_OPEN, EOD). Nicht für `monitor` (MIDDAY), `scanner`, `pre_signal`
  (XETRA_PRE: nur News/Sentiment).
- **Erwarteter Bar:** Intraday `last_us_trading_day(heute − 1)`, EOD
  `last_us_trading_day(heute)` — derselbe Kalender wie Ingest-Gate und Wächter.
- **Geprüft** werden nur Ticker mit Store-Zeilen (`get_daily_ohlc_max_dates`).
  Ticker ohne Store (Scanner-Tier-2) sind `unknown`, laufen weiter und landen
  ohnehin im laut geloggten yfinance-Fallback des TechnicalAgent.
- **Entscheidung:**
  - alle Store-Ticker aktuell → `ok`, eine INFO-Zeile;
  - einige veraltet → `skip`: diese Ticker werden aus der Session entfernt,
    ERROR-Log, Telegram „⚠️ … ticker(s) skipped — stale daily bars“,
    `session_runs.note = 'SKIPPED n ticker(s) …'`;
  - alle Store-Ticker veraltet (so sieht ein Ingest-Ausfall aus) → `abort`:
    keine Signale, ERROR-Log, Telegram „🛑 … ABORTED — stale daily bars“,
    `session_runs.note = 'ABORTED: …'`, `run_batch` wird nicht aufgerufen.
- **EOD wartet** bei Verdacht bis zu `BAR_FRESHNESS_EOD_WAIT_S` (600 s, Poll
  alle 30 s) auf den Ingest, bevor es urteilt.
- **Fail-open:** ein Crash im Check selbst (z. B. DB gesperrt) blockiert keine
  Session, wird aber als ERROR mit Traceback geloggt.
- **Wächter:** `check_sessions` liest `session_runs.note` (nur wenn die Spalte
  existiert). `ABORTED…` → Session-Check **fail** mit Detail (Alarm,
  6-h-Reminder, Recovery-Logik wie bei „nicht gelaufen“); `SKIPPED…` → ok mit
  Detail im Statusblock.
- Abschaltbar per `BAR_FRESHNESS_GATE_ENABLED=false` (Env; Notausgang für
  eine ungeplante Marktschließung, die der Kalender nicht kennt).

**Verhalten bei ausgefallenem Ingest (Nacht T):** 22:45 EOD → Store steht auf
T-1 → 10 min warten → `abort`, Telegram, `note`. 23:00 Wächter: `ohlc` STALE
(bestehender Check) **und** `session EOD` fail („claimed 22:45 but ABORTED…“).
Nächster Morgen: US_PRE 13:15, PEAD_OPEN 13:45 und US_OPEN 14:30 erwarten
Bar T, der Store hat T-1 → jede dieser Sessions bricht mit Alarm ab. Es gibt
an Tag T+1 **keine Einstiege**, bis der Ingest nachgeholt ist
(`systemctl --user start nts-ohlc-ingest.service`; das Fenster ist 7 Tage,
der Lauf heilt sich selbst). Exits laufen über den PositionManager weiter,
der Store-ATR für neue Stops bliebe sonst ebenfalls veraltet. Begründung für
„abbrechen statt still weiterlaufen“: ein Bar von gestern in RSI/ATR ist
genau der Fehler, den der externe Check vermutet hat — lieber ein Tag ohne
Signale mit vier Alarmen als ein Tag mit falschen.

Was nicht dupliziert wurde: das Ingest-Gate (Produzent, einmal 22:30), der
`ohlc`-Wächter-Check (alle 15 min ab 23:00). Dieses Gate ist die
Konsumentenseite.

---

## 5. RiskAgent loggt Skips ehrlich

Prod-Beleg: 3 089 `risk_calculations`-Zeilen mit `skipped=1` tragen
`event_risk_flag='none'` und `days_to_earnings=NULL` — darunter alle 20
echten Earnings-Skips (`skip_reason LIKE 'Earnings%'`). Ursache: beide
Skip-Pfade riefen `_no_position()` ohne die Werte auf und setzten sie nur im
zurückgegebenen Dict.

Jetzt: `event_risk_flag`, `days_to_earnings` und `regime` werden in beiden
Skip-Pfaden durchgereicht und persistiert. Liefert `get_days_to_earnings()`
None, steht im Log
`[TICKER] earnings date unknown (calendar returned None) — earnings filter open, event_risk_flag=none`.
Schwellen (≤ 2 / ≤ 5 Tage, Konfidenz < 50) und Multiplikatoren unverändert.

---

## 6. Preis-Fallback

`data/price_fallback.py` liest jetzt `ALPHA_VANTAGE_KEY` **oder**
`ALPHA_VANTAGE_API_KEY` (erste nicht-leere gewinnt). Die `.env` bleibt.

Befund, der die Frage „welche Stufen sind aktiv“ ändert: **Die gesamte
`PriceFallback`-Kette hängt nicht am Live-Pfad.** Kein Modul außerhalb der
Tests instanziiert sie. Der Live-Pfad für Sizing/Guard ist
`data/market_data.py::MarketData.fetch`:

| Stufe | Quelle | Status auf dem Live-Pfad |
|---|---|---|
| Alpaca latest trade | `AlpacaDataClient.get_current_price` | **aktiv** |
| Alpaca quote mid (bid/ask) | dito, Fallback im selben Aufruf | **aktiv** |
| danach | `price=None, degraded=True` → Coordinator nutzt Bar-Close nur für Analyse, `price_is_live=False` blockiert jede Order | aktiv (Blockade) |

Die vierstufige `PriceFallback`-Kette (L0 Alpaca, L1 Alpha Vantage, L2 Yahoo
JSON, L3 Cache/DB) ist damit vollständig **inaktiv** — mit dem Key-Fix ist L1
nutzbar, aber nicht genutzt. Sie einzuhängen würde die Preisquelle für das
Sizing ändern und ist bewusst nicht Teil dieses Deploys. Das Modul trägt jetzt
einen entsprechenden Hinweis im Docstring.

---

## 7. Deploy auf dem VPS (Lenni)

Kein Unit-File ändert sich, kein Timer. Die Crontab-Zeile für
`update_outcomes.py` bleibt gültig.

```bash
# 1. Code holen (main nach Merge)
ssh claw 'cd ~/news-trading-system && git pull --ff-only origin main && git rev-parse --short HEAD'

# 2. Daemon neu starten (liest config/sessions.py → EOD 22:45)
ssh claw 'systemctl --user restart nts-trading && sleep 3 && systemctl --user is-active nts-trading'
ssh claw 'journalctl --user -u nts-trading -n 20 --no-pager | grep -E "Next:|Daemon started"'
#    erwartet heute Abend: "Next: EOD at 2026-09-23 22:45 UTC"

# 3. Outcome-Tracker einmal manuell: erst Trockenlauf, dann echt
ssh claw 'cd ~/news-trading-system && .venv/bin/python3 -m analytics.outcome_tracker --dry-run'
ssh claw 'cd ~/news-trading-system && .venv/bin/python3 -m analytics.outcome_tracker'
#    erwartet: unevaluable=421 (no_store_bars=247, sentinel=174), filled=0

# 4. Verifikation nach der ersten EOD (ab ~22:48 UTC)
ssh claw 'journalctl --user -u nts-trading --since "22:40" --no-pager | grep -E "bar freshness|EOD outcome tracker|Claimed session slot for"'
ssh claw 'python3 -c "
import sqlite3; c=sqlite3.connect(\"file:/home/trading/trading-data/news_trading.db?mode=ro\", uri=True)
print(c.execute(\"SELECT session, started_at, note FROM session_runs WHERE run_date=date(\x27now\x27) ORDER BY started_at\").fetchall())
print(c.execute(\"SELECT ticker, strategy, price_at_signal, indicator_bar_date, indicator_bar_source, live_price, live_price_source FROM signal_events WHERE session=\x27EOD\x27 ORDER BY id DESC LIMIT 3\").fetchall())
"'
#    erwartet: EOD started_at 22:45:xx, note NULL; indicator_bar_date = heutiges Datum, source daily_ohlc
```

Rollback: `git checkout 70a1ade` + Restart; die neuen Spalten sind additiv und
stören den alten Code nicht.

---

## 8. Nicht geändert, gemeldet

- `outcome_correct` = Vorzeichen des zuletzt geschriebenen Horizonts (10d),
  während Kelly `outcome_5d_pct` als Betrag nutzt — historische Eigenart,
  beibehalten (Kelly-Eingaben unverändert).
- Der Coordinator ruft `get_days_to_earnings` zusätzlich zum RiskAgent auf
  (Doppelabfrage, beide Ergebnisse identisch) — nicht zusammengelegt.
- Der Daemon feuert Sessions auch an US-Feiertagen (bekannt seit
  2026-09-08); das Gate erwartet dann korrekt den letzten Handelstag und lässt
  die Session laufen.
- Die 8 002-Collectors und der `nts-earnings-probe`-Timer auf dem VPS sind
  nicht Teil dieses Themas.
