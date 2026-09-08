# OHLC-Ingest: Umstellung Polygon Starter → Alpaca (2026-09-08)

Anlass: Der Polygon-Stocks-Starter-Plan (~$29/Monat) wird gekündigt. Der
Querschnitts-Faktortest (`docs/XSEC_FACTOR_TEST_2026-09-07.md`) war der dritte
unabhängige Befund ohne Edge; NTS läuft als Side Project weiter, aber ohne
kostenpflichtige Datenquelle. `scripts/ingest_ohlc.py` muss trotzdem täglich
laufen — `daily_ohlc` ist seit 2026-09-01 die Preisbasis für Wilder-ATR(14),
Stops und Levels; ohne frische Bars fällt jede Level-Berechnung in den
Fixed-Stop-Fallback und das Freshness-Gate alarmiert täglich.

Stand: dev, **nicht** deployed. Code-Änderung nur im Ingest-Pfad; Live-Pfad,
RiskAgent, Freshness-Gate und Schema unverändert.

## 1. Cache-Sicherung (Ziel 1) — abgeschlossen und verifiziert

Der Polygon-Cache des Faktortests (`analysis/xsec_factor_test/cache/`, 2 680
Dateien, 1 674 091 661 Bytes) ist die einzige Kopie der Punkt-in-Zeit-Daten
ab 2021-09-08 (Polygons Fenster rollt, nach der Kündigung kein Zugang mehr).

| | |
|---|---|
| Archiv | `~/Archive/nts-xsec-polygon-cache-2026-09-08/` (außerhalb des Repos, Repo-Cache bleibt gitignored) |
| Inhalt | `cache/` 1:1, `README.md` (Erhebung, Zeitraum, Entitätsschlüssel FIGI, 9 bekannte Datenmängel), `MANIFEST.tsv` (sha256 + Bytes je Datei), `MANIFEST.sha256`, `VERIFY.log`, `provenance/` (Skripte, Logs, `out/`, Bericht, Git-Commit) |
| Verifikation | alle 2 680 Hashes OK, 0 Größenabweichungen, 0 fehlende, 0 überzählige Dateien (`VERIFY.log`, 2026-09-08 07:50 UTC) |
| Key-Check | Logs, Skripte, Outputs enthalten den Polygon-Key nicht |

Offen (Lenni): eine zweite Kopie außer Haus (z. B. R2 `nts-backup/`, rclone
liegt nur auf dem VPS). Das Archiv ist ein Verzeichnis, kein Tarball —
`tar -C ~/Archive -cf - nts-xsec-polygon-cache-2026-09-08 | rclone rcat …`.

## 2. Quellenwahl (Ziel 2)

Anforderungen: Tagesbars mit H/L/C, ≥ 15 Bars für ATR(14), roher Close
(D3-Konvention), Split-Adjustierung wie bisher, 20 Ticker einmal täglich im
22:30-UTC-Fenster, Freshness-Gate unverändert.

| Kandidat | Befund | Entscheidung |
|---|---|---|
| **Alpaca** (Basic, kostenlos; Zugang existiert in `.env`) | `GET /v2/stocks/bars`, `feed=sip` (konsolidiertes Band), `adjustment=raw` / `split` explizit; 200 Calls/min (Header gemessen); Historie bis 2016 geprobt; Bars stimmen mit dem Bestand überein (§3) | **gewählt** |
| Polygon Free (Basic) | gleiche Daten wie bisher, aber 5 Calls/min (Lauf ~8 min, schon heute so getaktet) und nicht prüfbar, ob der Key nach der Kündigung sauber auf Basic zurückfällt und der Tages-Bar um 22:30 UTC verfügbar ist | Rückfall: `OHLC_SOURCE=polygon` |
| yfinance 1.2.0 | liefert auch mit `auto_adjust=False` **split-adjustierte** Kurse (AMZN vor dem 20:1-Split 2022-06: Close 2 800 → 140, −95 %); H/L stimmen nur zu 6 % exakt (Float-Rundung); kein Vertrag, kein SLA, MCP-Server lokal bereits defekt | verworfen — verletzt die D3-Konvention |

Alpaca-Eigenheiten, die im Feed abgefangen sind (`data/alpaca_ohlc_feed.py`):

* **SIP-Recency-Regel des Free-Plans:** ein `end` innerhalb der letzten 15 min
  → HTTP 403 „subscription does not permit querying recent SIP data“. Ein
  reines Datum zählt als Tagesende und wird für *heute* abgelehnt. Der Feed
  klemmt `end` auf jetzt − 20 min; der 22:30-UTC-Lauf fragt bis 22:10 UTC,
  also nach dem Close (20:00/21:00 UTC), der Tages-Bar ist enthalten
  (gemessen: −16 min → 200, −14 min → 403). `feed=delayed_sip` existiert für
  diesen Endpoint nicht (400).
* **IEX-Feed nicht verwenden:** einzelne Venue, H/L sind nicht das Sitzungs-H/L
  (AAPL 2026-09-04: IEX High 328,895 vs SIP 328,93, Volumen 1,2 M vs 39,8 M).
* **`adj_close` bleibt split-only.** Der Bestand ist mit Polygon
  `adjusted=true` gefüllt, das keine Dividenden anwendet; Alpaca
  `adjustment=split` deckt sich damit auf 99,9 % der Bars innerhalb 0,01 %
  (95,7 % exakt). `adjustment=all` wäre dividendenbereinigt und bräche die
  Serie (PBR 2021-07: 10,67 vs 3,39). Nichts im Live-Pfad liest `adj_close`.
* 4xx außer 429 werden **nicht** wiederholt (Credentials, Recency, Symbol) —
  der Ticker fällt sofort durch, der Lauf endet mit Exit 1 + Telegram.

Taktung: 40 Calls bei 0,35 s Abstand; Testlauf 2026-09-08 gegen eine
Scratch-DB: 20/20 Ticker, 100 Zeilen, Freshness-Gate bestanden, **23 s**
Wandzeit (bisher ~8 min). Timer bleibt 22:30 UTC; `SAME_DAY_CUTOFF_UTC`,
Fenster, Gate und Alarmpfad unverändert.

## 3. Bar-Vergleich Alpaca vs. Bestand (Polygon Starter)

Referenz: `daily_ohlc` auf dem VPS (read-only gelesen), 20 Ticker, 2021-06-01
bis 2026-09-04, 26 349 Zeilen; 148 `TICKER_RECYCLE`-Zeilen (META vor
2022-06-09, anderer Emittent) ausgeschlossen → 26 201 Paare, 0 fehlende, 0
überzählige Daten (Alpaca hat zusätzlich 259 echte FB-Bars unter META).

| Feld | exakt gleich | \|Δ\| ≤ 0,01 % | > 0,1 % | > 1 % | P99 \|Δ\| | Max \|Δ\| | Mittel Δ |
|---|---|---|---|---|---|---|---|
| open | 99,9 % | 99,9 % | 0,05 % | 0,02 % | 0,000 % | 10,21 % | +0,0002 % |
| **high** | 99,8 % | 99,9 % | 0,13 % | 0,05 % | 0,000 % | 10,21 % | −0,0004 % |
| **low** | 99,8 % | 99,8 % | 0,13 % | 0,04 % | 0,000 % | 5,32 % | +0,0013 % |
| close | 100,0 % | 100,0 % | 0,01 % | 0,00 % | 0,000 % | 0,22 % | +0,0000 % |
| adj_close (split) | 95,7 % | 99,9 % | 0,01 % | 0,00 % | — | 0,22 % | — |
| volume | 44,6 % | 49,8 % | 39,8 % | 31,8 % | 31,9 % | 1 612 % | +3,46 % |

**Keine systematische H/L-Verzerrung:** Alpaca-High höher in 0,0 %, gleich
99,8 %, niedriger 0,2 %; Alpaca-Low höher 0,1 %, gleich 99,8 %, niedriger
0,0 %. Mittelwert der Abweichung in beiden Feldern ≈ 0.

**Wo es abweicht (51 von 26 201 Bars, 0,19 %):** fast alles liegt auf sechs
marktweiten Tagen, an denen beide Anbieter die Sitzungsspanne unterschiedlich
aus dem Band gefiltert haben:

| Datum | Muster | Beispiel |
|---|---|---|
| 2021-10-25, 2022-01-24/26, 2022-03-08, 2023-10-30 | Polygon-Spanne **weiter** (High höher, Low tiefer) bei 8–14 Tickern gleichzeitig; Close identisch | TOL 2022-03-08 High 50,87 vs 49,05 (−3,6 %), VRT Low 9,59 vs 10,10 (+5,3 %) |
| 2023-01-24 (NYSE-Eröffnungsauktions-Panne) | Alpaca behält die später für ungültig erklärten Auktionsprints, Polygon nicht | PBR Open/High 13,12 vs 14,46 (+10,2 %), XOM Open 109,16 vs 99,23 (−9,1 %) |
| Einzelfälle | 3 Close-Abweichungen: AAPL 2022-02-23/24 (Alpaca 160,24 vs 160,07; offizieller Close ist 160,07 → Alpaca falsch), VRT 2023-05-30 (Polygon-Volumen 584 k statt ~10 M → Polygon-Bar unvollständig) | |

**Auswirkung auf ATR(14) Wilder** (RiskAgent-Formel, 25 921 Ticker-Tage):
exakt gleich 74,2 %, |Δ| ≤ 0,1 % 94,0 %, > 1 % 2,18 %, P99 2,1 %, Max
14,1 % (PBR/TOL in den 14 Tagen nach 2023-01-24), Mittel −0,03 %. **Am
Stichtag 2026-09-04 weichen alle 20 ATR-Werte um < 0,01 % ab** — kein
laufender Stop verschiebt sich durch die Umstellung.

Volumen: Alpaca meldet im Median dasselbe, in ~5 % der Bars bis 16 % mehr
(P95). Volumen geht in keine Stop- oder Level-Rechnung ein.

Vollständige Tabellen inkl. yfinance: Vergleichsskript `compare_sources.py`
lief lokal (Scratch), Ausreißerliste 51 Zeilen; beides bei Bedarf
reproduzierbar (Alpaca-Credentials aus `.env`, Bestand per read-only
SELECT vom VPS).

## 4. Empfehlung zum Bestand: nicht neu backfillen

Der Polygon-Bestand bleibt stehen; Alpaca hängt ab dem ersten Lauf
inkrementell an (`source='alpaca'` je Zeile, Grenze 2026-09-04/08). Gründe:

* An der Nahtstelle sind die Bars identisch (Testlauf: 100/100 Zeilen
  O/H/L/C/adj_close exakt gleich).
* Ein `--backfill` mit Alpaca würde (a) die 2023-01-24-Auktionsprints in die
  Historie holen (ATR +14 % für zwei Wochen, nur historisch relevant) und
  (b) die 148 `TICKER_RECYCLE`-Zeilen von META mit echten FB-Bars
  überschreiben und dabei das Flag löschen (Upsert setzt `quality_flag`
  neu). Beides ist kein Gewinn für den Live-Pfad.
* Wer die Historie einmal komplett auf Alpaca ziehen will, macht das bewusst
  mit `--backfill` und lässt danach `scripts/flag_meta_recycle.py` **nicht**
  laufen (die FB-Bars sind dann echt).

## 5. Weitere Polygon-Abhängigkeiten nach der Kündigung

* `data/benzinga_feed.py` (`api.polygon.io/benzinga/v1/earnings`) wird vom
  Coordinator pro PEAD-Auswertung aufgerufen — **recorded-only, fail-safe**
  (jeder Fehler → `log.warning`, kein Einfluss auf Signal/Trade). Nach der
  Kündigung fehlen die Benzinga-Vergleichszeilen im `pead_signal_log`; sonst
  nichts. Der Timer `nts-preann-estimates` (Q-013) ist auf dem VPS **nicht
  installiert** (kein Unit in `list-timers`), also kein nächtlicher Alarm.
* `POLYGON_API_KEY` darf in `.env` bleiben (Rückfall `OHLC_SOURCE=polygon`,
  Benzinga-Miss bleibt still). Er ist nirgends im Repo.

## 6. Was auf dem VPS zu tun ist (Lenni, nach Merge dev → main)

Kein systemctl, keine Git-Writes und keine Credentials von meiner Seite —
alles unten ist Lennis Lauf. Reihenfolge:

1. **Vor der Kündigung** deployen, damit der erste Alpaca-Lauf noch neben
   einem funktionierenden Polygon-Rückfall liegt.
2. `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` stehen bereits in der VPS-`.env`
   (Alpaca-Data-Client des Daemons). Prüfen, dass es dieselben Werte wie
   lokal sind — gemessen wurde mit dem lokalen Paar. `OHLC_SOURCE` muss
   **nicht** gesetzt werden (Default `alpaca`).
3. ```
   sudo -u trading -i bash -c 'cd ~/news-trading-system && git pull --ff-only origin main \
     && cp deployment/systemd/nts-ohlc-ingest.service deployment/systemd/nts-ohlc-ingest.timer ~/.config/systemd/user/ \
     && XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user daemon-reload'
   ```
   Kein Restart des Daemons nötig; der Ingest ist ein oneshot-Timer.
4. **Einmal von Hand laufen lassen** (irgendwann nach 22:00 UTC eines
   Handelstags oder sofort — vor 22:00 UTC endet das Fenster auf gestern und
   das Gate erwartet den letzten abgeschlossenen Handelstag):
   ```
   sudo -u trading -i bash -c 'cd ~/news-trading-system && .venv/bin/python scripts/ingest_ohlc.py --incremental; echo exit=$?'
   ```
   Erwartung: `OHLC source: alpaca`, 20/20 Ticker, `Freshness gate passed`,
   Exit 0, ~25 s. Danach in der DB: `SELECT source, COUNT(*) FROM daily_ohlc
   GROUP BY 1` zeigt eine kleine `alpaca`-Gruppe neben `polygon`.
5. Nächste Nacht 22:30 UTC: `journalctl --user -u nts-ohlc-ingest.service`
   und der Watchdog-Check `ohlc` (unverändert) bestätigen den Lauf.
6. Erst dann Polygon kündigen. Rückfall bei Problemen: `OHLC_SOURCE=polygon`
   in die `.env` (Free-Tier, 5 Calls/min, Lauf ~8 min — im Fenster).

## 7. Dateien

* `data/alpaca_ohlc_feed.py` — neu (Feed, Recency-Klemme, Retry-Regeln)
* `scripts/ingest_ohlc.py` — Quellenwahl `build_feed(OHLC_SOURCE)`, `source`-Stempel je Zeile
* `config/settings.py` — `OHLC_SOURCE` (Default `alpaca`)
* `data/polygon_feed.py` — nur Docstring (Rückfall; adjusted = split-only)
* `deployment/systemd/nts-ohlc-ingest.{service,timer}` — Beschreibung
* `tests/test_ohlc_ingest.py` — 31 Tests (Alpaca-Parsing, Split-Pairing, Paging, 403 ohne Retry, Recency-Klemme, Credentials, Quellenwahl, Exit 1 + Alert ohne Credentials, `source`-Stempel)
