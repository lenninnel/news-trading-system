# Querschnitts-Faktortest (offline, read-only)

Reine Analyse. Kein Zugriff auf den NTS-Live-Pfad, keine NTS-Imports, keine
DB-Schreibzugriffe. Einzige externe Quelle: Polygon REST (read-only), komplett
unter `cache/` zwischengespeichert (gitignored, ~0,5 GB).

    python3 -m venv .venv && .venv/bin/pip install pandas numpy pyarrow requests scipy
    .venv/bin/python fetch_polygon.py --all     # resumierbar, überspringt Vorhandenes
    .venv/bin/python build_panel.py             # Cache -> cache/panel/*.parquet
    .venv/bin/python run_factors.py             # -> out/REPORT_TABLES.md, results.json, monthly_returns.csv

Der Polygon-Key wird aus `POLYGON_API_KEY` (env) oder der Repo-`.env` gelesen.
Alle festen Regeln stehen im Docstring von `run_factors.py`; Bericht in
`docs/XSEC_FACTOR_TEST_2026-09-07.md`.
