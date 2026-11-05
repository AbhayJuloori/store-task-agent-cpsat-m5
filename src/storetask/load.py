"""Load the M5 CSVs into DuckDB.

Tables
  calendar(d_num, date, wm_yr_wk, weekday, event_name_1)
  items(item_id, dept_id, cat_id)
  prices(store_id, item_id, wm_yr_wk, sell_price)
  sales(store_id, item_id, d_num, units)        -- d_num >= history_start_d
  price_changes(store_id, item_id, wm_yr_wk, effective_date, old_price, new_price)
"""
from __future__ import annotations

import time

import duckdb

from .config import DB_PATH, RAW, WORK, load_assumptions


def build(db_path=DB_PATH, raw=RAW, history_start_d: int | None = None) -> dict:
    a = load_assumptions()
    start = history_start_d or a["window"]["history_start_d"]
    WORK.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    t0 = time.time()
    con = duckdb.connect(str(db_path))
    con.execute("SET threads TO 4")
    con.execute(f"""
        CREATE TABLE calendar AS
        SELECT CAST(substr(d, 3) AS INTEGER) AS d_num, CAST(date AS DATE) AS date,
               wm_yr_wk, weekday, event_name_1
        FROM read_csv_auto('{raw}/calendar.csv')
    """)
    day_cols = ", ".join(f"d_{i}" for i in range(start, 1942))
    con.execute(f"""
        CREATE TABLE items AS
        SELECT DISTINCT item_id, dept_id, cat_id
        FROM read_csv_auto('{raw}/sales_train_evaluation.csv')
    """)
    con.execute(f"""
        CREATE TABLE sales AS
        SELECT store_id, item_id, CAST(substr(d, 3) AS INTEGER) AS d_num, CAST(units AS INTEGER) AS units
        FROM (
            SELECT store_id, item_id, {day_cols}
            FROM read_csv_auto('{raw}/sales_train_evaluation.csv')
        ) UNPIVOT (units FOR d IN ({day_cols}))
    """)
    con.execute(f"""
        CREATE TABLE prices AS
        SELECT store_id, item_id, wm_yr_wk, sell_price
        FROM read_csv_auto('{raw}/sell_prices.csv')
    """)
    con.execute("""
        CREATE TABLE price_changes AS
        WITH p AS (
            SELECT store_id, item_id, wm_yr_wk, sell_price,
                   lag(sell_price) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk) AS prev_price
            FROM prices
        ), wk AS (SELECT wm_yr_wk, min(date) AS effective_date FROM calendar GROUP BY 1)
        SELECT p.store_id, p.item_id, p.wm_yr_wk, wk.effective_date,
               p.prev_price AS old_price, p.sell_price AS new_price
        FROM p JOIN wk USING (wm_yr_wk)
        WHERE p.prev_price IS NOT NULL AND p.sell_price <> p.prev_price
    """)
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ["calendar", "items", "sales", "prices", "price_changes"]}
    con.close()
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


if __name__ == "__main__":
    print(build())
