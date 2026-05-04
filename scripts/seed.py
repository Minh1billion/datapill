from __future__ import annotations

import asyncio
import json
import math
import random
import string
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Linda", "Mallory", "Ninh", "Oscar", "Peggy",
    "Quan", "Ruth", "Steve", "Tuan", "Uma", "Victor", "Wendy", "Xuan",
    "Yen", "Zach", "An", "Binh", "Chi", "Duc", "Huong", "Khanh",
    "Lan", "Minh", "Nam", "Phuong", "Quynh", "Son", "Thu", "Van",
]
_LAST_NAMES = [
    "Nguyen", "Tran", "Le", "Pham", "Hoang", "Phan", "Vu", "Dang",
    "Bui", "Do", "Ho", "Ngo", "Duong", "Ly", "Smith", "Johnson",
    "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
]
_DEPTS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations", "Legal", "Product"]
_CITIES = ["Hanoi", "HCMC", "Danang", "Hue", "Cantho", "Haiphong", "Nhatrang", "Vungtau", "Dalat", "Bien Hoa"]
_COUNTRIES = ["VN", "US", "SG", "JP", "KR", "AU", "GB", "DE", "FR", "TH"]
_LEVELS = ["junior", "mid", "senior", "lead", "principal", "staff", "manager", "director", "vp", "c-level"]
_STATUSES = ["active", "inactive", "suspended", "pending"]
_CATEGORIES = ["electronics", "clothing", "food", "books", "toys", "sports", "beauty", "home", "auto", "health"]
_SUBCATEGORIES = {
    "electronics": ["phone", "laptop", "tablet", "tv", "audio", "camera", "gaming", "wearable"],
    "clothing":    ["shirt", "pants", "dress", "shoes", "hat", "jacket", "underwear", "socks"],
    "food":        ["snack", "beverage", "dairy", "meat", "vegetable", "fruit", "frozen", "canned"],
    "books":       ["fiction", "non-fiction", "textbook", "comic", "manga", "biography", "science", "history"],
    "toys":        ["action-figure", "board-game", "puzzle", "doll", "rc", "educational", "outdoor", "craft"],
    "sports":      ["running", "swimming", "cycling", "football", "basketball", "tennis", "yoga", "gym"],
    "beauty":      ["skincare", "makeup", "haircare", "fragrance", "nailcare", "bodycare", "sunscreen", "serum"],
    "home":        ["furniture", "kitchen", "bedding", "decor", "lighting", "storage", "cleaning", "garden"],
    "auto":        ["parts", "accessories", "oil", "tire", "battery", "wiper", "filter", "audio"],
    "health":      ["supplement", "medicine", "equipment", "personal-care", "dental", "vision", "first-aid", "protein"],
}
_PAYMENT_METHODS = ["credit_card", "debit_card", "bank_transfer", "cash", "e_wallet", "crypto", "bnpl", "voucher"]
_ORDER_STATUSES  = ["pending", "confirmed", "processing", "shipped", "delivered", "returned", "cancelled", "refunded"]
_EVENT_TYPES     = ["page_view", "click", "search", "add_to_cart", "remove_from_cart",
                    "checkout_start", "checkout_complete", "login", "logout", "signup",
                    "product_view", "review_submit", "wishlist_add", "share", "error"]
_LOG_LEVELS      = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_SERVICES        = ["api-gateway", "auth-service", "order-service", "payment-service",
                    "inventory-service", "notification-service", "search-service", "analytics-service"]
_BROWSERS        = ["Chrome", "Firefox", "Safari", "Edge", "Opera", "Samsung Browser", "UC Browser"]
_OS_LIST         = ["Windows", "macOS", "Linux", "Android", "iOS", "ChromeOS"]
_CHANNELS        = ["web", "mobile_app", "pos", "call_center", "partner_api", "marketplace"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_DATE = date(2020, 1, 1)
END_DATE  = date(2024, 12, 31)
TOTAL_DAYS = (END_DATE - BASE_DATE).days


def _rand_date(start: date = BASE_DATE, end: date = END_DATE) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def _rand_dt(start: date = BASE_DATE, end: date = END_DATE) -> datetime:
    d = _rand_date(start, end)
    return datetime(d.year, d.month, d.day,
                    random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))


def _rand_email(name: str, domain_pool: list[str] | None = None) -> str:
    domains = domain_pool or ["gmail.com", "yahoo.com", "outlook.com", "company.vn",
                               "corp.io", "mail.vn", "proton.me", "icloud.com"]
    slug = name.lower().replace(" ", ".") + str(random.randint(1, 999))
    return f"{slug}@{random.choice(domains)}"


def _rand_phone() -> str | None:
    if random.random() < 0.08:   # 8% null
        return None
    prefix = random.choice(["090", "091", "093", "094", "096", "097", "098", "070", "079", "077"])
    return prefix + "".join([str(random.randint(0, 9)) for _ in range(7)])


def _skewed_salary(dept: str) -> float:
    """Salary với distribution theo dept, có outlier ~2%."""
    base = {
        "Engineering": 3500, "Finance": 3000, "Legal": 3200, "Product": 3300,
        "Marketing": 2200, "Sales": 2000, "HR": 1800, "Operations": 1700,
    }.get(dept, 2000)
    if random.random() < 0.02:          # outlier
        return round(random.uniform(15000, 30000), 2)
    if random.random() < 0.03:          # near-zero outlier
        return round(random.uniform(100, 400), 2)
    return round(random.gauss(base, base * 0.25), 2)


def _rand_score(null_rate: float = 0.05) -> float | None:
    if random.random() < null_rate:
        return None
    return round(random.uniform(0, 100), 4)


def _rand_str_id(prefix: str = "", length: int = 8) -> str:
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _maybe_null(val: Any, rate: float = 0.05) -> Any:
    return None if random.random() < rate else val


def _rand_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _rand_ua(browser: str, os_: str) -> str:
    return f"Mozilla/5.0 ({os_}; rv:120.0) Gecko/20100101 {browser}/120.0"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _make_departments() -> list[dict]:
    return [
        {"dept_id": i + 1, "name": d,
         "budget": round(random.uniform(100_000, 2_000_000), 2),
         "headcount": random.randint(5, 120),
         "cost_center": _rand_str_id("CC", 6),
         "location": random.choice(_CITIES),
         "manager_id": None}   # filled later
        for i, d in enumerate(_DEPTS)
    ]


def _make_employees(n: int = 150_000) -> list[dict]:
    rows = []
    dept_ids = list(range(1, len(_DEPTS) + 1))
    for i in range(n):
        first = random.choice(_FIRST_NAMES)
        last  = random.choice(_LAST_NAMES)
        full  = f"{first} {last}"
        dept  = random.choice(_DEPTS)
        joined = _rand_date(date(2015, 1, 1), date(2024, 6, 1))
        terminated = None
        if random.random() < 0.18:      # 18% đã nghỉ
            terminated = _rand_date(joined + timedelta(days=90), date(2024, 12, 31))
        salary = _skewed_salary(dept)
        rows.append({
            "employee_id":   i + 1,
            "first_name":    first,
            "last_name":     last,
            "full_name":     full,
            "email":         _maybe_null(_rand_email(full), 0.01),
            "phone":         _rand_phone(),
            "age":           _maybe_null(random.randint(18, 65), 0.02),
            "dept":          dept,
            "dept_id":       random.choice(dept_ids),
            "level":         random.choice(_LEVELS),
            "city":          random.choice(_CITIES),
            "country":       _maybe_null(random.choice(_COUNTRIES), 0.03),
            "salary":        salary,
            "bonus_pct":     _maybe_null(round(random.uniform(0, 0.40), 4), 0.15),
            "performance":   _maybe_null(_rand_score(0.0), 0.10),
            "status":        random.choice(_STATUSES),
            "joined":        str(joined),
            "terminated":    str(terminated) if terminated else None,
            "is_remote":     random.choice([True, False, False]),   # biased false
            "manager_id":    _maybe_null(random.randint(1, min(i + 1, 500)), 0.05) if i > 0 else None,
            "tags":          _maybe_null(
                                 json.dumps(random.sample(["python","java","go","sql","ml","devops","pm","finance"], k=random.randint(0, 4))),
                                 0.20),
        })
    # inject duplicates (~0.5%)
    dupes = random.sample(rows, k=int(n * 0.005))
    for d in dupes:
        clone = {**d, "employee_id": n + rows.index(d) + 1}
        rows.append(clone)
    return rows


def _make_products(n: int = 5_000) -> list[dict]:
    rows = []
    for i in range(n):
        cat = random.choice(_CATEGORIES)
        sub = random.choice(_SUBCATEGORIES[cat])
        cost = round(random.uniform(1, 2000), 2)
        rows.append({
            "product_id":   i + 1,
            "sku":          _rand_str_id("SKU-", 10),
            "name":         f"{sub.title()} Product #{i+1}",
            "category":     cat,
            "subcategory":  sub,
            "brand":        _maybe_null(random.choice(["BrandA","BrandB","BrandC","BrandD","BrandE","NoName"]), 0.05),
            "cost":         cost,
            "price":        round(cost * random.uniform(1.1, 4.0), 2),
            "weight_kg":    _maybe_null(round(random.uniform(0.01, 50), 3), 0.12),
            "stock":        _maybe_null(random.randint(0, 10_000), 0.04),
            "rating":       _maybe_null(round(random.uniform(1, 5), 2), 0.08),
            "review_count": _maybe_null(random.randint(0, 50_000), 0.08),
            "active":       random.random() > 0.10,
            "created_at":   str(_rand_date(date(2019, 1, 1), date(2023, 12, 31))),
        })
    return rows


def _make_customers(n: int = 50_000) -> list[dict]:
    rows = []
    for i in range(n):
        first = random.choice(_FIRST_NAMES)
        last  = random.choice(_LAST_NAMES)
        full  = f"{first} {last}"
        rows.append({
            "customer_id":   i + 1,
            "full_name":     full,
            "email":         _maybe_null(_rand_email(full), 0.02),
            "phone":         _rand_phone(),
            "age":           _maybe_null(random.randint(16, 85), 0.07),
            "gender":        _maybe_null(random.choice(["M","F","O","prefer_not_to_say"]), 0.10),
            "city":          _maybe_null(random.choice(_CITIES), 0.04),
            "country":       random.choice(_COUNTRIES),
            "segment":       random.choice(["vip","loyal","regular","new","churned","at_risk"]),
            "lifetime_value": _maybe_null(round(random.uniform(0, 100_000), 2), 0.06),
            "signup_date":   str(_rand_date(date(2018, 1, 1), date(2024, 12, 1))),
            "last_active":   _maybe_null(str(_rand_date(date(2023, 1, 1), date(2024, 12, 31))), 0.10),
            "channel":       random.choice(_CHANNELS),
            "referral_code": _maybe_null(_rand_str_id("REF", 6), 0.70),
        })
    return rows


def _make_orders(n: int = 200_000, n_customers: int = 50_000, n_products: int = 5_000) -> list[dict]:
    rows = []
    for i in range(n):
        order_dt = _rand_dt()
        status   = random.choice(_ORDER_STATUSES)
        qty      = random.randint(1, 20)
        unit_price = round(random.uniform(1, 3000), 2)
        discount = _maybe_null(round(random.uniform(0, 0.5), 4), 0.60)
        rows.append({
            "order_id":       i + 1,
            "order_ref":      _rand_str_id("ORD-", 12),
            "customer_id":    _maybe_null(random.randint(1, n_customers), 0.03),
            "product_id":     random.randint(1, n_products),
            "qty":            qty,
            "unit_price":     unit_price,
            "discount":       discount,
            "gross_amount":   round(qty * unit_price, 2),
            "net_amount":     round(qty * unit_price * (1 - (discount or 0)), 2),
            "payment_method": random.choice(_PAYMENT_METHODS),
            "status":         status,
            "channel":        random.choice(_CHANNELS),
            "city":           _maybe_null(random.choice(_CITIES), 0.05),
            "country":        random.choice(_COUNTRIES),
            "ordered_at":     str(order_dt),
            "shipped_at":     _maybe_null(
                                  str(order_dt + timedelta(hours=random.randint(1, 120))),
                                  0.0 if status in ("shipped","delivered") else 0.70),
            "delivered_at":   _maybe_null(
                                  str(order_dt + timedelta(hours=random.randint(24, 480))),
                                  0.0 if status == "delivered" else 0.75),
            "notes":          _maybe_null(
                                  random.choice(["urgent","fragile","gift wrap","leave at door","call before","no note"]),
                                  0.80),
        })
    return rows


def _make_transactions(n: int = 300_000, n_customers: int = 50_000) -> list[dict]:
    rows = []
    for i in range(n):
        amount = round(random.expovariate(1 / 500), 2)   # right-skewed
        if random.random() < 0.01:
            amount = round(random.uniform(50_000, 500_000), 2)  # large outliers
        txn_dt = _rand_dt()
        rows.append({
            "txn_id":         i + 1,
            "txn_ref":        _rand_str_id("TXN-", 16),
            "customer_id":    _maybe_null(random.randint(1, n_customers), 0.05),
            "amount":         amount,
            "currency":       random.choice(["VND","USD","SGD","JPY","EUR","GBP","KRW","AUD"]),
            "method":         random.choice(_PAYMENT_METHODS),
            "status":         random.choice(["success","failed","pending","reversed","disputed"]),
            "gateway":        random.choice(["VNPay","Momo","ZaloPay","Stripe","PayPal","Adyen","2C2P"]),
            "fee":            _maybe_null(round(amount * random.uniform(0.005, 0.03), 4), 0.15),
            "country":        random.choice(_COUNTRIES),
            "ip":             _maybe_null(_rand_ip(), 0.10),
            "device":         _maybe_null(random.choice(["mobile","desktop","tablet","pos","api"]), 0.05),
            "is_fraud":       random.random() < 0.008,   # 0.8% fraud
            "created_at":     str(txn_dt),
            "settled_at":     _maybe_null(str(txn_dt + timedelta(hours=random.randint(0, 72))), 0.20),
        })
    return rows


def _make_events(n: int = 500_000, n_customers: int = 50_000, n_products: int = 5_000) -> list[dict]:
    rows = []
    for i in range(n):
        event_type = random.choice(_EVENT_TYPES)
        browser = random.choice(_BROWSERS)
        os_     = random.choice(_OS_LIST)
        rows.append({
            "event_id":     i + 1,
            "session_id":   _rand_str_id("SES-", 12),
            "customer_id":  _maybe_null(random.randint(1, n_customers), 0.30),  # 30% anonymous
            "event_type":   event_type,
            "page":         _maybe_null(random.choice(["/home","/search","/product","/cart","/checkout","/account","/promo"]), 0.05),
            "product_id":   _maybe_null(random.randint(1, n_products), 0.60),
            "search_query": _maybe_null(
                                random.choice(["iphone","laptop","dress","book","toy","headphone","sneaker","vitamin"]),
                                0.0 if event_type == "search" else 0.95),
            "referrer":     _maybe_null(random.choice(["google","facebook","tiktok","direct","email","sms","zalo",None]), 0.30),
            "browser":      browser,
            "os":           os_,
            "user_agent":   _rand_ua(browser, os_),
            "ip":           _maybe_null(_rand_ip(), 0.08),
            "country":      _maybe_null(random.choice(_COUNTRIES), 0.05),
            "duration_ms":  _maybe_null(
                                round(random.expovariate(1 / 2000), 1),
                                0.20),
            "created_at":   str(_rand_dt()),
        })
    return rows


def _make_logs(n: int = 100_000) -> list[dict]:
    rows = []
    error_msgs = [
        "connection timeout", "null pointer exception", "disk quota exceeded",
        "authentication failed", "rate limit exceeded", "service unavailable",
        "invalid payload", "db deadlock detected", "memory limit exceeded",
        "upstream error 502", "ssl handshake failed", "token expired",
    ]
    info_msgs = [
        "request processed", "cache hit", "cache miss", "health check ok",
        "scheduled job started", "scheduled job completed", "config reloaded",
        "user authenticated", "payment processed", "order shipped",
    ]
    for i in range(n):
        level   = random.choices(_LOG_LEVELS, weights=[20, 55, 15, 8, 2])[0]
        service = random.choice(_SERVICES)
        msg = random.choice(error_msgs if level in ("ERROR","CRITICAL") else info_msgs)
        rows.append({
            "log_id":        i + 1,
            "level":         level,
            "service":       service,
            "message":       msg,
            "trace_id":      _maybe_null(_rand_str_id("TR-", 16), 0.40),
            "latency_ms":    _maybe_null(round(random.expovariate(1 / 150), 2), 0.25),
            "status_code":   _maybe_null(random.choice([200,201,400,401,403,404,422,429,500,502,503]), 0.20),
            "host":          f"{service}-{random.randint(1,5):02d}.internal",
            "region":        random.choice(["ap-southeast-1","ap-northeast-1","us-east-1","eu-west-1"]),
            "created_at":    str(_rand_dt()),
        })
    return rows


def _make_reviews(n: int = 80_000, n_customers: int = 50_000, n_products: int = 5_000) -> list[dict]:
    rows = []
    sentiments = ["positive", "neutral", "negative"]
    templates = {
        "positive": ["Great product!", "Highly recommend.", "Very satisfied.", "Exceeded expectations.", "Fast delivery."],
        "neutral":  ["It's okay.", "Average quality.", "Nothing special.", "Does the job.", "Could be better."],
        "negative": ["Disappointed.", "Poor quality.", "Not as described.", "Slow delivery.", "Would not buy again."],
    }
    for i in range(n):
        sentiment = random.choices(sentiments, weights=[60, 25, 15])[0]
        rating    = {"positive": random.randint(4, 5), "neutral": 3, "negative": random.randint(1, 2)}[sentiment]
        rows.append({
            "review_id":   i + 1,
            "product_id":  random.randint(1, n_products),
            "customer_id": _maybe_null(random.randint(1, n_customers), 0.05),
            "rating":      rating,
            "sentiment":   sentiment,
            "title":       _maybe_null(random.choice(templates[sentiment]), 0.30),
            "body":        _maybe_null(random.choice(templates[sentiment]) + " " + random.choice(templates[sentiment]), 0.15),
            "helpful":     random.randint(0, 500),
            "verified":    random.random() > 0.25,
            "created_at":  str(_rand_date(date(2020, 1, 1), date(2024, 12, 31))),
        })
    return rows


def _make_inventory(n_products: int = 5_000) -> list[dict]:
    rows = []
    warehouses = ["WH-HAN-01", "WH-HCM-01", "WH-HCM-02", "WH-DAN-01", "WH-CAN-01"]
    for pid in range(1, n_products + 1):
        for wh in random.sample(warehouses, k=random.randint(1, len(warehouses))):
            qty = random.randint(0, 5_000)
            rows.append({
                "product_id":    pid,
                "warehouse":     wh,
                "qty_on_hand":   qty,
                "qty_reserved":  min(qty, random.randint(0, 500)),
                "reorder_point": random.randint(10, 200),
                "last_counted":  str(_rand_date(date(2024, 1, 1), date(2024, 12, 31))),
            })
    return rows


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _build_all() -> dict[str, pl.DataFrame]:
    print("  generating employees (150k + ~750 dupes)…", flush=True)
    emp_rows = _make_employees(150_000)
    emp_df   = pl.DataFrame(emp_rows)

    print("  generating products (5k)…", flush=True)
    prod_df = pl.DataFrame(_make_products(5_000))

    print("  generating customers (50k)…", flush=True)
    cust_df = pl.DataFrame(_make_customers(50_000))

    print("  generating orders (200k)…", flush=True)
    ord_df  = pl.DataFrame(_make_orders(200_000))

    print("  generating transactions (300k)…", flush=True)
    txn_df  = pl.DataFrame(_make_transactions(300_000))

    print("  generating events (500k)…", flush=True)
    evt_df  = pl.DataFrame(_make_events(500_000))

    print("  generating logs (100k)…", flush=True)
    log_df  = pl.DataFrame(_make_logs(100_000))

    print("  generating reviews (80k)…", flush=True)
    rev_df  = pl.DataFrame(_make_reviews(80_000))

    print("  generating inventory…", flush=True)
    inv_df  = pl.DataFrame(_make_inventory(5_000))

    print("  generating departments…", flush=True)
    dept_df = pl.DataFrame(_make_departments())

    total = sum(len(df) for df in [emp_df, prod_df, cust_df, ord_df, txn_df, evt_df, log_df, rev_df, inv_df, dept_df])
    print(f"  ✓ total rows across all tables: {total:,}")

    return {
        "employees":    emp_df,
        "products":     prod_df,
        "customers":    cust_df,
        "orders":       ord_df,
        "transactions": txn_df,
        "events":       evt_df,
        "logs":         log_df,
        "reviews":      rev_df,
        "inventory":    inv_df,
        "departments":  dept_df,
    }


# ---------------------------------------------------------------------------
# Seeders
# ---------------------------------------------------------------------------

def seed_local_files(tables: dict[str, pl.DataFrame]) -> None:
    print("→ Local files", flush=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # -- full dumps per table
    for name, df in tables.items():
        df.write_csv(FIXTURE_DIR / f"{name}.csv")
        df.write_parquet(FIXTURE_DIR / f"{name}.parquet")
        df.write_ndjson(FIXTURE_DIR / f"{name}.ndjson")
        print(f"   {name}: {len(df):,} rows → csv + parquet + ndjson")

    # -- partitioned employees by dept (simulates partitioned S3/local data lake)
    part_dir = FIXTURE_DIR / "employees_by_dept"
    part_dir.mkdir(exist_ok=True)
    emp_df = tables["employees"]
    for dept in _DEPTS:
        part = emp_df.filter(pl.col("dept") == dept)
        part.write_parquet(part_dir / f"dept={dept.lower().replace(' ','_')}.parquet")
    print(f"   employees partitioned by dept → {len(_DEPTS)} parquet files")

    # -- partitioned orders by year-month
    part_dir2 = FIXTURE_DIR / "orders_by_month"
    part_dir2.mkdir(exist_ok=True)
    ord_df = tables["orders"]
    ord_with_month = ord_df.with_columns(
        pl.col("ordered_at").str.slice(0, 7).alias("ym")
    )
    for ym in sorted(ord_with_month["ym"].unique().to_list()):
        part = ord_with_month.filter(pl.col("ym") == ym).drop("ym")
        safe = ym.replace("-", "_")
        part.write_parquet(part_dir2 / f"month={safe}.parquet")
    n_months = ord_with_month["ym"].n_unique()
    print(f"   orders partitioned by month → {n_months} parquet files")

    # -- dirty CSV for testing clean ops (inject extra nulls + corrupt values)
    dirty = tables["employees"].sample(fraction=0.10, seed=SEED).to_dicts()
    for row in dirty:
        if random.random() < 0.10:
            row["salary"]  = None
        if random.random() < 0.15:
            row["age"]     = random.choice([-1, 0, 999, None])
        if random.random() < 0.08:
            row["email"]   = random.choice(["not-an-email", "@@bad", "", None])
        if random.random() < 0.12:
            row["country"] = random.choice(["XX", "00", "??", None])
    pl.DataFrame(dirty).write_csv(FIXTURE_DIR / "employees_dirty.csv")
    print(f"   employees_dirty.csv: {len(dirty):,} rows with injected nulls/bad values")

    print(f"  ✓ all local files written to {FIXTURE_DIR}/")


def seed_sqlite(tables: dict[str, pl.DataFrame]) -> None:
    import sqlite3
    print("→ SQLite", flush=True)
    db = FIXTURE_DIR / "test.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)

    ddl = {
        "departments": """
            CREATE TABLE departments (
                dept_id INTEGER PRIMARY KEY, name TEXT, budget REAL,
                headcount INTEGER, cost_center TEXT, location TEXT, manager_id INTEGER
            )""",
        "employees": """
            CREATE TABLE employees (
                employee_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT,
                full_name TEXT, email TEXT, phone TEXT, age INTEGER,
                dept TEXT, dept_id INTEGER, level TEXT, city TEXT, country TEXT,
                salary REAL, bonus_pct REAL, performance REAL, status TEXT,
                joined TEXT, terminated TEXT, is_remote INTEGER,
                manager_id INTEGER, tags TEXT
            )""",
        "products": """
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY, sku TEXT, name TEXT,
                category TEXT, subcategory TEXT, brand TEXT,
                cost REAL, price REAL, weight_kg REAL, stock INTEGER,
                rating REAL, review_count INTEGER, active INTEGER, created_at TEXT
            )""",
        "customers": """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY, full_name TEXT, email TEXT,
                phone TEXT, age INTEGER, gender TEXT, city TEXT, country TEXT,
                segment TEXT, lifetime_value REAL, signup_date TEXT,
                last_active TEXT, channel TEXT, referral_code TEXT
            )""",
        "orders": """
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY, order_ref TEXT, customer_id INTEGER,
                product_id INTEGER, qty INTEGER, unit_price REAL, discount REAL,
                gross_amount REAL, net_amount REAL, payment_method TEXT,
                status TEXT, channel TEXT, city TEXT, country TEXT,
                ordered_at TEXT, shipped_at TEXT, delivered_at TEXT, notes TEXT
            )""",
        "transactions": """
            CREATE TABLE transactions (
                txn_id INTEGER PRIMARY KEY, txn_ref TEXT, customer_id INTEGER,
                amount REAL, currency TEXT, method TEXT, status TEXT,
                gateway TEXT, fee REAL, country TEXT, ip TEXT,
                device TEXT, is_fraud INTEGER, created_at TEXT, settled_at TEXT
            )""",
        "reviews": """
            CREATE TABLE reviews (
                review_id INTEGER PRIMARY KEY, product_id INTEGER,
                customer_id INTEGER, rating INTEGER, sentiment TEXT,
                title TEXT, body TEXT, helpful INTEGER, verified INTEGER, created_at TEXT
            )""",
        "inventory": """
            CREATE TABLE inventory (
                product_id INTEGER, warehouse TEXT, qty_on_hand INTEGER,
                qty_reserved INTEGER, reorder_point INTEGER, last_counted TEXT,
                PRIMARY KEY (product_id, warehouse)
            )""",
        "logs": """
            CREATE TABLE logs (
                log_id INTEGER PRIMARY KEY, level TEXT, service TEXT,
                message TEXT, trace_id TEXT, latency_ms REAL,
                status_code INTEGER, host TEXT, region TEXT, created_at TEXT
            )""",
    }

    for tname, create_sql in ddl.items():
        con.execute(f"DROP TABLE IF EXISTS {tname}")
        con.execute(create_sql)

        df = tables.get(tname)
        if df is None:
            continue

        rows = df.to_dicts()
        # Coerce bool → int for sqlite
        for row in rows:
            for k, v in row.items():
                if isinstance(v, bool):
                    row[k] = int(v)

        cols = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str      = ", ".join(cols)
        batch_size   = 5_000
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            con.executemany(
                f"INSERT OR IGNORE INTO {tname} ({col_str}) VALUES ({placeholders})",
                [[r[c] for c in cols] for r in batch],
            )
        con.commit()
        print(f"   {tname}: {len(rows):,} rows inserted")

    con.close()
    print(f"  ✓ {db}")


async def seed_postgres(tables: dict[str, pl.DataFrame]) -> None:
    import asyncpg
    print("→ PostgreSQL", flush=True)
    conn = await asyncpg.connect(
        host="localhost", port=5433,
        user="testuser", password="testpass", database="testdb",
    )

    ddl_stmts = [
        "DROP TABLE IF EXISTS inventory, reviews, logs, events, transactions, orders, customers, products, employees, departments CASCADE",
        """CREATE TABLE departments (
            dept_id SERIAL PRIMARY KEY, name TEXT, budget NUMERIC(14,2),
            headcount INT, cost_center TEXT, location TEXT, manager_id INT
        )""",
        """CREATE TABLE employees (
            employee_id BIGINT PRIMARY KEY, first_name TEXT, last_name TEXT,
            full_name TEXT, email TEXT, phone TEXT, age INT,
            dept TEXT, dept_id INT REFERENCES departments(dept_id),
            level TEXT, city TEXT, country CHAR(2),
            salary NUMERIC(12,2), bonus_pct NUMERIC(6,4), performance NUMERIC(8,4),
            status TEXT, joined DATE, terminated DATE,
            is_remote BOOLEAN, manager_id BIGINT, tags JSONB
        )""",
        """CREATE TABLE products (
            product_id INT PRIMARY KEY, sku TEXT UNIQUE, name TEXT,
            category TEXT, subcategory TEXT, brand TEXT,
            cost NUMERIC(12,2), price NUMERIC(12,2), weight_kg NUMERIC(8,3),
            stock INT, rating NUMERIC(4,2), review_count INT,
            active BOOLEAN, created_at DATE
        )""",
        """CREATE TABLE customers (
            customer_id INT PRIMARY KEY, full_name TEXT, email TEXT,
            phone TEXT, age INT, gender TEXT, city TEXT, country CHAR(2),
            segment TEXT, lifetime_value NUMERIC(14,2), signup_date DATE,
            last_active DATE, channel TEXT, referral_code TEXT
        )""",
        """CREATE TABLE orders (
            order_id BIGINT PRIMARY KEY, order_ref TEXT, customer_id INT,
            product_id INT, qty INT, unit_price NUMERIC(12,2),
            discount NUMERIC(6,4), gross_amount NUMERIC(14,2), net_amount NUMERIC(14,2),
            payment_method TEXT, status TEXT, channel TEXT, city TEXT, country TEXT,
            ordered_at TIMESTAMP, shipped_at TIMESTAMP, delivered_at TIMESTAMP, notes TEXT
        )""",
        """CREATE TABLE transactions (
            txn_id BIGINT PRIMARY KEY, txn_ref TEXT, customer_id INT,
            amount NUMERIC(16,2), currency CHAR(3), method TEXT, status TEXT,
            gateway TEXT, fee NUMERIC(12,4), country TEXT, ip INET,
            device TEXT, is_fraud BOOLEAN, created_at TIMESTAMP, settled_at TIMESTAMP
        )""",
        """CREATE TABLE reviews (
            review_id INT PRIMARY KEY, product_id INT, customer_id INT,
            rating SMALLINT, sentiment TEXT, title TEXT, body TEXT,
            helpful INT, verified BOOLEAN, created_at DATE
        )""",
        """CREATE TABLE inventory (
            product_id INT, warehouse TEXT, qty_on_hand INT,
            qty_reserved INT, reorder_point INT, last_counted DATE,
            PRIMARY KEY (product_id, warehouse)
        )""",
        """CREATE TABLE logs (
            log_id BIGINT PRIMARY KEY, level TEXT, service TEXT,
            message TEXT, trace_id TEXT, latency_ms NUMERIC(10,2),
            status_code INT, host TEXT, region TEXT, created_at TIMESTAMP
        )""",
    ]

    for stmt in ddl_stmts:
        await conn.execute(stmt)

    async def _bulk_insert(tname: str, df: pl.DataFrame, transform=None) -> None:
        rows = df.to_dicts()
        if transform:
            rows = [transform(r) for r in rows]
        if not rows:
            return
        cols   = list(rows[0].keys())
        params = ", ".join([f"${i+1}" for i in range(len(cols))])
        col_str = ", ".join(f'"{c}"' for c in cols)
        sql = f'INSERT INTO {tname} ({col_str}) VALUES ({params}) ON CONFLICT DO NOTHING'
        batch_size = 2_000
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            await conn.executemany(sql, [[r[c] for c in cols] for r in batch])
        print(f"   {tname}: {len(rows):,} rows")

    from datetime import date as _date, datetime as _datetime

    # DATE columns (need date object, not str)
    _DATE_COLS = {
        "employees":  {"joined", "terminated"},
        "products":   {"created_at"},
        "customers":  {"signup_date", "last_active"},
        "reviews":    {"created_at"},
        "inventory":  {"last_counted"},
    }
    # TIMESTAMP columns (need datetime object)
    _TS_COLS = {
        "orders":        {"ordered_at", "shipped_at", "delivered_at"},
        "transactions":  {"created_at", "settled_at"},
        "logs":          {"created_at"},
    }

    def _coerce_row(tname: str, r: dict) -> dict:
        date_cols = _DATE_COLS.get(tname, set())
        ts_cols   = _TS_COLS.get(tname, set())
        for k, v in r.items():
            if v is None:
                continue
            if k in date_cols:
                if isinstance(v, str):
                    try:
                        r[k] = _date.fromisoformat(v[:10])
                    except Exception:
                        r[k] = None
                elif isinstance(v, _datetime):
                    r[k] = v.date()
            elif k in ts_cols:
                if isinstance(v, str):
                    try:
                        r[k] = _datetime.fromisoformat(v)
                    except Exception:
                        r[k] = None
        return r

    def _fix_ip(r: dict) -> dict:
        ip = r.get("ip")
        if ip and not all(part.isdigit() for part in str(ip).split(".")):
            r["ip"] = None
        return r

    await _bulk_insert("departments", tables["departments"])
    await _bulk_insert("products",    tables["products"],
                       lambda r: _coerce_row("products", r))
    await _bulk_insert("customers",   tables["customers"],
                       lambda r: _coerce_row("customers", r))
    await _bulk_insert("employees",   tables["employees"],
                       lambda r: _coerce_row("employees", r))
    await _bulk_insert("orders",      tables["orders"],
                       lambda r: _coerce_row("orders", r))
    await _bulk_insert("transactions", tables["transactions"],
                       lambda r: _fix_ip(_coerce_row("transactions", r)))
    await _bulk_insert("reviews",     tables["reviews"],
                       lambda r: _coerce_row("reviews", r))
    await _bulk_insert("inventory",   tables["inventory"],
                       lambda r: _coerce_row("inventory", r))
    await _bulk_insert("logs",        tables["logs"],
                       lambda r: _coerce_row("logs", r))

    # indexes
    index_stmts = [
        "CREATE INDEX IF NOT EXISTS idx_emp_dept ON employees(dept)",
        "CREATE INDEX IF NOT EXISTS idx_emp_status ON employees(status)",
        "CREATE INDEX IF NOT EXISTS idx_ord_customer ON orders(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_ord_product ON orders(product_id)",
        "CREATE INDEX IF NOT EXISTS idx_ord_status ON orders(status)",
        "CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_txn_fraud ON transactions(is_fraud)",
        "CREATE INDEX IF NOT EXISTS idx_log_level ON logs(level)",
        "CREATE INDEX IF NOT EXISTS idx_log_service ON logs(service)",
    ]
    for stmt in index_stmts:
        await conn.execute(stmt)

    await conn.close()
    print("  ✓ PostgreSQL seeded + indexes created")


async def seed_mysql(tables: dict[str, pl.DataFrame]) -> None:
    import asyncmy
    print("→ MySQL", flush=True)
    conn = await asyncmy.connect(
        host="localhost", port=3307,
        user="testuser", password="testpass", db="testdb",
    )

    table_order = ["departments", "products", "customers", "employees",
                   "orders", "transactions", "reviews", "inventory", "logs"]

    ddl = {
        "departments": """CREATE TABLE `departments` (
            `dept_id` INT PRIMARY KEY, `name` VARCHAR(100), `budget` DECIMAL(14,2),
            `headcount` INT, `cost_center` VARCHAR(20), `location` VARCHAR(100), `manager_id` INT
        ) ENGINE=InnoDB""",
        "employees": """CREATE TABLE `employees` (
            `employee_id` BIGINT PRIMARY KEY, `first_name` VARCHAR(100), `last_name` VARCHAR(100),
            `full_name` VARCHAR(200), `email` VARCHAR(200), `phone` VARCHAR(30), `age` INT,
            `dept` VARCHAR(100), `dept_id` INT, `level` VARCHAR(50), `city` VARCHAR(100),
            `country` VARCHAR(10), `salary` DECIMAL(12,2), `bonus_pct` DECIMAL(6,4),
            `performance` DECIMAL(8,4), `status` VARCHAR(30), `joined` DATE, `terminated` DATE,
            `is_remote` TINYINT(1), `manager_id` BIGINT, `tags` TEXT
        ) ENGINE=InnoDB""",
        "products": """CREATE TABLE `products` (
            `product_id` INT PRIMARY KEY, `sku` VARCHAR(50), `name` VARCHAR(255),
            `category` VARCHAR(100), `subcategory` VARCHAR(100), `brand` VARCHAR(100),
            `cost` DECIMAL(12,2), `price` DECIMAL(12,2), `weight_kg` DECIMAL(8,3),
            `stock` INT, `rating` DECIMAL(4,2), `review_count` INT,
            `active` TINYINT(1), `created_at` DATE
        ) ENGINE=InnoDB""",
        "customers": """CREATE TABLE `customers` (
            `customer_id` INT PRIMARY KEY, `full_name` VARCHAR(200), `email` VARCHAR(200),
            `phone` VARCHAR(30), `age` INT, `gender` VARCHAR(30), `city` VARCHAR(100),
            `country` VARCHAR(10), `segment` VARCHAR(50), `lifetime_value` DECIMAL(14,2),
            `signup_date` DATE, `last_active` DATE, `channel` VARCHAR(50), `referral_code` VARCHAR(20)
        ) ENGINE=InnoDB""",
        "orders": """CREATE TABLE `orders` (
            `order_id` BIGINT PRIMARY KEY, `order_ref` VARCHAR(50), `customer_id` INT,
            `product_id` INT, `qty` INT, `unit_price` DECIMAL(12,2), `discount` DECIMAL(6,4),
            `gross_amount` DECIMAL(14,2), `net_amount` DECIMAL(14,2), `payment_method` VARCHAR(50),
            `status` VARCHAR(50), `channel` VARCHAR(50), `city` VARCHAR(100), `country` VARCHAR(10),
            `ordered_at` DATETIME, `shipped_at` DATETIME, `delivered_at` DATETIME, `notes` TEXT
        ) ENGINE=InnoDB""",
        "transactions": """CREATE TABLE `transactions` (
            `txn_id` BIGINT PRIMARY KEY, `txn_ref` VARCHAR(60), `customer_id` INT,
            `amount` DECIMAL(16,2), `currency` CHAR(3), `method` VARCHAR(50), `status` VARCHAR(30),
            `gateway` VARCHAR(50), `fee` DECIMAL(12,4), `country` VARCHAR(10), `ip` VARCHAR(45),
            `device` VARCHAR(30), `is_fraud` TINYINT(1), `created_at` DATETIME, `settled_at` DATETIME
        ) ENGINE=InnoDB""",
        "reviews": """CREATE TABLE `reviews` (
            `review_id` INT PRIMARY KEY, `product_id` INT, `customer_id` INT,
            `rating` TINYINT, `sentiment` VARCHAR(20), `title` TEXT, `body` TEXT,
            `helpful` INT, `verified` TINYINT(1), `created_at` DATE
        ) ENGINE=InnoDB""",
        "inventory": """CREATE TABLE `inventory` (
            `product_id` INT, `warehouse` VARCHAR(30), `qty_on_hand` INT,
            `qty_reserved` INT, `reorder_point` INT, `last_counted` DATE,
            PRIMARY KEY (`product_id`, `warehouse`)
        ) ENGINE=InnoDB""",
        "logs": """CREATE TABLE `logs` (
            `log_id` BIGINT PRIMARY KEY, `level` VARCHAR(20), `service` VARCHAR(50),
            `message` TEXT, `trace_id` VARCHAR(40), `latency_ms` DECIMAL(10,2),
            `status_code` INT, `host` VARCHAR(100), `region` VARCHAR(50), `created_at` DATETIME
        ) ENGINE=InnoDB""",
    }

    async with conn.cursor() as cur:
        await cur.execute("SET FOREIGN_KEY_CHECKS=0")
        for tname in reversed(table_order):
            await cur.execute(f"DROP TABLE IF EXISTS `{tname}`")
        await cur.execute("SET FOREIGN_KEY_CHECKS=1")
        for tname in table_order:
            await cur.execute(ddl[tname])

        for tname in table_order:
            df = tables.get(tname)
            if df is None:
                continue
            rows = df.to_dicts()
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, bool):
                        row[k] = int(v)
            cols = list(rows[0].keys())
            placeholders = ", ".join(["%s"] * len(cols))
            col_str = ", ".join(f"`{c}`" for c in cols)
            sql = f"INSERT IGNORE INTO {tname} ({col_str}) VALUES ({placeholders})"
            batch_size = 3_000
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                await cur.executemany(sql, [[r[c] for c in cols] for r in batch])
            print(f"   {tname}: {len(rows):,} rows")

    await conn.commit()
    conn.close()
    print("  ✓ MySQL seeded")


async def seed_s3(tables: dict[str, pl.DataFrame]) -> None:
    import io
    import aioboto3
    print("→ MinIO/S3", flush=True)
    session = aioboto3.Session(
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    )
    async with session.client("s3", endpoint_url="http://localhost:9000") as s3:
        # make sure bucket exists
        try:
            await s3.head_bucket(Bucket="testbucket")
        except Exception:
            await s3.create_bucket(Bucket="testbucket")

        for name, df in tables.items():
            # CSV
            buf = io.BytesIO()
            df.write_csv(buf); buf.seek(0)
            await s3.put_object(Bucket="testbucket", Key=f"data/{name}.csv", Body=buf.read())

            # Parquet
            buf = io.BytesIO()
            df.write_parquet(buf); buf.seek(0)
            await s3.put_object(Bucket="testbucket", Key=f"data/{name}.parquet", Body=buf.read())

            # NDJSON
            buf = io.BytesIO()
            df.write_ndjson(buf); buf.seek(0)
            await s3.put_object(Bucket="testbucket", Key=f"data/{name}.ndjson", Body=buf.read())
            print(f"   s3://testbucket/data/{name}.{{csv,parquet,ndjson}}")

        # Partitioned employees by dept
        emp_df = tables["employees"]
        for dept in _DEPTS:
            part = emp_df.filter(pl.col("dept") == dept)
            buf = io.BytesIO()
            part.write_parquet(buf); buf.seek(0)
            key = f"partitioned/employees/dept={dept.lower().replace(' ','_')}/data.parquet"
            await s3.put_object(Bucket="testbucket", Key=key, Body=buf.read())
        print(f"   partitioned employees → {len(_DEPTS)} keys under partitioned/employees/")

        # Partitioned orders by month
        ord_df = tables["orders"]
        ord_with_month = ord_df.with_columns(
            pl.col("ordered_at").str.slice(0, 7).alias("ym")
        )
        for ym in sorted(ord_with_month["ym"].unique().to_list()):
            part = ord_with_month.filter(pl.col("ym") == ym).drop("ym")
            buf = io.BytesIO()
            part.write_parquet(buf); buf.seek(0)
            safe = ym.replace("-", "_")
            key  = f"partitioned/orders/month={safe}/data.parquet"
            await s3.put_object(Bucket="testbucket", Key=key, Body=buf.read())
        n_months = ord_with_month["ym"].n_unique()
        print(f"   partitioned orders → {n_months} keys under partitioned/orders/")

    print("  ✓ MinIO/S3 seeded")


async def seed_kafka(tables: dict[str, pl.DataFrame]) -> None:
    from aiokafka import AIOKafkaProducer
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    print("→ Kafka", flush=True)

    topics_cfg = {
        "employees":    {"partitions": 4, "replication": 1},
        "orders":       {"partitions": 8, "replication": 1},
        "transactions": {"partitions": 8, "replication": 1},
        "events":       {"partitions": 16, "replication": 1},
        "logs":         {"partitions": 4, "replication": 1},
    }

    # Create topics
    admin = AIOKafkaAdminClient(bootstrap_servers="localhost:9092")
    await admin.start()
    try:
        existing = set(await admin.list_topics())
        new_topics = [
            NewTopic(name=t, num_partitions=cfg["partitions"], replication_factor=cfg["replication"])
            for t, cfg in topics_cfg.items() if t not in existing
        ]
        if new_topics:
            await admin.create_topics(new_topics)
    except Exception as e:
        print(f"   topic creation warning: {e}")
    finally:
        await admin.close()

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        request_timeout_ms=30_000,
        compression_type="gzip",
    )
    await producer.start()
    try:
        for topic, cfg in topics_cfg.items():
            df = tables.get(topic)
            if df is None:
                continue
            # For large tables, send a representative sample to Kafka
            sample_size = min(len(df), 20_000)
            sample = df.sample(n=sample_size, seed=SEED).to_dicts()
            for row in sample:
                key_col = {
                    "employees":    "employee_id",
                    "orders":       "order_id",
                    "transactions": "txn_id",
                    "events":       "event_id",
                    "logs":         "log_id",
                }.get(topic, "id")
                key = str(row.get(key_col, "")).encode()
                val = json.dumps(row, default=str).encode()
                await producer.send(topic, value=val, key=key)
            await producer.flush()
            print(f"   topic={topic}: {sample_size:,} messages ({cfg['partitions']} partitions)")
    finally:
        await producer.stop()

    print("  ✓ Kafka seeded")


def seed_configs() -> None:
    print("→ Configs", flush=True)
    cfg_dir = FIXTURE_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "sqlite.json": {
            "path": "tests/fixtures/test.db",
            "read_only": True,
        },
        "local.json": {
            "base_path": "tests/fixtures",
            "read_only": True,
        },
        "local_dirty.json": {
            "base_path": "tests/fixtures",
        },
        "postgres.json": {
            "host": "localhost", "port": 5433,
            "database": "testdb", "user": "testuser", "password": "testpass",
        },
        "mysql.json": {
            "host": "localhost", "port": 3307,
            "database": "testdb", "user": "testuser", "password": "testpass",
        },
        "s3.json": {
            "bucket": "testbucket", "region": "us-east-1",
            "access_key": "minioadmin", "secret_key": "minioadmin",
            "endpoint_url": "http://localhost:9000",
        },
        "kafka.json": {
            "brokers": ["localhost:9092"],
            "group_id": "datapill-test",
            "auto_offset_reset": "earliest",
        },
        "rest_api.json": {
            "base_url": "https://jsonplaceholder.typicode.com",
        },
    }

    for fname, cfg in configs.items():
        (cfg_dir / fname).write_text(json.dumps(cfg, indent=2))

    # -- ops files for preprocess testing
    ops_dir = FIXTURE_DIR / "ops"
    ops_dir.mkdir(exist_ok=True)

    ops_examples = {
        "clean_employees.json": [
            {"group": "clean",     "type": "drop_null",     "cols": ["email", "salary"]},
            {"group": "clean",     "type": "clip",          "cols": ["age"],    "min": 16, "max": 70},
            {"group": "clean",     "type": "clip",          "cols": ["salary"], "min": 100, "max": 20000},
            {"group": "clean",     "type": "fill_null",     "cols": ["bonus_pct"], "value": 0},
            {"group": "clean",     "type": "impute",        "cols": ["performance"], "strategy": "median"},
            {"group": "schema",    "type": "drop_columns",  "cols": ["tags", "phone"]},
            {"group": "parse",     "type": "trim",          "cols": ["full_name", "email"]},
            {"group": "parse",     "type": "lower",         "cols": ["email"]},
            {"group": "transform", "type": "log_transform", "cols": ["salary"], "base": "log10"},
            {"group": "reshape",   "type": "dedup",         "cols": ["email"], "keep": "first"},
        ],
        "feature_engineering.json": [
            {"group": "transform", "type": "normalize",     "cols": ["salary", "performance"]},
            {"group": "transform", "type": "standardize",   "cols": ["age", "bonus_pct"]},
            {"group": "transform", "type": "bin",           "col": "age", "breaks": [0, 25, 35, 45, 55, 100]},
            {"group": "compose",   "type": "feature_cross", "col_a": "dept", "col_b": "level"},
            {"group": "transform", "type": "encode",        "col": "dept",   "method": "onehot"},
            {"group": "transform", "type": "encode",        "col": "status", "method": "label"},
            {"group": "reshape",   "type": "sort",          "cols": ["salary"], "descending": True}
        ],
        "clean_orders.json": [
            {"group": "clean",   "type": "drop_null",          "cols": ["customer_id", "product_id"]},
            {"group": "clean",   "type": "clip",               "cols": ["qty"], "min": 1, "max": 100},
            {"group": "clean",   "type": "winsorize",          "cols": ["net_amount"], "lower": 0.01, "upper": 0.99},
            {"group": "parse",   "type": "parse_datetime",     "cols": ["ordered_at"], "format": "%Y-%m-%d %H:%M:%S"},
            {"group": "parse",   "type": "extract_datetime_part", "col": "ordered_at", "parts": ["year", "month", "dow"]},
            {"group": "reshape", "type": "dedup",              "cols": ["order_ref"], "keep": "first"},
            {"group": "reshape", "type": "sort",               "cols": ["ordered_at"], "descending": True}
        ],
        "aggregate_orders.json": [
            {"group": "compose",   "type": "group_agg",     "by": ["country", "status"],
             "aggs": [{"col": "net_amount", "fn": "sum", "out": "total_revenue"},
                      {"col": "order_id",   "fn": "count", "out": "order_count"},
                      {"col": "qty",        "fn": "mean",  "out": "avg_qty"}]},
        ],
    }

    for fname, ops in ops_examples.items():
        (ops_dir / fname).write_text(json.dumps(ops, indent=2))

    print(f"  ✓ {cfg_dir}/  ({len(configs)} config files)")
    print(f"  ✓ {ops_dir}/  ({len(ops_examples)} ops files)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="datapill - seed test data")
    parser.add_argument(
        "--only", nargs="+",
        choices=["postgres", "mysql", "s3", "kafka", "sqlite", "local"],
        help="seed only the specified targets (default: all)",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="skip DataFrame generation (useful if re-running a single target)",
    )
    args = parser.parse_args()
    only = set(args.only) if args.only else {"postgres", "mysql", "s3", "kafka", "sqlite", "local"}

    print("=" * 60)
    print("datapill  —  seeding test data")
    print("=" * 60)

    seed_configs()

    print("\nGenerating DataFrames…")
    tables = _build_all()

    print()
    errors: list[tuple[str, Exception]] = []

    if "local" in only:
        try:
            seed_local_files(tables)
        except Exception as exc:
            errors.append(("local", exc))

    if "sqlite" in only:
        try:
            seed_sqlite(tables)
        except Exception as exc:
            errors.append(("sqlite", exc))

    async_targets: list[tuple[str, Any]] = []
    if "postgres" in only:
        async_targets.append(("postgres", seed_postgres(tables)))
    if "mysql" in only:
        async_targets.append(("mysql", seed_mysql(tables)))
    if "s3" in only:
        async_targets.append(("s3", seed_s3(tables)))
    if "kafka" in only:
        async_targets.append(("kafka", seed_kafka(tables)))

    for name, coro in async_targets:
        print()
        try:
            await coro
        except Exception as exc:
            errors.append((name, exc))
            print(f"  ✗ {name}: {exc}")

    print()
    print("=" * 60)
    if errors:
        print(f"done with {len(errors)} error(s):")
        for name, exc in errors:
            print(f"  ✗ {name}: {exc}")
    else:
        total = sum(len(df) for df in tables.values())
        print(f"done — {total:,} total rows seeded across {len(tables)} tables")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())