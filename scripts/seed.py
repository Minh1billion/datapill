from __future__ import annotations

import asyncio
import json
import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

random.seed(42)
_names  = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
_depts  = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
_cities = ["Hanoi", "HCMC", "Danang", "Hue", "Cantho"]

ROWS = [
    {
        "id":     i + 1,
        "name":   random.choice(_names),
        "age":    random.randint(22, 55),
        "salary": round(random.uniform(800, 5000), 2),
        "dept":   random.choice(_depts),
        "city":   random.choice(_cities),
        "active": random.choice([True, False]),
        "joined": date(2018, 1, 1) + timedelta(days=random.randint(0, 2000)),
    }
    for i in range(100)
]
DF = pl.DataFrame(ROWS)


async def seed_postgres() -> None:
    import asyncpg
    print("→ PostgreSQL", end=" ", flush=True)
    conn = await asyncpg.connect(
        host="localhost", port=5433,
        user="testuser", password="testpass", database="testdb",
    )
    await conn.execute("""
        DROP TABLE IF EXISTS employees;
        CREATE TABLE employees (
            id      SERIAL PRIMARY KEY,
            name    TEXT,
            age     INTEGER,
            salary  NUMERIC(10,2),
            dept    TEXT,
            city    TEXT,
            active  BOOLEAN,
            joined  DATE
        );
        DROP TABLE IF EXISTS departments;
        CREATE TABLE departments (
            name      TEXT PRIMARY KEY,
            budget    NUMERIC(12,2),
            headcount INTEGER
        );
    """)
    await conn.executemany(
        "INSERT INTO employees(id,name,age,salary,dept,city,active,joined) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8)",
        [(r["id"], r["name"], r["age"], r["salary"],
          r["dept"], r["city"], r["active"], r["joined"]) for r in ROWS],
    )
    await conn.executemany(
        "INSERT INTO departments VALUES($1,$2,$3)",
        [("Engineering",500000,30),("Marketing",200000,15),
         ("Sales",350000,20),("HR",150000,10),("Finance",250000,12)],
    )
    await conn.close()
    print("✓ employees(100) + departments(5)")


async def seed_mysql() -> None:
    import asyncmy
    print("→ MySQL", end=" ", flush=True)
    conn = await asyncmy.connect(
        host="localhost", port=3307,
        user="testuser", password="testpass", db="testdb",
    )
    async with conn.cursor() as cur:
        await cur.execute("DROP TABLE IF EXISTS employees")
        await cur.execute("""
            CREATE TABLE employees (
                id      INT PRIMARY KEY,
                name    VARCHAR(100),
                age     INT,
                salary  DECIMAL(10,2),
                dept    VARCHAR(100),
                city    VARCHAR(100),
                active  BOOLEAN,
                joined  DATE
            )
        """)
        await cur.executemany(
            "INSERT INTO employees VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
            [(r["id"], r["name"], r["age"], r["salary"],
              r["dept"], r["city"], r["active"], r["joined"]) for r in ROWS],
        )
        await cur.execute("DROP TABLE IF EXISTS departments")
        await cur.execute("""
            CREATE TABLE departments (
                name      VARCHAR(100) PRIMARY KEY,
                budget    DECIMAL(12,2),
                headcount INT
            )
        """)
        await cur.executemany(
            "INSERT INTO departments VALUES(%s,%s,%s)",
            [("Engineering",500000,30),("Marketing",200000,15),
             ("Sales",350000,20),("HR",150000,10),("Finance",250000,12)],
        )
    await conn.commit()
    conn.close()
    print("✓ employees(100) + departments(5)")


async def seed_s3() -> None:
    import io
    import aioboto3
    print("→ MinIO/S3", end=" ", flush=True)
    session = aioboto3.Session(
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    )
    async with session.client("s3", endpoint_url="http://localhost:9000") as s3:
        buf = io.BytesIO()
        DF.write_csv(buf); buf.seek(0)
        await s3.put_object(Bucket="testbucket", Key="data/employees.csv", Body=buf.read())

        buf = io.BytesIO()
        DF.write_parquet(buf); buf.seek(0)
        await s3.put_object(Bucket="testbucket", Key="data/employees.parquet", Body=buf.read())

        buf = io.BytesIO()
        DF.write_ndjson(buf); buf.seek(0)
        await s3.put_object(Bucket="testbucket", Key="data/employees.ndjson", Body=buf.read())

    print("✓ data/employees.{csv,parquet,ndjson}")


async def seed_kafka() -> None:
    from aiokafka import AIOKafkaProducer
    print("→ Kafka", end=" ", flush=True)

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        request_timeout_ms=10000,
    )
    await producer.start()
    try:
        for row in ROWS:
            await producer.send(
                "employees",
                value=json.dumps(row, default=str).encode(),
                key=str(row["id"]).encode(),
            )
        await producer.flush()
    finally:
        await producer.stop()
    print("✓ topic=employees (100 messages)")


def seed_sqlite() -> None:
    import sqlite3
    print("→ SQLite", end=" ", flush=True)
    db = FIXTURE_DIR / "test.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("DROP TABLE IF EXISTS employees")
    con.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY, name TEXT, age INTEGER,
            salary REAL, dept TEXT, city TEXT, active INTEGER, joined TEXT
        )
    """)
    con.executemany(
        "INSERT INTO employees VALUES (:id,:name,:age,:salary,:dept,:city,:active,:joined)",
        [{**r, "active": int(r["active"]), "joined": str(r["joined"])} for r in ROWS],
    )
    con.execute("DROP TABLE IF EXISTS departments")
    con.execute("CREATE TABLE departments (name TEXT PRIMARY KEY, budget REAL, headcount INTEGER)")
    con.executemany("INSERT INTO departments VALUES (?,?,?)", [
        ("Engineering",500000,30), ("Marketing",200000,15),
        ("Sales",350000,20), ("HR",150000,10), ("Finance",250000,12),
    ])
    con.commit(); con.close()
    print(f"✓ {db}  employees(100) + departments(5)")


def seed_local_files() -> None:
    print("→ Local files", end=" ", flush=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    DF.write_csv(FIXTURE_DIR / "data.csv")
    DF.write_parquet(FIXTURE_DIR / "data.parquet")
    DF.write_ndjson(FIXTURE_DIR / "data.ndjson")
    big = pl.concat([DF] * 10).with_columns(
        pl.arange(0, len(DF) * 10, eager=True).alias("id")
    )
    big.write_csv(FIXTURE_DIR / "data_1k.csv")
    print(f"✓ {FIXTURE_DIR}/  csv + parquet + ndjson + data_1k.csv")


def seed_configs() -> None:
    import json
    print("→ Configs", end=" ", flush=True)
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
        "postgres.json": {
            "host": "localhost",
            "port": 5433,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        },
        "mysql.json": {
            "host": "localhost",
            "port": 3307,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        },
        "s3.json": {
            "bucket": "testbucket",
            "region": "us-east-1",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
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

    print(f"✓ {cfg_dir}/  ({len(configs)} files)")


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+",
                        choices=["postgres","mysql","s3","kafka","sqlite","local"])
    args = parser.parse_args()
    only = set(args.only) if args.only else {"postgres","mysql","s3","kafka","sqlite","local"}

    print("=" * 50)
    print("datapill - seeding test services")
    print("=" * 50)

    seed_configs()
    if "local" in only:
        seed_local_files()
    if "sqlite" in only:
        seed_sqlite()

    tasks = []
    if "postgres" in only:
        tasks.append(("postgres", seed_postgres()))
    if "mysql" in only:
        tasks.append(("mysql", seed_mysql()))
    if "s3" in only:
        tasks.append(("s3", seed_s3()))
    if "kafka" in only:
        tasks.append(("kafka", seed_kafka()))

    for name, coro in tasks:
        try:
            await coro
        except Exception as exc:
            print(f"✗ {name}: {exc}")

    print("=" * 50)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())