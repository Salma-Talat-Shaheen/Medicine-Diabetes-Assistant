#!/usr/bin/env python3
"""
Create the `Patients` table in a Postgres (Supabase) database.

This script follows the environment variable names `src/main.py`:
- `user`, `password`, `host`, `port`, `dbname` (loaded from `.env` via `python-dotenv`).

Usage:
  python3 scripts/create_patients_table_pg.py

"""
import os
from dotenv import load_dotenv
import psycopg2


load_dotenv()

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME)


def create_table(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS Patients (
        Patient_ID TEXT PRIMARY KEY,
        Name TEXT,
        Age INTEGER,
        Gender TEXT,
        Height_cm REAL,
        Weight_kg REAL,
        Diabetes_Type TEXT,
        Duration_Years REAL,
        Comorbidities TEXT,
        Latest_HbA1c REAL,
        Current_Meds TEXT,
        eGFR_ml_min REAL,
        Recent_Symptoms TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    """
    cur = conn.cursor()
    cur.execute(ddl)
    conn.commit()
    cur.close()


def main():
    try:
        conn = get_conn()
        print("Connected to Postgres — creating Patients table if missing...")
        create_table(conn)
        print("Patients table created (or already existed).")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
