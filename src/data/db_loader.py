"""
Database Data Loader
====================
Loads churn data from a real (production) database instead of the synthetic
generator. DB-agnostic: works with any SQLAlchemy-supported backend
(PostgreSQL recommended; SQLite handy for local testing).

Configuration is read from environment variables (see .env.example):
  - DB_URL        Full SQLAlchemy connection string. Takes precedence.
  - DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD
                  Used to assemble a PostgreSQL URL when DB_URL is unset.
  - SQL_QUERY_PATH  Path to a .sql file containing the SELECT to run.
                    Defaults to config/churn_query.sql.
"""

import logging
import os
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv optional; env vars may be set externally
    pass

logger = logging.getLogger("data.db_loader")

# Columns the downstream pipeline (feature engineering + model) requires.
REQUIRED_COLUMNS = [
    "customer_id",
    "subscription_type",
    "monthly_revenue",
    "tenure_months",
    "country",
    "industry",
    "sessions_30d",
    "sessions_90d",
    "days_since_last_login",
    "days_since_last_core_action",
    "login_frequency_change",
    "session_duration_change",
    "ticket_count",
    "support_sentiment",
    "error_rate",
    "failed_api_calls",
    "rage_click_count",
    "billing_page_visits",
    "downgrade_page_visits",
    "invoice_download_frequency",
    "trial_expiry_days",
    "tier_capacity_utilization",
    "export_frequency",
    "api_usage",
    "api_limit_usage",
    "seat_limit_usage",
    # "churned" is required for TRAINING but optional for scoring live data.
]

DEFAULT_QUERY_PATH = Path("config/churn_query.sql")


def build_db_url() -> str:
    """Return a SQLAlchemy connection string from environment variables."""
    url = os.getenv("DB_URL")
    if url:
        return url

    host = os.getenv("DB_HOST")
    name = os.getenv("DB_NAME")
    if not (host and name):
        raise ValueError(
            "No database configured. Set DB_URL, or DB_HOST/DB_NAME "
            "(plus DB_USER/DB_PASSWORD/DB_PORT) in your .env file."
        )

    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")
    port = os.getenv("DB_PORT", "5432")
    auth = f"{user}:{password}@" if user else ""
    return f"postgresql+psycopg2://{auth}{host}:{port}/{name}"


def load_query(query_path: str = None) -> str:
    """Read the SQL query text from a .sql file."""
    path = Path(query_path) if query_path else Path(
        os.getenv("SQL_QUERY_PATH", str(DEFAULT_QUERY_PATH))
    )
    if not path.exists():
        raise FileNotFoundError(
            f"SQL query file not found: {path}. "
            f"Create it or set SQL_QUERY_PATH in .env."
        )
    sql = path.read_text().strip()
    if not sql:
        raise ValueError(f"SQL query file is empty: {path}")
    return sql


def validate_schema(df: pd.DataFrame, require_label: bool = True) -> None:
    """Raise if required columns are missing from the loaded DataFrame."""
    required = list(REQUIRED_COLUMNS)
    if require_label:
        required = required + ["churned"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Database result is missing required columns: {missing}. "
            f"Adjust your SQL query (aliases) to match the expected schema."
        )


def load_from_db(
    query_path: str = None,
    require_label: bool = True,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Connect to the configured database, run the SQL query, validate the
    schema, and return a DataFrame. Optionally cache it to a CSV.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SQLAlchemy is required for database loading. "
            "Install it with: pip install sqlalchemy psycopg2-binary"
        ) from exc

    db_url = build_db_url()
    sql = load_query(query_path)

    # Redact credentials before logging.
    safe_url = db_url.split("@")[-1] if "@" in db_url else db_url
    logger.info(f"Connecting to database: …@{safe_url}")

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
    finally:
        engine.dispose()

    logger.info(f"Loaded {len(df):,} rows from database")
    validate_schema(df, require_label=require_label)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Cached database snapshot to {output_path}")

    return df
