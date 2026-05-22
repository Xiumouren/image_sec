from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.shared.config import DEFAULT_SQLITE_PATH, SQLITE_TIMEOUT_SECONDS


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS source_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_filename TEXT NOT NULL,
        archived_path TEXT NOT NULL,
        archived_relative_path TEXT NOT NULL,
        archived_url TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        phash TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_analysis_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id INTEGER NOT NULL,
        predicted_class_index INTEGER NOT NULL,
        predicted_label TEXT NOT NULL,
        has_exif INTEGER,
        exif_datetime_original TEXT,
        exif_make TEXT,
        exif_model TEXT,
        exif_software TEXT,
        ela_supported_for_detection INTEGER,
        ela_anomaly_ratio REAL,
        ela_is_tampered INTEGER,
        detection_payload TEXT NOT NULL,
        source_payload TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(asset_id) REFERENCES source_assets(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_asset_embeddings (
        asset_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        embedding_dim INTEGER NOT NULL,
        embedding_path TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(asset_id, model_name),
        FOREIGN KEY(asset_id) REFERENCES source_assets(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_roi_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id INTEGER NOT NULL,
        target_label TEXT NOT NULL,
        bbox_json TEXT NOT NULL,
        coverage_ratio REAL NOT NULL,
        roi_path TEXT NOT NULL,
        roi_relative_path TEXT NOT NULL,
        roi_url TEXT NOT NULL,
        roi_sha256 TEXT NOT NULL,
        roi_phash TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(asset_id) REFERENCES source_assets(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_roi_embeddings (
        roi_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        embedding_dim INTEGER NOT NULL,
        embedding_path TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(roi_id, model_name),
        FOREIGN KEY(roi_id) REFERENCES source_roi_assets(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_source_assets_sha256 ON source_assets(sha256)",
    "CREATE INDEX IF NOT EXISTS idx_source_assets_phash ON source_assets(phash)",
    "CREATE INDEX IF NOT EXISTS idx_source_analysis_asset_id ON source_analysis_records(asset_id)",
    "CREATE INDEX IF NOT EXISTS idx_source_asset_embeddings_model ON source_asset_embeddings(model_name)",
    "CREATE INDEX IF NOT EXISTS idx_source_roi_assets_asset_id ON source_roi_assets(asset_id)",
    "CREATE INDEX IF NOT EXISTS idx_source_roi_assets_sha256 ON source_roi_assets(roi_sha256)",
    "CREATE INDEX IF NOT EXISTS idx_source_roi_assets_phash ON source_roi_assets(roi_phash)",
    "CREATE INDEX IF NOT EXISTS idx_source_roi_embeddings_model ON source_roi_embeddings(model_name)",
)

MIGRATION_COLUMNS = {
    "source_analysis_records": {
        "has_exif": "INTEGER",
        "exif_datetime_original": "TEXT",
        "exif_make": "TEXT",
        "exif_model": "TEXT",
        "exif_software": "TEXT",
        "ela_supported_for_detection": "INTEGER",
        "ela_anomaly_ratio": "REAL",
        "ela_is_tampered": "INTEGER",
    }
}


def get_connection(database_path: Path = DEFAULT_SQLITE_PATH) -> sqlite3.Connection:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=SQLITE_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={int(SQLITE_TIMEOUT_SECONDS * 1000)}")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_database(database_path: Path = DEFAULT_SQLITE_PATH) -> None:
    with get_connection(database_path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_columns(connection)
        connection.commit()


def _ensure_columns(connection: sqlite3.Connection) -> None:
    for table_name, columns in MIGRATION_COLUMNS.items():
        existing_columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_type in columns.items():
            if column_name not in existing_columns:
                connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
