from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from backend.shared.config import DEFAULT_SQLITE_PATH
from backend.storage_layer.sqlite.db import get_connection, initialize_database
from backend.storage_layer.sqlite.models import (
    PersistSourceAnalysisResult,
    SourceAssetEmbeddingRecord,
    SourceAssetRecord,
    SourceRoiAssetRecord,
    SourceRoiEmbeddingRecord,
)


class SourceRepository:
    def __init__(self, database_path: Path = DEFAULT_SQLITE_PATH) -> None:
        self.database_path = Path(database_path)
        initialize_database(self.database_path)

    def list_assets(self) -> list[SourceAssetRecord]:
        return self.list_assets_except_sha256()

    def list_assets_except_sha256(self, sha256: str | None = None) -> list[SourceAssetRecord]:
        query = """
        SELECT id, original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size, created_at
        FROM source_assets
        {where_clause}
        ORDER BY id ASC
        """
        where_clause = "WHERE sha256 != ?" if sha256 else ""
        with get_connection(self.database_path) as connection:
            rows = connection.execute(query.format(where_clause=where_clause), ((sha256,) if sha256 else ())).fetchall()
        return [self._row_to_asset(row) for row in rows]

    def find_asset_by_sha256(self, sha256: str) -> SourceAssetRecord | None:
        query = """
        SELECT id, original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size, created_at
        FROM source_assets
        WHERE sha256 = ?
        ORDER BY id ASC
        LIMIT 1
        """
        with get_connection(self.database_path) as connection:
            row = connection.execute(query, (sha256,)).fetchone()
        return self._row_to_asset(row) if row is not None else None

    def upsert_asset(self, *, asset_payload: dict) -> tuple[SourceAssetRecord, bool]:
        with get_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN")
                asset_record = self._find_asset_by_sha256_with_connection(connection, asset_payload["sha256"])
                created = asset_record is None
                if asset_record is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO source_assets (
                            original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            asset_payload["original_filename"],
                            asset_payload["archived_path"],
                            asset_payload["archived_relative_path"],
                            asset_payload["archived_url"],
                            asset_payload["sha256"],
                            asset_payload["phash"],
                            asset_payload["file_size"],
                        ),
                    )
                    asset_row = connection.execute(
                        """
                        SELECT id, original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size, created_at
                        FROM source_assets
                        WHERE id = ?
                        """,
                        (int(cursor.lastrowid),),
                    ).fetchone()
                    asset_record = self._row_to_asset(asset_row)
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise
        return asset_record, created

    def find_embedding(self, asset_id: int, model_name: str) -> SourceAssetEmbeddingRecord | None:
        query = """
        SELECT asset_id, model_name, embedding_dim, embedding_path, created_at
        FROM source_asset_embeddings
        WHERE asset_id = ? AND model_name = ?
        """
        with get_connection(self.database_path) as connection:
            row = connection.execute(query, (asset_id, model_name)).fetchone()
        return self._row_to_embedding(row) if row is not None else None

    def upsert_embedding(
        self,
        *,
        asset_id: int,
        model_name: str,
        embedding_dim: int,
        embedding_path: str,
    ) -> SourceAssetEmbeddingRecord:
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO source_asset_embeddings (
                    asset_id, model_name, embedding_dim, embedding_path
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(asset_id, model_name) DO UPDATE SET
                    embedding_dim = excluded.embedding_dim,
                    embedding_path = excluded.embedding_path
                """,
                (asset_id, model_name, embedding_dim, embedding_path),
            )
            row = connection.execute(
                """
                SELECT asset_id, model_name, embedding_dim, embedding_path, created_at
                FROM source_asset_embeddings
                WHERE asset_id = ? AND model_name = ?
                """,
                (asset_id, model_name),
            ).fetchone()
            connection.commit()
        return self._row_to_embedding(row)

    def list_roi_assets(self, exclude_asset_id: int | None = None) -> list[SourceRoiAssetRecord]:
        query = """
        SELECT id, asset_id, target_label, bbox_json, coverage_ratio, roi_path, roi_relative_path, roi_url,
               roi_sha256, roi_phash, created_at
        FROM source_roi_assets
        {where_clause}
        ORDER BY id ASC
        """
        where_clause = "WHERE asset_id != ?" if exclude_asset_id is not None else ""
        params = (exclude_asset_id,) if exclude_asset_id is not None else ()
        with get_connection(self.database_path) as connection:
            rows = connection.execute(query.format(where_clause=where_clause), params).fetchall()
        return [self._row_to_roi_asset(row) for row in rows]

    def upsert_roi_asset(
        self,
        *,
        asset_id: int,
        target_label: str,
        bbox_json: str,
        coverage_ratio: float,
        roi_path: str,
        roi_relative_path: str,
        roi_url: str,
        roi_sha256: str,
        roi_phash: str,
    ) -> SourceRoiAssetRecord:
        with get_connection(self.database_path) as connection:
            existing = connection.execute(
                """
                SELECT id, asset_id, target_label, bbox_json, coverage_ratio, roi_path, roi_relative_path, roi_url,
                       roi_sha256, roi_phash, created_at
                FROM source_roi_assets
                WHERE asset_id = ? AND roi_sha256 = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (asset_id, roi_sha256),
            ).fetchone()
            if existing is not None:
                return self._row_to_roi_asset(existing)

            cursor = connection.execute(
                """
                INSERT INTO source_roi_assets (
                    asset_id, target_label, bbox_json, coverage_ratio, roi_path, roi_relative_path,
                    roi_url, roi_sha256, roi_phash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    target_label,
                    bbox_json,
                    coverage_ratio,
                    roi_path,
                    roi_relative_path,
                    roi_url,
                    roi_sha256,
                    roi_phash,
                ),
            )
            row = connection.execute(
                """
                SELECT id, asset_id, target_label, bbox_json, coverage_ratio, roi_path, roi_relative_path, roi_url,
                       roi_sha256, roi_phash, created_at
                FROM source_roi_assets
                WHERE id = ?
                """,
                (int(cursor.lastrowid),),
            ).fetchone()
            connection.commit()
        return self._row_to_roi_asset(row)

    def find_roi_embedding(self, roi_id: int, model_name: str) -> SourceRoiEmbeddingRecord | None:
        query = """
        SELECT roi_id, model_name, embedding_dim, embedding_path, created_at
        FROM source_roi_embeddings
        WHERE roi_id = ? AND model_name = ?
        """
        with get_connection(self.database_path) as connection:
            row = connection.execute(query, (roi_id, model_name)).fetchone()
        return self._row_to_roi_embedding(row) if row is not None else None

    def upsert_roi_embedding(
        self,
        *,
        roi_id: int,
        model_name: str,
        embedding_dim: int,
        embedding_path: str,
    ) -> SourceRoiEmbeddingRecord:
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO source_roi_embeddings (
                    roi_id, model_name, embedding_dim, embedding_path
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(roi_id, model_name) DO UPDATE SET
                    embedding_dim = excluded.embedding_dim,
                    embedding_path = excluded.embedding_path
                """,
                (roi_id, model_name, embedding_dim, embedding_path),
            )
            row = connection.execute(
                """
                SELECT roi_id, model_name, embedding_dim, embedding_path, created_at
                FROM source_roi_embeddings
                WHERE roi_id = ? AND model_name = ?
                """,
                (roi_id, model_name),
            ).fetchone()
            connection.commit()
        return self._row_to_roi_embedding(row)

    def delete_roi_assets_for_asset(self, asset_id: int) -> list[SourceRoiAssetRecord]:
        existing = [roi for roi in self.list_roi_assets() if roi.asset_id == asset_id]
        if not existing:
            return []
        with get_connection(self.database_path) as connection:
            connection.execute("BEGIN")
            try:
                connection.execute(
                    """
                    DELETE FROM source_roi_embeddings
                    WHERE roi_id IN (
                        SELECT id FROM source_roi_assets WHERE asset_id = ?
                    )
                    """,
                    (asset_id,),
                )
                connection.execute("DELETE FROM source_roi_assets WHERE asset_id = ?", (asset_id,))
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise
        return existing

    def persist_analysis(
        self,
        *,
        asset_payload: dict,
        predicted_class_index: int,
        predicted_label: str,
        detection_payload: dict,
        source_payload: dict,
    ) -> PersistSourceAnalysisResult:
        with get_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN")
                asset_record = self._find_asset_by_sha256_with_connection(connection, asset_payload["sha256"])
                asset_reused = asset_record is not None

                if asset_record is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO source_assets (
                            original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            asset_payload["original_filename"],
                            asset_payload["archived_path"],
                            asset_payload["archived_relative_path"],
                            asset_payload["archived_url"],
                            asset_payload["sha256"],
                            asset_payload["phash"],
                            asset_payload["file_size"],
                        ),
                    )
                    asset_id = int(cursor.lastrowid)
                    asset_row = connection.execute(
                        """
                        SELECT id, original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size, created_at
                        FROM source_assets
                        WHERE id = ?
                        """,
                        (asset_id,),
                    ).fetchone()
                    asset_record = self._row_to_asset(asset_row)

                exif_payload = source_payload.get("exif") or source_payload.get("signals", {}).get("exif", {})
                ela_payload = source_payload.get("ela") or source_payload.get("signals", {}).get("ela", {})
                exif_summary = exif_payload.get("summary", {})
                connection.execute(
                    """
                    INSERT INTO source_analysis_records (
                        asset_id,
                        predicted_class_index,
                        predicted_label,
                        has_exif,
                        exif_datetime_original,
                        exif_make,
                        exif_model,
                        exif_software,
                        ela_supported_for_detection,
                        ela_anomaly_ratio,
                        ela_is_tampered,
                        detection_payload,
                        source_payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_record.id,
                        predicted_class_index,
                        predicted_label,
                        int(bool(exif_payload.get("has_exif"))),
                        exif_summary.get("datetime_original"),
                        exif_summary.get("make"),
                        exif_summary.get("model"),
                        exif_summary.get("software"),
                        int(bool(ela_payload.get("supported_for_detection"))),
                        ela_payload.get("anomaly_ratio", 0.0),
                        int(bool(ela_payload.get("is_tampered"))),
                        json.dumps(detection_payload, ensure_ascii=False),
                        json.dumps(source_payload, ensure_ascii=False),
                    ),
                )
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise

        return PersistSourceAnalysisResult(asset=asset_record, record_saved=True, asset_reused=asset_reused)

    def _find_asset_by_sha256_with_connection(
        self,
        connection: sqlite3.Connection,
        sha256: str,
    ) -> SourceAssetRecord | None:
        row = connection.execute(
            """
            SELECT id, original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size, created_at
            FROM source_assets
            WHERE sha256 = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (sha256,),
        ).fetchone()
        return self._row_to_asset(row) if row is not None else None

    @staticmethod
    def _row_to_asset(row) -> SourceAssetRecord:
        return SourceAssetRecord(
            id=int(row["id"]),
            original_filename=str(row["original_filename"]),
            archived_path=str(row["archived_path"]),
            archived_relative_path=str(row["archived_relative_path"]),
            archived_url=str(row["archived_url"]),
            sha256=str(row["sha256"]),
            phash=str(row["phash"]),
            file_size=int(row["file_size"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_embedding(row) -> SourceAssetEmbeddingRecord:
        return SourceAssetEmbeddingRecord(
            asset_id=int(row["asset_id"]),
            model_name=str(row["model_name"]),
            embedding_dim=int(row["embedding_dim"]),
            embedding_path=str(row["embedding_path"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_roi_asset(row) -> SourceRoiAssetRecord:
        return SourceRoiAssetRecord(
            id=int(row["id"]),
            asset_id=int(row["asset_id"]),
            target_label=str(row["target_label"]),
            bbox_json=str(row["bbox_json"]),
            coverage_ratio=float(row["coverage_ratio"]),
            roi_path=str(row["roi_path"]),
            roi_relative_path=str(row["roi_relative_path"]),
            roi_url=str(row["roi_url"]),
            roi_sha256=str(row["roi_sha256"]),
            roi_phash=str(row["roi_phash"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_roi_embedding(row) -> SourceRoiEmbeddingRecord:
        return SourceRoiEmbeddingRecord(
            roi_id=int(row["roi_id"]),
            model_name=str(row["model_name"]),
            embedding_dim=int(row["embedding_dim"]),
            embedding_path=str(row["embedding_path"]),
            created_at=str(row["created_at"]),
        )
