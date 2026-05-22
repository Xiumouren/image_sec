from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

from backend.shared.config import PHASH_MATCH_THRESHOLD, PHASH_TOP_K
from backend.storage_layer.sqlite.models import SourceAssetRecord


class PHashService:
    def __init__(self, threshold: int = PHASH_MATCH_THRESHOLD, top_k: int = PHASH_TOP_K) -> None:
        self.threshold = threshold
        self.top_k = top_k

    def compute_hash(self, image_path: Path) -> str:
        with Image.open(image_path) as image:
            return str(imagehash.phash(image.convert("RGB")))

    def analyze(
        self,
        current_hash: str,
        historical_assets: list[SourceAssetRecord],
        exact_match_asset: SourceAssetRecord | None = None,
    ) -> dict[str, object]:
        matches: list[dict[str, object]] = []
        skipped_invalid_hashes = 0
        current_image_hash = imagehash.hex_to_hash(current_hash)
        total_bits = int(current_image_hash.hash.size)

        for asset in historical_assets:
            try:
                historical_hash = imagehash.hex_to_hash(asset.phash)
            except Exception:
                skipped_invalid_hashes += 1
                continue

            distance = int(current_image_hash - historical_hash)
            similarity = max(0.0, 1.0 - (distance / float(total_bits)))
            matches.append(
                {
                    "asset_id": asset.id,
                    "image_name": asset.original_filename,
                    "relative_path": asset.archived_relative_path,
                    "source_url": asset.archived_url,
                    "sha256": asset.sha256,
                    "distance": distance,
                    "similarity": round(similarity, 4),
                }
            )

        ranked = sorted(matches, key=lambda item: (int(item["distance"]), -float(item["similarity"])))[: self.top_k]
        exact_match = None
        if exact_match_asset is not None:
            exact_match = {
                "asset_id": exact_match_asset.id,
                "image_name": exact_match_asset.original_filename,
                "relative_path": exact_match_asset.archived_relative_path,
                "source_url": exact_match_asset.archived_url,
                "sha256": exact_match_asset.sha256,
                "distance": 0,
                "similarity": 1.0,
            }
            ranked = [exact_match, *ranked][: self.top_k]

        top_match = exact_match or (ranked[0] if ranked else None)
        error = None
        if skipped_invalid_hashes:
            error = f"Skipped {skipped_invalid_hashes} invalid historical pHash values."
        return {
            "hash": current_hash,
            "exact_match": exact_match is not None,
            "matched": bool(top_match and int(top_match["distance"]) <= self.threshold),
            "threshold": self.threshold,
            "top_match": top_match,
            "candidates": ranked,
            "error": error,
        }
