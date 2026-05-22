from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from backend.shared.config import OUTPUTS_DIR, REPORTS_DIR


class ReportService:
    def __init__(self, report_root: Path = REPORTS_DIR):
        self.report_root = Path(report_root)

    def build_report_paths(self, category: str) -> tuple[Path, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = self.report_root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{timestamp}.json", target_dir / f"{timestamp}.csv"

    def write_single_report(self, payload: dict) -> Path:
        target_dir = self.report_root / "single"
        target_dir.mkdir(parents=True, exist_ok=True)
        image_name = payload.get("image_name", "result")
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(image_name).stem}.json"
        output_path = target_dir / filename
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def write_batch_reports(self, rows: list[dict], payload: dict) -> tuple[Path, Path]:
        json_path, csv_path = self.build_report_paths("batch")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        fieldnames = [
            "image_name",
            "relative_path",
            "predicted_class_index",
            "predicted_label",
            "predicted_score",
            "method",
            "output_path",
            "error",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return json_path, csv_path

    @staticmethod
    def relative_to_outputs(path: Path) -> str:
        return path.relative_to(OUTPUTS_DIR).as_posix()
