from __future__ import annotations

from pathlib import Path

import exifread


SUMMARY_FIELD_MAP = {
    "datetime_original": "EXIF DateTimeOriginal",
    "make": "Image Make",
    "model": "Image Model",
    "software": "Image Software",
    "artist": "Image Artist",
}
RAW_ALLOWLIST = {
    "EXIF DateTimeOriginal",
    "Image Make",
    "Image Model",
    "Image Software",
    "Image Artist",
    "Image Orientation",
}
CORE_EXIF_KEYS = {
    "EXIF DateTimeOriginal",
    "Image Make",
    "Image Model",
    "Image Software",
}
POSTPROCESS_SOFTWARE_HINTS = (
    "photoshop",
    "lightroom",
    "gimp",
    "affinity",
    "canva",
    "picsart",
    "snapseed",
    "meitu",
)


class ExifService:
    def analyze(self, image_path: Path) -> dict[str, object]:
        with Path(image_path).open("rb") as handle:
            tags = exifread.process_file(handle, details=False)

        raw = {key: str(value) for key, value in tags.items() if key in RAW_ALLOWLIST}
        summary = {
            label: raw[source_key]
            for label, source_key in SUMMARY_FIELD_MAP.items()
            if raw.get(source_key)
        }
        has_exif = any(key in raw for key in CORE_EXIF_KEYS)
        software_value = summary.get("software", "")
        software_present = bool(software_value)
        possible_postprocess_hint = any(hint in software_value.lower() for hint in POSTPROCESS_SOFTWARE_HINTS)
        return {
            "has_exif": has_exif,
            "summary": summary,
            "raw": raw,
            "software_present": software_present,
            "possible_postprocess_hint": possible_postprocess_hint,
            "error": None,
        }
