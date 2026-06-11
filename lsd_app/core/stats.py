"""Statistical summaries for vectorized landslide predictions."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol


class VectorStatsFeature(Protocol):
    """Feature fields needed for vector summary tables."""

    area_m2: float
    mean_confidence: float | None


@dataclass(frozen=True)
class VectorStats:
    """One-row summary for vectorized landslide polygons."""

    polygon_count: int
    total_area_m2: float
    mean_area_m2: float
    mean_confidence: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a JSON-ready summary mapping."""

        return asdict(self)


def summarize_vector_features(features: list[VectorStatsFeature]) -> VectorStats:
    """Compute aggregate polygon statistics."""

    polygon_count = len(features)
    total_area_m2 = float(sum(feature.area_m2 for feature in features))
    mean_area_m2 = total_area_m2 / polygon_count if polygon_count else 0.0
    confidences = [feature.mean_confidence for feature in features if feature.mean_confidence is not None]
    mean_confidence = float(sum(confidences) / len(confidences)) if confidences else None
    return VectorStats(
        polygon_count=polygon_count,
        total_area_m2=total_area_m2,
        mean_area_m2=mean_area_m2,
        mean_confidence=mean_confidence,
    )


def write_vector_stats(stats: VectorStats, metrics_dir: Path) -> tuple[Path, Path]:
    """Write vector statistics to CSV and JSON files."""

    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "vector_stats.csv"
    json_path = metrics_dir / "vector_stats.json"
    row = stats.to_dict()

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["polygon_count", "total_area_m2", "mean_area_m2", "mean_confidence"],
        )
        writer.writeheader()
        writer.writerow({key: "" if value is None else value for key, value in row.items()})

    json_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, json_path
