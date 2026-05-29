"""Reproducibility exports for geometry and source grids."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .geometry import (
    WingGeometryParams,
    default_geometry,
    feather_root_properties,
    feather_sections,
    source_grid,
)


def geometry_metadata(
    params: WingGeometryParams | None = None,
    span_fractions: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    params = default_geometry() if params is None else params
    sections = feather_sections(params, span_fractions)
    roots = feather_root_properties(params)
    return {
        "units": "SI",
        "coordinate_mapping": {
            "paper_x": "Fusion Y",
            "paper_y": "Fusion X",
            "paper_z": "Fusion Z",
        },
        "params": {
            key: value
            for key, value in params.__dict__.items()
            if key != "per_feather_incidence_deg"
        }
        | {"per_feather_incidence_deg": params.per_feather_incidence_deg},
        "roots": [root.__dict__ for root in roots],
        "sections": [section.__dict__ for section in sections],
    }


def write_geometry_metadata_json(
    path: str | Path,
    params: WingGeometryParams | None = None,
    span_fractions: tuple[float, ...] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(geometry_metadata(params, span_fractions), indent=2),
        encoding="utf-8",
    )
    return output_path


def write_source_grid_csv(
    path: str | Path,
    params: WingGeometryParams | None = None,
    n_eta: int = 64,
    source_chord_fraction: float = 1.0,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = source_grid(params, n_eta=n_eta, source_chord_fraction=source_chord_fraction)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feather_id",
                "eta",
                "paper_x_m",
                "paper_y_m",
                "paper_z_m",
                "weight_m",
                "chord_m",
                "incidence_deg",
                "loading_x",
                "loading_y",
                "loading_z",
            ]
        )
        for index, point in enumerate(grid.points):
            direction = grid.loading_directions[index]
            writer.writerow(
                [
                    grid.feather_ids[index],
                    grid.etas[index],
                    point[0],
                    point[1],
                    point[2],
                    grid.weights[index],
                    grid.chords[index],
                    grid.incidence_deg[index],
                    direction[0],
                    direction[1],
                    direction[2],
                ]
            )
    return output_path
