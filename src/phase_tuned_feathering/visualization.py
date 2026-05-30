"""Dependency-free renders of the geometry seen by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import html
import math
from pathlib import Path

from .geometry import SourceGrid, WingGeometryParams, default_geometry, source_grid
from .validation import ComparisonResult, ComparisonRow


@dataclass(frozen=True)
class ProjectedPoint:
    u: float
    v: float


def _project(
    point: tuple[float, float, float],
    view: str,
) -> ProjectedPoint:
    x, y, z = point
    if view == "plan":
        return ProjectedPoint(y, x)
    if view == "side":
        return ProjectedPoint(y, z)
    if view == "front":
        return ProjectedPoint(x, z)
    if view == "isometric":
        return ProjectedPoint(y - 0.55 * x, z + 0.32 * x)
    raise ValueError("view must be one of 'plan', 'side', 'front', or 'isometric'.")


def _bounds(points: tuple[ProjectedPoint, ...]) -> tuple[float, float, float, float]:
    u_values = tuple(point.u for point in points)
    v_values = tuple(point.v for point in points)
    u_min = min(u_values)
    u_max = max(u_values)
    v_min = min(v_values)
    v_max = max(v_values)
    if u_max == u_min:
        u_max += 0.5
        u_min -= 0.5
    if v_max == v_min:
        v_max += 0.5
        v_min -= 0.5
    return u_min, u_max, v_min, v_max


def _svg_transform(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
):
    u_min, u_max, v_min, v_max = bounds
    span_u = u_max - u_min
    span_v = v_max - v_min
    scale = min((width - 2 * margin) / span_u, (height - 2 * margin) / span_v)
    plot_width = span_u * scale
    plot_height = span_v * scale
    offset_x = margin + 0.5 * (width - 2 * margin - plot_width)
    offset_y = margin + 0.5 * (height - 2 * margin - plot_height)

    def transform(point: ProjectedPoint) -> tuple[float, float]:
        sx = offset_x + (point.u - u_min) * scale
        sy = height - offset_y - (point.v - v_min) * scale
        return sx, sy

    return transform, scale


def _color_for_feather(feather_id: int) -> str:
    colors = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#bcbd22",
        "#e377c2",
    )
    return colors[(feather_id - 1) % len(colors)]


def _format_frequency(frequency_hz: float) -> str:
    if frequency_hz >= 1000.0:
        return f"{frequency_hz / 1000.0:g} kHz"
    return f"{frequency_hz:g} Hz"


def _angle_deg_xz(direction: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(direction[2], direction[0]))


def _observer_label(direction: tuple[float, float, float]) -> str:
    return (
        f"x={direction[0]:.2f}, y={direction[1]:.2f}, "
        f"z={direction[2]:.2f}, angle={_angle_deg_xz(direction):.1f} deg"
    )


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.0, stroke_dasharray: str = "") -> str:
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{stroke_dasharray}"' if stroke_dasharray else ""
    return (
        f'<polyline points="{coordinates}" fill="none" '
        f'stroke="{color}" stroke-width="{width:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} />'
    )


def _axis_glyph(view: str, x: float, y: float) -> str:
    vectors = {
        "plan": (("x", 0, -34), ("y", 42, 0)),
        "side": (("x", 42, 0), ("z", 0, -34)),
        "front": (("y", 42, 0), ("z", 0, -34)),
        "isometric": (("x", 42, 0), ("y", 28, 18), ("z", 0, -36)),
    }[view]
    items = [
        f'<text x="{x}" y="{y - 46}" font-family="Arial, sans-serif" font-size="13" fill="#333">paper axes</text>'
    ]
    for label, dx, dy in vectors:
        items.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + dx:.1f}" y2="{y + dy:.1f}" '
            'stroke="#222" stroke-width="1.4" marker-end="url(#arrow)" />'
        )
        items.append(
            f'<text x="{x + dx + 6:.1f}" y="{y + dy + 4:.1f}" '
            'font-family="Arial, sans-serif" font-size="13" fill="#222">'
            f"{html.escape(label)}</text>"
        )
    return "\n".join(items)


def source_grid_svg(
    grid: SourceGrid,
    view: str = "isometric",
    width: int = 1200,
    height: int = 800,
    title: str | None = None,
    show_loading_vectors: bool = True,
) -> str:
    projected = tuple(_project(point, view) for point in grid.points)
    bounds = _bounds(projected)
    transform, scale = _svg_transform(bounds, width, height, margin=64)
    title = title or f"Simulator Source Geometry - {view}"

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="32" y="38" font-family="Arial, sans-serif" font-size="24" '
        f'fill="#111111">{html.escape(title)}</text>',
    ]

    by_feather: dict[int, list[int]] = {}
    for index, feather_id in enumerate(grid.feather_ids):
        by_feather.setdefault(feather_id, []).append(index)

    chord_min = min(grid.chords)
    chord_max = max(grid.chords)
    chord_span = max(chord_max - chord_min, 1.0e-12)

    for feather_id, indices in by_feather.items():
        color = _color_for_feather(feather_id)
        path_points = [transform(projected[index]) for index in indices]
        svg.append(_polyline(path_points, color, width=2.4))

        for index in indices:
            x, y = transform(projected[index])
            chord_fraction = (grid.chords[index] - chord_min) / chord_span
            radius = 2.2 + 2.4 * chord_fraction
            if grid.etas[index] in (0.0, 1.0):
                radius += 1.1
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                f'fill="{color}" opacity="0.86">'
                f"<title>feather {feather_id}, eta={grid.etas[index]:.3f}, "
                f"chord={grid.chords[index]:.4f} m</title></circle>"
            )
        root_x, root_y = transform(projected[indices[0]])
        tip_x, tip_y = transform(projected[indices[-1]])
        svg.append(
            f'<text x="{root_x + 8:.1f}" y="{root_y - 7:.1f}" '
            'font-family="Arial, sans-serif" font-size="12" '
            f'fill="{color}">F{feather_id} root</text>'
        )
        svg.append(
            f'<text x="{tip_x + 8:.1f}" y="{tip_y + 4:.1f}" '
            'font-family="Arial, sans-serif" font-size="12" '
            f'fill="{color}">tip</text>'
        )

    if show_loading_vectors:
        vector_scale = min(0.10 * scale, 30.0)
        step = max(1, len(grid.points) // 80)
        for index in range(0, len(grid.points), step):
            point = projected[index]
            direction = grid.loading_directions[index]
            projected_direction = _project(direction, view)
            x0, y0 = transform(point)
            x1 = x0 + projected_direction.u * vector_scale
            y1 = y0 - projected_direction.v * vector_scale
            svg.append(
                f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                'stroke="#222222" stroke-width="1.2" opacity="0.55" '
                'marker-end="url(#arrow)" />'
            )

    svg.append(_axis_glyph(view, width - 150, height - 110))

    u_min, u_max, v_min, v_max = bounds
    scale_bar_m = 0.05
    scale_bar_px = scale_bar_m * scale
    while scale_bar_px > 180.0:
        scale_bar_m *= 0.5
        scale_bar_px = scale_bar_m * scale
    while scale_bar_px < 70.0:
        scale_bar_m *= 2.0
        scale_bar_px = scale_bar_m * scale
    bar_x = 34
    bar_y = height - 72
    svg.append(
        f'<line x1="{bar_x}" y1="{bar_y}" x2="{bar_x + scale_bar_px:.1f}" y2="{bar_y}" '
        'stroke="#222" stroke-width="2.2" />'
    )
    svg.append(
        f'<text x="{bar_x}" y="{bar_y - 8}" font-family="Arial, sans-serif" '
        f'font-size="13" fill="#333">{scale_bar_m:g} m</text>'
    )

    legend_items = []
    legend_x = 32
    legend_y = 74
    legend_items.append(
        f'<rect x="{legend_x - 10}" y="{legend_y - 24}" width="168" '
        f'height="{24 + 20 * len(by_feather)}" fill="#fff" stroke="#ddd" />'
    )
    legend_items.append(
        f'<text x="{legend_x}" y="{legend_y - 7}" font-family="Arial, sans-serif" '
        'font-size="13" fill="#333">feather source lines</text>'
    )
    for offset, feather_id in enumerate(sorted(by_feather), start=1):
        y = legend_y + 18 * offset
        color = _color_for_feather(feather_id)
        legend_items.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" />'
        )
        legend_items.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#333">F{feather_id}</text>'
        )
    svg.extend(legend_items)

    svg.insert(
        3,
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#222222" /></marker></defs>',
    )

    legend_x = 32
    legend_y = height - 34
    svg.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, sans-serif" '
        'font-size="14" fill="#333333">'
        f"view={html.escape(view)} | sources={grid.n} | "
        f"projected bounds: {u_min:.3f}..{u_max:.3f} m, {v_min:.3f}..{v_max:.3f} m | "
        f"chord range={chord_min:.4f}..{chord_max:.4f} m"
        "</text>"
    )
    svg.append("</svg>")
    return "\n".join(svg)


def write_source_grid_svg(
    path: str | Path,
    grid: SourceGrid,
    view: str = "isometric",
    width: int = 1200,
    height: int = 800,
    show_loading_vectors: bool = True,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        source_grid_svg(
            grid,
            view=view,
            width=width,
            height=height,
            show_loading_vectors=show_loading_vectors,
        ),
        encoding="utf-8",
    )
    return output_path


def write_simulator_geometry_renders(
    output_dir: str | Path,
    params: WingGeometryParams | None = None,
    n_eta: int = 24,
    source_chord_fraction: float = 1.0,
    views: tuple[str, ...] = ("plan", "side", "front", "isometric"),
) -> tuple[Path, ...]:
    params = default_geometry() if params is None else params
    output_path = Path(output_dir)
    grid = source_grid(
        params,
        n_eta=n_eta,
        source_chord_fraction=source_chord_fraction,
    )
    return tuple(
        write_source_grid_svg(
            output_path / f"simulator_geometry_{view}.svg",
            grid,
            view=view,
        )
        for view in views
    )


def _value_bounds(values: tuple[float, ...], pad_fraction: float = 0.08) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    pad = (upper - lower) * pad_fraction
    return lower - pad, upper + pad


def _plot_transform(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    width: int,
    height: int,
    margin_left: int = 82,
    margin_right: int = 34,
    margin_top: int = 60,
    margin_bottom: int = 68,
):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def transform(x_value: float, y_value: float) -> tuple[float, float]:
        x = margin_left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = height - margin_bottom - (y_value - y_min) / (y_max - y_min) * plot_height
        return x, y

    return transform


def _figure_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="28" y="36" font-family="Arial, sans-serif" font-size="22" '
        f'fill="#111111">{html.escape(title)}</text>',
    ]


def _axes(
    width: int,
    height: int,
    x_label: str,
    y_label: str,
    margin_left: int = 82,
    margin_right: int = 34,
    margin_top: int = 60,
    margin_bottom: int = 68,
) -> str:
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    return "\n".join(
        [
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="1.4" />',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="1.4" />',
            f'<text x="{0.5 * (x0 + x1):.1f}" y="{height - 20}" '
            'font-family="Arial, sans-serif" font-size="15" text-anchor="middle" '
            f'fill="#222">{html.escape(x_label)}</text>',
            f'<text x="22" y="{0.5 * (y0 + y1):.1f}" '
            'font-family="Arial, sans-serif" font-size="15" text-anchor="middle" '
            f'transform="rotate(-90 22 {0.5 * (y0 + y1):.1f})" '
            f'fill="#222">{html.escape(y_label)}</text>',
        ]
    )


def _tick_labels(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    transform,
    width: int,
    height: int,
    x_count: int = 5,
    y_count: int = 5,
    x_formatter=None,
    y_formatter=None,
) -> str:
    x_formatter = (lambda value: f"{value:.2g}") if x_formatter is None else x_formatter
    y_formatter = (lambda value: f"{value:.2f}") if y_formatter is None else y_formatter
    items: list[str] = []
    for index in range(x_count):
        value = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * index / max(x_count - 1, 1)
        x, y = transform(value, y_bounds[0])
        items.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + 6:.1f}" stroke="#222" />')
        items.append(
            f'<text x="{x:.1f}" y="{height - 45}" font-family="Arial, sans-serif" '
            f'font-size="12" text-anchor="middle" fill="#333">{html.escape(x_formatter(value))}</text>'
        )
    for index in range(y_count):
        value = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * index / max(y_count - 1, 1)
        x, y = transform(x_bounds[0], value)
        items.append(f'<line x1="{x - 6:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#222" />')
        items.append(
            f'<text x="72" y="{y + 4:.1f}" font-family="Arial, sans-serif" '
            f'font-size="12" text-anchor="end" fill="#333">{html.escape(y_formatter(value))}</text>'
        )
    return "\n".join(items)


def _split_color(split: str) -> str:
    return "#1f77b4" if split == "calibration" else "#d62728" if split == "validation" else "#555555"


def _signed_error_color(error_db: float, max_abs_error: float) -> str:
    fraction = min(abs(error_db) / max(max_abs_error, 1.0e-12), 1.0)
    if error_db >= 0.0:
        red = 230
        green = int(230 - 132 * fraction)
        blue = int(230 - 152 * fraction)
    else:
        red = int(230 - 160 * fraction)
        green = int(230 - 100 * fraction)
        blue = 230
    return f"#{red:02x}{green:02x}{blue:02x}"


def _log_frequency_label(log_frequency: float) -> str:
    return _format_frequency(10.0 ** log_frequency)


def _comparison_context(result: ComparisonResult) -> str:
    frequencies = sorted({row.frequency_hz for row in result.rows})
    directions = {_observer_label(row.direction) for row in result.rows}
    splits = sorted({row.split for row in result.rows})
    return (
        f"rows={len(result.rows)} | frequencies={len(frequencies)} "
        f"({_format_frequency(frequencies[0])}..{_format_frequency(frequencies[-1])}) | "
        f"observer directions={len(directions)} | splits={', '.join(splits)}"
    )


def _footer(width: int, height: int, text: str) -> str:
    return (
        f'<text x="{width / 2:.1f}" y="{height - 10}" '
        'font-family="Arial, sans-serif" font-size="12" text-anchor="middle" '
        f'fill="#444">{html.escape(text)}</text>'
    )


def theory_vs_simulation_scatter_svg(
    result: ComparisonResult,
    width: int = 980,
    height: int = 760,
) -> str:
    rows = result.rows
    values = tuple(row.simulated_level_db for row in rows) + tuple(row.theory_level_db for row in rows)
    bounds = _value_bounds(values)
    transform = _plot_transform(bounds, bounds, width, height)
    svg = _figure_header(width, height, "Theory vs Simulated Spectral Level")
    svg.append(_axes(width, height, "Simulated level (dB re p_ref^2/Hz)", "Theory level (dB re p_ref^2/Hz)"))
    svg.append(_tick_labels(bounds, bounds, transform, width, height))
    x0, y0 = transform(bounds[0], bounds[0])
    x1, y1 = transform(bounds[1], bounds[1])
    svg.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#777" stroke-dasharray="6 6" />')
    for offset_db in (-10.0, -5.0, 5.0, 10.0):
        a0, b0 = transform(bounds[0], bounds[0] + offset_db)
        a1, b1 = transform(bounds[1], bounds[1] + offset_db)
        svg.append(
            f'<line x1="{a0:.1f}" y1="{b0:.1f}" x2="{a1:.1f}" y2="{b1:.1f}" '
            'stroke="#d1d5db" stroke-dasharray="3 7" />'
        )
    for row in rows:
        x, y = transform(row.simulated_level_db, row.theory_level_db)
        _, y_identity = transform(row.simulated_level_db, row.simulated_level_db)
        frequency_scale = 1.0 + 1.5 * (
            math.log10(row.frequency_hz) - math.log10(min(item.frequency_hz for item in rows))
        ) / max(
            math.log10(max(item.frequency_hz for item in rows))
            - math.log10(min(item.frequency_hz for item in rows)),
            1.0e-12,
        )
        svg.append(
            f'<line x1="{x:.2f}" y1="{y_identity:.2f}" x2="{x:.2f}" y2="{y:.2f}" '
            'stroke="#9ca3af" stroke-width="0.9" opacity="0.45" />'
        )
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.0 + frequency_scale:.2f}" '
            f'fill="{_split_color(row.split)}" opacity="0.72">'
            f"<title>{row.split}, {_format_frequency(row.frequency_hz)}, "
            f"{_observer_label(row.direction)}, error={row.error_db:.3f} dB</title></circle>"
        )
    svg.append(_legend(width - 210, 74))
    svg.append(
        f'<text x="{width - 210}" y="150" font-family="Arial, sans-serif" '
        'font-size="12" fill="#444">circle size increases with frequency</text>'
    )
    svg.append(_footer(width, height, _comparison_context(result)))
    svg.append("</svg>")
    return "\n".join(svg)


def _legend(x: float, y: float) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y - 20}" width="178" height="58" fill="#fff" stroke="#ddd" />',
            f'<circle cx="{x + 16}" cy="{y}" r="5" fill="#1f77b4" /><text x="{x + 30}" y="{y + 4}" font-family="Arial, sans-serif" font-size="13">calibration</text>',
            f'<circle cx="{x + 16}" cy="{y + 24}" r="5" fill="#d62728" /><text x="{x + 30}" y="{y + 28}" font-family="Arial, sans-serif" font-size="13">validation</text>',
        ]
    )


def error_by_frequency_svg(
    result: ComparisonResult,
    width: int = 1040,
    height: int = 680,
) -> str:
    grouped: dict[tuple[str, float], list[ComparisonRow]] = {}
    for row in result.rows:
        grouped.setdefault((row.split, row.frequency_hz), []).append(row)
    points = []
    for (split, frequency), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        weights = tuple(row.weight for row in rows)
        weight_sum = sum(weights)
        mae = sum(abs(row.error_db) * row.weight for row in rows) / weight_sum
        bias = sum(row.error_db * row.weight for row in rows) / weight_sum
        points.append((split, frequency, mae, bias))

    x_values = tuple(math.log10(item[1]) for item in points)
    y_values = tuple(item[2] for item in points) + tuple(item[3] for item in points)
    x_bounds = _value_bounds(x_values, 0.05)
    y_abs = max(abs(value) for value in y_values) if y_values else 1.0
    y_bounds = (-1.15 * y_abs, 1.15 * y_abs)
    transform = _plot_transform(x_bounds, y_bounds, width, height)
    svg = _figure_header(width, height, "Error by Frequency and Split")
    svg.append(_axes(width, height, "Frequency", "Signed error and mean absolute error (dB)"))
    svg.append(
        _tick_labels(
            x_bounds,
            y_bounds,
            transform,
            width,
            height,
            x_formatter=_log_frequency_label,
        )
    )
    x_zero0, y_zero = transform(x_bounds[0], 0.0)
    x_zero1, _ = transform(x_bounds[1], 0.0)
    svg.append(
        f'<line x1="{x_zero0:.1f}" y1="{y_zero:.1f}" x2="{x_zero1:.1f}" y2="{y_zero:.1f}" '
        'stroke="#6b7280" stroke-dasharray="5 5" />'
    )
    for split, frequency, mae, bias in points:
        x, y_mae_pos = transform(math.log10(frequency), mae)
        _, y_mae_neg = transform(math.log10(frequency), -mae)
        _, y_zero = transform(math.log10(frequency), 0.0)
        bar_width = 12.0
        offset = -8.0 if split == "calibration" else 8.0
        svg.append(
            f'<line x1="{x + offset:.2f}" y1="{y_mae_neg:.2f}" '
            f'x2="{x + offset:.2f}" y2="{y_mae_pos:.2f}" '
            f'stroke="{_split_color(split)}" stroke-width="{bar_width:.2f}" opacity="0.22">'
            f"<title>{split}, {_format_frequency(frequency)}, +/- MAE={mae:.3f} dB</title></line>"
        )
        x_bias, y_bias = transform(math.log10(frequency), abs(bias))
        _, y_signed_bias = transform(math.log10(frequency), bias)
        svg.append(
            f'<circle cx="{x_bias + offset:.2f}" cy="{y_signed_bias:.2f}" r="4.0" '
            f'fill="{_split_color(split)}" stroke="#111" stroke-width="0.8">'
            f"<title>{split}, {_format_frequency(frequency)}, signed mean={bias:.3f} dB, "
            f"MAE={mae:.3f} dB</title></circle>"
        )
    svg.append(_legend(width - 210, 74))
    svg.append(
        f'<text x="{width - 252}" y="150" font-family="Arial, sans-serif" '
        'font-size="12" fill="#444">vertical bars show +/- mean absolute error; dots show signed mean</text>'
    )
    svg.append(_footer(width, height, _comparison_context(result)))
    svg.append("</svg>")
    return "\n".join(svg)


def spectrum_overlay_svg(
    result: ComparisonResult,
    split: str = "validation",
    width: int = 1080,
    height: int = 720,
) -> str:
    rows = tuple(row for row in result.rows if row.split == split)
    if not rows:
        rows = result.rows
        split = "all"
    grouped: dict[float, list[ComparisonRow]] = {}
    for row in rows:
        grouped.setdefault(row.frequency_hz, []).append(row)

    frequencies = tuple(sorted(grouped))
    x_values = tuple(math.log10(frequency) for frequency in frequencies)
    sim_values = tuple(
        sum(row.simulated_level_db * row.weight for row in grouped[frequency])
        / sum(row.weight for row in grouped[frequency])
        for frequency in frequencies
    )
    theory_values = tuple(
        sum(row.theory_level_db * row.weight for row in grouped[frequency])
        / sum(row.weight for row in grouped[frequency])
        for frequency in frequencies
    )
    sim_minmax = tuple(
        (
            min(row.simulated_level_db for row in grouped[frequency]),
            max(row.simulated_level_db for row in grouped[frequency]),
        )
        for frequency in frequencies
    )
    theory_minmax = tuple(
        (
            min(row.theory_level_db for row in grouped[frequency]),
            max(row.theory_level_db for row in grouped[frequency]),
        )
        for frequency in frequencies
    )
    x_bounds = _value_bounds(x_values, 0.05)
    spread_values = tuple(value for pair in sim_minmax + theory_minmax for value in pair)
    y_bounds = _value_bounds(sim_values + theory_values + spread_values, 0.12)
    transform = _plot_transform(x_bounds, y_bounds, width, height)
    svg = _figure_header(width, height, f"Spectrum Overlay - {split}")
    svg.append(_axes(width, height, "Frequency", "Spectral level (dB re p_ref^2/Hz)"))
    svg.append(
        _tick_labels(
            x_bounds,
            y_bounds,
            transform,
            width,
            height,
            x_formatter=_log_frequency_label,
        )
    )

    sim_points = [transform(math.log10(f), value) for f, value in zip(frequencies, sim_values)]
    theory_points = [transform(math.log10(f), value) for f, value in zip(frequencies, theory_values)]
    for frequency, (sim_low, sim_high), (theory_low, theory_high) in zip(frequencies, sim_minmax, theory_minmax):
        x, y_sim_low = transform(math.log10(frequency), sim_low)
        _, y_sim_high = transform(math.log10(frequency), sim_high)
        _, y_theory_low = transform(math.log10(frequency), theory_low)
        _, y_theory_high = transform(math.log10(frequency), theory_high)
        svg.append(
            f'<line x1="{x - 5:.2f}" y1="{y_sim_low:.2f}" x2="{x - 5:.2f}" y2="{y_sim_high:.2f}" '
            'stroke="#d62728" stroke-width="5" opacity="0.20">'
            f"<title>simulated observer range, {_format_frequency(frequency)}: "
            f"{sim_low:.2f}..{sim_high:.2f} dB</title></line>"
        )
        svg.append(
            f'<line x1="{x + 5:.2f}" y1="{y_theory_low:.2f}" x2="{x + 5:.2f}" y2="{y_theory_high:.2f}" '
            'stroke="#1f77b4" stroke-width="5" opacity="0.20">'
            f"<title>theory observer range, {_format_frequency(frequency)}: "
            f"{theory_low:.2f}..{theory_high:.2f} dB</title></line>"
        )
    svg.append(_polyline(sim_points, "#d62728", width=3.0))
    svg.append(_polyline(theory_points, "#1f77b4", width=3.0))
    for frequency, (x, y) in zip(frequencies, sim_points):
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#d62728">'
            f"<title>simulated weighted mean, {_format_frequency(frequency)}: "
            f"{grouped[frequency][0].split}</title></circle>"
        )
    for x, y in theory_points:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#1f77b4" />')
    svg.append(
        f'<rect x="{width - 230}" y="54" width="198" height="58" fill="#fff" stroke="#ddd" />'
        f'<line x1="{width - 210}" y1="76" x2="{width - 170}" y2="76" stroke="#d62728" stroke-width="3" />'
        f'<text x="{width - 160}" y="80" font-family="Arial, sans-serif" font-size="13">simulated</text>'
        f'<line x1="{width - 210}" y1="100" x2="{width - 170}" y2="100" stroke="#1f77b4" stroke-width="3" />'
        f'<text x="{width - 160}" y="104" font-family="Arial, sans-serif" font-size="13">theory</text>'
    )
    svg.append(
        f'<text x="{width - 230}" y="132" font-family="Arial, sans-serif" '
        'font-size="12" fill="#444">thick pale bars show min..max across observer directions</text>'
    )
    svg.append(_footer(width, height, _comparison_context(result)))
    svg.append("</svg>")
    return "\n".join(svg)


def error_histogram_svg(
    result: ComparisonResult,
    width: int = 980,
    height: int = 660,
) -> str:
    errors = tuple(row.error_db for row in result.rows)
    x_bounds = _value_bounds(errors, 0.1)
    bin_count = 16
    splits = tuple(sorted({row.split for row in result.rows}))
    bins_by_split = {split: [0 for _ in range(bin_count)] for split in splits}
    for row in result.rows:
        error = row.error_db
        index = int((error - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * bin_count)
        bins_by_split[row.split][min(max(index, 0), bin_count - 1)] += 1
    stacked_bins = [
        sum(bins_by_split[split][index] for split in splits)
        for index in range(bin_count)
    ]
    y_bounds = (0.0, max(stacked_bins) * 1.2 if stacked_bins else 1.0)
    transform = _plot_transform(x_bounds, y_bounds, width, height)
    svg = _figure_header(width, height, "Signed Error Distribution")
    svg.append(_axes(width, height, "Theory - simulated error (dB)", "Count"))
    svg.append(_tick_labels(x_bounds, y_bounds, transform, width, height, y_formatter=lambda value: f"{value:.0f}"))
    x_zero, y_zero_axis = transform(0.0, 0.0)
    if x_bounds[0] <= 0.0 <= x_bounds[1]:
        _, y_top = transform(0.0, y_bounds[1])
        svg.append(
            f'<line x1="{x_zero:.1f}" y1="{y_zero_axis:.1f}" x2="{x_zero:.1f}" y2="{y_top:.1f}" '
            'stroke="#6b7280" stroke-dasharray="5 5" />'
        )
    bin_width_value = (x_bounds[1] - x_bounds[0]) / bin_count
    for index, count in enumerate(stacked_bins):
        left = x_bounds[0] + index * bin_width_value
        right = left + bin_width_value
        x_left, _ = transform(left, count)
        x_right, y_zero = transform(right, 0.0)
        cumulative = 0
        for split in splits:
            split_count = bins_by_split[split][index]
            if split_count == 0:
                continue
            y_top = transform(left, cumulative + split_count)[1]
            y_bottom = transform(left, cumulative)[1]
            svg.append(
                f'<rect x="{x_left:.2f}" y="{y_top:.2f}" '
                f'width="{max(x_right - x_left - 2.0, 1.0):.2f}" '
                f'height="{max(y_bottom - y_top, 0.0):.2f}" '
                f'fill="{_split_color(split)}" opacity="0.78">'
                f"<title>{split}, {left:.2f}..{right:.2f} dB, count={split_count}</title></rect>"
            )
            cumulative += split_count
    svg.append(_legend(width - 210, 74))
    svg.append(_footer(width, height, _comparison_context(result)))
    svg.append("</svg>")
    return "\n".join(svg)


def directivity_comparison_svg(
    result: ComparisonResult,
    frequency_hz: float | None = None,
    split: str = "validation",
    width: int = 860,
    height: int = 820,
) -> str:
    candidate_rows = tuple(row for row in result.rows if row.split == split)
    if not candidate_rows:
        candidate_rows = result.rows
        split = "all"
    if frequency_hz is None:
        frequency_hz = sorted({row.frequency_hz for row in candidate_rows})[0]
    rows = tuple(
        sorted(
            (row for row in candidate_rows if row.frequency_hz == frequency_hz),
            key=lambda row: _angle_deg_xz(row.direction),
        )
    )
    if not rows:
        rows = candidate_rows
    center_x = width / 2
    center_y = height / 2 + 18
    radius = min(width, height) * 0.34
    max_level = max(max(row.theory_level_db, row.simulated_level_db) for row in rows)
    min_level = min(min(row.theory_level_db, row.simulated_level_db) for row in rows)
    level_span = max(max_level - min_level, 1.0)

    def polar_point(row: ComparisonRow, level: float) -> tuple[float, float]:
        angle = math.atan2(row.direction[2], row.direction[0])
        local_radius = radius * (0.12 + 0.88 * (level - min_level) / level_span)
        return (
            center_x + local_radius * math.cos(angle),
            center_y - local_radius * math.sin(angle),
        )

    def direction_label(
        angle_deg: float,
        text: str,
        offset: float = 56.0,
    ) -> str:
        angle = math.radians(angle_deg)
        return (
            f'<text x="{center_x + (radius + offset) * math.cos(angle):.1f}" '
            f'y="{center_y - (radius + offset) * math.sin(angle) + 5:.1f}" '
            'font-family="Arial, sans-serif" font-size="13" font-weight="600" '
            'text-anchor="middle" fill="#222">'
            f"{html.escape(text)}</text>"
        )

    svg = _figure_header(width, height, f"Directivity Comparison - {split}, {frequency_hz:g} Hz")
    for fraction in (0.25, 0.5, 0.75, 1.0):
        svg.append(
            f'<circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="{radius * fraction:.1f}" '
            'fill="none" stroke="#e5e7eb" />'
        )
        level_value = min_level + (level_span * (fraction - 0.12) / 0.88)
        svg.append(
            f'<text x="{center_x + 6:.1f}" y="{center_y - radius * fraction + 4:.1f}" '
            'font-family="Arial, sans-serif" font-size="11" fill="#666">'
            f"{level_value:.1f} dB</text>"
        )
    svg.append(f'<line x1="{center_x - radius:.1f}" y1="{center_y:.1f}" x2="{center_x + radius:.1f}" y2="{center_y:.1f}" stroke="#d1d5db" />')
    svg.append(f'<line x1="{center_x:.1f}" y1="{center_y - radius:.1f}" x2="{center_x:.1f}" y2="{center_y + radius:.1f}" stroke="#d1d5db" />')
    
    # Overlay the side-view wing render (since plot uses X and Z for angles)
    svg.append(
        f'<image href="simulator_geometry_side.svg" '
        f'x="{center_x - radius:.1f}" y="{center_y - radius:.1f}" '
        f'width="{radius * 2:.1f}" height="{radius * 2:.1f}" opacity="0.15" />'
    )
    for angle_deg in (-90, -45, 0, 45, 90, 135, 180):
        angle = math.radians(angle_deg)
        x = center_x + radius * math.cos(angle)
        y = center_y - radius * math.sin(angle)
        svg.append(
            f'<line x1="{center_x:.1f}" y1="{center_y:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
            'stroke="#f0f0f0" />'
        )
        svg.append(
            f'<text x="{center_x + (radius + 24) * math.cos(angle):.1f}" '
            f'y="{center_y - (radius + 24) * math.sin(angle) + 4:.1f}" '
            'font-family="Arial, sans-serif" font-size="12" text-anchor="middle" '
            f'fill="#555">{angle_deg} deg</text>'
        )
    svg.append(direction_label(0, "+x / tip"))
    svg.append(direction_label(180, "-x / root"))
    svg.append(direction_label(90, "+z / up"))
    svg.append(direction_label(-90, "-z / down"))
    sim_points = [polar_point(row, row.simulated_level_db) for row in rows]
    theory_points = [polar_point(row, row.theory_level_db) for row in rows]
    if len(sim_points) > 1:
        svg.append(_polyline(sim_points + [sim_points[0]], "#d62728", width=2.4))
        svg.append(_polyline(theory_points + [theory_points[0]], "#1f77b4", width=2.4))
    for row, sim_point, theory_point in zip(rows, sim_points, theory_points):
        svg.append(
            f'<line x1="{sim_point[0]:.2f}" y1="{sim_point[1]:.2f}" '
            f'x2="{theory_point[0]:.2f}" y2="{theory_point[1]:.2f}" '
            'stroke="#9ca3af" stroke-width="1.0" opacity="0.55">'
            f"<title>{_observer_label(row.direction)}, error={row.error_db:.3f} dB</title></line>"
        )
    for row, (x, y) in zip(rows, sim_points):
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="#d62728" opacity="0.82">'
            f"<title>simulated, {_observer_label(row.direction)}, "
            f"{row.simulated_level_db:.3f} dB</title></circle>"
        )
    for row, (x, y) in zip(rows, theory_points):
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="#1f77b4" opacity="0.82">'
            f"<title>theory, {_observer_label(row.direction)}, "
            f"{row.theory_level_db:.3f} dB</title></circle>"
        )
    has_target = any(row.target_level_db is not None for row in rows)
    if has_target:
        sim_peak = max(row.simulated_level_db for row in rows)
        target_raw_peak = max(row.target_level_db for row in rows if row.target_level_db is not None)
        offset = sim_peak - target_raw_peak
        
        target_points = [polar_point(row, row.target_level_db + offset) for row in rows if row.target_level_db is not None]
        if len(target_points) > 1:
            svg.append(_polyline(target_points + [target_points[0]], "#2ca02c", width=2.4, stroke_dasharray="6,4"))
            
        svg.append(
            f'<rect x="{width - 190}" y="58" width="150" height="82" fill="#fff" stroke="#ddd" />'
            f'<line x1="{width - 172}" y1="80" x2="{width - 132}" y2="80" stroke="#d62728" stroke-width="3" />'
            f'<text x="{width - 122}" y="84" font-family="Arial, sans-serif" font-size="13">simulated</text>'
            f'<line x1="{width - 172}" y1="104" x2="{width - 132}" y2="104" stroke="#1f77b4" stroke-width="3" />'
            f'<text x="{width - 122}" y="108" font-family="Arial, sans-serif" font-size="13">theory</text>'
            f'<line x1="{width - 172}" y1="128" x2="{width - 132}" y2="128" stroke="#2ca02c" stroke-width="3" stroke-dasharray="6,4" />'
            f'<text x="{width - 122}" y="132" font-family="Arial, sans-serif" font-size="13">target shape</text>'
        )
    else:
        svg.append(
            f'<rect x="{width - 190}" y="58" width="150" height="58" fill="#fff" stroke="#ddd" />'
            f'<line x1="{width - 172}" y1="80" x2="{width - 132}" y2="80" stroke="#d62728" stroke-width="3" />'
            f'<text x="{width - 122}" y="84" font-family="Arial, sans-serif" font-size="13">simulated</text>'
            f'<line x1="{width - 172}" y1="104" x2="{width - 132}" y2="104" stroke="#1f77b4" stroke-width="3" />'
            f'<text x="{width - 122}" y="108" font-family="Arial, sans-serif" font-size="13">theory</text>'
        )
    svg.append(
        f'<text x="{center_x:.1f}" y="{height - 28}" font-family="Arial, sans-serif" '
        'font-size="13" text-anchor="middle" fill="#333">'
        f"radial scale: {min_level:.1f} to {max_level:.1f} dB, x-z observer plane projection</text>"
    )
    svg.append("</svg>")
    return "\n".join(svg)


def error_heatmap_svg(
    result: ComparisonResult,
    split: str = "validation",
    width: int = 1060,
    height: int = 720,
) -> str:
    rows = tuple(row for row in result.rows if row.split == split)
    if not rows:
        rows = result.rows
        split = "all"
    frequencies = tuple(sorted({row.frequency_hz for row in rows}))
    directions = tuple(
        sorted(
            {row.direction for row in rows},
            key=lambda direction: (_angle_deg_xz(direction), direction[1]),
        )
    )
    lookup = {
        (row.frequency_hz, row.direction): row
        for row in rows
    }
    max_abs_error = max(abs(row.error_db) for row in rows)
    margin_left = 132
    margin_right = 96
    margin_top = 86
    margin_bottom = 106
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    cell_width = plot_width / max(len(frequencies), 1)
    cell_height = plot_height / max(len(directions), 1)

    svg = _figure_header(width, height, f"Signed Error Heatmap - {split}")
    svg.append(
        f'<text x="{margin_left}" y="62" font-family="Arial, sans-serif" '
        'font-size="13" fill="#444">'
        "cell value is theory level minus simulated level in dB</text>"
    )
    for frequency_index, frequency in enumerate(frequencies):
        x = margin_left + frequency_index * cell_width
        label_x = x + 0.5 * cell_width
        svg.append(
            f'<text x="{label_x:.1f}" y="{height - 66}" font-family="Arial, sans-serif" '
            'font-size="12" text-anchor="end" transform="rotate(-35 '
            f'{label_x:.1f} {height - 66})" fill="#333">'
            f"{html.escape(_format_frequency(frequency))}</text>"
        )
    for direction_index, direction in enumerate(directions):
        y = margin_top + direction_index * cell_height
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 0.5 * cell_height + 4:.1f}" '
            'font-family="Arial, sans-serif" font-size="12" text-anchor="end" '
            f'fill="#333">{_angle_deg_xz(direction):.1f} deg</text>'
        )
        svg.append(
            f'<title>{_observer_label(direction)}</title>'
        )
    for direction_index, direction in enumerate(directions):
        for frequency_index, frequency in enumerate(frequencies):
            row = lookup.get((frequency, direction))
            x = margin_left + frequency_index * cell_width
            y = margin_top + direction_index * cell_height
            if row is None:
                fill = "#f3f4f6"
                title = "no row"
                text = ""
            else:
                fill = _signed_error_color(row.error_db, max_abs_error)
                title = (
                    f"{split}, {_format_frequency(frequency)}, "
                    f"{_observer_label(direction)}, error={row.error_db:.3f} dB"
                )
                text = f"{row.error_db:+.1f}" if cell_width >= 44 and cell_height >= 22 else ""
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" '
                f'height="{cell_height:.2f}" fill="{fill}" stroke="#ffffff" stroke-width="1">'
                f"<title>{html.escape(title)}</title></rect>"
            )
            if text:
                svg.append(
                    f'<text x="{x + 0.5 * cell_width:.1f}" y="{y + 0.5 * cell_height + 4:.1f}" '
                    'font-family="Arial, sans-serif" font-size="11" text-anchor="middle" '
                    f'fill="#111">{html.escape(text)}</text>'
                )
    svg.append(
        f'<text x="{margin_left - 12}" y="{margin_top - 16}" '
        'font-family="Arial, sans-serif" font-size="13" text-anchor="end" fill="#333">'
        "observer angle</text>"
    )
    svg.append(
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 20}" '
        'font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#222">'
        "frequency</text>"
    )

    legend_x = width - 76
    legend_y = margin_top
    legend_height = plot_height
    steps = 32
    for index in range(steps):
        fraction0 = index / steps
        error_value = max_abs_error * (1.0 - 2.0 * fraction0)
        fill = _signed_error_color(error_value, max_abs_error)
        y = legend_y + index * legend_height / steps
        svg.append(
            f'<rect x="{legend_x}" y="{y:.1f}" width="20" '
            f'height="{legend_height / steps + 0.5:.1f}" fill="{fill}" />'
        )
    svg.append(
        f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="Arial, sans-serif" '
        f'font-size="11" fill="#333">+{max_abs_error:.1f}</text>'
    )
    svg.append(
        f'<text x="{legend_x + 28}" y="{legend_y + legend_height / 2 + 4:.1f}" '
        'font-family="Arial, sans-serif" font-size="11" fill="#333">0</text>'
    )
    svg.append(
        f'<text x="{legend_x + 28}" y="{legend_y + legend_height:.1f}" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#333">-{max_abs_error:.1f}</text>'
    )
    svg.append(
        f'<text x="{legend_x - 4}" y="{legend_y - 12}" font-family="Arial, sans-serif" '
        'font-size="11" text-anchor="end" fill="#333">error dB</text>'
    )
    svg.append(_footer(width, height, _comparison_context(result)))
    svg.append("</svg>")
    return "\n".join(svg)


def write_validation_figures(
    output_dir: str | Path,
    result: ComparisonResult,
) -> tuple[Path, ...]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures = {
        "validation_scatter.svg": theory_vs_simulation_scatter_svg(result),
        "validation_error_by_frequency.svg": error_by_frequency_svg(result),
        "validation_spectrum_overlay.svg": spectrum_overlay_svg(result),
        "validation_error_histogram.svg": error_histogram_svg(result),
        "validation_directivity.svg": directivity_comparison_svg(result),
        "validation_error_heatmap.svg": error_heatmap_svg(result),
    }
    paths: list[Path] = []
    for filename, svg in figures.items():
        path = output_path / filename
        path.write_text(svg, encoding="utf-8")
        paths.append(path)
    return tuple(paths)
