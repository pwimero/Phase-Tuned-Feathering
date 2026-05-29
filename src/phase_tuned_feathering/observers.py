"""Observer grids and angular sectors for directivity metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def unit_direction(vector: Sequence[float]) -> tuple[float, float, float]:
    if len(vector) != 3:
        raise ValueError("Directions must have three components.")
    raw = (float(vector[0]), float(vector[1]), float(vector[2]))
    length = _norm(raw)
    if length <= 0.0:
        raise ValueError("Direction cannot be zero length.")
    return (raw[0] / length, raw[1] / length, raw[2] / length)


@dataclass(frozen=True)
class ObserverGrid:
    directions: tuple[tuple[float, float, float], ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.directions) != len(self.weights):
            raise ValueError("directions and weights must have the same length.")
        if not self.directions:
            raise ValueError("ObserverGrid cannot be empty.")
        normalized = tuple(unit_direction(direction) for direction in self.directions)
        object.__setattr__(self, "directions", normalized)
        object.__setattr__(self, "weights", tuple(float(weight) for weight in self.weights))

    @classmethod
    def from_directions(
        cls,
        directions: Iterable[Sequence[float]],
        weights: Iterable[float] | None = None,
    ) -> "ObserverGrid":
        direction_tuple = tuple(unit_direction(direction) for direction in directions)
        if weights is None:
            weight_tuple = tuple(1.0 for _ in direction_tuple)
        else:
            weight_tuple = tuple(float(weight) for weight in weights)
        return cls(direction_tuple, weight_tuple)

    @classmethod
    def six_axis(cls) -> "ObserverGrid":
        directions = (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
        return cls.from_directions(directions)

    @classmethod
    def spherical(
        cls,
        n_polar: int = 9,
        n_azimuth: int = 16,
    ) -> "ObserverGrid":
        if n_polar < 2 or n_azimuth < 3:
            raise ValueError("n_polar >= 2 and n_azimuth >= 3 are required.")
        directions: list[tuple[float, float, float]] = []
        weights: list[float] = []
        dtheta = math.pi / (n_polar - 1)
        dphi = 2.0 * math.pi / n_azimuth
        for polar_index in range(n_polar):
            theta = polar_index * dtheta
            sin_theta = math.sin(theta)
            polar_weight = max(sin_theta * dtheta * dphi, 1.0e-12)
            for azimuth_index in range(n_azimuth):
                phi = azimuth_index * dphi
                directions.append(
                    (
                        sin_theta * math.cos(phi),
                        sin_theta * math.sin(phi),
                        math.cos(theta),
                    )
                )
                weights.append(polar_weight)
        return cls(tuple(directions), tuple(weights))

