"""Study-area geometry and end-device (ED) placement.

Reproduces Section 3.3/3.4 of Correia et al. (2023): EDs are placed randomly
inside the plantation area. The paper's simulator (LoRaWANSim) took the
area as a circle of radius R = 9 km (Table 2); the real polygonal outline
of the Nilo Coelho / Maria Tereza plantation (Figure 3) is only published
as a map image, not as coordinates, so it cannot be reproduced exactly.
"""
from __future__ import annotations

import numpy as np


def sample_points_in_circle(n: int, radius_m: float, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample n points inside a circle centered at the origin."""
    r = radius_m * np.sqrt(rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def sample_points_in_polygon(n: int, polygon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample n points inside an arbitrary simple polygon.

    polygon: (P, 2) array of vertex coordinates (not necessarily closed).
    Uses rejection sampling against the polygon's bounding box.
    """
    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    points = np.empty((0, 2))
    while points.shape[0] < n:
        batch = rng.uniform(min_xy, max_xy, size=(max(n * 2, 64), 2))
        inside = _points_in_polygon(batch, polygon)
        points = np.vstack([points, batch[inside]])
    return points[:n]


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test."""
    x, y = points[:, 0], points[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    px, py = polygon[:, 0], polygon[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        )
        inside ^= intersect
        j = i
    return inside
