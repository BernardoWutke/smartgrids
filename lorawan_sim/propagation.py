"""Log-distance path-loss / shadowing propagation model.

Implements Eq. (2) of Correia et al. (2023):

    Pr = Pt + Gt + Gr - 10 n log10(d) + X_sigma

with n = 3.66 and sigma = 3 dB, fitted by the authors from Radio Mobile
terrain data for the Petrolina (Nilo Coelho) plantation (Section 3.3,
Figure 5). Distance d is in meters (matches Figure 5, where RSSI reaches
about -130 dBm at d = 15 km with these parameters).
"""
from __future__ import annotations

import numpy as np

N_PATH_LOSS_EXPONENT = 3.66  # fitted by the authors (Section 3.4), R^2 = 0.97
SIGMA_SHADOWING_DB = 3.0     # Table 2


def received_power_dbm(
    tx_power_dbm: np.ndarray | float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    distance_m: np.ndarray,
    n: float = N_PATH_LOSS_EXPONENT,
    sigma_db: float = SIGMA_SHADOWING_DB,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Received power for one or many links.

    If `rng` is given, an independent log-normal shadowing sample is drawn
    per link (stochastic channel realization). If `rng` is None, the mean
    (shadow-free) link budget is returned, which is what the ADR margin
    calculation should use.
    """
    distance_m = np.maximum(np.asarray(distance_m, dtype=float), 1.0)  # avoid log(0)
    path_loss_db = 10.0 * n * np.log10(distance_m)
    shadow_db = 0.0
    if rng is not None:
        shadow_db = rng.normal(0.0, sigma_db, size=distance_m.shape)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db + shadow_db


def distances_m(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between two point sets (meters)."""
    diff = a_xy[:, None, :] - b_xy[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))
