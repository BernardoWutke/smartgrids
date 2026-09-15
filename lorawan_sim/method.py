"""Orchestration pipeline matching Figure 6 of Correia et al. (2023):

    for G = 1 .. G_max:
        place G gateways with each clustering algorithm
        run the LoRaWAN MAC/PHY simulation -> ULDR
        fit the theoretical energy model to the simulated energy sample
        chi-square test -> p-value ("smallest p-value = best model fit,
        i.e. energy consumption closest to the theoretical minimum")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clustering import ALGORITHMS, place_gateways
from energy_model import fit_theoretical_energy_model, chi_square_goodness_of_fit
from simulator import run_simulation, SimulationResult


@dataclass
class GwCountResult:
    n_gw: int
    algorithm: str
    uldr: float
    mean_energy_mJ: float
    min_energy_mJ: float
    max_energy_mJ: float
    p_value: float
    sim: SimulationResult


def evaluate_gw_count(
    ed_xy: np.ndarray,
    algorithm: str,
    n_gw: int,
    *,
    cycle_period_s: float,
    n_warmup_cycles: int,
    n_measure_cycles: int,
    payload_bytes: int,
    seed: int,
    orthogonal_sf: bool = True,
) -> GwCountResult:
    gw_xy = place_gateways(
        algorithm, ed_xy, n_gw, seed,
        cycle_period_s=cycle_period_s, payload_bytes=payload_bytes,
    )
    sim = run_simulation(
        ed_xy, gw_xy,
        cycle_period_s=cycle_period_s,
        n_warmup_cycles=n_warmup_cycles,
        n_measure_cycles=n_measure_cycles,
        payload_bytes=payload_bytes,
        orthogonal_sf=orthogonal_sf,
        seed=seed,
    )

    energies_mJ = np.array([b.total_mJ for b in sim.energy_breakdowns])
    pdf, _betas = fit_theoretical_energy_model(sim.energy_breakdowns)
    _chi2, p_value = chi_square_goodness_of_fit(energies_mJ, pdf)

    return GwCountResult(
        n_gw=n_gw,
        algorithm=algorithm,
        uldr=sim.uldr,
        mean_energy_mJ=float(energies_mJ.mean()),
        min_energy_mJ=float(energies_mJ.min()),
        max_energy_mJ=float(energies_mJ.max()),
        p_value=p_value,
        sim=sim,
    )


def run_method(
    ed_xy: np.ndarray,
    *,
    g_max: int,
    cycle_period_s: float,
    n_warmup_cycles: int,
    n_measure_cycles: int,
    payload_bytes: int,
    seed: int,
    algorithms: tuple[str, ...] = ALGORITHMS,
    orthogonal_sf: bool = True,
) -> dict[str, list[GwCountResult]]:
    """Runs the full G=1..g_max sweep for every clustering algorithm."""
    results: dict[str, list[GwCountResult]] = {alg: [] for alg in algorithms}
    for alg in algorithms:
        for n_gw in range(1, g_max + 1):
            results[alg].append(
                evaluate_gw_count(
                    ed_xy, alg, n_gw,
                    cycle_period_s=cycle_period_s,
                    n_warmup_cycles=n_warmup_cycles,
                    n_measure_cycles=n_measure_cycles,
                    payload_bytes=payload_bytes,
                    orthogonal_sf=orthogonal_sf,
                    seed=seed + n_gw,  # decorrelate shadowing/traffic across G
                )
            )
    return results


def recommended_gw_count(records: list[GwCountResult]) -> int:
    """Section 3.5: the method's suggested number of GWs is the one at
    which the chi-square p-value (model-vs-simulation fit) is minimized.
    NaN p-values (degenerate/near-zero-variance energy samples, which can
    happen once ADR has driven the whole population to SF7/min power) are
    excluded from the search."""
    valid = [r for r in records if not np.isnan(r.p_value)]
    pool = valid if valid else records
    return min(pool, key=lambda r: r.p_value).n_gw
