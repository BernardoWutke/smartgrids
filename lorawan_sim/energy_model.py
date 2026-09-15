"""Per-cycle ED energy consumption and the stochastic energy model (Eq. 1).

IMPORTANT CAVEAT (read before trusting absolute mJ numbers):
Correia et al. (2023) get their energy-per-state parameters from a
companion paper (ref. [34]: Correia, Alencar & Assis, 2023, "Stochastic
Modeling and Analysis of the Energy Consumption of Wireless Sensor
Networks", IEEE Lat. Am. Trans.), which is not reproduced in the article
we have and was not available to us. The device currents below are
therefore *typical SX1276-class datasheet figures*, not the authors'
calibrated values. Consequently:
  - the *methodology* here (per-state energy, Eq. 1 hypoexponential PDF,
    chi-square goodness-of-fit) is a faithful reproduction of Sections 3.2
    and 4 of the paper;
  - the *absolute* mJ values will NOT match Tables 5-6 -- only the
    qualitative trends (energy falls as SF/Tx power drop via ADR, mean
    energy converges as the GW count grows, etc.) are expected to match.
Swap the constants below for ref. [34]'s values if you obtain that paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from lora_phy import BANDWIDTH_HZ, time_on_air_s

V_SUPPLY = 3.3  # V

I_SLEEP_A = 2e-6     # deep sleep, radio + MCU off
I_PROC_A = 6e-3      # MCU active + sensor acquisition before each uplink
T_PROC_S = 0.05      # measurement + processing time before transmitting

# Approximate SX1276 RFO-pin current draw vs. output power (Table 2 levels)
I_TX_A = {2: 0.017, 4: 0.019, 6: 0.021, 8: 0.024, 10: 0.028, 12: 0.033, 14: 0.040}
I_RX_A = 0.0113      # continuous-receive current

RX_WINDOW_SYMBOLS = 8   # preamble-detect timeout, in symbol periods
SF_RX2_DEFAULT = 8      # fixed RX2 data rate (RX1DROffset = 0 keeps RX1 at the uplink SF)


@dataclass
class CycleEnergyBreakdown:
    t_proc: float
    t_tx: float
    t_rx1: float
    t_rx2: float
    t_sleep: float
    p_proc: float
    p_tx: float
    p_rx1: float
    p_rx2: float
    p_sleep: float

    @property
    def total_mJ(self) -> float:
        energy_j = (
            self.t_proc * self.p_proc
            + self.t_tx * self.p_tx
            + self.t_rx1 * self.p_rx1
            + self.t_rx2 * self.p_rx2
            + self.t_sleep * self.p_sleep
        )
        return energy_j * 1000.0


def symbol_time_s(sf: int, bw_hz: float = BANDWIDTH_HZ) -> float:
    return (2**sf) / bw_hz


def cycle_energy(
    sf: int, tx_power_dbm: int, payload_bytes: int, cycle_period_s: float,
    rng: np.random.Generator | None = None,
) -> CycleEnergyBreakdown:
    """ToA (Tx) and the RX-window lengths are fixed by the LoRaWAN spec and
    the SF, so they are deterministic. The pre-transmission sensor-read /
    MCU processing time is not spec-defined and genuinely varies cycle to
    cycle (sensor conversion time, MCU wake jitter); when `rng` is given it
    is sampled from Exponential(mean=T_PROC_S), which is what turns the
    per-ED energy into the continuous-ish random variable that Eq. (1)'s
    hypoexponential model assumes (see the module docstring: this is the
    one state we can legitimately treat as stochastic without contradicting
    the deterministic PHY timing)."""
    t_tx = time_on_air_s(payload_bytes, sf)
    t_rx1 = RX_WINDOW_SYMBOLS * symbol_time_s(sf)
    t_rx2 = RX_WINDOW_SYMBOLS * symbol_time_s(SF_RX2_DEFAULT)
    t_proc = rng.exponential(T_PROC_S) if rng is not None else T_PROC_S
    t_sleep = max(cycle_period_s - t_proc - t_tx - t_rx1 - t_rx2, 0.0)

    return CycleEnergyBreakdown(
        t_proc=t_proc, t_tx=t_tx, t_rx1=t_rx1, t_rx2=t_rx2, t_sleep=t_sleep,
        p_proc=V_SUPPLY * I_PROC_A,
        p_tx=V_SUPPLY * I_TX_A[tx_power_dbm],
        p_rx1=V_SUPPLY * I_RX_A,
        p_rx2=V_SUPPLY * I_RX_A,
        p_sleep=V_SUPPLY * I_SLEEP_A,
    )


def hypoexponential_betas(mean_state_energy_j: dict[str, float]) -> np.ndarray:
    """beta_k = 1 / mean_energy_k (Eq. 1: beta_k = alpha_k / mean_power_k,
    where alpha_k = 1 / mean_time_k, so beta_k = 1 / (mean_time_k * mean_power_k)
    = 1 / mean_energy_k)."""
    betas = np.array([max(e, 1e-12) for e in mean_state_energy_j.values()])
    betas = 1.0 / betas
    # Eq. (1) has a removable singularity when two betas coincide; nudge apart.
    order = np.argsort(betas)
    sorted_betas = betas[order]
    for i in range(1, len(sorted_betas)):
        if np.isclose(sorted_betas[i], sorted_betas[i - 1], rtol=1e-9):
            sorted_betas[i] *= 1.0 + 1e-6 * (i + 1)
    betas[order] = sorted_betas
    return betas


def hypoexponential_pdf(betas: np.ndarray):
    """Returns p_E(e) for e in Joules (Eq. 1), vectorized over e."""
    M = len(betas)

    def pdf(e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=float)
        total = np.zeros_like(e)
        for i in range(M):
            denom = np.prod([betas[j] - betas[i] for j in range(M) if j != i])
            coeff = np.prod(betas) / denom if denom != 0 else 0.0
            total += coeff * np.exp(-betas[i] * e)
        total = np.where(e >= 0, total, 0.0)
        return total

    return pdf


def fit_theoretical_energy_model(breakdowns: list[CycleEnergyBreakdown]):
    """Population-average per-state (time, power) -> theoretical hypoexponential
    PDF of total per-cycle energy (Joules), following Section 3.2 / Eq. (1)."""
    states = ["proc", "tx", "rx1", "rx2", "sleep"]
    mean_energy_j = {}
    for s in states:
        t = np.mean([getattr(b, f"t_{s}") for b in breakdowns])
        p = np.mean([getattr(b, f"p_{s}") for b in breakdowns])
        mean_energy_j[s] = t * p
    betas = hypoexponential_betas(mean_energy_j)
    return hypoexponential_pdf(betas), betas


def chi_square_goodness_of_fit(samples_mJ: np.ndarray, pdf_joules, n_bins: int = 12, min_expected: float = 5.0):
    """Chi-square test comparing the empirical energy histogram to the
    theoretical hypoexponential PDF (Section 3.5: 'the smallest p-value
    indicated the best adherence of the model to the data')."""
    samples_j = samples_mJ / 1000.0
    edges = np.linspace(samples_j.min(), samples_j.max() * 1.001, n_bins + 1)

    observed = np.histogram(samples_j, bins=edges)[0].astype(float)
    expected = np.array([
        _integrate(pdf_joules, edges[i], edges[i + 1]) for i in range(n_bins)
    ]) * len(samples_j)

    observed, expected = _merge_small_bins(observed, expected, min_expected)
    expected *= observed.sum() / max(expected.sum(), 1e-12)  # renormalize
    chi2, p_value = stats.chisquare(observed, expected)
    return chi2, p_value


def _integrate(f, a: float, b: float, n: int = 64) -> float:
    x = np.linspace(a, b, n)
    return np.trapz(f(x), x)


def battery_lifetime_days(mean_energy_mJ: float, cycle_period_s: float, battery_capacity_mAh: float = 3000.0) -> float:
    """Field-maintenance metric (innovation over the base paper, which
    stops at per-cycle mJ): the paper reports energy per uplink cycle but
    never turns that into "how long before someone has to walk out to the
    plantation and change a battery" -- the number an actual deployment
    decision hinges on. Default capacity matches a common LoRa-node
    primary cell (3.6 V ER18505 Li-SOCl2, ~3000 mAh); energy is drawn at
    V_SUPPLY, consistent with `cycle_energy`."""
    battery_energy_j = battery_capacity_mAh * 1e-3 * V_SUPPLY * 3600.0
    cycles_per_day = 86400.0 / cycle_period_s
    energy_per_day_j = (mean_energy_mJ / 1000.0) * cycles_per_day
    return battery_energy_j / energy_per_day_j


def _merge_small_bins(observed: np.ndarray, expected: np.ndarray, min_expected: float):
    obs, exp = list(observed), list(expected)
    i = 0
    while i < len(exp) - 1:
        if exp[i] < min_expected:
            exp[i + 1] += exp[i]
            obs[i + 1] += obs[i]
            del exp[i], obs[i]
        else:
            i += 1
    if len(exp) > 1 and exp[-1] < min_expected:
        exp[-2] += exp[-1]
        obs[-2] += obs[-1]
        del exp[-1], obs[-1]
    return np.array(obs), np.array(exp)
