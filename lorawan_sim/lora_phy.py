"""LoRa PHY primitives: time-on-air, receiver sensitivity and the ADR loop.

Sensitivity / required-SNR values are the standard SX127x (BW=125 kHz)
figures used across the LoRaWAN simulation literature (e.g. Bor et al. 2016,
"Do LoRa Low-Power Wide-Area Networks Scale?", and the FLoRa/LoRaSim
simulators) since Correia et al. (2023) delegate this table to their
simulator (LoRaWANSim) without reprinting it.

The ADR step algorithm follows the Semtech "ADR-Simple" recommendation
referenced by the paper ([35] in the article) with a 3 dB step per SNR
margin unit and a 10 dB fixed device margin, as commonly implemented in
open-source LoRaWAN simulators (e.g. TTN, FLoRa).
"""
from __future__ import annotations

import numpy as np

BANDWIDTH_HZ = 125_000.0          # Table 2: BW = 125 kHz
CODING_RATE = 1                   # Table 2: CR = 1 -> actual code rate 4/(4+1)
PREAMBLE_SYMBOLS = 8              # Table 2: L_p = 8 bytes/symbols
HEADER_ENABLED = True             # Table 2: H = true (explicit header)
CRC_ENABLED = True                # standard for uplinks

TX_POWER_LEVELS_DBM = [14, 12, 10, 8, 6, 4, 2]  # Table 2: P_T^ED
SF_LEVELS = [7, 8, 9, 10, 11, 12]

# SX127x sensitivity (dBm) and required SNR (dB) at BW = 125 kHz
SENSITIVITY_DBM = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}
SNR_REQUIRED_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

NOISE_FIGURE_DB = 6.0
NOISE_FLOOR_DBM = -174.0 + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB

ADR_MARGIN_DB = 10.0  # fixed device margin, Semtech ADR-Simple recommendation
ADR_STEP_DB = 3.0
ADR_HISTORY_LEN = 20  # "After 20 transmissions, the algorithm iterates" (Section 2.1)

# --- Non-orthogonal SF interference (innovation over the base paper) ---------
# Correia et al. (2023), like most LoRaWAN capacity studies that build on
# LoRaSim/FLoRa-style models, treat different spreading factors as fully
# orthogonal: a SF9 transmission never destroys a SF7 one, however strong.
# Real LoRa chirps are only *quasi*-orthogonal (Croce et al. 2018; Reynders
# & Pollin 2016): a much stronger interferer on a different SF still
# desensitizes the receiver. SIR_ISOLATION_DB(v, i) is the minimum
# victim-minus-interferer power ratio (dB) the victim SF `v` needs to
# survive an overlapping transmission on SF `i`. It is a monotonic model
# (isolation grows with |SF separation|) in the qualitative pattern
# reported in that literature; the exact dB figures vary by chip and are
# not reproduced verbatim here -- treat them as illustrative, and set
# `orthogonal_sf=True` in simulator.run_simulation to fall back to the
# paper's simplifying assumption.
SF_ISOLATION_BASE_DB = -8.0
SF_ISOLATION_STEP_DB = -4.0


def sir_threshold_db(victim_sf: int, interferer_sf: int, capture_threshold_db: float = 6.0) -> float:
    if victim_sf == interferer_sf:
        return capture_threshold_db
    delta = abs(victim_sf - interferer_sf)
    return SF_ISOLATION_BASE_DB + SF_ISOLATION_STEP_DB * (delta - 1)


def time_on_air_s(
    payload_bytes: int,
    sf: int,
    bw_hz: float = BANDWIDTH_HZ,
    cr: int = CODING_RATE,
    preamble_symbols: int = PREAMBLE_SYMBOLS,
    header_enabled: bool = HEADER_ENABLED,
    crc_enabled: bool = CRC_ENABLED,
) -> float:
    """Semtech AN1200.22 time-on-air formula."""
    t_sym = (2**sf) / bw_hz
    t_preamble = (preamble_symbols + 4.25) * t_sym

    low_dr_optimize = 1 if (sf >= 11 and bw_hz <= 125_000) else 0
    h = 0 if header_enabled else 1
    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc_enabled) - 20 * h
    denominator = 4 * (sf - 2 * low_dr_optimize)
    n_payload = 8 + max(np.ceil(numerator / denominator) * (cr + 4), 0)
    t_payload = n_payload * t_sym
    return t_preamble + t_payload


def initial_sf(mean_rx_power_dbm: np.ndarray, margin_db: float = ADR_MARGIN_DB) -> np.ndarray:
    """Network-aware initial SF assignment: the lowest SF whose mean
    (shadow-free) link budget to the nearest gateway clears sensitivity by
    at least `margin_db`, falling back to SF12 if none does. This mirrors
    how a network-aware deployment (Table 2 GWs are placed with exact
    knowledge of ED positions, same as related work [29] cited in the
    paper) would provision devices, and avoids the unrealistic "cold start
    at SF12" collision deadlock that a naive ADR bootstrap would suffer at
    high offered load."""
    mean_rx_power_dbm = np.asarray(mean_rx_power_dbm, dtype=float)
    sf_out = np.full(mean_rx_power_dbm.shape, SF_LEVELS[-1], dtype=int)
    resolved = np.zeros(mean_rx_power_dbm.shape, dtype=bool)
    for sf in SF_LEVELS:  # ascending: prefer the fastest/lowest-SF that clears margin
        ok = (~resolved) & (mean_rx_power_dbm - SENSITIVITY_DBM[sf] >= margin_db)
        sf_out[ok] = sf
        resolved |= ok
    return sf_out


def snr_margin_db(sf, tx_power_dbm, rx_power_dbm):
    """Vectorized over `sf` (scalar or array, broadcast against the power args)."""
    required = np.array([SNR_REQUIRED_DB[int(s)] for s in np.atleast_1d(sf)])
    snr_db = np.asarray(rx_power_dbm) - NOISE_FLOOR_DBM
    return snr_db - required - ADR_MARGIN_DB


def adr_step(sf: int, tx_power_dbm: float, mean_margin_db: float) -> tuple[int, float]:
    """One Semtech-style ADR adjustment given the mean SNR margin over the
    last ADR_HISTORY_LEN uplinks. Steps SF down first, then Tx power down
    (to save energy); steps power up first, then SF up, on the way back."""
    n_steps = int(np.floor(mean_margin_db / ADR_STEP_DB))
    sf_idx = SF_LEVELS.index(sf)
    pw_idx = TX_POWER_LEVELS_DBM.index(tx_power_dbm)

    while n_steps > 0:
        if sf_idx > 0:
            sf_idx -= 1
        elif pw_idx < len(TX_POWER_LEVELS_DBM) - 1:
            pw_idx += 1  # walk toward the lowest power (list is high->low)
        else:
            break
        n_steps -= 1
    while n_steps < 0:
        if pw_idx > 0:
            pw_idx -= 1  # walk toward higher power
        elif sf_idx < len(SF_LEVELS) - 1:
            sf_idx += 1
        else:
            break
        n_steps += 1

    return SF_LEVELS[sf_idx], TX_POWER_LEVELS_DBM[pw_idx]
