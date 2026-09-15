"""LoRaWAN MAC/PHY simulation core.

This is the module that stands in for LoRaWANSim (Marini et al. 2021),
the MATLAB simulator used by Correia et al. (2023). Its internal collision
and ADR algorithms are not published in the paper, so this module
implements the standard academic model used across the LoRaWAN simulation
literature (Bor et al. 2016; Croce et al. 2018; the FLoRa/LoRaSim family):

  - Unslotted-ALOHA-style channel access: each ED transmits once per cycle
    at a random offset within the cycle (unsynchronized clocks).
  - A gateway receives a packet if its power is above the SF's sensitivity
    AND it does not collide destructively with another same-SF packet that
    overlaps in time at that gateway (capture effect: if the power
    difference exceeds `capture_threshold_db`, the stronger packet
    survives; otherwise, all overlapping packets on that SF are lost at
    that gateway).
  - Packets on different SFs are treated as (quasi-)orthogonal, as is
    standard in this class of simplified models.
  - A packet is uplink-delivered if received by ANY gateway (network
    server de-duplicates), matching ULDR's definition (Eq., Section 3.1).
  - The Network Server runs Semtech-style ADR every 20 uplinks per ED
    (lora_phy.ADR_HISTORY_LEN), based on the best SNR margin observed
    across the gateways that received it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import lora_phy as phy
import propagation as prop
from energy_model import cycle_energy, CycleEnergyBreakdown

CAPTURE_THRESHOLD_DB = 6.0  # standard LoRa co-SF capture margin (Croce et al. 2018)


@dataclass
class SimulationResult:
    delivered: np.ndarray       # (n_measure_cycles, n_ed) bool
    transmitted: np.ndarray     # (n_measure_cycles, n_ed) bool (always True here, kept for generality)
    sf_history: np.ndarray      # (n_measure_cycles, n_ed) int
    tx_power_history: np.ndarray  # (n_measure_cycles, n_ed) int
    energy_breakdowns: list = field(default_factory=list)  # one CycleEnergyBreakdown per (cycle, ed)

    @property
    def uldr(self) -> float:
        return self.delivered.sum() / self.transmitted.sum()


def _overlap_clusters(start: np.ndarray, toa: np.ndarray) -> np.ndarray:
    """Sweep-line grouping of overlapping [start, start+toa) intervals.
    Returns a cluster id per item (in the *original* index order)."""
    order = np.argsort(start)
    cluster_id = np.empty(len(start), dtype=int)
    current_id = -1
    current_end = -np.inf
    for idx in order:
        if start[idx] >= current_end:
            current_id += 1
        cluster_id[idx] = current_id
        current_end = max(current_end, start[idx] + toa[idx])
    return cluster_id


def _resolve_reception(
    cluster_id: np.ndarray, rx_power_dbm: np.ndarray, sensitivity_dbm: float, capture_threshold_db: float
) -> np.ndarray:
    """Per gateway, per SF group: within each time-overlap cluster, the
    strongest packet survives only if it beats the runner-up by
    >= capture_threshold_db; isolated packets (cluster size 1) just need
    to clear sensitivity. Used when orthogonal_sf=True (the base paper's
    simplifying assumption: different SFs never interfere)."""
    received = np.zeros(len(cluster_id), dtype=bool)
    for cid in np.unique(cluster_id):
        members = np.where(cluster_id == cid)[0]
        powers = rx_power_dbm[members]
        if len(members) == 1:
            received[members[0]] = powers[0] >= sensitivity_dbm
            continue
        order = np.argsort(powers)[::-1]
        strongest, runner_up = powers[order[0]], powers[order[1]]
        if strongest >= sensitivity_dbm and (strongest - runner_up) >= capture_threshold_db:
            received[members[order[0]]] = True
    return received


def _resolve_reception_cross_sf(
    cluster_id: np.ndarray, sf_arr: np.ndarray, rx_power_dbm: np.ndarray, capture_threshold_db: float
) -> np.ndarray:
    """Per gateway, across ALL SFs jointly (orthogonal_sf=False): a packet
    survives only if, against every *other* time-overlapping packet
    (regardless of its SF), the victim-minus-interferer power ratio clears
    lora_phy.sir_threshold_db(victim_sf, interferer_sf). Clusters are
    small (concurrent transmissions are rare relative to the cycle
    period), so the O(k^2) pairwise check per cluster is cheap.

    Simplification: cluster membership (built once, ignoring SF, purely
    from time overlap) is used as a proxy for "every pair in this group
    could interfere" -- for a 3+-way cluster this can include pairs that
    do not directly overlap in time (only transitively, via a third
    packet), which slightly *overestimates* collisions. Conservative and
    standard practice for this class of simplified simulator."""
    n = len(cluster_id)
    received = np.zeros(n, dtype=bool)
    for cid in np.unique(cluster_id):
        members = np.where(cluster_id == cid)[0]
        powers = rx_power_dbm[members]
        sfs = sf_arr[members]
        for local_i, victim in enumerate(members):
            if powers[local_i] < phy.SENSITIVITY_DBM[sfs[local_i]]:
                continue
            survives = True
            for local_j in range(len(members)):
                if local_j == local_i:
                    continue
                required = phy.sir_threshold_db(sfs[local_i], sfs[local_j], capture_threshold_db)
                if (powers[local_i] - powers[local_j]) < required:
                    survives = False
                    break
            received[victim] = survives
    return received


def run_simulation(
    ed_xy: np.ndarray,
    gw_xy: np.ndarray,
    *,
    cycle_period_s: float,
    n_warmup_cycles: int,
    n_measure_cycles: int,
    payload_bytes: int = 20,
    ed_antenna_gain_dbi: float = 5.0,
    gw_antenna_gain_dbi: float = 5.0,
    capture_threshold_db: float = CAPTURE_THRESHOLD_DB,
    orthogonal_sf: bool = True,
    seed: int = 0,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    n_ed, n_gw = len(ed_xy), len(gw_xy)
    if n_gw == 0:
        raise ValueError("At least one gateway is required")

    dist_m = prop.distances_m(ed_xy, gw_xy)  # (n_ed, n_gw)

    tx_power = np.full(n_ed, 14, dtype=int)
    nearest_dist_m = dist_m.min(axis=1)
    mean_rx_power_nearest = prop.received_power_dbm(
        tx_power, ed_antenna_gain_dbi, gw_antenna_gain_dbi, nearest_dist_m, rng=None
    )
    sf = phy.initial_sf(mean_rx_power_nearest)
    snr_margin_buffer = [[] for _ in range(n_ed)]
    cycles_since_adr = np.zeros(n_ed, dtype=int)

    total_cycles = n_warmup_cycles + n_measure_cycles
    delivered = np.zeros((n_measure_cycles, n_ed), dtype=bool)
    sf_hist = np.zeros((n_measure_cycles, n_ed), dtype=int)
    pw_hist = np.zeros((n_measure_cycles, n_ed), dtype=int)
    energy_breakdowns: list[CycleEnergyBreakdown] = []

    for cycle in range(total_cycles):
        measuring = cycle >= n_warmup_cycles
        start = rng.uniform(0.0, cycle_period_s, size=n_ed)
        toa = np.array([phy.time_on_air_s(payload_bytes, s) for s in sf])

        rx_power = prop.received_power_dbm(
            tx_power[:, None], ed_antenna_gain_dbi, gw_antenna_gain_dbi, dist_m, rng=rng
        )  # (n_ed, n_gw)

        ed_received_any = np.zeros(n_ed, dtype=bool)
        best_snr_margin = np.full(n_ed, -np.inf)

        if orthogonal_sf:
            for s in phy.SF_LEVELS:
                members = np.where(sf == s)[0]
                if len(members) == 0:
                    continue
                cluster_id = _overlap_clusters(start[members], toa[members])
                for g in range(n_gw):
                    received_mask = _resolve_reception(
                        cluster_id, rx_power[members, g], phy.SENSITIVITY_DBM[s], capture_threshold_db
                    )
                    got = members[received_mask]
                    ed_received_any[got] = True
                    margins = phy.snr_margin_db(s, tx_power[got], rx_power[got, g])
                    for ed_i, m in zip(got, margins):
                        best_snr_margin[ed_i] = max(best_snr_margin[ed_i], m)
        else:
            cluster_id = _overlap_clusters(start, toa)  # global: mixes all SFs
            for g in range(n_gw):
                received_mask = _resolve_reception_cross_sf(cluster_id, sf, rx_power[:, g], capture_threshold_db)
                got = np.where(received_mask)[0]
                ed_received_any[got] = True
                margins = phy.snr_margin_db(sf[got], tx_power[got], rx_power[got, g])
                best_snr_margin[got] = np.maximum(best_snr_margin[got], margins)

        cycles_since_adr += 1
        for i in range(n_ed):
            if ed_received_any[i]:
                snr_margin_buffer[i].append(best_snr_margin[i])
            # The NS re-evaluates ADR every ADR_HISTORY_LEN transmit
            # opportunities using whatever uplinks it actually received
            # (not "20 successful receptions", which could stall an ED in
            # a persistently bad/congested link at SF12 indefinitely).
            if cycles_since_adr[i] >= phy.ADR_HISTORY_LEN:
                if snr_margin_buffer[i]:
                    mean_margin = float(np.mean(snr_margin_buffer[i]))
                    sf[i], tx_power[i] = phy.adr_step(sf[i], tx_power[i], mean_margin)
                snr_margin_buffer[i] = []
                cycles_since_adr[i] = 0

        if measuring:
            m_idx = cycle - n_warmup_cycles
            delivered[m_idx] = ed_received_any
            sf_hist[m_idx] = sf
            pw_hist[m_idx] = tx_power
            if cycle == total_cycles - 1:  # energy snapshot: converged population, one sample per ED
                for i in range(n_ed):
                    energy_breakdowns.append(
                        cycle_energy(int(sf[i]), int(tx_power[i]), payload_bytes, cycle_period_s, rng=rng)
                    )

    return SimulationResult(
        delivered=delivered,
        transmitted=np.ones_like(delivered),
        sf_history=sf_hist,
        tx_power_history=pw_hist,
        energy_breakdowns=energy_breakdowns,
    )
