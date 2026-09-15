"""End-to-end reproduction of the Correia et al. (2023) simulation study.

Usage:
    python run_scenario.py --scenario both --out-dir outputs

Produces, per scenario, under <out-dir>/scenario<N>/:
  - uldr_table.csv, energy_table.csv   (analogous to Tables 3-6)
  - recommended_gw_count.png           (Figure 7)
  - uldr_by_algorithm.png              (Figure 8)
  - uldr_vs_gw.png                     (Figure 9)
  - energy_vs_gw.png                   (Figure 11)

See environment.py / energy_model.py module docstrings for the two
approximations that were required because they are not in the published
paper: the exact plantation polygon, and the device energy parameters
behind ref. [34]. Everything else (path-loss model, LoRaWAN parameters,
clustering algorithms, ADR, ULDR, energy-model methodology) follows
Correia et al. (2023) directly.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clustering import ALGORITHMS_WITH_PSO
from environment import sample_points_in_circle
from method import run_method, recommended_gw_count, GwCountResult

SCENARIOS = {
    1: dict(cycle_period_s=600.0, g_max=10, label="Scenario 1 (T=600 s, low-medium concurrency)"),
    2: dict(cycle_period_s=60.0, g_max=20, label="Scenario 2 (T=60 s, high-medium concurrency)"),
}

N_ED = 500          # Table 2
AREA_RADIUS_M = 9000.0  # Table 2: R = 9 km
PAYLOAD_BYTES = 20  # Table 2: B_UL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=["1", "2", "both"], default="both")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--n-ed", type=int, default=N_ED)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-cycles", type=int, default=60, help="several ADR_HISTORY_LEN windows so ADR converges")
    parser.add_argument("--measure-cycles", type=int, default=5, help="more cycles = smoother ULDR/energy stats, slower")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS_WITH_PSO)[:4], choices=list(ALGORITHMS_WITH_PSO))
    parser.add_argument("--non-orthogonal-sf", action="store_true", help="enable cross-SF interference (innovation; default follows the paper's orthogonal-SF assumption)")
    args = parser.parse_args()

    scenario_ids = [1, 2] if args.scenario == "both" else [int(args.scenario)]

    rng = np.random.default_rng(args.seed)
    ed_xy = sample_points_in_circle(args.n_ed, AREA_RADIUS_M, rng)

    for sid in scenario_ids:
        cfg = SCENARIOS[sid]
        print(f"\n=== {cfg['label']} ===")
        out_dir = os.path.join(args.out_dir, f"scenario{sid}")
        os.makedirs(out_dir, exist_ok=True)

        results = run_method(
            ed_xy,
            g_max=cfg["g_max"],
            cycle_period_s=cfg["cycle_period_s"],
            n_warmup_cycles=args.warmup_cycles,
            n_measure_cycles=args.measure_cycles,
            payload_bytes=PAYLOAD_BYTES,
            seed=args.seed,
            algorithms=tuple(args.algorithms),
            orthogonal_sf=not args.non_orthogonal_sf,
        )

        write_uldr_table(results, out_dir)
        write_energy_table(results, out_dir)
        recommended = {alg: recommended_gw_count(recs) for alg, recs in results.items()}
        print("Recommended GW count per algorithm:", recommended)

        plot_recommended_gw_count(recommended, out_dir, cfg["label"])
        plot_uldr_vs_gw(results, out_dir, cfg["label"])
        plot_energy_vs_gw(results, out_dir, cfg["label"])
        plot_uldr_at_recommended(results, recommended, out_dir, cfg["label"])

        print(f"Outputs written to: {out_dir}")


def write_uldr_table(results: dict[str, list[GwCountResult]], out_dir: str) -> None:
    algs = list(results.keys())
    g_max = len(results[algs[0]])
    with open(os.path.join(out_dir, "uldr_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GW"] + [f"v_{a}" for a in algs] + [f"p_{a}" for a in algs])
        for i in range(g_max):
            row = [i + 1]
            row += [f"{results[a][i].uldr:.4f}" for a in algs]
            row += [f"{results[a][i].p_value:.4f}" for a in algs]
            w.writerow(row)


def write_energy_table(results: dict[str, list[GwCountResult]], out_dir: str) -> None:
    algs = list(results.keys())
    g_max = len(results[algs[0]])
    with open(os.path.join(out_dir, "energy_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["GW"]
            + [f"mean_{a}_mJ" for a in algs]
            + [f"min_{a}_mJ" for a in algs]
            + [f"max_{a}_mJ" for a in algs]
        )
        for i in range(g_max):
            row = [i + 1]
            row += [f"{results[a][i].mean_energy_mJ:.2f}" for a in algs]
            row += [f"{results[a][i].min_energy_mJ:.2f}" for a in algs]
            row += [f"{results[a][i].max_energy_mJ:.2f}" for a in algs]
            w.writerow(row)


def plot_recommended_gw_count(recommended: dict[str, int], out_dir: str, title: str) -> None:
    fig, ax = plt.subplots()
    ax.bar(list(recommended.keys()), list(recommended.values()), edgecolor="black", fill=False)
    ax.set_xlabel("Clustering algorithm")
    ax.set_ylabel("Number of GWs")
    ax.set_title(f"Recommended GW count - {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recommended_gw_count.png"), dpi=150)
    plt.close(fig)


def plot_uldr_vs_gw(results: dict[str, list[GwCountResult]], out_dir: str, title: str) -> None:
    fig, ax = plt.subplots()
    for alg, recs in results.items():
        ax.plot([r.n_gw for r in recs], [r.uldr for r in recs], marker="o", label=alg)
    ax.set_xlabel("G (number of gateways)")
    ax.set_ylabel("ULDR")
    ax.set_title(f"ULDR vs. GW count - {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uldr_vs_gw.png"), dpi=150)
    plt.close(fig)


def plot_energy_vs_gw(results: dict[str, list[GwCountResult]], out_dir: str, title: str) -> None:
    fig, ax = plt.subplots()
    for alg, recs in results.items():
        ax.plot([r.n_gw for r in recs], [r.mean_energy_mJ for r in recs], marker="o", label=f"mean ({alg})")
    ax.set_xlabel("G (number of gateways)")
    ax.set_ylabel("Energy per cycle [mJ]")
    ax.set_title(f"Mean energy vs. GW count - {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energy_vs_gw.png"), dpi=150)
    plt.close(fig)


def plot_uldr_at_recommended(
    results: dict[str, list[GwCountResult]], recommended: dict[str, int], out_dir: str, title: str
) -> None:
    fig, ax = plt.subplots()
    algs = list(results.keys())
    values = [results[a][recommended[a] - 1].uldr for a in algs]
    ax.bar(algs, values, edgecolor="black", fill=False)
    ax.set_xlabel("Clustering algorithm")
    ax.set_ylabel("ULDR at recommended GW count")
    ax.set_title(f"ULDR at recommended GW count - {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uldr_by_algorithm.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
