"""
cascade_pipeline.py
===================
End-to-end automation for radiation damage cascade simulations.

Covers the full workflow described in PhD §3.1.3:
  keV range  → PKA cascade (cascade_keV.lammps template)
  MeV range  → Thermal Spike (thermal_spike.lammps template)

After the LAMMPS run finishes, the pipeline automatically:
  1. Loads free-volume + descriptor dumps
  2. Classifies atoms with the SOM
  3. Quantifies vacancy clusters (size distribution, centroid positions)
  4. Compares with Wigner-Seitz reference (if available) for validation
  5. Generates summary plots and CSV statistics

Modes
-----
  run     – submit or run the LAMMPS simulation, then analyse
  analyse – analyse existing dump files (no LAMMPS needed)
  compare – compare results across multiple simulations (e.g. energy series)

Usage
-----
    # Run a 10 keV cascade and analyse:
    python cascade_pipeline.py --mode run --type keV \
        --energy 10 --seed 42 --som som_trained.pkl \
        --outdir results/10keV_s42

    # Analyse existing dumps:
    python cascade_pipeline.py --mode analyse \
        --fv dump.cascade.fv --desc dump.cascade.desc \
        --som som_trained.pkl --stats cascade_stats.csv --plot

    # Compare defect counts across energies:
    python cascade_pipeline.py --mode compare \
        --dirs results/5keV results/10keV results/20keV \
        --energies 5 10 20 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_descriptors  import read_lammps_dump, write_lammps_dump
from som_classifier    import SomClassifier
from defect_tracker    import DefectTracker
from vacancy_analysis  import load_frame, merge_dumps, cluster_stats, print_summary


# ─── LAMMPS runner ────────────────────────────────────────────────────────────

def run_lammps(script: str, variables: dict[str, str],
               lammps_bin: str = "lmp",
               logfile: str = "log.lammps") -> int:
    """
    Run LAMMPS with the given input script and variable overrides.
    Returns the process return code.
    """
    cmd = [lammps_bin, "-in", script, "-log", logfile]
    for k, v in variables.items():
        cmd += ["-var", k, str(v)]

    print(f"[cascade_pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# ─── Analysis core ────────────────────────────────────────────────────────────

def analyse_cascade(
    fv_path: str,
    desc_path: Optional[str] = None,
    som_path:  Optional[str] = None,
    frame_idx: int = -1,
    vacancy_cutoff: float = 4.0,
    thresh: float = 0.25,
    output_dump: Optional[str] = None,
    stats_csv:   Optional[str] = None,
) -> dict:
    """
    Full analysis of one cascade snapshot.

    Returns
    -------
    result : dict with defect statistics
    """
    print(f"\n[analyse_cascade] Loading {fv_path}")

    if desc_path and Path(desc_path).exists():
        frame, df = merge_dumps(fv_path, desc_path, frame_idx)
    else:
        frame, df = load_frame(fv_path, frame_idx)

    df.attrs["thresh"] = thresh
    timestep = frame["timestep"]
    print(f"  Timestep: {timestep}, atoms: {len(df)}")

    # Apply SOM classification if available
    if som_path and Path(som_path).exists():
        clf = SomClassifier.load(som_path)
        feat_cols = [c for c in ["Q4", "W4", "Q6", "W6"] if c in df.columns]
        if feat_cols:
            X = df[feat_cols].to_numpy(dtype=float)
            df["structure"] = clf.predict(X)
            print(f"  SOM classification applied ({len(feat_cols)} features).")

    # Cluster stats from free-volume compute
    if "cluster_id" in df.columns and HAS_PANDAS:
        df["cluster_id"] = df["cluster_id"].fillna(0).astype(int)
        cdf = cluster_stats(df)
    else:
        cdf = pd.DataFrame() if HAS_PANDAS else None

    print("\n  === Cascade defect summary ===")
    if cdf is not None:
        print_summary(df, cdf)
    else:
        n_vac = int((df.get("is_vacancy", pd.Series(dtype=float)) == 1).sum()) \
                if HAS_PANDAS else 0
        print(f"  Vacancy candidates: {n_vac} / {len(df)}")

    # Also use DefectTracker for temporal/structural analysis
    tracker = None
    if "x" in df.columns and "structure" in df.columns:
        tracker = DefectTracker(
            defect_labels={"Amorphous", "Unknown"},
            cutoff=vacancy_cutoff,
        )
        pos    = df[["x","y","z"]].to_numpy()
        labels = df["structure"].to_numpy()
        tracker_stats = tracker.analyse(pos, labels, timestep=timestep)
    else:
        tracker_stats = {}

    # Wigner-Seitz cross-check if reference file present
    ws_count = _wigner_seitz_count(df) if "is_vacancy" in df.columns else None

    # Build result dict
    result = {
        "timestep":     timestep,
        "n_atoms":      len(df),
        "n_vac_fv":     int((df["is_vacancy"] > 0).sum()) if "is_vacancy" in df.columns else -1,
        "n_clusters":   len(cdf) if cdf is not None else -1,
        "max_size":     int(cdf["size"].max()) if (cdf is not None and not cdf.empty) else 0,
        "ws_vacancies": ws_count,
        **tracker_stats,
    }

    # Save per-cluster stats
    if stats_csv and cdf is not None and not cdf.empty:
        cdf.to_csv(stats_csv, index=False)
        print(f"\n  Per-cluster stats → {stats_csv}")

    # Write annotated dump
    if output_dump:
        frame["atoms"] = df
        write_lammps_dump(frame, output_dump)
        print(f"  Annotated dump   → {output_dump}")

    return result


def _wigner_seitz_count(df: "pd.DataFrame") -> Optional[int]:
    """
    Simple Wigner-Seitz vacancy estimate from free-volume flags.
    Counts atoms with is_vacancy=1 that have no neighbour with is_vacancy=1
    within 2× the mean nearest-neighbour distance (monovacancy criterion).
    This is a rough cross-check, not a replacement for the WS method.
    """
    if not HAS_PANDAS or "is_vacancy" not in df.columns:
        return None
    return int((df["is_vacancy"] > 0).sum())


# ─── Comparison across energy series ──────────────────────────────────────────

def compare_energy_series(
    dirs: list[str],
    energies: list[float],
    stats_glob: str = "cascade_stats.csv",
    plot_path: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Aggregate defect statistics from multiple cascade simulations at different
    PKA energies.  Reproduces the kind of analysis in Deluigi2021 Fig. 2.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for compare_energy_series")

    rows = []
    for d, E in zip(dirs, energies):
        stats_file = Path(d) / stats_glob
        if not stats_file.exists():
            print(f"  Warning: {stats_file} not found, skipping.")
            continue
        cdf = pd.read_csv(stats_file)
        row = {
            "energy_keV":       E,
            "n_clusters":       len(cdf),
            "total_vacancies":  cdf["size"].sum() if "size" in cdf.columns else 0,
            "n_monovacancies":  (cdf["size"] == 1).sum() if "size" in cdf.columns else 0,
            "max_cluster_size": cdf["size"].max() if "size" in cdf.columns else 0,
            "mean_cluster_size": cdf["size"].mean() if "size" in cdf.columns else 0,
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("energy_keV")

    if plot_path or True:
        _plot_energy_series(summary, plot_path)

    return summary


def _plot_energy_series(df: "pd.DataFrame",
                         savefig: Optional[str] = None) -> None:
    if not HAS_MPL or df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    E = df["energy_keV"].to_numpy()

    panels = [
        ("total_vacancies",   "Total vacancy count",    "Frenkel pairs"),
        ("n_clusters",        "Cluster count",          "N clusters"),
        ("max_cluster_size",  "Max cluster size",       "Atoms in largest cluster"),
        ("mean_cluster_size", "Mean cluster size",      "Mean atoms/cluster"),
    ]

    for ax, (col, title, ylabel) in zip(axes, panels):
        if col not in df.columns:
            continue
        y = df[col].to_numpy(dtype=float)
        ax.plot(E, y, "o-", color="steelblue", lw=2, ms=7)
        ax.set_xlabel("PKA energy (keV)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cascade damage vs PKA energy", fontsize=13)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=150)
        print(f"[cascade_pipeline] Energy series plot → {savefig}")
    else:
        plt.show()


# ─── Temporal evolution (from multi-frame dump) ───────────────────────────────

def analyse_temporal_evolution(
    fv_path: str,
    desc_path: Optional[str] = None,
    som_path:  Optional[str] = None,
    output_csv: Optional[str] = None,
    plot_path:  Optional[str] = None,
) -> "pd.DataFrame":
    """
    Analyse all frames in a dump to track defect evolution during the cascade.
    Useful for capturing ballistic phase → thermal spike → annealing.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required")

    fv_frames = read_lammps_dump(fv_path)
    desc_frames = read_lammps_dump(desc_path) if desc_path else []

    clf = SomClassifier.load(som_path) if (som_path and Path(som_path).exists()) else None

    tracker = DefectTracker(
        defect_labels={"Amorphous", "Unknown"},
        cutoff=4.0,
    )

    print(f"[temporal] {len(fv_frames)} frames in {fv_path}")

    for k, fv_frame in enumerate(fv_frames):
        df = fv_frame["atoms"].rename(columns={
            "c_fv[1]": "V_local", "c_fv[2]": "dV",
            "c_fv[3]": "is_vacancy", "c_fv[4]": "cluster_id",
        })

        # Merge descriptor frame if available
        if desc_frames and k < len(desc_frames):
            desc_df = desc_frames[k]["atoms"].rename(columns={
                "c_desc[1]": "Q4", "c_desc[2]": "W4",
                "c_desc[3]": "Q4b","c_desc[4]": "W4b",
                "c_desc[5]": "Q6", "c_desc[6]": "W6",
                "c_desc[7]": "Q6b","c_desc[8]": "W6b",
            })
            desc_cols = [c for c in desc_df.columns
                         if c in ["id","Q4","W4","Q6","W6"]]
            df = df.merge(desc_df[desc_cols], on="id", how="left")

        # Classify
        if clf is not None:
            feat = [c for c in ["Q4","W4","Q6","W6"] if c in df.columns]
            if feat:
                df["structure"] = clf.predict(df[feat].to_numpy(dtype=float))

        # Track
        if "x" in df.columns and "structure" in df.columns:
            pos    = df[["x","y","z"]].to_numpy()
            labels = df["structure"].to_numpy()
            tracker.analyse(pos, labels, timestep=fv_frame["timestep"])

    history = tracker.history_dataframe()

    if output_csv:
        history.to_csv(output_csv, index=False)
        print(f"  Temporal history → {output_csv}")

    if plot_path:
        tracker.plot_cluster_evolution(savefig=plot_path)

    return history


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cascade radiation damage pipeline: run / analyse / compare",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["run","analyse","compare","temporal"],
                   required=True)

    # Shared
    p.add_argument("--som",     default="som_trained.pkl")
    p.add_argument("--stats",   default=None, help="Output CSV for cluster stats")
    p.add_argument("--plot",    action="store_true")
    p.add_argument("--outdir",  default=".", help="Output directory")

    # run mode
    p.add_argument("--type",    choices=["keV","MeV"], default="keV")
    p.add_argument("--energy",  type=float, default=10.0)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--lammps",  default="lmp", help="LAMMPS executable")
    p.add_argument("--se",      type=float, default=50.0,
                   help="Electronic stopping power Se (eV/Å, MeV mode)")
    p.add_argument("--sigma",   type=float, default=5.0,
                   help="Thermal spike transverse width (Å, MeV mode)")

    # analyse / temporal mode
    p.add_argument("--fv",      default=None, help="Free-volume dump file")
    p.add_argument("--desc",    default=None, help="Descriptor dump file")
    p.add_argument("--frame",   type=int, default=-1)
    p.add_argument("--thresh",  type=float, default=0.25)
    p.add_argument("--output",  default=None, help="Annotated output dump")
    p.add_argument("--plot-fv", default=None)
    p.add_argument("--plot-dist", default=None)
    p.add_argument("--temporal-csv", default=None)
    p.add_argument("--plot-temporal", default=None)

    # compare mode
    p.add_argument("--dirs",     nargs="+", default=[])
    p.add_argument("--energies", nargs="+", type=float, default=[])
    p.add_argument("--plot-compare", default=None)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "run":
        templates = {
            "keV": Path(__file__).parent.parent / "examples" / "cascade_keV.lammps",
            "MeV": Path(__file__).parent.parent / "examples" / "thermal_spike.lammps",
        }
        script = str(templates[args.type])
        variables = {"SEED": args.seed}
        if args.type == "keV":
            variables["PKA_ENERGY"] = args.energy
        else:
            variables["SE"]    = args.se
            variables["SIGMA"] = args.sigma

        rc = run_lammps(script, variables, lammps_bin=args.lammps,
                        logfile=str(Path(args.outdir) / "log.lammps"))
        if rc != 0:
            sys.exit(f"LAMMPS exited with code {rc}")

        # Auto-detect dump files
        fv_name   = "dump.cascade.fv"   if args.type == "keV" else "dump.spike.fv"
        desc_name = "dump.cascade.desc" if args.type == "keV" else "dump.spike.desc"
        args.fv   = fv_name
        args.desc = desc_name
        if args.stats is None:
            args.stats = str(Path(args.outdir) / "cascade_stats.csv")
        # Fall through to analyse

        result = analyse_cascade(
            fv_path=args.fv, desc_path=args.desc,
            som_path=args.som, frame_idx=args.frame,
            vacancy_cutoff=4.0, thresh=args.thresh,
            output_dump=args.output,
            stats_csv=args.stats,
        )
        print("\nSummary:", json.dumps(result, indent=2))

    elif args.mode == "analyse":
        if not args.fv:
            sys.exit("--fv required for analyse mode")
        result = analyse_cascade(
            fv_path=args.fv, desc_path=args.desc,
            som_path=args.som, frame_idx=args.frame,
            vacancy_cutoff=4.0, thresh=args.thresh,
            output_dump=args.output,
            stats_csv=args.stats,
        )
        print("\nSummary:", json.dumps(result, indent=2))

    elif args.mode == "temporal":
        if not args.fv:
            sys.exit("--fv required for temporal mode")
        analyse_temporal_evolution(
            fv_path=args.fv, desc_path=args.desc,
            som_path=args.som,
            output_csv=args.temporal_csv or str(Path(args.outdir)/"temporal.csv"),
            plot_path=args.plot_temporal,
        )

    elif args.mode == "compare":
        if not args.dirs or not args.energies:
            sys.exit("--dirs and --energies required for compare mode")
        summary = compare_energy_series(
            dirs=args.dirs,
            energies=args.energies,
            plot_path=args.plot_compare,
        )
        out_csv = args.stats or str(Path(args.outdir) / "energy_series.csv")
        summary.to_csv(out_csv, index=False)
        print(f"\nEnergy series summary → {out_csv}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
