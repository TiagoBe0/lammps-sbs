"""
vacancy_analysis.py
===================
Post-process LAMMPS dumps produced by compute local_freevol and/or
compute local_descriptor.

Combines free-volume data with ML structure labels to produce a richer
vacancy characterisation than either method alone:

  - Free volume (from local_freevol) detects *where* vacancies are.
  - ML labels  (from local_descriptor + SOM) identify *which structure*
    surrounds each vacancy, enabling e.g. vacancy-in-FCC vs vacancy-in-BCC
    discrimination in multi-phase samples.

Key analyses
------------
1. Cluster size distribution  (monovacancies, di-, tri-vacancies, ...)
2. Spatial map of clusters    (centroid positions + sizes → for OVITO)
3. Vacancy volume distribution (Voronoi free volume per cluster)
4. Time-series: cluster count, max size, defect fraction
5. Joint analysis: free-volume threshold + SOM label agreement rate

Usage
-----
    python vacancy_analysis.py dump.fv [dump.desc] [options]

    # Single-file mode (only free-volume dump):
    python vacancy_analysis.py dump.fv --plot --stats stats.csv

    # Joint mode (free-volume + descriptor dump side by side):
    python vacancy_analysis.py dump.fv --desc dump.desc \
        --som som_trained.pkl --plot --output annotated.dump
"""

from __future__ import annotations

import argparse
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
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Re-use existing modules from this package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_descriptors import read_lammps_dump, write_lammps_dump


# Column names produced by compute local_freevol
FREEVOL_COLS = {
    "c_fv[1]": "V_local",
    "c_fv[2]": "dV",
    "c_fv[3]": "is_vacancy",
    "c_fv[4]": "cluster_id",
}

# Column names produced by compute local_descriptor (l=4,6, average=yes)
DESC_COLS = {
    "c_desc[1]": "Q4",  "c_desc[2]": "W4",
    "c_desc[3]": "Q4b", "c_desc[4]": "W4b",
    "c_desc[5]": "Q6",  "c_desc[6]": "W6",
    "c_desc[7]": "Q6b", "c_desc[8]": "W6b",
}


# ─── Loading helpers ──────────────────────────────────────────────────────────

def load_frame(dump_path: str, frame_idx: int = -1) -> tuple[dict, "pd.DataFrame"]:
    frames = read_lammps_dump(dump_path)
    if not frames:
        sys.exit(f"No frames in {dump_path}")
    frame = frames[frame_idx]
    df = frame["atoms"].rename(columns={**FREEVOL_COLS, **DESC_COLS})
    return frame, df


def merge_dumps(fv_path: str, desc_path: str,
                frame_idx: int = -1) -> tuple[dict, "pd.DataFrame"]:
    """
    Merge free-volume and descriptor dumps by atom id.
    Both dumps must cover the same timestep.
    """
    fv_frame,   fv_df   = load_frame(fv_path,   frame_idx)
    desc_frame, desc_df = load_frame(desc_path,  frame_idx)

    merged = fv_df.merge(
        desc_df[[c for c in desc_df.columns
                 if c in list(DESC_COLS.values()) + ["id"]]],
        on="id", how="left",
    )
    return fv_frame, merged


# ─── Cluster statistics ───────────────────────────────────────────────────────

def cluster_stats(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Per-cluster summary: size, mean/std free volume, centroid, composition.

    Parameters
    ----------
    df : DataFrame with columns cluster_id (>0), V_local, dV,
         and optionally x y z and structure.

    Returns
    -------
    cdf : DataFrame indexed by cluster_id with descriptive statistics.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required")

    vac = df[df["cluster_id"] > 0].copy()
    if vac.empty:
        return pd.DataFrame()

    agg = {"dV": ["count", "mean", "std", "max"],
           "V_local": ["mean", "sum"]}
    if "x" in vac.columns:
        agg.update({"x": "mean", "y": "mean", "z": "mean"})
    if "structure" in vac.columns:
        # dominant structure label per cluster
        def dominant(s):
            return s.value_counts().index[0]
        extra = vac.groupby("cluster_id")["structure"].agg(dominant)
    else:
        extra = None

    cdf = vac.groupby("cluster_id").agg(agg)
    cdf.columns = ["_".join(c).strip("_") for c in cdf.columns]
    cdf = cdf.rename(columns={"dV_count": "size",
                               "dV_mean":  "mean_dV",
                               "dV_std":   "std_dV",
                               "dV_max":   "max_dV",
                               "V_local_mean": "mean_V_local",
                               "V_local_sum":  "total_V_free",
                               "x_mean": "cx", "y_mean": "cy", "z_mean": "cz"})
    if extra is not None:
        cdf["dominant_structure"] = extra

    return cdf.reset_index()


def size_distribution(cdf: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray]:
    """Return (sizes, counts) arrays from cluster DataFrame."""
    if cdf.empty:
        return np.array([]), np.array([])
    sizes, counts = np.unique(cdf["size"].to_numpy(), return_counts=True)
    return sizes, counts


def print_summary(df: "pd.DataFrame", cdf: "pd.DataFrame") -> None:
    n_total  = len(df)
    n_vac    = int((df["cluster_id"] > 0).sum()) if "cluster_id" in df else 0
    n_clust  = len(cdf)

    print(f"  Total atoms       : {n_total}")
    print(f"  Vacancy candidates: {n_vac}  ({100*n_vac/n_total:.2f}%)")
    print(f"  Clusters found    : {n_clust}")

    if not cdf.empty:
        sizes, counts = size_distribution(cdf)
        print(f"  Size distribution :")
        for s, c in zip(sizes, counts):
            tag = "(monovacancy)" if s == 1 else \
                  "(di-vacancy)"  if s == 2 else \
                  "(tri-vacancy)" if s == 3 else ""
            bar = "█" * min(c, 40)
            print(f"    size {s:4d}: {c:5d}  {bar} {tag}")

        if "dominant_structure" in cdf.columns:
            print("\n  Structure around vacancies:")
            print("  " + cdf["dominant_structure"].value_counts().to_string())


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_freevol_distribution(df: "pd.DataFrame",
                               savefig: Optional[str] = None) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    dv = df["dV"].to_numpy()
    vl = df["V_local"].to_numpy()

    # δV histogram
    ax = axes[0]
    ax.hist(dv, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0.0,   color="k",    lw=1,   ls="--", label="δV = 0")
    thresh = df.attrs.get("thresh", 0.30)
    ax.axvline(thresh, color="red", lw=1.5, ls="--", label=f"threshold ({thresh})")
    ax.set_xlabel("δV  (normalised excess volume)")
    ax.set_ylabel("Atom count")
    ax.set_title("Local free-volume distribution")
    ax.legend()

    # Spatial map (x-y projection) coloured by cluster_id
    ax = axes[1]
    if "x" in df.columns and "cluster_id" in df.columns:
        bulk = df[df["cluster_id"] == 0]
        vacs = df[df["cluster_id"] >  0]
        ax.scatter(bulk["x"], bulk["y"], s=1, c="lightgray", alpha=0.3, label="bulk")
        if not vacs.empty:
            sc = ax.scatter(vacs["x"], vacs["y"], s=10,
                            c=vacs["cluster_id"], cmap="tab20",
                            norm=mcolors.Normalize(vmin=1, vmax=vacs["cluster_id"].max()),
                            label="vacancy cluster", zorder=5)
            plt.colorbar(sc, ax=ax, label="cluster id")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_title("Vacancy clusters (x-y projection)")
        ax.legend(markerscale=4, fontsize=8)
    else:
        ax.text(0.5, 0.5, "x/y columns not available",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
        print(f"  Plot saved → {savefig}")
    else:
        plt.show()


def plot_size_distribution(cdf: "pd.DataFrame",
                            log_y: bool = True,
                            savefig: Optional[str] = None) -> None:
    if not HAS_MPL or cdf.empty:
        return
    sizes, counts = size_distribution(cdf)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sizes, counts, color="steelblue", edgecolor="k", alpha=0.8)
    ax.set_xlabel("Cluster size (number of atoms)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Vacancy cluster size distribution")
    if log_y and counts.max() > 10:
        ax.set_yscale("log")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
        print(f"  Size distribution plot → {savefig}")
    else:
        plt.show()


def plot_joint_freevol_structure(df: "pd.DataFrame",
                                  savefig: Optional[str] = None) -> None:
    """
    For each structure label, show the δV distribution.
    Reveals whether certain crystal phases are more prone to vacancy formation.
    """
    if not HAS_MPL or "structure" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    structure_colors = {
        "FCC": "green", "HCP": "red", "BCC": "blue",
        "ICO": "purple", "Amorphous": "gray", "Unknown": "black",
    }
    for lbl, grp in df.groupby("structure"):
        c = structure_colors.get(lbl, "gray")
        ax.hist(grp["dV"], bins=50, alpha=0.5, color=c, label=lbl, density=True)

    ax.set_xlabel("δV (normalised excess volume)")
    ax.set_ylabel("Probability density")
    ax.set_title("Free-volume distribution by crystal phase")
    ax.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    else:
        plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse vacancy clusters from compute local_freevol dumps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dump_fv",
                   help="LAMMPS dump with c_fv[1..4] columns")
    p.add_argument("--desc",       default=None,
                   help="Optional dump with c_desc[1..8] for joint analysis")
    p.add_argument("--som",        default=None,
                   help="SOM model (.pkl) for structure classification "
                        "(used only with --desc)")
    p.add_argument("--frame",      type=int, default=-1,
                   help="Frame index to load (-1 = last)")
    p.add_argument("--stats",      default=None,
                   help="Output per-cluster stats CSV")
    p.add_argument("--output",     default=None,
                   help="Output annotated dump file")
    p.add_argument("--plot",       action="store_true",
                   help="Show interactive plots")
    p.add_argument("--savefig",    default=None,
                   help="Save free-volume/cluster map to file")
    p.add_argument("--save-size",  default=None,
                   help="Save size-distribution plot to file")
    p.add_argument("--save-joint", default=None,
                   help="Save joint free-volume/structure plot to file")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    print(f"Loading free-volume dump: {args.dump_fv}")

    if args.desc:
        frame, df = merge_dumps(args.dump_fv, args.desc, args.frame)
        # Optionally apply SOM classifier to descriptor columns
        if args.som and Path(args.som).exists():
            from som_classifier import SomClassifier
            clf = SomClassifier.load(args.som)
            feat_cols = [c for c in ["Q4","W4","Q6","W6"] if c in df.columns]
            if feat_cols:
                X = df[feat_cols].to_numpy(dtype=float)
                df["structure"] = clf.predict(X)
                print("  Applied SOM classifier → structure labels assigned.")
    else:
        frame, df = load_frame(args.dump_fv, args.frame)

    print(f"  Timestep {frame['timestep']}: {len(df)} atoms")
    print(f"  Columns: {list(df.columns)}")

    # Cluster stats
    if "cluster_id" not in df.columns:
        print("  Warning: 'cluster_id' column not found. "
              "Run with compute local_freevol vcut <value>.")
        cdf = pd.DataFrame() if HAS_PANDAS else None
    else:
        df["cluster_id"] = df["cluster_id"].fillna(0).astype(int)
        cdf = cluster_stats(df) if HAS_PANDAS else None

    print("\n=== Vacancy summary ===")
    if cdf is not None:
        print_summary(df, cdf)
    else:
        n_vac = int((df.get("is_vacancy", pd.Series(dtype=float)) == 1).sum())
        print(f"  Vacancy candidates: {n_vac} / {len(df)}")

    # Save stats
    if args.stats and cdf is not None and not cdf.empty:
        cdf.to_csv(args.stats, index=False)
        print(f"\n  Per-cluster stats → {args.stats}")

    # Write annotated dump
    if args.output:
        frame["atoms"] = df
        write_lammps_dump(frame, args.output)
        print(f"  Annotated dump → {args.output}")

    # Plots
    if args.plot or args.savefig:
        plot_freevol_distribution(df, savefig=args.savefig)
    if (args.plot or args.save_size) and cdf is not None:
        plot_size_distribution(cdf, savefig=args.save_size)
    if (args.plot or args.save_joint) and "structure" in df.columns:
        plot_joint_freevol_structure(df, savefig=args.save_joint)


if __name__ == "__main__":
    main()
