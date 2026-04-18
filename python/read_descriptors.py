"""
read_descriptors.py
====================
Load LAMMPS dump files produced by compute local_descriptor, then
classify atomic environments with a simple scikit-learn classifier.

Typical workflow
----------------
1.  Load dump → DataFrame with one row per atom, columns Q4 W4 Q6 W6
    (and optionally the neighbour-averaged versions Q4b W4b Q6b W6b).
2.  Optionally assign ground-truth labels via a perfect-crystal reference.
3.  Train a classifier (or use the hard-coded Steinhardt thresholds).
4.  Predict phases / defects for every atom and write a new dump for OVITO.

Usage
-----
    python read_descriptors.py dump.desc --output classified.dump --plot

Dependencies
------------
    numpy, pandas, scikit-learn, matplotlib (optional)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── Optional imports ─────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Structure labels and ideal Steinhardt values ─────────────────────────────
# Rows: Q4, W4_hat, Q6, W6_hat  (local, non-averaged)
IDEAL_DESCRIPTORS = {
    "FCC":       np.array([0.191, -0.159,  0.574, -0.013]),
    "HCP":       np.array([0.097, -0.134,  0.485, -0.012]),
    "BCC":       np.array([0.036,  0.159,  0.511,  0.013]),
    "ICO":       np.array([0.000,  0.000,  0.663,  0.017]),
    "Amorphous": np.array([0.050,  0.000,  0.350,  0.000]),
}

STRUCTURE_COLORS = {
    "FCC": "green",
    "HCP": "red",
    "BCC": "blue",
    "ICO": "purple",
    "Amorphous": "gray",
    "Unknown": "black",
}


# ─── I/O ──────────────────────────────────────────────────────────────────────

def read_lammps_dump(path: str | Path) -> list[dict]:
    """Parse a LAMMPS custom dump file; return list of frames as dicts.

    Each dict has keys:
        'timestep': int
        'box':      (3,2) array  [[xlo,xhi], [ylo,yhi], [zlo,zhi]]
        'atoms':    DataFrame
    """
    frames = []
    current: dict = {}
    lines: list[str] = []

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "ITEM: TIMESTEP":
                if current:
                    frames.append(_finalise_frame(current, lines))
                current = {}
                lines = []
                current["_section"] = "timestep"
            elif line.startswith("ITEM: NUMBER"):
                current["_section"] = "natoms"
            elif line.startswith("ITEM: BOX"):
                current["_section"] = "box"
                current["_box"] = []
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]  # column names
                current["_cols"] = cols
                current["_section"] = "atoms"
                current["_atom_lines"] = []
            else:
                _parse_section(current, line)

    if current:
        frames.append(_finalise_frame(current, lines))

    return frames


def _parse_section(state: dict, line: str) -> None:
    sec = state.get("_section", "")
    if sec == "timestep":
        state["timestep"] = int(line)
    elif sec == "natoms":
        state["natoms"] = int(line)
    elif sec == "box":
        vals = list(map(float, line.split()))
        state["_box"].append(vals[:2])
    elif sec == "atoms":
        state["_atom_lines"].append(line.split())


def _finalise_frame(state: dict, _lines: list) -> dict:
    df = pd.DataFrame(state.get("_atom_lines", []),
                      columns=state.get("_cols", []))
    # Cast numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    return {
        "timestep": state.get("timestep", 0),
        "box": np.array(state.get("_box", [[0,1],[0,1],[0,1]])),
        "atoms": df,
    }


def write_lammps_dump(frame: dict, path: str | Path,
                      extra_cols: Optional[list[str]] = None) -> None:
    """Write a single frame back as a LAMMPS dump file."""
    df = frame["atoms"]
    box = frame["box"]
    cols = list(df.columns)
    with open(path, "w") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{frame['timestep']}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{len(df)}\n")
        fh.write("ITEM: BOX BOUNDS pp pp pp\n")
        for lo, hi in box:
            fh.write(f"{lo} {hi}\n")
        fh.write("ITEM: ATOMS " + " ".join(cols) + "\n")
        df.to_csv(fh, sep=" ", index=False, header=False)


# ─── Classification ───────────────────────────────────────────────────────────

def nearest_ideal(row: np.ndarray,
                  feature_cols: list[str]) -> str:
    """Assign structure label by minimum Euclidean distance to ideal values."""
    best, best_d = "Unknown", np.inf
    for label, ideal in IDEAL_DESCRIPTORS.items():
        n = min(len(row), len(ideal))
        d = np.linalg.norm(row[:n] - ideal[:n])
        if d < best_d:
            best_d = d
            best = label
    return best


def classify_nearest(df: pd.DataFrame,
                      feature_cols: list[str]) -> pd.Series:
    """Apply nearest-ideal classifier to every atom."""
    feats = df[feature_cols].to_numpy(dtype=float)
    labels = [nearest_ideal(feats[i], feature_cols) for i in range(len(feats))]
    return pd.Series(labels, index=df.index, name="structure")


def train_random_forest(df: pd.DataFrame,
                         feature_cols: list[str],
                         label_col: str = "structure",
                         **rf_kwargs) -> tuple:
    """Train a Random Forest on labelled data; return (model, scaler, report)."""
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for ML training")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                               test_size=0.2,
                                               random_state=42,
                                               stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=rf_kwargs.get("n_estimators", 200),
                                 max_depth=rf_kwargs.get("max_depth", None),
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    report = classification_report(y_te, rf.predict(X_te_s))
    return rf, scaler, report


def predict_rf(df: pd.DataFrame,
               feature_cols: list[str],
               model, scaler) -> pd.Series:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required")
    X = scaler.transform(df[feature_cols].to_numpy(dtype=float))
    return pd.Series(model.predict(X), index=df.index, name="structure")


# ─── Defect analysis ──────────────────────────────────────────────────────────

def count_defects(df: pd.DataFrame,
                  structure_col: str = "structure",
                  defect_labels: tuple[str, ...] = ("Amorphous",)) -> dict:
    """Return counts and fraction of each label considered a 'defect'."""
    n_total  = len(df)
    n_defect = df[structure_col].isin(defect_labels).sum()
    return {
        "total_atoms": n_total,
        "defect_atoms": int(n_defect),
        "defect_fraction": n_defect / n_total if n_total else 0.0,
        "by_label": df[structure_col].value_counts().to_dict(),
    }


def cluster_vacancies(df: pd.DataFrame,
                      defect_col: str = "structure",
                      defect_label: str = "Amorphous",
                      cutoff: float = 4.0) -> pd.DataFrame:
    """
    Simple distance-based clustering of defect atoms.
    Returns df with added column 'cluster_id' (-1 = not a defect).
    Requires: x, y, z columns in df.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("scikit-learn DBSCAN required for cluster analysis")

    mask = df[defect_col] == defect_label
    df = df.copy()
    df["cluster_id"] = -1

    if mask.sum() == 0:
        return df

    coords = df.loc[mask, ["x", "y", "z"]].to_numpy(dtype=float)
    labels = DBSCAN(eps=cutoff, min_samples=1).fit_predict(coords)
    df.loc[mask, "cluster_id"] = labels
    return df


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_q4_q6(df: pd.DataFrame,
               label_col: Optional[str] = None,
               savefig: Optional[str] = None) -> None:
    if not HAS_MPL:
        print("matplotlib not available – skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    q4_col = "c_desc[1]" if "c_desc[1]" in df.columns else "Q4"
    q6_col = "c_desc[5]" if "c_desc[5]" in df.columns else "Q6"

    if label_col and label_col in df.columns:
        for lbl, grp in df.groupby(label_col):
            color = STRUCTURE_COLORS.get(lbl, "gray")
            ax.scatter(grp[q4_col], grp[q6_col],
                       s=2, alpha=0.4, color=color, label=lbl)
        ax.legend(markerscale=6)
    else:
        ax.scatter(df[q4_col], df[q6_col], s=2, alpha=0.3, color="steelblue")

    # Overlay ideal reference points
    for lbl, vals in IDEAL_DESCRIPTORS.items():
        ax.plot(vals[0], vals[2], "*", ms=14, color=STRUCTURE_COLORS[lbl],
                markeredgecolor="k", label=f"{lbl} (ideal)")

    ax.set_xlabel("Q4", fontsize=12)
    ax.set_ylabel("Q6", fontsize=12)
    ax.set_title("Q4 vs Q6 – local structure map")
    ax.legend(markerscale=1, fontsize=8)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=150)
        print(f"Saved plot → {savefig}")
    else:
        plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load LAMMPS descriptor dump, classify atoms, optionally plot."
    )
    p.add_argument("dump", help="LAMMPS dump file from compute local_descriptor")
    p.add_argument("--frame", type=int, default=-1,
                   help="Frame index to analyse (-1 = last)")
    p.add_argument("--method", choices=["nearest", "rf"], default="nearest",
                   help="Classification method")
    p.add_argument("--output", help="Write classified dump to this file")
    p.add_argument("--plot", action="store_true", help="Show Q4 vs Q6 scatter plot")
    p.add_argument("--savefig", help="Save plot to file instead of showing")
    p.add_argument("--vacancy-cutoff", type=float, default=4.0,
                   help="Cutoff (Å) for vacancy cluster DBSCAN")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    print(f"Reading {args.dump} …")
    frames = read_lammps_dump(args.dump)
    if not frames:
        sys.exit("No frames found in dump file.")
    frame = frames[args.frame]
    df = frame["atoms"]
    print(f"  Timestep {frame['timestep']}: {len(df)} atoms, "
          f"columns: {list(df.columns)}")

    # Detect descriptor columns produced by compute local_descriptor
    desc_cols = [c for c in df.columns if c.startswith("c_desc")]
    if len(desc_cols) < 2:
        sys.exit("Expected columns c_desc[1]…c_desc[N] in dump. "
                 "Did you dump `c_desc[*]`?")

    # Map to friendly names (Q4, W4, Q4b, W4b, Q6, W6, Q6b, W6b …)
    # Assumes: l=4 first, l=6 second, and average=yes (8 columns)
    rename = {}
    col_pairs = [("Q4","W4","Q4b","W4b"), ("Q6","W6","Q6b","W6b")]
    for i, pair in enumerate(col_pairs):
        for j, name in enumerate(pair):
            key = f"c_desc[{i*4 + j + 1}]"
            if key in df.columns:
                rename[key] = name
    df = df.rename(columns=rename)

    feature_cols = [c for c in ["Q4", "W4", "Q6", "W6"] if c in df.columns]
    print(f"  Using features: {feature_cols}")

    # Classify
    if args.method == "nearest" or not HAS_SKLEARN:
        df["structure"] = classify_nearest(df, feature_cols)
    else:
        print("  [RF mode] labels from nearest-ideal used as training data.")
        df["structure"] = classify_nearest(df, feature_cols)
        model, scaler, report = train_random_forest(df, feature_cols)
        print(report)
        df["structure"] = predict_rf(df, feature_cols, model, scaler)

    # Defect stats
    stats = count_defects(df)
    print(f"\nDefect analysis:")
    print(f"  Total atoms  : {stats['total_atoms']}")
    print(f"  Defect atoms : {stats['defect_atoms']}  "
          f"({100*stats['defect_fraction']:.2f}%)")
    print("  By label:")
    for lbl, cnt in stats["by_label"].items():
        print(f"    {lbl:12s}: {cnt}")

    # Vacancy clustering
    if "x" in df.columns and HAS_SKLEARN:
        df = cluster_vacancies(df, cutoff=args.vacancy_cutoff)
        n_clusters = df["cluster_id"].max() + 1
        print(f"\nVacancy clusters (DBSCAN ε={args.vacancy_cutoff} Å): {n_clusters}")

    # Write output dump
    if args.output:
        frame["atoms"] = df
        write_lammps_dump(frame, args.output)
        print(f"\nClassified dump → {args.output}")

    # Plot
    if args.plot or args.savefig:
        plot_q4_q6(df, label_col="structure", savefig=args.savefig)


if __name__ == "__main__":
    main()
