"""
pipeline.py
===========
LAMMPS-Python integration driver.

Runs a LAMMPS simulation and, at user-defined intervals, extracts per-atom
Steinhardt descriptors (from compute local_descriptor), classifies atomic
environments with a SOM-based classifier, and records defect statistics.

Typical usage
-------------
    from pipeline import LammpsMLPipeline
    from som_classifier import SomClassifier
    from defect_tracker import DefectTracker

    clf = SomClassifier(grid_x=8, grid_y=8)
    clf.load("som_trained.pkl")          # pre-trained SOM
    tracker = DefectTracker(cutoff=4.0)

    pipe = LammpsMLPipeline(
        input_script="descriptor_test.lammps",
        compute_id="desc",
        ncols=8,                         # 8 cols: Q4 W4 Q4b W4b Q6 W6 Q6b W6b
        classifier=clf,
        tracker=tracker,
        dump_classified="classified.dump",
    )
    pipe.run(total_steps=10000, interval=500)
    pipe.save_stats("stats.csv")

LAMMPS Python API notes
-----------------------
Requires LAMMPS built as a shared library with Python bindings enabled.
Set LD_LIBRARY_PATH or use the lammps conda/pip package.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ─── LAMMPS import ────────────────────────────────────────────────────────────
try:
    from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY, LMP_TYPE_VECTOR
    HAS_LAMMPS = True
except ImportError:
    HAS_LAMMPS = False
    # Allow import for testing / documentation without a LAMMPS installation.
    print("[pipeline] Warning: lammps Python module not found. "
          "LammpsMLPipeline will raise NotImplementedError when run().")


# Feature column names produced by compute local_descriptor with l=4,6 average=yes
DEFAULT_FEAT_NAMES = ["Q4", "W4", "Q4b", "W4b", "Q6", "W6", "Q6b", "W6b"]

# Columns used for classification (non-averaged local only is faster;
# averaged is more robust – use all 8 for best accuracy)
DEFAULT_CLF_COLS = [0, 1, 4, 5]          # Q4 W4 Q6 W6  (local)
AVERAGED_CLF_COLS = [0, 1, 2, 3, 4, 5, 6, 7]  # all 8 (local + averaged)


class LammpsMLPipeline:
    """
    Drive a LAMMPS MD run, classify atoms on-the-fly, and track defects.

    Parameters
    ----------
    input_script : str | Path
        LAMMPS input file.  It must define a compute named `compute_id`
        using the local_descriptor pair style.
    compute_id : str
        Name of the LAMMPS compute (default "desc").
    ncols : int
        Number of output columns per atom from the compute (default 8).
    classifier : object or None
        Object with a .predict(X) method returning structure labels.
        If None, nearest-ideal Steinhardt classifier is used.
    tracker : DefectTracker or None
        If provided, cluster analysis is run at each checkpoint.
    dump_classified : str or None
        Path for the LAMMPS-format dump with per-atom structure labels.
    clf_cols : list[int] or None
        Which descriptor columns to pass to the classifier (0-indexed).
        Defaults to [0,1,4,5] (Q4 W4 Q6 W6).
    lammps_args : list[str]
        Extra arguments passed to the lammps() constructor.
    """

    def __init__(
        self,
        input_script: str | Path,
        compute_id: str = "desc",
        ncols: int = 8,
        classifier=None,
        tracker=None,
        dump_classified: Optional[str] = None,
        clf_cols: Optional[list[int]] = None,
        lammps_args: Optional[list[str]] = None,
    ):
        self.input_script   = Path(input_script)
        self.compute_id     = compute_id
        self.ncols          = ncols
        self.classifier     = classifier
        self.tracker        = tracker
        self.dump_classified = dump_classified
        self.clf_cols       = clf_cols if clf_cols is not None else DEFAULT_CLF_COLS
        self.lammps_args    = lammps_args or []

        self._lmp: Optional[object] = None
        self._stats: list[dict]     = []   # one dict per checkpoint

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_lammps(self) -> None:
        if not HAS_LAMMPS:
            raise NotImplementedError("LAMMPS Python module is not installed.")
        self._lmp = lammps(cmdargs=self._lammps_args)
        self._lmp.file(str(self.input_script))

    def close(self) -> None:
        if self._lmp is not None:
            self._lmp.close()
            self._lmp = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, total_steps: int, interval: int = 500) -> None:
        """
        Run `total_steps` MD steps, calling _checkpoint() every `interval`.
        """
        self._init_lammps()
        lmp = self._lmp

        n_intervals = total_steps // interval
        remainder   = total_steps %  interval

        print(f"[pipeline] Starting MD run: {total_steps} steps, "
              f"checkpoint every {interval} steps.")

        for k in range(n_intervals):
            t0 = time.perf_counter()
            lmp.command(f"run {interval} pre no post no")
            step = lmp.get_thermo("step")
            self._checkpoint(int(step))
            dt = time.perf_counter() - t0
            print(f"  step {int(step):8d}  checkpoint #{k+1}  ({dt:.2f}s)")

        if remainder:
            lmp.command(f"run {remainder} pre no post no")
            step = lmp.get_thermo("step")
            self._checkpoint(int(step))

        print("[pipeline] Run complete.")
        self.close()

    # ── Checkpoint logic ──────────────────────────────────────────────────────

    def _checkpoint(self, timestep: int) -> None:
        """Extract descriptors, classify, track defects, record stats."""
        pos, types, desc = self._extract_data()
        natoms = len(types)

        # Classify
        X = desc[:, self.clf_cols]
        if self.classifier is not None:
            labels = self.classifier.predict(X)
        else:
            labels = _nearest_ideal_classify(X, self.clf_cols)

        # Defect tracking
        cluster_info = {}
        if self.tracker is not None:
            cluster_info = self.tracker.analyse(pos, labels, timestep)

        # Statistics
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts.tolist()))
        stat = {
            "timestep": timestep,
            "natoms":   natoms,
            **{f"n_{k}": v for k, v in label_counts.items()},
            **cluster_info,
        }
        self._stats.append(stat)

        # Write classified dump
        if self.dump_classified:
            self._write_dump(timestep, pos, types, desc, labels)

    # ── Data extraction ───────────────────────────────────────────────────────

    def _extract_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions [N,3], types [N], descriptors [N, ncols])."""
        lmp = self._lmp

        # Positions and atom types
        x     = lmp.numpy.extract_atom("x")          # (N, 3)
        atype = lmp.numpy.extract_atom("type")        # (N,)

        # Per-atom compute array
        desc = lmp.numpy.extract_compute(
            self.compute_id,
            LMP_STYLE_ATOM,
            LMP_TYPE_ARRAY,
        )  # (N, ncols)

        if desc is None or desc.shape[1] != self.ncols:
            raise RuntimeError(
                f"Compute '{self.compute_id}' returned unexpected shape "
                f"{None if desc is None else desc.shape}; expected ncols={self.ncols}"
            )

        return np.array(x), np.array(atype, dtype=int), np.array(desc)

    # ── Dump writer ───────────────────────────────────────────────────────────

    def _write_dump(self, timestep: int,
                    pos: np.ndarray,
                    types: np.ndarray,
                    desc: np.ndarray,
                    labels: np.ndarray) -> None:
        """Append one frame to the classified dump file."""
        label_to_int = {lbl: i for i, lbl in
                        enumerate(sorted(set(labels)))}

        mode = "a" if Path(self.dump_classified).exists() else "w"
        with open(self.dump_classified, mode) as fh:
            N = len(types)
            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{timestep}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{N}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            # Approximate box from positions
            for dim in range(3):
                lo = pos[:, dim].min() - 1.0
                hi = pos[:, dim].max() + 1.0
                fh.write(f"{lo:.4f} {hi:.4f}\n")

            col_header = " ".join(DEFAULT_FEAT_NAMES[:self.ncols])
            fh.write(f"ITEM: ATOMS id type x y z structure_id {col_header}\n")
            for i in range(N):
                sid = label_to_int[labels[i]]
                desc_str = " ".join(f"{v:.6f}" for v in desc[i])
                fh.write(
                    f"{i+1} {types[i]} "
                    f"{pos[i,0]:.4f} {pos[i,1]:.4f} {pos[i,2]:.4f} "
                    f"{sid} {desc_str}\n"
                )

    # ── Output ────────────────────────────────────────────────────────────────

    def save_stats(self, path: str) -> None:
        """Write per-checkpoint statistics to a CSV file."""
        if not self._stats:
            print("[pipeline] No statistics to save.")
            return
        keys = list(self._stats[0].keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._stats)
        print(f"[pipeline] Stats saved → {path}")

    @property
    def stats(self) -> list[dict]:
        return self._stats


# ─── Fallback classifier (no sklearn/SOM needed) ─────────────────────────────

_IDEAL = {
    "FCC":       np.array([0.191, -0.159, 0.574, -0.013]),
    "HCP":       np.array([0.097, -0.134, 0.485, -0.012]),
    "BCC":       np.array([0.036,  0.159, 0.511,  0.013]),
    "Amorphous": np.array([0.050,  0.000, 0.350,  0.000]),
}


def _nearest_ideal_classify(X: np.ndarray,
                             col_indices: list[int]) -> np.ndarray:
    """Classify N atoms (rows of X) by nearest Steinhardt ideal value."""
    labels = []
    ideal_arr = np.array(list(_IDEAL.values()))   # (nstructs, 4)
    keys      = list(_IDEAL.keys())

    for row in X:
        n = min(len(row), ideal_arr.shape[1])
        dists = np.linalg.norm(ideal_arr[:, :n] - row[:n], axis=1)
        labels.append(keys[int(np.argmin(dists))])
    return np.array(labels)
