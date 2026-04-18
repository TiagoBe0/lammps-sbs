"""
defect_tracker.py
=================
Vacancy and defect cluster quantification for atomistic simulations.

This module addresses the key gap identified in the PhD proposal (§3.3.1):
  "complementar con estudio detallado del volumen libre para obtener
   distribución de tamaño de clusters de vacancias"

Standard methods (Wigner-Seitz, PTM/CNA) fail under extreme conditions.
Here we identify defect atoms from ML-assigned labels and then:
  1. Cluster defect atoms spatially (DBSCAN).
  2. Characterise each cluster: size, free volume, shape (gyration radius,
     asphericity), spatial distribution.
  3. Track evolution over simulation timesteps.

No reference configuration is required — the method is inherently robust
to large deformations, rotations, and high temperatures.

Usage
-----
    from defect_tracker import DefectTracker

    tracker = DefectTracker(
        defect_labels={"Amorphous"},
        cutoff=4.0,          # Å, DBSCAN epsilon
        min_cluster_size=1,  # include monovacancies
    )

    # At each checkpoint (pos: (N,3) array, labels: (N,) str array):
    stats = tracker.analyse(pos, labels, timestep=500)

    # After simulation:
    df = tracker.history_dataframe()
    df.to_csv("defect_history.csv", index=False)
    tracker.plot_cluster_evolution()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ClusterInfo:
    """Per-cluster descriptors at one timestep."""
    cluster_id: int
    size: int                       # number of atoms
    centroid: np.ndarray            # (3,) centre of mass
    gyration_radius: float          # Rg = sqrt(mean |r - centroid|²)
    asphericity: float              # 0 = sphere, 1 = rod
    free_volume: float              # Voronoi-based estimate (ų), -1 if unavailable
    label_composition: dict         # {label: count} for atoms in cluster


@dataclass
class TimestepStats:
    """Aggregate defect statistics at one timestep."""
    timestep: int
    n_atoms: int
    n_defect: int
    defect_fraction: float
    n_clusters: int
    cluster_size_hist: dict         # {size: count}
    mean_cluster_size: float
    max_cluster_size: int
    clusters: list[ClusterInfo] = field(default_factory=list)


# ─── Main class ───────────────────────────────────────────────────────────────

class DefectTracker:
    """
    Track defect clusters across MD timesteps without a reference configuration.

    Parameters
    ----------
    defect_labels : set[str]
        Structure labels (from SOM classifier) considered as defects.
    cutoff : float
        DBSCAN epsilon (Å).  Should be ~ 1.5× nearest-neighbour distance.
    min_cluster_size : int
        Minimum atoms per cluster (DBSCAN min_samples).
    compute_free_volume : bool
        Estimate local free volume via Voronoi analysis (requires scipy).
    box : (3,2) array or None
        Simulation box bounds for periodic Voronoi.  If None, open boundaries.
    """

    def __init__(
        self,
        defect_labels: Optional[set[str]] = None,
        cutoff: float = 4.0,
        min_cluster_size: int = 1,
        compute_free_volume: bool = True,
        box: Optional[np.ndarray] = None,
    ):
        self.defect_labels       = defect_labels or {"Amorphous"}
        self.cutoff              = cutoff
        self.min_cluster_size    = min_cluster_size
        self.compute_free_volume = compute_free_volume
        self.box                 = box  # (3,2) [[xlo,xhi],[ylo,yhi],[zlo,zhi]]

        self._history: list[TimestepStats] = []

    # ── Main interface ────────────────────────────────────────────────────────

    def analyse(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        timestep: int = 0,
    ) -> dict:
        """
        Identify and characterise defect clusters.

        Parameters
        ----------
        positions : (N, 3)
        labels    : (N,)  str array from SOM/ML classifier
        timestep  : int

        Returns
        -------
        stats_dict : flat dict suitable for CSV/logging
        """
        pos    = np.asarray(positions, dtype=float)
        labels = np.asarray(labels)
        N      = len(labels)

        # --- defect mask ---
        defect_mask = np.zeros(N, dtype=bool)
        for lbl in self.defect_labels:
            defect_mask |= (labels == lbl)

        n_defect = int(defect_mask.sum())

        if n_defect == 0:
            ts = TimestepStats(
                timestep=timestep, n_atoms=N,
                n_defect=0, defect_fraction=0.0,
                n_clusters=0, cluster_size_hist={},
                mean_cluster_size=0.0, max_cluster_size=0,
            )
            self._history.append(ts)
            return self._to_dict(ts)

        defect_pos = pos[defect_mask]
        defect_lbl = labels[defect_mask]

        # --- cluster defect atoms ---
        cluster_ids = self._cluster(defect_pos)

        # --- per-cluster descriptors ---
        unique_ids = [c for c in np.unique(cluster_ids) if c >= 0]
        clusters   = []
        for cid in unique_ids:
            mask_c = cluster_ids == cid
            info   = self._describe_cluster(
                cid,
                defect_pos[mask_c],
                defect_lbl[mask_c],
            )
            clusters.append(info)

        # --- aggregate stats ---
        sizes = [c.size for c in clusters]
        size_hist = {}
        for s in sizes:
            size_hist[s] = size_hist.get(s, 0) + 1

        ts = TimestepStats(
            timestep=timestep,
            n_atoms=N,
            n_defect=n_defect,
            defect_fraction=n_defect / N,
            n_clusters=len(clusters),
            cluster_size_hist=size_hist,
            mean_cluster_size=float(np.mean(sizes)) if sizes else 0.0,
            max_cluster_size=int(max(sizes)) if sizes else 0,
            clusters=clusters,
        )
        self._history.append(ts)
        return self._to_dict(ts)

    # ── Clustering ────────────────────────────────────────────────────────────

    def _cluster(self, pos: np.ndarray) -> np.ndarray:
        """Return per-atom cluster IDs (-1 = noise) using DBSCAN."""
        if not HAS_SKLEARN:
            warnings.warn(
                "scikit-learn not available; treating each defect atom "
                "as its own cluster (no spatial grouping)."
            )
            return np.arange(len(pos))

        db = DBSCAN(
            eps=self.cutoff,
            min_samples=self.min_cluster_size,
            algorithm="ball_tree",
            metric="euclidean",
        ).fit(pos)
        return db.labels_

    # ── Cluster descriptors ───────────────────────────────────────────────────

    def _describe_cluster(
        self,
        cid: int,
        pos: np.ndarray,
        labels: np.ndarray,
    ) -> ClusterInfo:
        """Compute geometric and compositional descriptors for one cluster."""
        centroid = pos.mean(axis=0)
        delta    = pos - centroid

        # Gyration radius
        rg = float(np.sqrt((delta**2).sum(axis=1).mean()))

        # Asphericity from gyration tensor eigenvalues
        asp = self._asphericity(delta)

        # Free volume estimate (Voronoi)
        fv = -1.0
        if self.compute_free_volume and len(pos) >= 4:
            fv = self._voronoi_volume(pos)

        # Label composition
        unique_lbl, cnts = np.unique(labels, return_counts=True)
        composition = dict(zip(unique_lbl.tolist(), cnts.tolist()))

        return ClusterInfo(
            cluster_id=cid,
            size=len(pos),
            centroid=centroid,
            gyration_radius=rg,
            asphericity=asp,
            free_volume=fv,
            label_composition=composition,
        )

    @staticmethod
    def _asphericity(delta: np.ndarray) -> float:
        """
        Asphericity b = λ1 - (λ2+λ3)/2  from gyration tensor eigenvalues.
        Normalised to [0,1].  0 = sphere, 1 = perfect rod.
        """
        if len(delta) < 2:
            return 0.0
        T = (delta.T @ delta) / len(delta)      # 3×3 gyration tensor
        eigvals = np.sort(np.linalg.eigvalsh(T))[::-1]  # descending
        denom = eigvals.sum()
        if denom < 1e-14:
            return 0.0
        b = eigvals[0] - 0.5 * (eigvals[1] + eigvals[2])
        return float(b / denom)

    @staticmethod
    def _voronoi_volume(pos: np.ndarray) -> float:
        """
        Estimate total free volume of a cluster as the sum of Voronoi cell
        volumes of the constituent atoms.  Requires scipy.
        """
        try:
            from scipy.spatial import ConvexHull, Voronoi
            if len(pos) < 4:
                return -1.0
            vor = Voronoi(pos)
            total_vol = 0.0
            for region_idx in vor.point_region:
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    continue  # open region near boundary
                verts = vor.vertices[region]
                try:
                    hull = ConvexHull(verts)
                    total_vol += hull.volume
                except Exception:
                    pass
            return float(total_vol)
        except Exception:
            return -1.0

    # ── History & output ──────────────────────────────────────────────────────

    @property
    def history(self) -> list[TimestepStats]:
        return self._history

    def history_dataframe(self):
        """Return a pandas DataFrame of per-timestep aggregate stats."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for history_dataframe()")
        rows = [self._to_dict(ts) for ts in self._history]
        return pd.DataFrame(rows)

    def cluster_size_distribution(self, timestep: Optional[int] = None):
        """
        Return (sizes, counts) arrays for the size distribution.

        If timestep is None, aggregate over all checkpoints (useful for
        generating a time-averaged distribution for the PhD analysis).
        """
        if timestep is not None:
            match = [ts for ts in self._history if ts.timestep == timestep]
            hist  = match[-1].cluster_size_hist if match else {}
        else:
            hist: dict[int, int] = {}
            for ts in self._history:
                for s, c in ts.cluster_size_hist.items():
                    hist[s] = hist.get(s, 0) + c

        if not hist:
            return np.array([]), np.array([])
        sizes  = np.array(sorted(hist.keys()))
        counts = np.array([hist[s] for s in sizes])
        return sizes, counts

    def monovacancy_count(self) -> np.ndarray:
        """Return array of monovacancy counts per checkpoint."""
        return np.array([ts.cluster_size_hist.get(1, 0)
                          for ts in self._history])

    def total_defect_fraction(self) -> np.ndarray:
        return np.array([ts.defect_fraction for ts in self._history])

    def timesteps(self) -> np.ndarray:
        return np.array([ts.timestep for ts in self._history])

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_cluster_evolution(self, savefig: Optional[str] = None) -> None:
        """Plot defect fraction and cluster count vs time."""
        if not HAS_MPL:
            print("[DefectTracker] matplotlib not available.")
            return
        if not self._history:
            print("[DefectTracker] No data to plot.")
            return

        t = self.timesteps()
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        # 1. Defect fraction
        axes[0].plot(t, self.total_defect_fraction() * 100,
                     color="tomato", lw=2)
        axes[0].set_ylabel("Defect fraction (%)")
        axes[0].set_title("Defect evolution over MD run")

        # 2. Cluster count vs size category
        n_mono  = self.monovacancy_count()
        n_small = np.array([
            sum(c for s, c in ts.cluster_size_hist.items() if 2 <= s <= 5)
            for ts in self._history
        ])
        n_large = np.array([
            sum(c for s, c in ts.cluster_size_hist.items() if s > 5)
            for ts in self._history
        ])
        axes[1].stackplot(t, n_mono, n_small, n_large,
                          labels=["Monovacancy", "Small (2–5)", "Large (>5)"],
                          alpha=0.75)
        axes[1].set_ylabel("Cluster count")
        axes[1].legend(loc="upper left", fontsize=8)

        # 3. Max cluster size
        axes[2].plot(t, [ts.max_cluster_size for ts in self._history],
                     color="steelblue", lw=2)
        axes[2].set_ylabel("Max cluster size")
        axes[2].set_xlabel("Timestep")

        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=150)
            print(f"[DefectTracker] Plot saved → {savefig}")
        else:
            plt.show()

    def plot_size_distribution(self, timestep: Optional[int] = None,
                                log_scale: bool = True,
                                savefig: Optional[str] = None) -> None:
        """Bar chart of vacancy cluster size distribution."""
        if not HAS_MPL:
            return
        sizes, counts = self.cluster_size_distribution(timestep)
        if len(sizes) == 0:
            print("[DefectTracker] No clusters to plot.")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(sizes, counts, color="steelblue", edgecolor="k", alpha=0.8)
        ax.set_xlabel("Cluster size (# atoms)")
        ax.set_ylabel("Count")
        title = "Vacancy cluster size distribution"
        if timestep is not None:
            title += f" (step {timestep})"
        else:
            title += " (all timesteps)"
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")

        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=150)
        else:
            plt.show()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(ts: TimestepStats) -> dict:
        return {
            "timestep":         ts.timestep,
            "n_atoms":          ts.n_atoms,
            "n_defect":         ts.n_defect,
            "defect_fraction":  ts.defect_fraction,
            "n_clusters":       ts.n_clusters,
            "mean_cluster_size": ts.mean_cluster_size,
            "max_cluster_size": ts.max_cluster_size,
            "n_monovacancies":  ts.cluster_size_hist.get(1, 0),
        }
