"""
som_classifier.py
=================
Self-Organizing Map (SOM) based classifier for atomic structure identification.

Inspired by MultiSOM (Aquistapace et al., Comput. Mat. Sci. 227, 112263, 2023).
Uses Steinhardt bond-order parameters (Q4, W4, Q6, W6 and averaged versions)
as input features and maps each atom to a structural phase or defect type.

Two backends are supported:
  1. MiniSOM  (pip install minisom) – recommended
  2. Built-in  NumPy SOM            – no extra dependencies, slower

Usage
-----
    from som_classifier import SomClassifier

    clf = SomClassifier(grid_x=10, grid_y=10, sigma=1.0, lr=0.5)

    # Train on reference data (perfect crystals at various T)
    clf.train(X_train, epochs=200)

    # Optionally assign phase labels to SOM nodes
    clf.assign_labels(X_train, y_train)

    # Predict on new data
    labels = clf.predict(X_new)

    # Persist
    clf.save("som_trained.pkl")
    clf.load("som_trained.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# Optional MiniSOM backend
try:
    from minisom import MiniSom
    HAS_MINISOM = True
except ImportError:
    HAS_MINISOM = False


# Default ideal Steinhardt values for label assignment when no training labels
# are provided.  (Q4, W4_hat, Q6, W6_hat) – local non-averaged.
IDEAL_STRUCTURES = {
    "FCC":       np.array([0.191, -0.159,  0.574, -0.013]),
    "HCP":       np.array([0.097, -0.134,  0.485, -0.012]),
    "BCC":       np.array([0.036,  0.159,  0.511,  0.013]),
    "ICO":       np.array([0.000,  0.000,  0.663,  0.017]),
    "Amorphous": np.array([0.050,  0.000,  0.350,  0.000]),
}


class SomClassifier:
    """
    SOM-based classifier for local atomic structure identification.

    Parameters
    ----------
    grid_x, grid_y : int
        SOM grid dimensions.  Larger grids → finer resolution, more training.
    sigma : float
        Initial neighbourhood radius.
    lr : float
        Initial learning rate.
    backend : "minisom" | "numpy"
        Which SOM implementation to use. "minisom" is preferred.
    random_seed : int
        Reproducibility seed.
    """

    def __init__(
        self,
        grid_x: int = 10,
        grid_y: int = 10,
        sigma: float = 1.0,
        lr: float = 0.5,
        backend: str = "minisom",
        random_seed: int = 42,
    ):
        self.grid_x      = grid_x
        self.grid_y      = grid_y
        self.sigma       = sigma
        self.lr          = lr
        self.random_seed = random_seed
        self.backend     = backend if (backend == "minisom" and HAS_MINISOM) else "numpy"

        self._som        = None   # SOM weights (numpy: [gx, gy, nfeat])
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None
        self._node_labels: Optional[np.ndarray] = None  # [gx, gy] → str label
        self._trained    = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 200,
        verbose: bool = True,
    ) -> "SomClassifier":
        """
        Train the SOM on feature matrix X (N × nfeatures).

        Parameters
        ----------
        X : (N, nfeatures) array
            Descriptor matrix. Typical columns: Q4 W4 Q4b W4b Q6 W6 Q6b W6b.
        y : (N,) array of str labels, optional
            If provided, each SOM node is labelled by majority vote.
            If None, nodes are labelled by nearest ideal Steinhardt value.
        epochs : int
            Training epochs (full passes over data).
        """
        X = np.asarray(X, dtype=float)
        self._fit_scaler(X)
        Xs = self._scale(X)

        if self.backend == "minisom":
            self._train_minisom(Xs, epochs, verbose)
        else:
            self._train_numpy(Xs, epochs, verbose)

        # Assign labels to nodes
        if y is not None:
            self._assign_labels_from_data(Xs, y)
        else:
            self._assign_labels_from_ideal()

        self._trained = True
        return self

    def _train_minisom(self, Xs: np.ndarray, epochs: int, verbose: bool) -> None:
        nfeat = Xs.shape[1]
        som = MiniSom(
            self.grid_x, self.grid_y, nfeat,
            sigma=self.sigma,
            learning_rate=self.lr,
            random_seed=self.random_seed,
        )
        som.pca_weights_init(Xs)
        som.train(Xs, num_iteration=epochs * len(Xs), verbose=verbose)
        self._som = som

    def _train_numpy(self, Xs: np.ndarray, epochs: int, verbose: bool) -> None:
        """Minimal batch SOM implemented in NumPy."""
        nfeat = Xs.shape[1]
        rng   = np.random.default_rng(self.random_seed)
        W     = rng.standard_normal((self.grid_x, self.grid_y, nfeat))
        # Normalise rows
        W /= (np.linalg.norm(W, axis=2, keepdims=True) + 1e-12)

        N = len(Xs)
        coords = np.array([[i, j]
                            for i in range(self.grid_x)
                            for j in range(self.grid_y)], dtype=float)
        max_iter = epochs * N

        for it in range(max_iter):
            frac = it / max_iter
            lr_t    = self.lr    * np.exp(-frac * 3)
            sigma_t = self.sigma * np.exp(-frac * 3)

            x = Xs[rng.integers(0, N)]
            # Find BMU
            diff = W - x  # (gx, gy, nfeat)
            dist2 = (diff**2).sum(axis=2)  # (gx, gy)
            bmu_flat = int(dist2.ravel().argmin())
            bmu = np.array([bmu_flat // self.grid_y,
                             bmu_flat %  self.grid_y], dtype=float)

            # Neighbourhood
            node_coords = coords.reshape(self.grid_x, self.grid_y, 2)
            d2_nodes = ((node_coords - bmu)**2).sum(axis=2)
            h = np.exp(-d2_nodes / (2 * sigma_t**2))[:, :, None]  # (gx, gy, 1)

            W += lr_t * h * (x - W)

            if verbose and (it % (max_iter // 10) == 0):
                print(f"  SOM epoch {it*100//max_iter:3d}%", end="\r")

        if verbose:
            print()
        self._som = W  # store raw weight matrix for numpy backend

    # ── Label assignment ──────────────────────────────────────────────────────

    def _assign_labels_from_data(self, Xs: np.ndarray,
                                  y: np.ndarray) -> None:
        """Label each node by majority vote among mapped training atoms."""
        from collections import Counter
        node_votes: dict[tuple, Counter] = {}

        for xi, yi in zip(Xs, y):
            bmu = self._get_bmu(xi)
            node_votes.setdefault(bmu, Counter())[yi] += 1

        self._node_labels = np.full((self.grid_x, self.grid_y),
                                     "Unknown", dtype=object)
        for (i, j), counter in node_votes.items():
            self._node_labels[i, j] = counter.most_common(1)[0][0]

    def _assign_labels_from_ideal(self) -> None:
        """Label each SOM node by nearest ideal Steinhardt descriptor."""
        ideal_arr = np.array(list(IDEAL_STRUCTURES.values()))
        keys      = list(IDEAL_STRUCTURES.keys())

        self._node_labels = np.full((self.grid_x, self.grid_y),
                                     "Unknown", dtype=object)
        weights = self._get_weights()  # (gx, gy, nfeat)

        for i in range(self.grid_x):
            for j in range(self.grid_y):
                w = weights[i, j]
                # Use first min(nfeat, 4) components for comparison
                n = min(len(w), ideal_arr.shape[1])
                dists = np.linalg.norm(ideal_arr[:, :n] - w[:n], axis=1)
                self._node_labels[i, j] = keys[int(np.argmin(dists))]

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify atoms.

        Parameters
        ----------
        X : (N, nfeatures) array

        Returns
        -------
        labels : (N,) array of str
        """
        if not self._trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        Xs = self._scale(np.asarray(X, dtype=float))
        labels = []
        for x in Xs:
            bmu = self._get_bmu(x)
            labels.append(self._node_labels[bmu])
        return np.array(labels)

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return per-atom soft probabilities via inverse-distance weighting
        over SOM nodes, grouped by label.
        """
        if not self._trained:
            raise RuntimeError("SOM not trained.")
        Xs     = self._scale(np.asarray(X, dtype=float))
        weights = self._get_weights()                    # (gx, gy, nfeat)
        unique_labels = np.unique(self._node_labels)
        W_flat = weights.reshape(-1, weights.shape[2])  # (gx*gy, nfeat)
        lbl_flat = self._node_labels.ravel()

        proba = {lbl: np.zeros(len(Xs)) for lbl in unique_labels}

        for ii, x in enumerate(Xs):
            dist2 = ((W_flat - x)**2).sum(axis=1)
            # avoid div-by-zero
            inv_d = 1.0 / (np.sqrt(dist2) + 1e-12)
            total = inv_d.sum()
            for lbl in unique_labels:
                mask = lbl_flat == lbl
                proba[lbl][ii] = inv_d[mask].sum() / total

        return proba

    # ── U-Matrix (for visualisation) ──────────────────────────────────────────

    def umatrix(self) -> np.ndarray:
        """Return (grid_x, grid_y) unified distance matrix."""
        W = self._get_weights()
        um = np.zeros((self.grid_x, self.grid_y))
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                neighbours = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.grid_x and 0 <= nj < self.grid_y:
                        neighbours.append(
                            np.linalg.norm(W[i,j] - W[ni,nj])
                        )
                um[i, j] = np.mean(neighbours) if neighbours else 0.0
        return um

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        state = {
            "grid_x":       self.grid_x,
            "grid_y":       self.grid_y,
            "sigma":        self.sigma,
            "lr":           self.lr,
            "backend":      self.backend,
            "random_seed":  self.random_seed,
            "_som":         self._som,
            "_scaler_mean": self._scaler_mean,
            "_scaler_std":  self._scaler_std,
            "_node_labels": self._node_labels,
            "_trained":     self._trained,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        print(f"[SomClassifier] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SomClassifier":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(
            grid_x=state["grid_x"], grid_y=state["grid_y"],
            sigma=state["sigma"],   lr=state["lr"],
            backend=state["backend"], random_seed=state["random_seed"],
        )
        for k in ("_som","_scaler_mean","_scaler_std",
                  "_node_labels","_trained"):
            setattr(obj, k, state[k])
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fit_scaler(self, X: np.ndarray) -> None:
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std  = X.std(axis=0) + 1e-12

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._scaler_mean is None:
            return X
        return (X - self._scaler_mean) / self._scaler_std

    def _get_bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Best Matching Unit for a single sample x (already scaled)."""
        if self.backend == "minisom":
            return self._som.winner(x)
        else:
            diff = self._som - x
            dist2 = (diff**2).sum(axis=2)
            idx = int(dist2.ravel().argmin())
            return (idx // self.grid_y, idx % self.grid_y)

    def _get_weights(self) -> np.ndarray:
        """Return weight matrix (grid_x, grid_y, nfeat)."""
        if self.backend == "minisom":
            return self._som.get_weights()
        return self._som


# ─── Reference data generator (for training on perfect crystals) ──────────────

def generate_reference_data(
    structures: dict[str, np.ndarray] = None,
    n_per_struct: int = 2000,
    noise_std: float = 0.015,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data by adding Gaussian noise to ideal
    Steinhardt values.  Useful for quick training without a LAMMPS run.

    Returns
    -------
    X : (N, 4) array  [Q4, W4, Q6, W6]
    y : (N,)   array of str labels
    """
    if structures is None:
        structures = IDEAL_STRUCTURES

    rng = np.random.default_rng(random_seed)
    Xs, ys = [], []
    for label, ideal in structures.items():
        noise = rng.normal(0, noise_std, size=(n_per_struct, len(ideal)))
        Xs.append(ideal + noise)
        ys.extend([label] * n_per_struct)

    return np.vstack(Xs), np.array(ys)
