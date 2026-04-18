"""
run_pipeline.py
===============
End-to-end example: LAMMPS MD + on-the-fly ML structure identification +
vacancy cluster quantification.

Three modes selectable via --mode:
  train     – generate synthetic reference data and train the SOM
  run       – drive a LAMMPS simulation with the trained SOM
  analyse   – post-process an existing dump.desc without running LAMMPS

Usage
-----
    # 1. Train the SOM on perfect-crystal reference data
    python run_pipeline.py --mode train --som som_trained.pkl

    # 2. Run LAMMPS + on-the-fly classification (needs LAMMPS Python bindings)
    python run_pipeline.py --mode run \
        --input descriptor_test.lammps \
        --som   som_trained.pkl \
        --steps 10000 --interval 500 \
        --dump  classified.dump \
        --stats defect_stats.csv

    # 3. Analyse an existing dump produced by compute local_descriptor
    python run_pipeline.py --mode analyse \
        --dump dump.desc \
        --som  som_trained.pkl \
        --output classified_out.dump
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure python/ is on the path when called from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from som_classifier import SomClassifier, generate_reference_data, IDEAL_STRUCTURES
from defect_tracker import DefectTracker
from read_descriptors import read_lammps_dump, write_lammps_dump, plot_q4_q6


# ─── Mode: train ──────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    """
    Train a SOM on synthetic reference data (Gaussian noise around ideal
    Steinhardt values).  Saves the trained model to --som.
    """
    print("=== Training SOM ===")
    print(f"  Grid      : {args.gx} × {args.gy}")
    print(f"  Structures: {list(IDEAL_STRUCTURES.keys())}")
    print(f"  N/struct  : {args.n_ref}")

    X_train, y_train = generate_reference_data(
        n_per_struct=args.n_ref,
        noise_std=args.noise,
        random_seed=0,
    )
    print(f"  Total training atoms: {len(X_train)}")

    clf = SomClassifier(
        grid_x=args.gx,
        grid_y=args.gy,
        sigma=args.sigma,
        lr=args.lr,
        backend="minisom" if not args.numpy_som else "numpy",
        random_seed=42,
    )
    clf.train(X_train, y=y_train, epochs=args.epochs, verbose=True)
    clf.save(args.som)

    # Quick accuracy check
    preds = clf.predict(X_train)
    acc   = (preds == y_train).mean()
    print(f"  Training accuracy : {acc*100:.1f}%")

    # Node distribution summary
    from collections import Counter
    node_lbl_flat = clf._node_labels.ravel()
    print("  Node label distribution:")
    for lbl, cnt in sorted(Counter(node_lbl_flat).items()):
        print(f"    {lbl:12s}: {cnt} nodes")


# ─── Mode: run ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """
    Drive a LAMMPS simulation with on-the-fly ML classification.
    """
    try:
        from pipeline import LammpsMLPipeline
    except ImportError as e:
        sys.exit(f"Cannot import pipeline module: {e}")

    print("=== LAMMPS + ML Pipeline ===")

    # Load classifier
    if Path(args.som).exists():
        clf = SomClassifier.load(args.som)
        print(f"  Loaded SOM from {args.som}")
    else:
        print("  No SOM found – using nearest-ideal fallback classifier.")
        clf = None

    # Defect tracker
    tracker = DefectTracker(
        defect_labels=set(args.defect_labels.split(",")),
        cutoff=args.vacancy_cutoff,
        min_cluster_size=1,
        compute_free_volume=True,
    )

    pipe = LammpsMLPipeline(
        input_script=args.input,
        compute_id="desc",
        ncols=8,
        classifier=clf,
        tracker=tracker,
        dump_classified=args.dump if args.dump else None,
    )

    pipe.run(total_steps=args.steps, interval=args.interval)

    if args.stats:
        pipe.save_stats(args.stats)

    if args.plot_evolution:
        tracker.plot_cluster_evolution(savefig=args.plot_evolution)

    if args.plot_distribution:
        tracker.plot_size_distribution(savefig=args.plot_distribution)

    # Print summary
    df = tracker.history_dataframe() if hasattr(tracker, "history_dataframe") else None
    if df is not None:
        print("\n=== Defect summary (last 5 checkpoints) ===")
        print(df[["timestep","n_defect","defect_fraction",
                   "n_clusters","max_cluster_size"]].tail(5).to_string(index=False))


# ─── Mode: analyse ────────────────────────────────────────────────────────────

def cmd_analyse(args: argparse.Namespace) -> None:
    """
    Post-process an existing dump file (no LAMMPS required).
    """
    print(f"=== Analysing dump: {args.dump} ===")
    frames = read_lammps_dump(args.dump)
    if not frames:
        sys.exit("No frames found.")

    frame_idx = args.frame if args.frame >= 0 else len(frames) + args.frame
    frame = frames[frame_idx]
    df    = frame["atoms"]
    print(f"  Timestep {frame['timestep']}: {len(df)} atoms")

    # Detect descriptor columns
    desc_cols = [c for c in df.columns if c.startswith("c_desc")]
    if len(desc_cols) < 4:
        sys.exit("Expected at least 4 c_desc[*] columns.")

    rename = {}
    labels_map = [("Q4","W4","Q4b","W4b"), ("Q6","W6","Q6b","W6b")]
    for i, grp in enumerate(labels_map):
        for j, nm in enumerate(grp):
            k = f"c_desc[{i*4+j+1}]"
            if k in df.columns:
                rename[k] = nm
    df = df.rename(columns=rename)

    feat_cols = [c for c in ["Q4","W4","Q6","W6"] if c in df.columns]

    # Load or build classifier
    if args.som and Path(args.som).exists():
        clf = SomClassifier.load(args.som)
        X = df[feat_cols].to_numpy(dtype=float)
        df["structure"] = clf.predict(X)
    else:
        from read_descriptors import classify_nearest
        df["structure"] = classify_nearest(df, feat_cols)

    # Defect analysis
    if all(c in df.columns for c in ["x","y","z"]):
        tracker = DefectTracker(
            defect_labels=set(args.defect_labels.split(",")),
            cutoff=args.vacancy_cutoff,
        )
        pos    = df[["x","y","z"]].to_numpy()
        labels = df["structure"].to_numpy()
        stats  = tracker.analyse(pos, labels, timestep=frame["timestep"])

        print("\n  Defect stats:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

        if args.plot_distribution:
            tracker.plot_size_distribution(savefig=args.plot_distribution)
        if args.plot_q4q6:
            plot_q4_q6(df, label_col="structure", savefig=args.plot_q4q6)

    # Write output
    if args.output:
        frame["atoms"] = df
        write_lammps_dump(frame, args.output)
        print(f"\n  Classified dump → {args.output}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LAMMPS ML pipeline: train / run / analyse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train","run","analyse"],
                   required=True)

    # Shared
    p.add_argument("--som", default="som_trained.pkl",
                   help="Path to SOM model file (.pkl)")
    p.add_argument("--defect-labels", default="Amorphous",
                   help="Comma-separated labels treated as defects")
    p.add_argument("--vacancy-cutoff", type=float, default=4.0,
                   help="DBSCAN cutoff for vacancy clustering (Å)")

    # Train-specific
    p.add_argument("--gx", type=int, default=10, help="SOM grid x")
    p.add_argument("--gy", type=int, default=10, help="SOM grid y")
    p.add_argument("--sigma", type=float, default=1.0, help="SOM initial sigma")
    p.add_argument("--lr", type=float, default=0.5, help="SOM learning rate")
    p.add_argument("--epochs", type=int, default=200, help="SOM training epochs")
    p.add_argument("--n-ref", type=int, default=2000,
                   help="Reference atoms per structure for training")
    p.add_argument("--noise", type=float, default=0.015,
                   help="Gaussian noise std for reference data")
    p.add_argument("--numpy-som", action="store_true",
                   help="Force NumPy SOM backend (no minisom required)")

    # Run-specific
    p.add_argument("--input", default="descriptor_test.lammps",
                   help="LAMMPS input script")
    p.add_argument("--steps", type=int, default=10000,
                   help="Total MD steps")
    p.add_argument("--interval", type=int, default=500,
                   help="Steps between ML checkpoints")
    p.add_argument("--stats", default=None,
                   help="Output CSV for per-checkpoint stats")
    p.add_argument("--plot-evolution", default=None,
                   help="Save defect evolution plot to file")
    p.add_argument("--plot-distribution", default=None,
                   help="Save cluster size distribution plot to file")

    # Analyse-specific
    p.add_argument("--dump", default=None,
                   help="Input dump file (for analyse mode)")
    p.add_argument("--frame", type=int, default=-1,
                   help="Frame index to analyse (-1 = last)")
    p.add_argument("--output", default=None,
                   help="Output classified dump")
    p.add_argument("--plot-q4q6", default=None,
                   help="Save Q4 vs Q6 scatter plot to file")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "run":
        cmd_run(args)
    elif args.mode == "analyse":
        cmd_analyse(args)


if __name__ == "__main__":
    main()
