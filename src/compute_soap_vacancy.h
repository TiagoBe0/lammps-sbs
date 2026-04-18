/* compute_soap_vacancy  –  grid-based vacancy detection for BCC bulk cascades
 *
 * Scans a uniform grid inside the simulation box.  Grid points farther than
 * vac_dist from any atom are classified as vacant.  Vacant grid points are
 * merged into physical vacancy events by a greedy PBC-aware clustering step.
 *
 * Algorithm (FaVaD §2.3.2 adapted for LAMMPS):
 *   1. Build a linked-cell list from all local + ghost atoms.
 *   2. Each MPI rank scans grid points inside its own subdomain.
 *   3. Vacant points are gathered across all ranks (MPI_Allgatherv).
 *   4. Greedy O(M²) clustering with minimum-image PBC.
 *   5. Results broadcast; output vector has three components:
 *        c_vac[1]  number of vacant grid points
 *        c_vac[2]  void volume  [Å³]  = n_pts * grid³
 *        c_vac[3]  number of vacancy clusters
 *   6. Optional per-timestep cluster CSV (append mode).
 *
 * Does NOT require a LAMMPS neighbor list — builds its own cell list from
 * atom->x (local + ghost atoms), so no pair_style dependency.
 *
 * Syntax:
 *   compute ID group soap_vacancy vac_dist
 *                  [grid G]            (default 0.5 Å)
 *                  [cluster_radius R]  (default = vac_dist)
 *                  [file FILE]         (append cluster data each invocation)
 *
 * Typical BCC Fe:  1NN = 2.48 Å → vac_dist ≈ 1.2 Å
 *
 * Example:
 *   compute vac all soap_vacancy 1.2 grid 0.4 file vacancies.csv
 *   thermo_style custom step temp c_vac[1] c_vac[2] c_vac[3]
 */
#pragma once

#include "compute.h"

#include <array>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class ComputeSOAPVacancy : public Compute {
 public:
  ComputeSOAPVacancy(class LAMMPS *, int, char **);
  ~ComputeSOAPVacancy() override;

  void   init() override;
  void   compute_vector() override;
  double memory_usage() override;

 private:
  double vac_dist_;        // vacancy threshold [Å]
  double grid_spacing_;    // grid spacing     [Å]
  double cluster_radius_;  // greedy merge radius [Å]
  std::string out_file_;
  bool   has_file_;

  // ---- compact linked-cell list (for nearest-atom queries) ----
  // Built from local + ghost atom positions; no PBC bookkeeping needed
  // because LAMMPS ghost atoms already account for periodic images.
  struct VacCellList {
    double xlo, ylo, zlo;
    double inv_cell;
    int    nx, ny, nz;
    std::vector<int>    head; // head[cell_id], -1 = empty
    std::vector<int>    next; // next[atom_id], -1 = end of chain
    std::vector<double> ax, ay, az; // atom position copies

    void build(double **x, int natoms,
               double xlo_, double ylo_, double zlo_,
               double xhi_, double yhi_, double zhi_,
               double cell_size);

    // Minimum squared distance from query point to any atom.
    double nearestDist2(double px, double py, double pz) const;
  };

  struct VacPt {
    double x, y, z, d_near;
  };

  struct Cluster {
    double x, y, z;        // grid point with max d_near in this cluster
    double d_near_max;
    int    n_pts;
  };

  // Build cell list from all atom->x (nlocal + nghost).
  void buildCellList(VacCellList &cl) const;

  // Scan local subdomain grid; return vacant grid points.
  std::vector<VacPt> scanGrid(const VacCellList &cl) const;

  // Greedy PBC-aware clustering.
  void clusterPoints(const std::vector<VacPt> &pts,
                     std::vector<Cluster> &clusters) const;

  // Append cluster rows to out_file_ (rank 0 only).
  void appendClusterFile(const std::vector<Cluster> &clusters) const;
};

} // namespace LAMMPS_NS
