/* compute_local_freevol.h
 *
 * Per-atom local free-volume estimator and vacancy cluster identifier.
 *
 * Method (no Voro++ or reference configuration required):
 *   For atom i within cutoff r_c:
 *     V_local(i)  = (4π/3 * r_c³) / (N_neigh(i) + 1)
 *     δV(i)       = [V_local(i) - V_ref] / V_ref
 *     is_vacancy  = δV(i) > threshold
 *
 *   V_ref is either set by the user or auto-computed as the mean V_local
 *   over the bulk-like atoms (first call = equilibrated configuration).
 *
 *   Connected vacancy-candidate atoms (within vacancy_cutoff of each other)
 *   are assigned the same cluster_id using a union-find algorithm.
 *
 * Syntax:
 *   compute ID group local_freevol cutoff [keyword value] ...
 *
 * Keywords:
 *   vref    <float>   Reference atomic volume (Å³). 0 = auto (default).
 *   thresh  <float>   δV threshold above which an atom is a vacancy
 *                     candidate. Default 0.3 (30% excess volume).
 *   vcut    <float>   Cutoff for vacancy-candidate clustering (Å).
 *                     Default = same as cutoff.
 *
 * Output columns per atom (4 total):
 *   [1] V_local     Local atomic volume estimate (Å³)
 *   [2] dV          Normalised excess volume (δV)
 *   [3] is_vacancy  1 if δV > thresh, else 0
 *   [4] cluster_id  Integer cluster label (≥1); 0 = not a vacancy candidate
 *
 * Example:
 *   compute fv all local_freevol 4.0 thresh 0.25
 *   dump    1  all custom 500 dump.fv id type x y z &
 *              c_fv[1] c_fv[2] c_fv[3] c_fv[4]
 *
 * References:
 *   Local packing fraction approach: Anikeenko & Medvedev, PRL 98, 235504 (2007)
 *   Union-Find for cluster labelling: Hoshen & Kopelman, PRB 14, 3438 (1976)
 */

#pragma once

#include "compute.h"

namespace LAMMPS_NS {

class ComputeLocalFreevol : public Compute {
 public:
  ComputeLocalFreevol(class LAMMPS *, int, char **);
  ~ComputeLocalFreevol() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

 private:
  double cutsq;          // neighbour cutoff squared (Å²)
  double vcut_sq;        // clustering cutoff squared (Å²)
  double vref;           // reference atomic volume (Å³); 0 = auto
  double thresh;         // δV threshold for vacancy candidates
  int    auto_vref;      // 1 if vref should be computed automatically

  int    nmax;           // allocated size of array_atom
  static constexpr int NCOLS = 4;

  class NeighList *list;

  // Union-Find helpers for connected-component clustering
  std::vector<int> _uf_parent;
  std::vector<int> _uf_rank;
  int  _uf_find(int x);
  void _uf_union(int a, int b);
  void _uf_init(int n);
};

}  // namespace LAMMPS_NS
