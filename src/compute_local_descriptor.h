/* Compute local_descriptor – Steinhardt bond-order descriptors for ML
 *
 * Computes per-atom Q_l and W_hat_l (optionally neighbor-averaged Q_bar_l
 * and W_bar_hat_l) for a list of spherical-harmonic degrees l.
 *
 * Syntax:
 *   compute ID group local_descriptor cutoff l1 [l2 ...] [average yes|no]
 *
 * Output columns (in order, for each l):
 *   Q_l  W_hat_l  [Q_bar_l  W_bar_hat_l]   (brackets when average=yes)
 *
 * Example:
 *   compute desc all local_descriptor 4.0 4 6 average yes
 *   dump    1   all custom 100 dump.desc id type x y z c_desc[1] c_desc[2] c_desc[3] c_desc[4] c_desc[5] c_desc[6] c_desc[7] c_desc[8]
 *
 * References:
 *   Steinhardt et al., PRB 28, 784 (1983)
 *   Lechner & Dellago, JCP 129, 114707 (2008)  [averaged version]
 */

#pragma once

#include "compute.h"

namespace LAMMPS_NS {

class ComputeLocalDescriptor : public Compute {
 public:
  ComputeLocalDescriptor(class LAMMPS *, int, char **);
  ~ComputeLocalDescriptor() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

  // Forward communication of q_lm to ghost atoms (for averaging)
  int pack_forward_comm(int, int *, double *, int, int *) override;
  int unpack_forward_comm(int, int, double *) override;

 private:
  // --- user parameters ---
  int    nnn;         // number of l-degrees requested
  int   *llist;       // array of l values
  int    do_average;  // 1 = also compute neighbour-averaged Q_bar / W_bar
  double cutsq;       // squared cutoff radius

  // --- internal book-keeping ---
  int nmax;           // allocated rows in array_atom / Qlm arrays
  int ncols;          // columns per atom in output
  int total_m;        // Σ(2l+1) over all l values

  class NeighList *list;

  /* Per-atom q_lm storage (flattened over all l,m).
   * Layout: for atom i, q_lm for l=llist[0] from m=-l..+l,
   *         then l=llist[1], etc.
   * _r = real part, _i = imaginary part.
   */
  double **Qlm_r, **Qlm_i;         // local  q_lm
  double **Qlm_bar_r, **Qlm_bar_i; // averaged q_lm (after comm)

  // --- spherical harmonics helpers ---
  // Compute Y_lm for a single (l,m) given direction (x,y,z).
  // Stores real part in yr, imaginary part in yi.
  void   ylm(int l, int m, double x, double y, double z, double &yr, double &yi) const;

  // Associated Legendre polynomial P_l^m(x) – m >= 0.
  double plm(int l, int m, double x) const;

  // Normalisation factor C_lm = sqrt[(2l+1)(l-|m|)! / (4π(l+|m|)!)]
  double clm(int l, int m) const;

  // --- Wigner 3j ---
  // Compute (l l l; m1 m2 m3) using the Racah formula (log-factorial).
  double wigner3j_lll(int l, int m1, int m2, int m3) const;

  // log(n!) via lgamma
  static double logfact(int n) { return std::lgamma(n + 1.0); }

  // Given q_lm for one l (stored in r[]/i[], length 2l+1, index = m+l),
  // compute Q_l and W_hat_l.
  void ql_wl(int l, const double *r, const double *i,
             double &ql_out, double &wl_out) const;
};

}  // namespace LAMMPS_NS
