/* compute_soap_classify  –  per-atom SOAP defect classification
 *
 * Reads a reference file produced by fix_soap_reference and, for every atom,
 * computes three quantities:
 *
 *   Column 1  dist_to_ref   ||q̃^i - q̄(T)||₂
 *   Column 2  defect_prob   1 - P_norm(d^i | k, σ)  from chi-distribution
 *   Column 3  defect_type   0 = Lattice   1 = Defect/Unknown
 *
 * Atoms with dist_to_ref < threshold are labelled Lattice (0); the rest are
 * labelled Defect (1).  The threshold defaults to 0.15 but can be overridden.
 *
 * Reference file format (written by fix_soap_reference):
 *   Line 1:  n_max l_max r_cut sigma chi_k chi_sigma dv_size
 *   Line 2+: dv_size whitespace-separated mean DV values
 *
 * Syntax:
 *   compute ID group soap_classify ref_file r_cut n_max l_max
 *                  [sigma S] [threshold T]
 *
 * Example (cascade run):
 *   compute cls all soap_classify ref_params.dat 5.5 9 9 threshold 0.15
 *   dump 1 all custom 100 cascade.dump id type x y z &
 *                c_cls[1] c_cls[2] c_cls[3]
 */
#pragma once

#include "compute.h"
#include "soap_radial_basis.h"
#include "soap_spherical_harmonics.h"
#include "soap_statistics.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class ComputeSOAPClassify : public Compute {
 public:
  ComputeSOAPClassify(class LAMMPS *, int, char **);
  ~ComputeSOAPClassify() override;

  void   init() override;
  void   init_list(int, class NeighList *) override;
  void   compute_peratom() override;
  double memory_usage() override;

 private:
  // user parameters
  std::string ref_file_;
  double      rcut_, sigma_, threshold_;
  int         n_max_, l_max_;

  // derived sizes
  int    n_pairs_, sh_size_, dv_size_;
  double cutsq_;

  // SOAP math objects
  SOAP::RadialBasis        rb_;
  SOAP::SphericalHarmonics sh_;

  // reference data loaded from file
  std::vector<double> mean_dv_;
  double chi_k_, chi_sigma_;

  int nmax_;
  class NeighList *list;

  void loadReference();

  // Compute SOAP DV for atom i into dv_out; also return dist_to_ref.
  double computeAndClassify(int i,
                            double **x,
                            int *numneigh, int **firstneigh,
                            double *c_buf,
                            double *rb_buf,
                            double *sh_buf,
                            double *dv_buf) const;
};

} // namespace LAMMPS_NS
