/* fix_soap_reference  –  build and write SOAP reference file from equilibration
 *
 * At the end of the run this fix:
 *   1. Forces compute_soap_descriptor to evaluate DVs for all local atoms.
 *   2. MPI-reduces the per-atom DV sum to obtain the global mean q̄(T).
 *   3. Computes distances ||q̃^i - q̄(T)|| for every atom and reduces the
 *      E[d²] and E[d⁴] moments needed for the chi-distribution fit.
 *   4. Writes the reference file (rank 0 only).
 *
 * Reference file format:
 *   Line 1:  n_max l_max r_cut sigma chi_k chi_sigma dv_size
 *   Line 2+: dv_size mean DV values (one per token, space-separated)
 *
 * This file is consumed by compute_soap_classify.
 *
 * Syntax:
 *   fix ID group soap_reference compute_id outfile
 *
 *   compute_id   ID of an existing compute soap_descriptor
 *   outfile      path for the reference parameter file
 *
 * Example (equilibration script):
 *   compute    dv   all soap_descriptor 5.5 9 9
 *   fix        ref  all soap_reference  dv  ref_params.dat
 *   run        5000
 *   unfix      ref
 *   uncompute  dv
 */
#pragma once

#include "fix.h"
#include "soap_statistics.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixSOAPReference : public Fix {
 public:
  FixSOAPReference(class LAMMPS *, int, char **);
  ~FixSOAPReference() override = default;

  int  setmask() override;
  void init() override;
  void end_of_run() override;

 private:
  std::string compute_id_;
  std::string outfile_;

  class ComputeSOAPDescriptor *c_soap_;

  void writeReference(const std::vector<double> &mean_dv,
                      double chi_k, double chi_sigma) const;
};

} // namespace LAMMPS_NS
