/* compute_soap_descriptor  –  per-atom normalised SOAP descriptor vector
 *
 * Computes the rotationally-invariant SOAP power-spectrum fingerprint for
 * every atom.  Output is a normalised vector of length
 *   dv_size = n_max*(n_max+1)/2 * (l_max+1)
 * (450 components for the default n_max=9, l_max=9).
 *
 * Syntax:
 *   compute ID group soap_descriptor r_cut n_max l_max [sigma S]
 *
 *   r_cut  cutoff radius [Å]
 *   n_max  radial basis functions  (default 9)
 *   l_max  max angular momentum    (default 9)
 *   sigma  Gaussian width [Å]; omit or -1 → r_cut/(n_max+1)
 *
 * Output: array_atom[i][0..dv_size-1]
 *
 * Example:
 *   compute soap all soap_descriptor 5.5 9 9
 *   dump 1 all custom 100 dump.soap id type x y z c_soap[1] c_soap[2] ...
 *
 * Use fix_soap_reference to build the reference file from this compute.
 * Use compute_soap_classify to classify atoms against a reference file.
 */
#pragma once

#include "compute.h"
#include "soap_radial_basis.h"
#include "soap_spherical_harmonics.h"

namespace LAMMPS_NS {

class ComputeSOAPDescriptor : public Compute {
 public:
  ComputeSOAPDescriptor(class LAMMPS *, int, char **);
  ~ComputeSOAPDescriptor() override;

  void   init() override;
  void   init_list(int, class NeighList *) override;
  void   compute_peratom() override;
  double memory_usage() override;

  int    dv_size() const { return dv_size_; }
  int    nmax()    const { return n_max_; }
  int    lmax()    const { return l_max_; }
  double rcut()    const { return rcut_; }
  double sigma()   const { return rb_.sigma(); }

 private:
  // user parameters
  double rcut_, sigma_;
  int    n_max_, l_max_;

  // derived sizes
  int    n_pairs_;   // n_max*(n_max+1)/2
  int    sh_size_;   // (l_max+1)^2
  int    dv_size_;   // n_pairs_ * (l_max+1)
  double cutsq_;

  // SOAP math objects
  SOAP::RadialBasis        rb_;
  SOAP::SphericalHarmonics sh_;

  int nmax_;    // rows currently allocated in array_atom
  class NeighList *list;

  // Compute and normalise the SOAP DV for atom i; store into array_atom[i].
  void computeAtom(int i,
                   double **x,
                   int *numneigh, int **firstneigh,
                   double *c_buf,
                   double *rb_buf,
                   double *sh_buf) const;
};

} // namespace LAMMPS_NS
