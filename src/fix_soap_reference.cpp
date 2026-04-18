/* fix_soap_reference.cpp */

#include "fix_soap_reference.h"
#include "compute_soap_descriptor.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
FixSOAPReference::FixSOAPReference(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), c_soap_(nullptr)
{
  if (narg != 5)
    error->all(FLERR,
               "Syntax: fix ID group soap_reference compute_id outfile");

  compute_id_ = std::string(arg[3]);
  outfile_    = std::string(arg[4]);
}

/* ---------------------------------------------------------------------- */
int FixSOAPReference::setmask()
{
  // END_OF_RUN fires once when the run command finishes
  return END_OF_RUN;
}

/* ---------------------------------------------------------------------- */
void FixSOAPReference::init()
{
  // Locate the compute by ID and verify it is a soap_descriptor compute
  auto *c = modify->get_compute_by_id(compute_id_);
  if (!c)
    error->all(FLERR,
               ("fix soap_reference: compute '" + compute_id_ + "' not found").c_str());

  c_soap_ = dynamic_cast<ComputeSOAPDescriptor *>(c);
  if (!c_soap_)
    error->all(FLERR,
               ("fix soap_reference: compute '" + compute_id_ +
                "' is not a soap_descriptor compute").c_str());
}

/* ---------------------------------------------------------------------- */
void FixSOAPReference::end_of_run()
{
  // --- Step 1: ensure DVs are up to date for the last timestep ---
  if (c_soap_->invoked_peratom != update->ntimestep)
    c_soap_->compute_peratom();

  const int    dv_size = c_soap_->dv_size();
  const int    nlocal  = atom->nlocal;
  double     **arr     = c_soap_->array_atom;

  // --- Step 2: local sum of all DVs for global mean ---
  std::vector<double> local_sum(dv_size, 0.0);
  for (int i = 0; i < nlocal; i++)
    for (int d = 0; d < dv_size; d++)
      local_sum[d] += arr[i][d];

  long long local_natoms = static_cast<long long>(nlocal);
  long long total_natoms = 0;
  MPI_Allreduce(&local_natoms, &total_natoms, 1, MPI_LONG_LONG, MPI_SUM, world);

  std::vector<double> mean_dv(dv_size);
  MPI_Allreduce(local_sum.data(), mean_dv.data(), dv_size,
                MPI_DOUBLE, MPI_SUM, world);

  if (total_natoms > 0) {
    const double inv_n = 1.0 / static_cast<double>(total_natoms);
    for (int d = 0; d < dv_size; d++) mean_dv[d] *= inv_n;
  }

  // --- Step 3: local E[d²] and E[d⁴] for chi-distribution fit ---
  double local_ed2 = 0.0, local_ed4 = 0.0;
  for (int i = 0; i < nlocal; i++) {
    double d2 = 0.0;
    for (int d = 0; d < dv_size; d++) {
      const double diff = arr[i][d] - mean_dv[d];
      d2 += diff * diff;
    }
    local_ed2 += d2;
    local_ed4 += d2 * d2;
  }

  double global_ed2 = 0.0, global_ed4 = 0.0;
  MPI_Allreduce(&local_ed2, &global_ed2, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&local_ed4, &global_ed4, 1, MPI_DOUBLE, MPI_SUM, world);

  if (total_natoms > 0) {
    global_ed2 /= static_cast<double>(total_natoms);
    global_ed4 /= static_cast<double>(total_natoms);
  }

  // Method-of-moments chi fit:  k = 2*E[d²]²/Var(d²),  sigma = sqrt(E[d²]/k)
  // Note: the distances here are d (not d²), so we use d² moments directly.
  // Let X = d²: E[X] = global_ed2, E[X²] = global_ed4
  double chi_k = 7.0, chi_sigma = 0.1;
  const double var_d2 = global_ed4 - global_ed2 * global_ed2;
  if (var_d2 > 0.0 && global_ed2 > 0.0) {
    // d is sqrt(X), so distances are sqrt(global_ed2[i]) per atom.
    // Recast: using distances vector E[d²] and Var(d²) through chi moments.
    // Direct form from soap_statistics.h fitChiParams logic:
    chi_k     = 2.0 * global_ed2 * global_ed2 / var_d2;
    chi_sigma = std::sqrt(global_ed2 / chi_k);
  }

  // --- Step 4: rank 0 writes the reference file ---
  if (comm->me == 0)
    writeReference(mean_dv, chi_k, chi_sigma);

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "fix soap_reference: wrote '{}' (chi_k={:.3f}, chi_sigma={:.4f}, "
                   "{} atoms)\n",
                   outfile_, chi_k, chi_sigma, total_natoms);
}

/* ---------------------------------------------------------------------- */
void FixSOAPReference::writeReference(const std::vector<double> &mean_dv,
                                       double chi_k, double chi_sigma) const
{
  const int dv_size = static_cast<int>(mean_dv.size());

  std::ofstream f(outfile_);
  if (!f)
    error->one(FLERR,
               ("fix soap_reference: cannot write to '" + outfile_ + "'").c_str());

  // Header: n_max l_max r_cut sigma chi_k chi_sigma dv_size
  f << std::setprecision(10);
  f << c_soap_->nmax() << " "
    << c_soap_->lmax() << " "
    << c_soap_->rcut() << " "
    << c_soap_->sigma() << " "
    << chi_k           << " "
    << chi_sigma       << " "
    << dv_size         << "\n";

  // Mean DV
  for (int d = 0; d < dv_size; d++) {
    f << mean_dv[d];
    f << ((d + 1 < dv_size) ? " " : "\n");
  }

  if (!f)
    error->one(FLERR,
               ("fix soap_reference: write error on '" + outfile_ + "'").c_str());
}
