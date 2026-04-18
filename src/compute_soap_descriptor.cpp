/* compute_soap_descriptor.cpp
 *
 * Build: copy compute_soap_descriptor.h/.cpp and the soap_*.h helpers into
 *        LAMMPS src/ and recompile with CMake.  No external dependencies
 *        beyond what LAMMPS already requires.
 *
 * References:
 *   Bartók, Kondor & Csányi, Phys. Rev. B 87, 184115 (2013)  [SOAP]
 *   Domínguez-Gutiérrez & von Toussaint, NME 22, 100724 (2019)
 */

#include "compute_soap_descriptor.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
ComputeSOAPDescriptor::ComputeSOAPDescriptor(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg),
      rcut_(5.0), sigma_(-1.0), n_max_(9), l_max_(9),
      rb_(9, 5.0), sh_(9),
      nmax_(0), list(nullptr)
{
  // Minimum: compute ID group soap_descriptor r_cut n_max l_max
  if (narg < 6)
    error->all(FLERR,
               "Syntax: compute ID group soap_descriptor r_cut n_max l_max [sigma S]");

  rcut_  = utils::numeric(FLERR, arg[3], false, lmp);
  n_max_ = utils::inumeric(FLERR, arg[4], false, lmp);
  l_max_ = utils::inumeric(FLERR, arg[5], false, lmp);

  if (rcut_ <= 0.0) error->all(FLERR, "compute soap_descriptor: r_cut must be > 0");
  if (n_max_ < 1)   error->all(FLERR, "compute soap_descriptor: n_max must be >= 1");
  if (l_max_ < 0)   error->all(FLERR, "compute soap_descriptor: l_max must be >= 0");

  sigma_ = -1.0;
  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sigma") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute soap_descriptor: missing value after 'sigma'");
      sigma_ = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "compute soap_descriptor: unknown keyword");
    }
  }

  // Rebuild math objects with final parameters
  rb_ = SOAP::RadialBasis(n_max_, rcut_, sigma_);
  sh_ = SOAP::SphericalHarmonics(l_max_);

  cutsq_   = rcut_ * rcut_;
  n_pairs_ = n_max_ * (n_max_ + 1) / 2;
  sh_size_ = (l_max_ + 1) * (l_max_ + 1);
  dv_size_ = n_pairs_ * (l_max_ + 1);

  peratom_flag      = 1;
  size_peratom_cols = dv_size_;
}

/* ---------------------------------------------------------------------- */
ComputeSOAPDescriptor::~ComputeSOAPDescriptor()
{
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPDescriptor::init()
{
  if (!force->pair)
    error->all(FLERR, "compute soap_descriptor requires a pair style");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPDescriptor::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPDescriptor::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  const int nlocal = atom->nlocal;

  // Grow output array if needed
  if (nlocal > nmax_) {
    memory->destroy(array_atom);
    nmax_ = nlocal;
    memory->create(array_atom, nmax_, dv_size_, "soap_descriptor:array_atom");
  }

  double **x         = atom->x;
  int    *numneigh   = list->numneigh;
  int   **firstneigh = list->firstneigh;

  // Each OpenMP thread gets its own scratch buffers; no data races on
  // array_atom because threads write to disjoint rows (different i).
#if defined(_OPENMP)
#pragma omp parallel
  {
    std::vector<double> c_buf(n_max_ * sh_size_);
    std::vector<double> rb_buf(n_max_);
    std::vector<double> sh_buf(sh_size_);
#pragma omp for schedule(dynamic, 32)
    for (int i = 0; i < nlocal; i++)
      computeAtom(i, x, numneigh, firstneigh,
                  c_buf.data(), rb_buf.data(), sh_buf.data());
  }
#else
  {
    std::vector<double> c_buf(n_max_ * sh_size_);
    std::vector<double> rb_buf(n_max_);
    std::vector<double> sh_buf(sh_size_);
    for (int i = 0; i < nlocal; i++)
      computeAtom(i, x, numneigh, firstneigh,
                  c_buf.data(), rb_buf.data(), sh_buf.data());
  }
#endif
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPDescriptor::computeAtom(
    int i,
    double **x,
    int *numneigh, int **firstneigh,
    double *c_buf,
    double *rb_buf,
    double *sh_buf) const
{
  // Zero expansion coefficient matrix  c[n][s]
  std::fill(c_buf, c_buf + n_max_ * sh_size_, 0.0);

  const int jnum   = numneigh[i];
  const int *jlist = firstneigh[i];

  for (int jj = 0; jj < jnum; jj++) {
    const int j  = jlist[jj] & NEIGHMASK;
    const double dx = x[j][0] - x[i][0];
    const double dy = x[j][1] - x[i][1];
    const double dz = x[j][2] - x[i][2];
    const double r2 = dx*dx + dy*dy + dz*dz;
    if (r2 >= cutsq_) continue;

    rb_.computeInto(std::sqrt(r2), rb_buf);
    sh_.computeInto(dx, dy, dz, sh_buf);

    for (int n = 0; n < n_max_; ++n) {
      const double rb_n = rb_buf[n];
      if (rb_n == 0.0) continue;
      double *cp = c_buf + n * sh_size_;
      for (int s = 0; s < sh_size_; ++s)
        cp[s] += rb_n * sh_buf[s];
    }
  }

  // Power spectrum  p_{nn'l} = pi*sqrt(8/(2l+1)) * sum_m c_{nlm} * c_{n'lm}
  double *dv = array_atom[i];
  std::fill(dv, dv + dv_size_, 0.0);

  for (int l = 0; l <= l_max_; ++l) {
    const double factor = M_PI * std::sqrt(8.0 / (2.0*l + 1.0));
    const int    l_base = l*l + l;   // index of m=0 in sh buffer
    for (int n = 0; n < n_max_; ++n) {
      const double *cn = c_buf + n * sh_size_;
      for (int np = n; np < n_max_; ++np) {
        const double *cnp = c_buf + np * sh_size_;
        double pnnl = 0.0;
        for (int m = -l; m <= l; ++m)
          pnnl += cn[l_base + m] * cnp[l_base + m];
        // upper-triangle pair index: n*(n_max - (n-1)/2) + (np-n)
        const int pair_idx = n * n_max_ - n*(n-1)/2 + (np - n);
        dv[l * n_pairs_ + pair_idx] = factor * pnnl;
      }
    }
  }

  // Normalise to unit sphere
  double norm2 = 0.0;
  for (int d = 0; d < dv_size_; ++d) norm2 += dv[d] * dv[d];
  if (norm2 > 0.0) {
    const double inv_norm = 1.0 / std::sqrt(norm2);
    for (int d = 0; d < dv_size_; ++d) dv[d] *= inv_norm;
  }
}

/* ---------------------------------------------------------------------- */
double ComputeSOAPDescriptor::memory_usage()
{
  return static_cast<double>(nmax_ * dv_size_) * sizeof(double);
}
