/* compute_soap_classify.cpp */

#include "compute_soap_classify.h"

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
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace LAMMPS_NS;

static constexpr int NCOLS = 3; // dist_to_ref, defect_prob, defect_type

/* ---------------------------------------------------------------------- */
ComputeSOAPClassify::ComputeSOAPClassify(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg),
      rcut_(5.0), sigma_(-1.0), threshold_(0.15),
      n_max_(9), l_max_(9),
      chi_k_(7.0), chi_sigma_(0.1),
      rb_(9, 5.0), sh_(9),
      nmax_(0), list(nullptr)
{
  // Minimum: compute ID group soap_classify ref_file r_cut n_max l_max
  if (narg < 7)
    error->all(FLERR,
               "Syntax: compute ID group soap_classify ref_file r_cut n_max l_max "
               "[sigma S] [threshold T]");

  ref_file_ = std::string(arg[3]);
  rcut_     = utils::numeric(FLERR, arg[4], false, lmp);
  n_max_    = utils::inumeric(FLERR, arg[5], false, lmp);
  l_max_    = utils::inumeric(FLERR, arg[6], false, lmp);

  if (rcut_ <= 0.0) error->all(FLERR, "compute soap_classify: r_cut must be > 0");
  if (n_max_ < 1)   error->all(FLERR, "compute soap_classify: n_max must be >= 1");
  if (l_max_ < 0)   error->all(FLERR, "compute soap_classify: l_max must be >= 0");

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sigma") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute soap_classify: missing value after 'sigma'");
      sigma_ = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "threshold") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute soap_classify: missing value after 'threshold'");
      threshold_ = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (threshold_ <= 0.0)
        error->all(FLERR, "compute soap_classify: threshold must be > 0");
      iarg += 2;
    } else {
      error->all(FLERR, "compute soap_classify: unknown keyword");
    }
  }

  rb_ = SOAP::RadialBasis(n_max_, rcut_, sigma_);
  sh_ = SOAP::SphericalHarmonics(l_max_);

  cutsq_   = rcut_ * rcut_;
  n_pairs_ = n_max_ * (n_max_ + 1) / 2;
  sh_size_ = (l_max_ + 1) * (l_max_ + 1);
  dv_size_ = n_pairs_ * (l_max_ + 1);

  // Load reference on rank 0, then broadcast
  loadReference();

  peratom_flag      = 1;
  size_peratom_cols = NCOLS;
}

/* ---------------------------------------------------------------------- */
ComputeSOAPClassify::~ComputeSOAPClassify()
{
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPClassify::loadReference()
{
  // Only rank 0 reads; then broadcast to all ranks
  int dv_size_file = 0;
  double chi_k_in = 7.0, chi_sigma_in = 0.1;
  std::vector<double> mean_dv_in;

  if (comm->me == 0) {
    std::ifstream f(ref_file_);
    if (!f)
      error->one(FLERR, ("compute soap_classify: cannot open reference file: "
                         + ref_file_).c_str());

    // Header line: n_max l_max r_cut sigma chi_k chi_sigma dv_size
    int    hdr_nmax, hdr_lmax;
    double hdr_rcut, hdr_sigma;
    f >> hdr_nmax >> hdr_lmax >> hdr_rcut >> hdr_sigma
      >> chi_k_in >> chi_sigma_in >> dv_size_file;

    if (!f)
      error->one(FLERR, "compute soap_classify: malformed reference file header");

    if (hdr_nmax != n_max_ || hdr_lmax != l_max_)
      error->one(FLERR,
                 "compute soap_classify: reference file n_max/l_max mismatch");

    if (dv_size_file != dv_size_)
      error->one(FLERR,
                 "compute soap_classify: reference file dv_size mismatch");

    mean_dv_in.resize(dv_size_file);
    for (int d = 0; d < dv_size_file; ++d) {
      f >> mean_dv_in[d];
      if (!f)
        error->one(FLERR, "compute soap_classify: truncated mean DV in reference file");
    }
  }

  // Broadcast chi params and mean DV
  MPI_Bcast(&chi_k_in,     1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&chi_sigma_in, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&dv_size_file, 1, MPI_INT,    0, world);

  if (comm->me != 0) mean_dv_in.resize(dv_size_file);
  MPI_Bcast(mean_dv_in.data(), dv_size_file, MPI_DOUBLE, 0, world);

  chi_k_    = chi_k_in;
  chi_sigma_ = chi_sigma_in;
  mean_dv_  = std::move(mean_dv_in);
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPClassify::init()
{
  if (!force->pair)
    error->all(FLERR, "compute soap_classify requires a pair style");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPClassify::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPClassify::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  const int nlocal = atom->nlocal;

  if (nlocal > nmax_) {
    memory->destroy(array_atom);
    nmax_ = nlocal;
    memory->create(array_atom, nmax_, NCOLS, "soap_classify:array_atom");
  }

  double **x         = atom->x;
  int    *numneigh   = list->numneigh;
  int   **firstneigh = list->firstneigh;

  std::vector<double> c_buf(n_max_ * sh_size_);
  std::vector<double> rb_buf(n_max_);
  std::vector<double> sh_buf(sh_size_);
  std::vector<double> dv_buf(dv_size_);

  for (int i = 0; i < nlocal; i++) {
    const double d = computeAndClassify(
        i, x, numneigh, firstneigh,
        c_buf.data(), rb_buf.data(), sh_buf.data(), dv_buf.data());

    array_atom[i][0] = d;
    array_atom[i][1] = 1.0 - SOAP::chiProbNorm(d, chi_k_, chi_sigma_);
    array_atom[i][2] = (d < threshold_) ? 0.0 : 1.0;
  }
}

/* ---------------------------------------------------------------------- */
double ComputeSOAPClassify::computeAndClassify(
    int i,
    double **x,
    int *numneigh, int **firstneigh,
    double *c_buf,
    double *rb_buf,
    double *sh_buf,
    double *dv_buf) const
{
  std::fill(c_buf, c_buf + n_max_ * sh_size_, 0.0);

  const int  jnum  = numneigh[i];
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

  // Power spectrum
  std::fill(dv_buf, dv_buf + dv_size_, 0.0);
  for (int l = 0; l <= l_max_; ++l) {
    const double factor = M_PI * std::sqrt(8.0 / (2.0*l + 1.0));
    const int    l_base = l*l + l;
    for (int n = 0; n < n_max_; ++n) {
      const double *cn = c_buf + n * sh_size_;
      for (int np = n; np < n_max_; ++np) {
        const double *cnp = c_buf + np * sh_size_;
        double pnnl = 0.0;
        for (int m = -l; m <= l; ++m)
          pnnl += cn[l_base + m] * cnp[l_base + m];
        const int pair_idx = n * n_max_ - n*(n-1)/2 + (np - n);
        dv_buf[l * n_pairs_ + pair_idx] = factor * pnnl;
      }
    }
  }

  // Normalise
  double norm2 = 0.0;
  for (int d = 0; d < dv_size_; ++d) norm2 += dv_buf[d] * dv_buf[d];
  if (norm2 > 0.0) {
    const double inv = 1.0 / std::sqrt(norm2);
    for (int d = 0; d < dv_size_; ++d) dv_buf[d] *= inv;
  }

  return SOAP::euclideanDist(dv_buf, mean_dv_.data(), dv_size_);
}

/* ---------------------------------------------------------------------- */
double ComputeSOAPClassify::memory_usage()
{
  return static_cast<double>(nmax_ * NCOLS) * sizeof(double)
       + static_cast<double>(dv_size_) * sizeof(double); // mean_dv_
}
