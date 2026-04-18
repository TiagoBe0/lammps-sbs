/* compute_local_descriptor.cpp
 *
 * Steinhardt Q_l / W_hat_l bond-order descriptors for ML classification
 * of local atomic environments.
 *
 * Build:
 *   Copy compute_local_descriptor.h/.cpp into the LAMMPS src/ directory
 *   and recompile.  No external dependencies.
 */

#include "compute_local_descriptor.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static const double MY_4PI = 4.0 * M_PI;

/* ---------------------------------------------------------------------- */
ComputeLocalDescriptor::ComputeLocalDescriptor(LAMMPS *lmp,
                                               int narg, char **arg)
    : Compute(lmp, narg, arg),
      llist(nullptr),
      Qlm_r(nullptr), Qlm_i(nullptr),
      Qlm_bar_r(nullptr), Qlm_bar_i(nullptr)
{
  // Minimum: compute ID group local_descriptor cutoff l1
  if (narg < 5)
    error->all(FLERR,
               "Syntax: compute ID group local_descriptor cutoff l1 [l2 ...] "
               "[average yes|no]");

  double cutoff = utils::numeric(FLERR, arg[3], false, lmp);
  if (cutoff <= 0.0)
    error->all(FLERR, "compute local_descriptor: cutoff must be positive");
  cutsq = cutoff * cutoff;

  // Count l values (positive integers before any keyword)
  nnn = 0;
  do_average = 1;  // default on
  int lstart = 4;
  int iarg   = 4;

  while (iarg < narg && arg[iarg][0] >= '0' && arg[iarg][0] <= '9') {
    nnn++;
    iarg++;
  }
  if (nnn == 0)
    error->all(FLERR, "compute local_descriptor: need at least one l value");

  llist = new int[nnn];
  for (int k = 0; k < nnn; k++) {
    llist[k] = utils::inumeric(FLERR, arg[lstart + k], false, lmp);
    if (llist[k] < 1 || llist[k] > 10)
      error->all(FLERR, "compute local_descriptor: l must be 1..10");
  }

  // Parse optional keywords
  while (iarg < narg) {
    if (strcmp(arg[iarg], "average") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute local_descriptor: missing value after 'average'");
      if (strcmp(arg[iarg + 1], "yes") == 0)      do_average = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0)  do_average = 0;
      else
        error->all(FLERR, "compute local_descriptor: 'average' must be yes or no");
      iarg += 2;
    } else {
      error->all(FLERR, "compute local_descriptor: unknown keyword");
    }
  }

  // total_m: number of complex q_lm per atom
  total_m = 0;
  for (int k = 0; k < nnn; k++) total_m += 2 * llist[k] + 1;

  // Output columns: Q_l, W_hat_l per degree; doubled if averaging
  ncols = nnn * 2 * (do_average ? 2 : 1);

  peratom_flag    = 1;
  size_peratom_cols = ncols;
  // If averaging, we need to communicate q_lm of ghost atoms
  comm_forward    = do_average ? total_m * 2 : 0;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */
ComputeLocalDescriptor::~ComputeLocalDescriptor()
{
  delete[] llist;
  memory->destroy(array_atom);
  memory->destroy(Qlm_r);
  memory->destroy(Qlm_i);
  if (do_average) {
    memory->destroy(Qlm_bar_r);
    memory->destroy(Qlm_bar_i);
  }
}

/* ---------------------------------------------------------------------- */
void ComputeLocalDescriptor::init()
{
  if (!force->pair)
    error->all(FLERR, "compute local_descriptor requires a pair style");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */
void ComputeLocalDescriptor::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void ComputeLocalDescriptor::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  int nlocal = atom->nlocal;
  int nall   = nlocal + atom->nghost;

  // --- Grow per-atom arrays if needed ---
  if (nall > nmax) {
    memory->destroy(array_atom);
    memory->destroy(Qlm_r);
    memory->destroy(Qlm_i);
    if (do_average) {
      memory->destroy(Qlm_bar_r);
      memory->destroy(Qlm_bar_i);
    }
    nmax = nall;
    memory->create(array_atom, nmax, ncols,    "local_descriptor:array_atom");
    memory->create(Qlm_r,      nmax, total_m,  "local_descriptor:Qlm_r");
    memory->create(Qlm_i,      nmax, total_m,  "local_descriptor:Qlm_i");
    if (do_average) {
      memory->create(Qlm_bar_r, nmax, total_m, "local_descriptor:Qlm_bar_r");
      memory->create(Qlm_bar_i, nmax, total_m, "local_descriptor:Qlm_bar_i");
    }
  }

  // Zero Qlm arrays for all atoms (local + ghost needed for comm)
  for (int i = 0; i < nall; i++)
    for (int m = 0; m < total_m; m++) {
      Qlm_r[i][m] = 0.0;
      Qlm_i[i][m] = 0.0;
    }

  double **x       = atom->x;
  int    *numneigh = list->numneigh;
  int   **firstneigh = list->firstneigh;

  // --- Pass 1: accumulate q_lm for every atom ---
  for (int i = 0; i < nlocal; i++) {
    int    jnum  = numneigh[i];
    int   *jlist = firstneigh[i];
    int    nb    = 0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      double r2 = dx*dx + dy*dy + dz*dz;
      if (r2 >= cutsq) continue;

      double r    = std::sqrt(r2);
      double rinv = 1.0 / r;
      dx *= rinv; dy *= rinv; dz *= rinv;  // unit vector

      // Accumulate Y_lm contributions
      int moff = 0;
      for (int k = 0; k < nnn; k++) {
        int l = llist[k];
        for (int mi = -l; mi <= l; mi++) {
          double yr, yi;
          ylm(l, mi, dx, dy, dz, yr, yi);
          Qlm_r[i][moff + mi + l] += yr;
          Qlm_i[i][moff + mi + l] += yi;
        }
        moff += 2*l + 1;
      }
      nb++;
    }

    // Normalise by number of neighbours
    if (nb > 0) {
      double inv = 1.0 / nb;
      for (int m = 0; m < total_m; m++) {
        Qlm_r[i][m] *= inv;
        Qlm_i[i][m] *= inv;
      }
    }
  }

  // --- Pass 2 (averaging): sum q_lm over neighbours → Q_bar_lm ---
  if (do_average) {
    // First communicate local q_lm to ghost atoms
    comm->forward_comm(this);

    for (int i = 0; i < nlocal; i++) {
      int    jnum  = numneigh[i];
      int   *jlist = firstneigh[i];

      // Start with self contribution
      for (int m = 0; m < total_m; m++) {
        Qlm_bar_r[i][m] = Qlm_r[i][m];
        Qlm_bar_i[i][m] = Qlm_i[i][m];
      }

      int nb = 1;  // count self
      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj] & NEIGHMASK;
        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= cutsq) continue;

        for (int m = 0; m < total_m; m++) {
          Qlm_bar_r[i][m] += Qlm_r[j][m];
          Qlm_bar_i[i][m] += Qlm_i[j][m];
        }
        nb++;
      }

      double inv = 1.0 / nb;
      for (int m = 0; m < total_m; m++) {
        Qlm_bar_r[i][m] *= inv;
        Qlm_bar_i[i][m] *= inv;
      }
    }
  }

  // --- Pass 3: compute final descriptors and fill array_atom ---
  for (int i = 0; i < nlocal; i++) {
    int col  = 0;
    int moff = 0;
    for (int k = 0; k < nnn; k++) {
      int l = llist[k];
      // Local Q_l and W_hat_l
      double ql, wl;
      ql_wl(l, &Qlm_r[i][moff], &Qlm_i[i][moff], ql, wl);
      array_atom[i][col++] = ql;
      array_atom[i][col++] = wl;

      // Averaged Q_bar_l and W_bar_hat_l
      if (do_average) {
        double qlb, wlb;
        ql_wl(l, &Qlm_bar_r[i][moff], &Qlm_bar_i[i][moff], qlb, wlb);
        array_atom[i][col++] = qlb;
        array_atom[i][col++] = wlb;
      }

      moff += 2*l + 1;
    }
  }
}

/* ---------------------------------------------------------------------- */
/* Communication of q_lm to ghost atoms (needed for averaging)            */
/* ---------------------------------------------------------------------- */
int ComputeLocalDescriptor::pack_forward_comm(int n, int *list_buf,
                                              double *buf, int /*pbc_flag*/,
                                              int * /*pbc*/)
{
  int m = 0;
  for (int ii = 0; ii < n; ii++) {
    int i = list_buf[ii];
    for (int k = 0; k < total_m; k++) {
      buf[m++] = Qlm_r[i][k];
      buf[m++] = Qlm_i[i][k];
    }
  }
  return m;
}

int ComputeLocalDescriptor::unpack_forward_comm(int n, int first, double *buf)
{
  int m    = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    for (int k = 0; k < total_m; k++) {
      Qlm_r[i][k] = buf[m++];
      Qlm_i[i][k] = buf[m++];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */
/* Spherical harmonic Y_lm(θ,φ) for direction (nx, ny, nz) unit vector.  */
/* Uses real & imaginary parts of the complex Y_lm.                       */
/* ---------------------------------------------------------------------- */
void ComputeLocalDescriptor::ylm(int l, int m, double nx, double ny, double nz,
                                 double &yr, double &yi) const
{
  int am = std::abs(m);
  double costheta = nz;  // z-component of unit vector = cos θ
  double p = plm(l, am, costheta);
  double c = clm(l, am);
  double mag = c * p;

  // φ = atan2(ny, nx)
  double phi = std::atan2(ny, nx);
  double cosmphi = std::cos(am * phi);
  double sinmphi = std::sin(am * phi);

  if (m == 0) {
    yr = mag;
    yi = 0.0;
  } else if (m > 0) {
    // Y_lm = C * P * e^{imφ} * sqrt(2) for the real convention
    // Use complex form directly: Y_lm = C_lm * Plm * (cos mφ + i sin mφ)
    yr = mag * cosmphi;
    yi = mag * sinmphi;
  } else {
    // Y_l(-m) = (-1)^m * conj(Y_lm)
    double sign = (am % 2 == 0) ? 1.0 : -1.0;
    yr =  sign * mag * cosmphi;
    yi = -sign * mag * sinmphi;
  }
}

/* ---------------------------------------------------------------------- */
/* Associated Legendre polynomial P_l^m(x) for m >= 0, via recurrence.   */
/* ---------------------------------------------------------------------- */
double ComputeLocalDescriptor::plm(int l, int m, double x) const
{
  // Bootstrap: P_m^m
  double pmm = 1.0;
  if (m > 0) {
    double sq = std::sqrt(1.0 - x*x);
    double fac = 1.0;
    for (int i = 1; i <= m; i++) {
      pmm *= -(2*i - 1) * sq;
      // We drop the (-1)^m Condon-Shortley phase here since clm() absorbs it
    }
    // Remove sign: pmm = (2m-1)!! * sin^m θ  (unsigned)
    pmm = std::fabs(pmm);
  }
  if (l == m) return pmm;

  // P_{m+1}^m
  double pmm1 = x * (2*m + 1) * pmm;
  if (l == m + 1) return pmm1;

  // Recurse upward
  double result = 0.0;
  for (int ll = m + 2; ll <= l; ll++) {
    result = ((2*ll - 1)*x*pmm1 - (ll + m - 1)*pmm) / (ll - m);
    pmm  = pmm1;
    pmm1 = result;
  }
  return result;
}

/* ---------------------------------------------------------------------- */
/* Normalisation coefficient C_lm (absorbs Condon-Shortley phase).        */
/* ---------------------------------------------------------------------- */
double ComputeLocalDescriptor::clm(int l, int m) const
{
  // C_lm = sqrt[(2l+1)/(4π) * (l-m)!/(l+m)!]
  double log_c = 0.5 * (std::log((2*l + 1) / MY_4PI)
                         + logfact(l - m) - logfact(l + m));
  return std::exp(log_c);
}

/* ---------------------------------------------------------------------- */
/* Q_l and W_hat_l from q_lm stored in r[0..2l], i[0..2l]                */
/* Index convention: r[m+l] for m = -l..+l                                */
/* ---------------------------------------------------------------------- */
void ComputeLocalDescriptor::ql_wl(int l, const double *r, const double *i,
                                   double &ql_out, double &wl_out) const
{
  // |q_lm|^2 sum → Q_l
  double sum2 = 0.0;
  for (int m = -l; m <= l; m++) {
    double re = r[m + l], im = i[m + l];
    sum2 += re*re + im*im;
  }
  ql_out = std::sqrt(MY_4PI / (2*l + 1) * sum2);

  // W_l = Σ_{m1+m2+m3=0} (l l l;m1 m2 m3) q_m1 q_m2 q_m3
  // W_hat_l = W_l / sum2^(3/2)
  double w3j_sum_r = 0.0, w3j_sum_i = 0.0;
  for (int m1 = -l; m1 <= l; m1++) {
    for (int m2 = -l; m2 <= l; m2++) {
      int m3 = -(m1 + m2);
      if (m3 < -l || m3 > l) continue;

      double w = wigner3j_lll(l, m1, m2, m3);
      if (w == 0.0) continue;

      // q_m1 * q_m2 * q_m3 (complex multiplication)
      double r1 = r[m1+l], i1 = i[m1+l];
      double r2 = r[m2+l], i2 = i[m2+l];
      double r3 = r[m3+l], i3 = i[m3+l];

      // (r1+ii1)(r2+ii2) = r1r2-i1i2 + i(r1i2+r2i1)
      double pr = r1*r2 - i1*i2;
      double pi = r1*i2 + r2*i1;
      // × (r3+ii3)
      double wr = pr*r3 - pi*i3;
      double wi = pr*i3 + pi*r3;

      w3j_sum_r += w * wr;
      w3j_sum_i += w * wi;
    }
  }

  // W_l should be real for invariant quantities; imaginary part ~ 0
  double wl = w3j_sum_r;
  double denom = std::pow(sum2, 1.5);
  wl_out = (denom > 1.0e-14) ? wl / denom : 0.0;
}

/* ---------------------------------------------------------------------- */
/* Wigner 3j symbol (l l l; m1 m2 m3) via Racah formula.                 */
/* Returns 0 if triangle rule or m1+m2+m3≠0 violated.                    */
/* ---------------------------------------------------------------------- */
double ComputeLocalDescriptor::wigner3j_lll(int l, int m1, int m2, int m3) const
{
  if (m1 + m2 + m3 != 0) return 0.0;
  if (std::abs(m1) > l || std::abs(m2) > l || std::abs(m3) > l) return 0.0;

  // Triangle condition for (l,l,l): need 0 <= l+l-l = l, 1<=2l+1 always true
  // and l+l+l = 3l must satisfy triangle; ok by construction.

  // Racah formula:
  // C-G prefactor
  // (l l l; m1 m2 m3) = (-1)^{l-l+m3} * sqrt[Δ(l,l,l)] * ...
  // For j1=j2=j3=l: Δ(l,l,l) = [(l!)^3] / (3l+1)!

  double log_delta = 3.0 * logfact(l) - logfact(3*l + 1);

  // Row factor: (j1+m1)!(j1-m1)!(j2+m2)!(j2-m2)!(j3+m3)!(j3-m3)!
  double log_row = logfact(l + m1) + logfact(l - m1)
                 + logfact(l + m2) + logfact(l - m2)
                 + logfact(l + m3) + logfact(l - m3);

  double prefactor = std::pow(-1.0, l - l + m3)
                   * std::exp(0.5 * (log_delta + log_row));

  // Sum over s (valid range determined by factorial non-negativity)
  int smin = std::max({0, l - l + m2 - 0, l - l - m1});  // simplified: all = 0
  smin = std::max(0, std::max(-(l - l + m3), -(l - l - m2)));
  // General bounds:
  // s >= 0
  // s <= l+l-l = l
  // s <= l-m1  (= l+m1 reflected since j1-m1-s >= 0)
  // s <= l+m2  (j2+m2-s >= 0)
  // l-l+m1+s >= 0 → s >= l-l-m1 = -m1 → s >= max(0,-m1) ... already s>=0
  // l-l-m2+s >= 0 → s >= m2 (if m2>0)
  int s_lo = std::max(0, std::max(m2, -m1 + l - l));  // = max(0, m2 if m2>0 else 0, ...)
  int s_hi = std::min({l, l - m1, l + m2});

  // Re-derive correctly for j1=j2=j3=j=l:
  // denominator terms: s!, (j1+j2-j3-s)!=(l-s)!, (j1-m1-s)!=(l-m1-s)!,
  //                    (j2+m2-s)!=(l+m2-s)!, (j3-j2+m1+s)!=(m1+s)!,
  //                    (j3-j1-m2+s)!=(s-m2)!   ← need s-m2 >= 0 → s>=m2
  s_lo = std::max({0, -m1, m2});      // s >= 0, s >= -m1 (if m1<0), s >= m2 (if m2>0)
  s_lo = std::max(0, std::max(-m1, m2));
  s_hi = std::min({l, l - m1, l + m2});

  double sum = 0.0;
  for (int s = s_lo; s <= s_hi; s++) {
    // factorial denominators (log to avoid overflow)
    double log_denom = logfact(s) + logfact(l - s) + logfact(l - m1 - s)
                     + logfact(l + m2 - s) + logfact(m1 + s) + logfact(s - m2);
    double term = std::pow(-1.0, s) * std::exp(-log_denom);
    sum += term;
  }

  return prefactor * sum;
}

/* ---------------------------------------------------------------------- */
double ComputeLocalDescriptor::memory_usage()
{
  double bytes = (double)nmax * ncols * sizeof(double);     // array_atom
  bytes       += 2.0 * nmax * total_m * sizeof(double);    // Qlm_r, Qlm_i
  if (do_average)
    bytes     += 2.0 * nmax * total_m * sizeof(double);    // Qlm_bar
  return bytes;
}
