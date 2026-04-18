/* fix_thermal_spike.cpp
 *
 * Thermal Spike initial condition for MeV ion radiation damage.
 * See fix_thermal_spike.h for full documentation.
 */

#include "fix_thermal_spike.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double eV_to_eV   = 1.0;           // metal units: eV
static constexpr double kcal_to_eV = 1.0 / 23.0605; // real units

/* ---------------------------------------------------------------------- */
FixThermalSpike::FixThermalSpike(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), random(nullptr), applied(false)
{
  // fix ID group thermal_spike Se Ltrack x0 y0 z0 ux uy uz sigma [kw val ...]
  if (narg < 13)
    error->all(FLERR,
               "Syntax: fix ID group thermal_spike Se Ltrack "
               "x0 y0 z0 ux uy uz sigma [sigma_long <v>] [seed <n>] [units <str>]");

  Se     = utils::numeric(FLERR, arg[3],  false, lmp);
  Ltrack = utils::numeric(FLERR, arg[4],  false, lmp);
  r0[0]  = utils::numeric(FLERR, arg[5],  false, lmp);
  r0[1]  = utils::numeric(FLERR, arg[6],  false, lmp);
  r0[2]  = utils::numeric(FLERR, arg[7],  false, lmp);
  double ux = utils::numeric(FLERR, arg[8],  false, lmp);
  double uy = utils::numeric(FLERR, arg[9],  false, lmp);
  double uz = utils::numeric(FLERR, arg[10], false, lmp);
  sigma  = utils::numeric(FLERR, arg[11], false, lmp);

  if (Se     <= 0.0) error->all(FLERR, "fix thermal_spike: Se must be > 0");
  if (Ltrack <= 0.0) error->all(FLERR, "fix thermal_spike: Ltrack must be > 0");
  if (sigma  <= 0.0) error->all(FLERR, "fix thermal_spike: sigma must be > 0");

  // Normalise direction
  double norm = std::sqrt(ux*ux + uy*uy + uz*uz);
  if (norm < 1e-12)
    error->all(FLERR, "fix thermal_spike: direction vector has zero length");
  uhat[0] = ux / norm;
  uhat[1] = uy / norm;
  uhat[2] = uz / norm;

  // Defaults for optional keywords
  sigma_long          = 0.0;
  seed                = 42;
  energy_units_factor = eV_to_eV;  // metal units default

  int iarg = 12;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sigma_long") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix thermal_spike: missing value for 'sigma_long'");
      sigma_long = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "seed") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix thermal_spike: missing value for 'seed'");
      seed = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix thermal_spike: missing value for 'units'");
      if (strcmp(arg[iarg+1], "metal") == 0)
        energy_units_factor = eV_to_eV;
      else if (strcmp(arg[iarg+1], "real") == 0)
        energy_units_factor = kcal_to_eV;
      else
        error->all(FLERR, "fix thermal_spike: unknown units (metal|real)");
      iarg += 2;
    } else {
      error->all(FLERR, "fix thermal_spike: unknown keyword");
    }
  }

  random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */
int FixThermalSpike::setmask()
{
  // We hook into PRE_FORCE so we can modify velocities before the first step.
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixThermalSpike::init() {}

/* ---------------------------------------------------------------------- */
void FixThermalSpike::setup(int /*vflag*/)
{
  deposit_energy();  // apply before the first force evaluation
}

/* ---------------------------------------------------------------------- */
void FixThermalSpike::deposit_energy()
{
  if (applied) return;  // apply exactly once
  applied = true;

  int     nlocal = atom->nlocal;
  double **x     = atom->x;
  double **v     = atom->v;
  double  *mass  = atom->mass;
  int     *type  = atom->type;
  int     *mask  = atom->mask;

  // Total energy deposited = Se (eV/Å) × Ltrack (Å), in LAMMPS units
  double E_total_eV = Se * Ltrack * energy_units_factor;
  // Convert eV → LAMMPS energy (metal: 1 eV = 1, real: 1 kcal/mol)
  double E_total = E_total_eV; // already in correct units for metal

  double s_centre = Ltrack * 0.5; // centre of track along u

  // Compute weights for each local atom in the group
  std::vector<double> weight(nlocal, 0.0);
  double w_sum_local = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double s = along_track(x[i]);
    // Only atoms inside the track region contribute
    if (s < 0.0 || s > Ltrack) continue;

    double dp2 = perp_distance_sq(x[i]);
    double w   = std::exp(-dp2 / (2.0 * sigma * sigma));

    // Longitudinal Gaussian (optional)
    if (sigma_long > 1e-6) {
      double ds = s - s_centre;
      w *= std::exp(-ds*ds / (2.0 * sigma_long * sigma_long));
    }

    weight[i]    = w;
    w_sum_local += w;
  }

  // Global sum of weights across MPI ranks
  double w_sum_global = 0.0;
  MPI_Allreduce(&w_sum_local, &w_sum_global, 1, MPI_DOUBLE, MPI_SUM, world);

  if (w_sum_global < 1e-30) {
    if (comm->me == 0)
      error->warning(FLERR,
          "fix thermal_spike: no atoms inside track region — "
          "check r0, ux/uy/uz, Ltrack, sigma.");
    return;
  }

  // Distribute energy and add velocity increments
  double inv_w = E_total / w_sum_global;

  for (int i = 0; i < nlocal; i++) {
    if (weight[i] <= 0.0) continue;

    double dKE  = weight[i] * inv_w;           // kinetic energy to add (eV / LAMMPS unit)
    double m    = mass[type[i]];               // mass in LAMMPS units

    // Additional speed: ΔKE = ½ m Δv²  → |Δv| = sqrt(2 ΔKE / m)
    double dv_mag = std::sqrt(2.0 * dKE / m);

    // Random direction for the velocity boost (isotropic in 3D)
    // Using Marsaglia method for uniform sphere sampling
    double rx, ry, rz, rsq;
    do {
      rx = 2.0 * random->uniform() - 1.0;
      ry = 2.0 * random->uniform() - 1.0;
      rz = 2.0 * random->uniform() - 1.0;
      rsq = rx*rx + ry*ry + rz*rz;
    } while (rsq > 1.0 || rsq < 1e-12);

    double r_inv = dv_mag / std::sqrt(rsq);
    v[i][0] += rx * r_inv;
    v[i][1] += ry * r_inv;
    v[i][2] += rz * r_inv;
  }

  if (comm->me == 0) {
    double E_keV = E_total_eV / 1000.0;
    utils::logmesg(lmp,
        "fix thermal_spike: deposited {:.2f} keV along track "
        "({:.1f} eV/Å × {:.1f} Å), σ_⊥ = {:.1f} Å\n",
        E_keV, Se, Ltrack, sigma);
  }
}

/* ---------------------------------------------------------------------- */
/* Perpendicular distance² from ion track axis                            */
/* ---------------------------------------------------------------------- */
double FixThermalSpike::perp_distance_sq(const double *ri) const
{
  // Vector from track entry r0 to atom ri
  double dx = ri[0] - r0[0];
  double dy = ri[1] - r0[1];
  double dz = ri[2] - r0[2];

  // Project onto track direction
  double proj = dx*uhat[0] + dy*uhat[1] + dz*uhat[2];

  // Perpendicular component
  double px = dx - proj * uhat[0];
  double py = dy - proj * uhat[1];
  double pz = dz - proj * uhat[2];

  return px*px + py*py + pz*pz;
}

/* ---------------------------------------------------------------------- */
/* Coordinate along track axis (signed, from entry point r0)              */
/* ---------------------------------------------------------------------- */
double FixThermalSpike::along_track(const double *ri) const
{
  double dx = ri[0] - r0[0];
  double dy = ri[1] - r0[1];
  double dz = ri[2] - r0[2];
  return dx*uhat[0] + dy*uhat[1] + dz*uhat[2];
}
