/* fix_thermal_spike.h
 *
 * Thermal Spike initial condition for MeV heavy-ion radiation damage.
 *
 * Models the energy deposition of a swift heavy ion (MeV range) along a
 * linear track using the inelastic thermal spike (iTS) model.
 * Reference: Prada et al., Eur. Phys. J. D 77, 18 (2023)
 *            Papaleo et al., Phys. Rev. Lett. 114, 118302 (2015)
 *
 * The fix deposits kinetic energy to atoms near the ion track at the
 * beginning of the run (applied once), simulating the instantaneous
 * electronic energy transfer to the atomic subsystem.
 *
 * Energy profile:
 *   ΔKE(i) = E_total * w(i) / Σ_j w(j)
 *   w(i)   = exp( -d_perp(i)² / (2 σ²) )
 *            × exp( -(s(i) - s_centre)² / (2 σ_long²) )  [if sigma_long > 0]
 *
 *   where d_perp(i) = perpendicular distance from ion track axis
 *         s(i)      = position along track axis
 *
 * The energy increment is added as a random-direction velocity boost,
 * preserving the pre-existing thermal velocity of each atom.
 *
 * Syntax:
 *   fix ID group thermal_spike Se Ltrack x0 y0 z0 ux uy uz sigma &
 *       [sigma_long <σ_L>] [seed <N>] [units <eV|keV|metal>]
 *
 * Required arguments:
 *   Se      Effective stopping power (eV/Å) — electronic stopping
 *   Ltrack  Track length (Å) — total energy = Se × Ltrack
 *   x0 y0 z0   Track entry point (Å)
 *   ux uy uz   Track direction (will be normalised)
 *   sigma      Transverse Gaussian width (Å), typically 3–6 Å for metals
 *
 * Optional keywords:
 *   sigma_long <σ_L>  Longitudinal Gaussian width (Å). 0 = uniform (default).
 *   seed <N>          RNG seed for random velocity directions (default 42).
 *   units <str>       Energy units: "metal" (eV, default) or "real" (kcal/mol).
 *
 * Output columns via fix property/atom (accessible in LAMMPS):
 *   None – modifies atom velocities in-place.
 *   Use thermo keywords pe/ke to monitor energy deposition.
 *
 * Example (Cu, 100 keV Au ion along z, σ = 4 Å):
 *   fix spike all thermal_spike 50.0 720.0 18.0 18.0 0.0 0 0 1 4.0 seed 777
 *   run  30000
 *   unfix spike   # (apply once → no-op after first step, but safe to unfix)
 */

#pragma once

#include "fix.h"

namespace LAMMPS_NS {

class FixThermalSpike : public Fix {
 public:
  FixThermalSpike(class LAMMPS *, int, char **);
  ~FixThermalSpike() override = default;

  int  setmask() override;
  void init() override;
  void setup(int) override;          // called before first run step

 private:
  // Track geometry
  double r0[3];   // entry point
  double uhat[3]; // unit direction vector

  // Energy parameters
  double Se;          // stopping power (eV/Å)
  double Ltrack;      // track length (Å)
  double sigma;       // transverse Gaussian width (Å)
  double sigma_long;  // longitudinal width (Å), 0 = uniform

  // Internal
  int    seed;
  double energy_units_factor; // 1.0 for metal, 1/23.06 for real
  bool   applied;             // apply only once

  class RanMars *random;

  // Helpers
  void   deposit_energy();
  double perp_distance_sq(const double *ri) const;
  double along_track(const double *ri)      const;
};

}  // namespace LAMMPS_NS
