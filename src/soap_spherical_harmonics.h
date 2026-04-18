/* soap_spherical_harmonics.h  –  Real tesseral spherical harmonics (header-only)
 *
 * QUIP/libatoms convention.  Index of (l,m): l^2 + l + m,  m in [-l, l].
 * Total size: (l_max+1)^2.
 *
 * Hot-path optimisations:
 *   - Bonnet recurrence for associated Legendre polynomials  O(l_max^2)
 *   - Trigonometric recurrence for cos/sin(m*phi)  – avoids atan2
 *   - thread_local scratch buffers: no heap allocation per call
 */
#pragma once
#include <cmath>
#include <vector>

namespace SOAP {

class SphericalHarmonics {
 public:
  explicit SphericalHarmonics(int l_max) : l_max_(l_max)
  {
    const int sz = (l_max + 1) * (l_max + 2) / 2;
    norm_.resize(sz);
    for (int l = 0; l <= l_max; ++l) {
      double fact_ratio = 1.0;
      norm_[l*(l+1)/2 + 0] = std::sqrt((2.0*l + 1.0) / (4.0*M_PI) * fact_ratio);
      for (int m = 1; m <= l; ++m) {
        fact_ratio /= static_cast<double>((l + m) * (l - m + 1));
        norm_[l*(l+1)/2 + m] = std::sqrt((2.0*l + 1.0) / (4.0*M_PI) * fact_ratio);
      }
    }
  }

  // Fill out[(l_max+1)^2] with Y_lm(dx,dy,dz).
  void computeInto(double dx, double dy, double dz, double *out) const
  {
    const int N = (l_max_+1)*(l_max_+1);
    const double r = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (r < 1e-12) {
      for (int i = 0; i < N; ++i) out[i] = 0.0;
      out[0] = norm_[0];
      return;
    }

    const double cos_theta = dz / r;
    const double rxy = std::sqrt(dx*dx + dy*dy);
    double cos_phi, sin_phi;
    if (rxy < 1e-12) { cos_phi = 1.0; sin_phi = 0.0; }
    else              { cos_phi = dx / rxy; sin_phi = dy / rxy; }
    const double sin_theta = rxy / r;

    // --- associated Legendre polynomials (unnormalised) ---
    const int plm_sz = (l_max_+1)*(l_max_+2)/2;
    thread_local std::vector<double> plm;
    plm.assign(plm_sz, 0.0);
    plm[0] = 1.0;

    for (int m = 0; m < l_max_; ++m) {
      plm[(m+1)*(m+2)/2+(m+1)] = -(2.0*m + 1.0) * sin_theta * plm[m*(m+1)/2+m];
      plm[(m+1)*(m+2)/2+m]     =  cos_theta * (2.0*m + 1.0) * plm[m*(m+1)/2+m];
    }
    for (int m = 0; m <= l_max_; ++m)
      for (int l = m+2; l <= l_max_; ++l)
        plm[l*(l+1)/2+m] = (cos_theta*(2.0*l-1.0)*plm[(l-1)*l/2+m]
                            - (l+m-1.0)*plm[(l-2)*(l-1)/2+m]) / (l-m);

    // --- trig recurrence for cos/sin(m*phi) ---
    thread_local std::vector<double> cp, sp;
    cp.resize(l_max_+1); sp.resize(l_max_+1);
    cp[0] = 1.0; sp[0] = 0.0;
    for (int m = 0; m < l_max_; ++m) {
      cp[m+1] = cp[m]*cos_phi - sp[m]*sin_phi;
      sp[m+1] = sp[m]*cos_phi + cp[m]*sin_phi;
    }

    // --- assemble output ---
    for (int i = 0; i < N; ++i) out[i] = 0.0;
    for (int l = 0; l <= l_max_; ++l) {
      const int base = l*l + l;
      out[base] = norm_[l*(l+1)/2] * plm[l*(l+1)/2];
      for (int m = 1; m <= l; ++m) {
        const double common = std::sqrt(2.0) * norm_[l*(l+1)/2+m] * plm[l*(l+1)/2+m];
        out[base + m] = common * cp[m];
        out[base - m] = common * sp[m];
      }
    }
  }

  int lmax() const { return l_max_; }
  int size()  const { return (l_max_+1)*(l_max_+1); }

 private:
  int l_max_;
  std::vector<double> norm_;
};

} // namespace SOAP
