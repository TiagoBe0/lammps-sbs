/* soap_radial_basis.h  –  Gaussian radial basis functions for SOAP (header-only)
 *
 * Centres placed equidistantly in (0, r_cut]:
 *   r_n = (n+1)*r_cut/(n_max+1),  n = 0..n_max-1
 *
 * Basis function:
 *   phi_n(r) = f_cut(r) * exp(-(r-r_n)^2 / (2*sigma^2))
 *   f_cut(r) = 0.5*(1 + cos(pi*r/r_cut))  for r < r_cut,  else 0
 *
 * If sigma <= 0 it defaults to r_cut/(n_max+1).
 */
#pragma once
#include <cmath>
#include <vector>

namespace SOAP {

class RadialBasis {
 public:
  RadialBasis(int n_max, double r_cut, double sigma = -1.0)
      : n_max_(n_max), r_cut_(r_cut)
  {
    sigma_    = (sigma <= 0.0) ? r_cut_ / (n_max_ + 1.0) : sigma;
    inv2sig2_ = 1.0 / (2.0 * sigma_ * sigma_);
    r_n_.resize(n_max_);
    for (int n = 0; n < n_max_; ++n)
      r_n_[n] = (n + 1.0) * r_cut_ / (n_max_ + 1.0);
  }

  // Fill out[0..n_max-1] with phi_n(r).
  void computeInto(double r, double *out) const {
    if (r >= r_cut_) {
      for (int n = 0; n < n_max_; ++n) out[n] = 0.0;
      return;
    }
    const double fcut = 0.5 * (1.0 + std::cos(M_PI * r / r_cut_));
    for (int n = 0; n < n_max_; ++n) {
      const double dr = r - r_n_[n];
      out[n] = fcut * std::exp(-dr * dr * inv2sig2_);
    }
  }

  int    nmax()  const { return n_max_; }
  double rcut()  const { return r_cut_; }
  double sigma() const { return sigma_; }

 private:
  int    n_max_;
  double r_cut_, sigma_, inv2sig2_;
  std::vector<double> r_n_;
};

} // namespace SOAP
