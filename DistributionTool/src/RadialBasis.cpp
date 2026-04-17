#include "RadialBasis.h"
#include <cmath>

RadialBasis::RadialBasis(int n_max, double r_cut, double sigma)
    : n_max_(n_max), r_cut_(r_cut)
{
    if (sigma <= 0.0)
        sigma_ = r_cut_ / (n_max_ + 1.0);
    else
        sigma_ = sigma;
    inv2sig2_ = 1.0 / (2.0 * sigma_ * sigma_);
    r_n_.resize(n_max_);
    for (int n = 0; n < n_max_; ++n)
        r_n_[n] = (n + 1.0) * r_cut_ / (n_max_ + 1.0);
}

void RadialBasis::computeInto(double r, double* out) const {
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
