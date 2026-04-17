#include "SphericalHarmonics.h"
#include <cmath>

SphericalHarmonics::SphericalHarmonics(int l_max) : l_max_(l_max) {
    int sz = (l_max + 1) * (l_max + 2) / 2;
    norm_.resize(sz);
    for (int l = 0; l <= l_max; ++l) {
        // factorial ratio: (l-m)! / (l+m)!
        // build incrementally from m=0
        double fact_ratio = 1.0; // (l-0)!/(l+0)! = 1
        norm_[l*(l+1)/2 + 0] = std::sqrt((2.0*l+1.0) / (4.0*M_PI) * fact_ratio);
        for (int m = 1; m <= l; ++m) {
            // (l-m)!/(l+m)! = (l-(m-1))!/(l+(m-1))! * 1/((l+m)*(l-m+1))
            fact_ratio /= static_cast<double>((l + m) * (l - m + 1));
            norm_[l*(l+1)/2 + m] = std::sqrt((2.0*l+1.0) / (4.0*M_PI) * fact_ratio);
        }
    }
}

void SphericalHarmonics::computeInto(double dx, double dy, double dz, double* out) const {
    const int N = (l_max_+1)*(l_max_+1);
    const double r2 = dx*dx + dy*dy + dz*dz;
    const double r  = std::sqrt(r2);

    if (r < 1e-12) {
        for (int i = 0; i < N; ++i) out[i] = 0.0;
        out[0] = norm_[0]; // Y_0^0 = N_{0,0}
        return;
    }

    const double cos_theta = dz / r;
    const double rxy = std::sqrt(dx*dx + dy*dy);
    double cos_phi, sin_phi;
    if (rxy < 1e-12) { cos_phi = 1.0; sin_phi = 0.0; }
    else              { cos_phi = dx / rxy; sin_phi = dy / rxy; }

    const double sin_theta = rxy / r;

    // Compute unnormalized associated Legendre P_l^m for 0<=m<=l<=l_max
    // stored in plm[l*(l+1)/2 + m]
    const int plm_sz = (l_max_+1)*(l_max_+2)/2;
    // thread_local avoids heap allocation on every call (hot path)
    thread_local std::vector<double> plm;
    plm.assign(plm_sz, 0.0);

    plm[0] = 1.0; // P_0^0

    for (int m = 0; m < l_max_; ++m) {
        // Diagonal: P_{m+1}^{m+1} = -(2m+1)*sin_theta*P_m^m
        plm[(m+1)*(m+2)/2 + (m+1)] = -(2.0*m + 1.0) * sin_theta * plm[m*(m+1)/2 + m];
        // Superdiagonal: P_{m+1}^m = cos_theta*(2m+1)*P_m^m
        plm[(m+1)*(m+2)/2 + m] = cos_theta * (2.0*m + 1.0) * plm[m*(m+1)/2 + m];
    }

    // General recurrence for l >= m+2
    for (int m = 0; m <= l_max_; ++m) {
        for (int l = m + 2; l <= l_max_; ++l) {
            plm[l*(l+1)/2 + m] = (cos_theta * (2.0*l - 1.0) * plm[(l-1)*l/2 + m]
                                  - (l + m - 1.0) * plm[(l-2)*(l-1)/2 + m]) / (l - m);
        }
    }

    // Build trig recurrences: cos(m*phi), sin(m*phi)
    thread_local std::vector<double> cp, sp;
    cp.resize(l_max_+1); sp.resize(l_max_+1);
    cp[0] = 1.0; sp[0] = 0.0;
    for (int m = 0; m < l_max_; ++m) {
        cp[m+1] = cp[m]*cos_phi - sp[m]*sin_phi;
        sp[m+1] = sp[m]*cos_phi + cp[m]*sin_phi;
    }

    // Fill output array out[l^2+l+m]
    for (int i = 0; i < N; ++i) out[i] = 0.0;

    for (int l = 0; l <= l_max_; ++l) {
        const int base = l*l + l;
        // m = 0
        out[base] = norm_[l*(l+1)/2] * plm[l*(l+1)/2];
        // m > 0
        for (int m = 1; m <= l; ++m) {
            const double n_lm   = norm_[l*(l+1)/2 + m];
            const double p_val  = plm[l*(l+1)/2 + m];
            const double common = std::sqrt(2.0) * n_lm * p_val;
            out[base + m]  = common * cp[m];   // Y_l^m
            out[base - m]  = common * sp[m];   // Y_l^{-m}
        }
    }
}
