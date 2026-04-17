#pragma once
#include <vector>

class RadialBasis {
public:
    // sigma=-1 means use r_cut/(n_max+1)
    RadialBasis(int n_max, double r_cut, double sigma = -1.0);

    // Fill out[0..n_max-1] = phi_n(r) = f_cut(r) * exp(-(r-r_n)^2/(2*sigma^2))
    // f_cut(r) = 0.5*(1+cos(pi*r/r_cut)) for r<r_cut else 0
    void computeInto(double r, double* out) const;

    int    nmax()  const { return n_max_; }
    double rcut()  const { return r_cut_; }
    double sigma() const { return sigma_; }

private:
    int    n_max_;
    double r_cut_;
    double sigma_;
    double inv2sig2_;           // 1/(2*sigma^2)
    std::vector<double> r_n_;  // centre positions
};
