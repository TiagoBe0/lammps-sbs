#pragma once
#include <vector>

class SphericalHarmonics {
public:
    explicit SphericalHarmonics(int l_max);

    // Fill out[(l_max+1)^2] with real tesseral Y_lm(dx,dy,dz).
    // Index of (l,m): l*l + l + m  (m in [-l,l]).
    void computeInto(double dx, double dy, double dz, double* out) const;

    int  lmax() const { return l_max_; }
    // Total number of Y_lm values
    static int size(int l_max) { return (l_max + 1) * (l_max + 1); }
    int size() const { return size(l_max_); }

private:
    int l_max_;
    // Normalisation factors norm_[l*(l+1)/2 + m] for 0<=m<=l
    std::vector<double> norm_;
};
