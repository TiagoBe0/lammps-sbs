#pragma once
#include <vector>
#include <utility>

namespace Statistics {
    // Arithmetic mean of a set of descriptor vectors
    std::vector<double> meanDV(const std::vector<std::vector<double>>& dvs);

    // Euclidean distance between two equal-length vectors
    double euclideanDist(const std::vector<double>& a, const std::vector<double>& b);

    // Fit chi-distribution parameters k and sigma from distances d_i
    // using method of moments: k = 2*E[d^2]^2/Var(d^2), sigma=sqrt(E[d^2]/k)
    std::pair<double,double> fitChiParams(const std::vector<double>& distances);

    // Normalised chi probability at d (normalised to 1 at mode d_peak)
    // Returns value in [0,1]; defect_prob = 1 - chiProbNorm(d, k, sigma)
    double chiProbNorm(double d, double k, double sigma);
}
