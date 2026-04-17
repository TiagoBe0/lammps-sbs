#include "Statistics.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace Statistics {

std::vector<double> meanDV(const std::vector<std::vector<double>>& dvs) {
    if (dvs.empty()) return {};
    const size_t D = dvs[0].size();
    std::vector<double> mean(D, 0.0);
    for (const auto& dv : dvs)
        for (size_t i = 0; i < D; ++i) mean[i] += dv[i];
    const double inv_n = 1.0 / dvs.size();
    for (auto& v : mean) v *= inv_n;
    return mean;
}

double euclideanDist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s);
}

std::pair<double,double> fitChiParams(const std::vector<double>& distances) {
    if (distances.size() < 2)
        return {7.0, 0.1};
    // compute E[d^2] and E[d^4]
    double ed2 = 0.0, ed4 = 0.0;
    for (double d : distances) {
        double d2 = d * d;
        ed2 += d2;
        ed4 += d2 * d2;
    }
    const double n = static_cast<double>(distances.size());
    ed2 /= n;
    ed4 /= n;
    const double var_d2 = ed4 - ed2 * ed2;
    if (var_d2 <= 0.0) return {7.0, std::sqrt(ed2 / 7.0)};
    const double k     = 2.0 * ed2 * ed2 / var_d2;
    const double sigma = std::sqrt(ed2 / k);
    return {k, sigma};
}

double chiProbNorm(double d, double k, double sigma) {
    if (k <= 2.0 || sigma <= 0.0) return 1.0;
    const double d_peak = sigma * std::sqrt(k - 2.0);
    if (d_peak <= 0.0) return 1.0;
    const double ratio = d / d_peak;
    const double exp_term = -(d*d - d_peak*d_peak) / (2.0 * sigma * sigma);
    return std::pow(ratio, k - 2.0) * std::exp(exp_term);
}

} // namespace Statistics
