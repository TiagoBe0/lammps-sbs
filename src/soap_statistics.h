/* soap_statistics.h  –  Statistics helpers for SOAP defect analysis (header-only)
 *
 * Provides:
 *   euclideanDist  – distance between two descriptor vectors
 *   fitChiParams   – fit chi-distribution k and sigma from distances
 *   chiProbNorm    – normalised chi probability (1 at the mode)
 *
 * Chi-distribution model (FaVaD, Eq. 6):
 *   P(d|k,sigma) ∝ d^(k-2) * exp(-d^2/(2*sigma^2))
 *   Mode: d_peak = sigma*sqrt(k-2)
 *   Method of moments: k = 2*E[d^2]^2 / Var(d^2),  sigma = sqrt(E[d^2]/k)
 */
#pragma once
#include <cmath>
#include <utility>
#include <vector>

namespace SOAP {

// Euclidean distance between two equal-length arrays.
inline double euclideanDist(const double *a, const double *b, int D)
{
  double s = 0.0;
  for (int i = 0; i < D; ++i) { double d = a[i] - b[i]; s += d*d; }
  return std::sqrt(s);
}

// Fit chi-distribution parameters from a vector of distances.
// Returns {k, sigma}; falls back to {7.0, 0.1} for degenerate input.
inline std::pair<double,double> fitChiParams(const std::vector<double> &distances)
{
  if (distances.size() < 2) return {7.0, 0.1};
  double ed2 = 0.0, ed4 = 0.0;
  for (double d : distances) {
    const double d2 = d * d;
    ed2 += d2;
    ed4 += d2 * d2;
  }
  const double n = static_cast<double>(distances.size());
  ed2 /= n; ed4 /= n;
  const double var_d2 = ed4 - ed2 * ed2;
  if (var_d2 <= 0.0) return {7.0, std::sqrt(ed2 / 7.0)};
  const double k     = 2.0 * ed2 * ed2 / var_d2;
  const double sigma = std::sqrt(ed2 / k);
  return {k, sigma};
}

// Normalised chi probability at d (== 1 at mode d_peak = sigma*sqrt(k-2)).
// Returns value in [0,1].  defect_prob = 1 - chiProbNorm(d, k, sigma).
inline double chiProbNorm(double d, double k, double sigma)
{
  if (k <= 2.0 || sigma <= 0.0) return 1.0;
  const double d_peak = sigma * std::sqrt(k - 2.0);
  if (d_peak <= 0.0) return 1.0;
  const double ratio    = d / d_peak;
  const double exp_term = -(d*d - d_peak*d_peak) / (2.0 * sigma * sigma);
  return std::pow(ratio, k - 2.0) * std::exp(exp_term);
}

} // namespace SOAP
