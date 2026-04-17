#pragma once
#include "AtomData.h"
#include "Statistics.h"
#include <array>
#include <string>
#include <vector>

struct VacancyPoint {
    std::array<double,3> pos;
    double d_near; // distance to nearest atom [Å]
};

struct VacancyCluster {
    std::array<double,3> center;  // grid point with maximum d_near
    double               d_near_max;
    int                  n_pts;
};

class DefectClassifier {
public:
    DefectClassifier() = default;

    // Compute reference mean DV and fit chi-distribution parameters from pristine frame.
    // Must be called before classify().
    void buildReference(const Frame& ref_frame);

    // Classify all atoms in frame:
    //   - compute dist_to_ref = ||q̃^i - q̄||
    //   - compute defect_prob = 1 - chiProbNorm(d, k, sigma)
    //   - label as Lattice (d < threshold) or secondary type
    // Secondary classification: if ref DVs provided, nearest reference DV wins;
    // otherwise label Unknown.
    void classify(Frame& frame, double threshold) const;

    // Grid-based vacancy detection. Returns VacancyPoints where d_near > vac_dist.
    std::vector<VacancyPoint> findVacanciesGrid(const Frame& frame,
                                                 double grid_spacing,
                                                 double vac_dist) const;

    // Greedy clustering of vacancy grid points with PBC support.
    // r_cluster: merge radius. Uses minimum-image PBC.
    std::vector<VacancyCluster> clusterVacancyPoints(
        const std::vector<VacancyPoint>& pts,
        double r_cluster,
        const SimBox& box) const;

    // Reference DVs for secondary classification (set before calling classify())
    std::vector<double> ref_sia;   // SIA reference DV
    std::vector<double> ref_antv;  // ANtV reference DV
    std::vector<double> ref_typea; // TypeA reference DV

    // Fitted chi-distribution parameters (available after buildReference)
    double chi_k     = 7.0;
    double chi_sigma = 0.1;

    // Reference mean DV
    std::vector<double> mean_dv;

private:
    double euclidDist(const std::vector<double>& a, const std::vector<double>& b) const;
};
