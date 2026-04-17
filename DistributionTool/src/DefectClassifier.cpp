#include "DefectClassifier.h"
#include "CellList.h"
#include "Statistics.h"
#include <algorithm>
#include <cmath>
#include <numeric>

void DefectClassifier::buildReference(const Frame& ref_frame) {
    std::vector<std::vector<double>> dvs;
    dvs.reserve(ref_frame.atoms.size());
    for (const auto& a : ref_frame.atoms)
        if (!a.dv.empty()) dvs.push_back(a.dv);

    mean_dv = Statistics::meanDV(dvs);

    std::vector<double> distances;
    distances.reserve(dvs.size());
    for (const auto& dv : dvs)
        distances.push_back(Statistics::euclideanDist(dv, mean_dv));

    auto [k, sigma] = Statistics::fitChiParams(distances);
    chi_k     = k;
    chi_sigma = sigma;
}

void DefectClassifier::classify(Frame& frame, double threshold) const {
    const bool have_sia   = !ref_sia.empty();
    const bool have_antv  = !ref_antv.empty();
    const bool have_typea = !ref_typea.empty();
    const bool have_any   = have_sia || have_antv || have_typea;

    for (auto& a : frame.atoms) {
        if (a.dv.empty()) continue;
        const double d = Statistics::euclideanDist(a.dv, mean_dv);
        a.dist_to_ref = d;
        a.defect_prob = 1.0 - Statistics::chiProbNorm(d, chi_k, chi_sigma);

        if (d < threshold) {
            a.defect_type = DefectType::Lattice;
        } else if (!have_any) {
            a.defect_type = DefectType::Unknown;
        } else {
            // Secondary: nearest reference DV
            double best = 1e300;
            DefectType best_type = DefectType::Unknown;
            auto check = [&](const std::vector<double>& ref, DefectType type) {
                if (ref.empty()) return;
                double dist = euclidDist(a.dv, ref);
                if (dist < best) { best = dist; best_type = type; }
            };
            check(ref_sia,   DefectType::Interstitial);
            check(ref_antv,  DefectType::VacancyAdj);
            check(ref_typea, DefectType::TypeA);
            a.defect_type = best_type;
        }
    }
}

std::vector<VacancyPoint> DefectClassifier::findVacanciesGrid(
        const Frame& frame, double grid_spacing, double vac_dist) const {

    CellList cl(frame, vac_dist * 1.5 + grid_spacing);
    const auto& box = frame.box;
    const double lx = box.lx(), ly = box.ly(), lz = box.lz();
    const int nx = static_cast<int>(std::ceil(lx / grid_spacing));
    const int ny = static_cast<int>(std::ceil(ly / grid_spacing));
    const int nz = static_cast<int>(std::ceil(lz / grid_spacing));

    const double vd2 = vac_dist * vac_dist;
    std::vector<VacancyPoint> result;

    for (int iz = 0; iz < nz; ++iz)
    for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
        double px = box.xb[0] + (ix + 0.5) * grid_spacing;
        double py = box.yb[0] + (iy + 0.5) * grid_spacing;
        double pz = box.zb[0] + (iz + 0.5) * grid_spacing;
        double d2 = cl.nearestDist2FromPoint(px, py, pz);
        if (d2 > vd2) {
            VacancyPoint vp;
            vp.pos   = {px, py, pz};
            vp.d_near = std::sqrt(d2);
            result.push_back(vp);
        }
    }
    return result;
}

std::vector<VacancyCluster> DefectClassifier::clusterVacancyPoints(
        const std::vector<VacancyPoint>& pts, double r_cluster,
        const SimBox& box) const {

    if (pts.empty()) return {};

    // Sort by d_near descending (deepest void first)
    std::vector<int> order(pts.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pts[a].d_near > pts[b].d_near;
    });

    const double r2 = r_cluster * r_cluster;
    const double lx = box.lx(), ly = box.ly(), lz = box.lz();
    std::vector<bool> assigned(pts.size(), false);
    std::vector<VacancyCluster> clusters;

    auto minImageDist2 = [&](const std::array<double,3>& a,
                              const std::array<double,3>& b) -> double {
        double dx = b[0]-a[0], dy = b[1]-a[1], dz = b[2]-a[2];
        if (box.periodic[0]) dx -= lx * std::round(dx / lx);
        if (box.periodic[1]) dy -= ly * std::round(dy / ly);
        if (box.periodic[2]) dz -= lz * std::round(dz / lz);
        return dx*dx + dy*dy + dz*dz;
    };

    for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
        const int seed = order[oi];
        if (assigned[seed]) continue;
        assigned[seed] = true;

        VacancyCluster cl;
        cl.center    = pts[seed].pos;
        cl.d_near_max = pts[seed].d_near;
        cl.n_pts      = 1;

        for (int oj = oi + 1; oj < static_cast<int>(order.size()); ++oj) {
            const int j = order[oj];
            if (assigned[j]) continue;
            if (minImageDist2(pts[seed].pos, pts[j].pos) <= r2) {
                assigned[j] = true;
                ++cl.n_pts;
                // center stays at the seed (max d_near) point
            }
        }
        clusters.push_back(cl);
    }
    return clusters;
}

double DefectClassifier::euclidDist(const std::vector<double>& a,
                                     const std::vector<double>& b) const {
    return Statistics::euclideanDist(a, b);
}
