#pragma once
#include <array>
#include <string>
#include <vector>

enum class DefectType { Lattice, Interstitial, VacancyAdj, TypeA, Unknown };

inline std::string defectTypeName(DefectType t) {
    switch (t) {
        case DefectType::Lattice:      return "Lattice";
        case DefectType::Interstitial: return "Interstitial";
        case DefectType::VacancyAdj:   return "VacancyAdj";
        case DefectType::TypeA:        return "TypeA";
        default:                       return "Unknown";
    }
}

struct Atom {
    int    id = 0, type = 0;
    double x = 0, y = 0, z = 0;
    std::vector<double> dv;
    double dist_to_ref = 0.0;
    double defect_prob = 0.0;
    DefectType defect_type = DefectType::Unknown;
};

struct SimBox {
    std::array<double,2> xb{}, yb{}, zb{};
    bool periodic[3] = {true, true, true};
    double lx() const { return xb[1] - xb[0]; }
    double ly() const { return yb[1] - yb[0]; }
    double lz() const { return zb[1] - zb[0]; }
};

struct Frame {
    int timestep = 0;
    SimBox box;
    std::vector<Atom> atoms;
};
