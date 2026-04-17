#include "AtomData.h"
#include "SOAPDescriptor.h"
#include "DefectClassifier.h"
#include "Statistics.h"
#include "PCA.h"
#include "LammpsDumpReader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
    int    n_max       = 9;
    int    l_max       = 9;
    double r_cut       = 5.0;
    double sigma       = -1.0;
    double threshold   = 0.15;
    double vac_dist    = -1.0;   // -1 = 0.4*r_cut
    double grid_spacing = 0.5;
    double vac_cluster_radius = -1.0; // -1 = vac_dist
    std::string ref_dump;
    std::string dam_dump;
    std::string ref_sia_file;
    std::string ref_antv_file;
    std::string ref_typea_file;
    int    save_dv_id  = -1;
    std::string save_dv_file;
    std::string output = "output.csv";
    bool   do_pca      = false;
    int    pca_n       = 2;
    bool   do_hist     = false;
    int    hist_bins   = 50;
};

static void printHelp() {
    std::cout <<
R"(Usage: distool [options] <reference.dump> <damaged.dump>

Options:
  --n-max N            Radial basis functions (default: 9)
  --l-max L            Max angular momentum (default: 9)
  --r-cut R            Cutoff radius in Angstrom (default: 5.0)
  --sigma S            Gaussian width (default: r_cut/(n_max+1))
  --threshold T        Distance threshold for defect classification (default: 0.15)
  --vac-dist D         Vacancy detection distance in Angstrom (default: 0.4*r_cut)
  --grid-spacing G     Vacancy grid spacing in Angstrom (default: 0.5)
  --vac-cluster-radius R  Cluster merge radius (default: vac-dist)
  --ref-sia FILE       Reference DV file for SIA atoms
  --ref-antv FILE      Reference DV file for ANtV atoms
  --ref-typea FILE     Reference DV file for TypeA atoms
  --save-dv ID FILE    Save DV of atom ID (damaged frame) to FILE
  --output FILE        Main output CSV (default: output.csv)
  --pca [N]            Run PCA with N components (default: 2)
  --hist [B]           Write distance histogram with B bins (default: 50)
  --help               Print this help
)";
}

static Config parseArgs(int argc, char** argv) {
    Config cfg;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextStr = [&]() -> std::string {
            if (i+1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        auto nextInt = [&]() { return std::stoi(nextStr()); };
        auto nextDbl = [&]() { return std::stod(nextStr()); };

        if (arg == "--help" || arg == "-h") { printHelp(); exit(0); }
        else if (arg == "--n-max")   cfg.n_max   = nextInt();
        else if (arg == "--l-max")   cfg.l_max   = nextInt();
        else if (arg == "--r-cut")   cfg.r_cut   = nextDbl();
        else if (arg == "--sigma")   cfg.sigma   = nextDbl();
        else if (arg == "--threshold") cfg.threshold = nextDbl();
        else if (arg == "--vac-dist")  cfg.vac_dist  = nextDbl();
        else if (arg == "--grid-spacing") cfg.grid_spacing = nextDbl();
        else if (arg == "--vac-cluster-radius") cfg.vac_cluster_radius = nextDbl();
        else if (arg == "--ref-sia")   cfg.ref_sia_file   = nextStr();
        else if (arg == "--ref-antv")  cfg.ref_antv_file  = nextStr();
        else if (arg == "--ref-typea") cfg.ref_typea_file = nextStr();
        else if (arg == "--output")    cfg.output         = nextStr();
        else if (arg == "--save-dv") {
            cfg.save_dv_id   = nextInt();
            cfg.save_dv_file = nextStr();
        }
        else if (arg == "--pca") {
            cfg.do_pca = true;
            // optional numeric argument
            if (i+1 < argc && argv[i+1][0] != '-') {
                try { cfg.pca_n = std::stoi(argv[++i]); } catch (...) { --i; }
            }
        }
        else if (arg == "--hist") {
            cfg.do_hist = true;
            if (i+1 < argc && argv[i+1][0] != '-') {
                try { cfg.hist_bins = std::stoi(argv[++i]); } catch (...) { --i; }
            }
        }
        else if (arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        }
        else positional.push_back(arg);
    }
    if (positional.size() < 2) throw std::runtime_error("Need <reference.dump> <damaged.dump>");
    cfg.ref_dump = positional[0];
    cfg.dam_dump = positional[1];
    if (cfg.vac_dist < 0)            cfg.vac_dist = 0.4 * cfg.r_cut;
    if (cfg.vac_cluster_radius < 0)  cfg.vac_cluster_radius = cfg.vac_dist;
    return cfg;
}

static void writeAtomCSV(const std::string& path, const Frame& frame) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "id,type,x,y,z,dist_to_ref,defect_prob,defect_type\n";
    for (const auto& a : frame.atoms) {
        f << a.id << ',' << a.type << ','
          << a.x  << ',' << a.y   << ',' << a.z << ','
          << a.dist_to_ref << ',' << a.defect_prob << ','
          << defectTypeName(a.defect_type) << '\n';
    }
}

static void writeVacancyCSV(const std::string& path,
                             const std::vector<VacancyCluster>& clusters) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "x,y,z,d_near_max,n_grid_pts\n";
    for (const auto& vc : clusters)
        f << vc.center[0] << ',' << vc.center[1] << ',' << vc.center[2] << ','
          << vc.d_near_max << ',' << vc.n_pts << '\n';
}

static void writeHistCSV(const std::string& path, const Frame& frame, int nbins) {
    std::vector<double> dists;
    for (const auto& a : frame.atoms) dists.push_back(a.dist_to_ref);
    double dmax = *std::max_element(dists.begin(), dists.end());
    double dmin = *std::min_element(dists.begin(), dists.end());
    if (dmax == dmin) dmax = dmin + 1e-6;
    const double bw = (dmax - dmin) / nbins;
    std::vector<int> counts(nbins, 0);
    for (double d : dists) {
        int b = std::min((int)((d - dmin)/bw), nbins-1);
        ++counts[b];
    }
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "bin_centre,count\n";
    for (int b = 0; b < nbins; ++b)
        f << (dmin + (b+0.5)*bw) << ',' << counts[b] << '\n';
}

static void writePCACSV(const std::string& path, const Frame& frame,
                         const std::vector<std::vector<double>>& proj, int n_comp) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "id,type";
    for (int c = 1; c <= n_comp; ++c) f << ",pc" << c;
    f << ",dist_to_ref,defect_type\n";
    for (size_t i = 0; i < frame.atoms.size(); ++i) {
        const auto& a = frame.atoms[i];
        f << a.id << ',' << a.type;
        for (int c = 0; c < n_comp; ++c)
            f << ',' << (i < proj.size() ? proj[i][c] : 0.0);
        f << ',' << a.dist_to_ref << ',' << defectTypeName(a.defect_type) << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        Config cfg = parseArgs(argc, argv);

        SOAPDescriptor soap(cfg.n_max, cfg.l_max, cfg.r_cut, cfg.sigma);
        std::cout << "SOAP descriptor size: " << soap.descriptorSize()
                  << " (n_max=" << cfg.n_max << " l_max=" << cfg.l_max << ")\n";

        // --- Reference frame ---
        std::cout << "Reading reference: " << cfg.ref_dump << "\n";
        Frame ref_frame = LammpsDumpReader::readFirstFrame(cfg.ref_dump);
        std::cout << "  " << ref_frame.atoms.size() << " atoms\n";
        std::cout << "Computing reference SOAP descriptors...\n";
        soap.computeAll(ref_frame);

        DefectClassifier clf;
        if (!cfg.ref_sia_file.empty())
            clf.ref_sia   = SOAPDescriptor::loadDV(cfg.ref_sia_file);
        if (!cfg.ref_antv_file.empty())
            clf.ref_antv  = SOAPDescriptor::loadDV(cfg.ref_antv_file);
        if (!cfg.ref_typea_file.empty())
            clf.ref_typea = SOAPDescriptor::loadDV(cfg.ref_typea_file);

        clf.buildReference(ref_frame);
        std::cout << "Chi-distribution fit: k=" << clf.chi_k
                  << " sigma=" << clf.chi_sigma << "\n";

        // --- Damaged frame ---
        std::cout << "Reading damaged: " << cfg.dam_dump << "\n";
        Frame dam_frame = LammpsDumpReader::readFirstFrame(cfg.dam_dump);
        std::cout << "  " << dam_frame.atoms.size() << " atoms\n";
        std::cout << "Computing damaged SOAP descriptors...\n";
        soap.computeAll(dam_frame);

        // Optional: save DV of specific atom
        if (cfg.save_dv_id >= 0) {
            soap.saveDV(dam_frame, cfg.save_dv_id, cfg.save_dv_file);
            std::cout << "Saved DV of atom " << cfg.save_dv_id
                      << " to " << cfg.save_dv_file << "\n";
        }

        // --- Classify ---
        std::cout << "Classifying atoms (threshold=" << cfg.threshold << ")...\n";
        clf.classify(dam_frame, cfg.threshold);

        // Count defects
        int n_lattice = 0, n_defect = 0;
        for (const auto& a : dam_frame.atoms)
            (a.defect_type == DefectType::Lattice ? n_lattice : n_defect)++;
        std::cout << "  Lattice: " << n_lattice << "  Defect: " << n_defect << "\n";

        // --- Vacancy detection ---
        std::cout << "Detecting vacancies (grid=" << cfg.grid_spacing
                  << " Å, vac_dist=" << cfg.vac_dist << " Å)...\n";
        auto vac_pts = clf.findVacanciesGrid(dam_frame, cfg.grid_spacing, cfg.vac_dist);
        auto clusters = clf.clusterVacancyPoints(vac_pts, cfg.vac_cluster_radius, dam_frame.box);
        std::cout << "  Vacancy grid points: " << vac_pts.size()
                  << "  Clusters: " << clusters.size() << "\n";

        // --- Write outputs ---
        writeAtomCSV(cfg.output, dam_frame);
        std::cout << "Written: " << cfg.output << "\n";

        const std::string base = cfg.output.substr(0, cfg.output.rfind('.'));
        const std::string vac_path  = base + "_vacancies.csv";
        const std::string hist_path = base + "_hist.csv";
        const std::string pca_path  = base + "_pca.csv";

        writeVacancyCSV(vac_path, clusters);
        std::cout << "Written: " << vac_path << "\n";

        if (cfg.do_hist) {
            writeHistCSV(hist_path, dam_frame, cfg.hist_bins);
            std::cout << "Written: " << hist_path << "\n";
        }

        if (cfg.do_pca) {
            std::cout << "Running PCA (n_components=" << cfg.pca_n << ")...\n";
            PCA pca(cfg.pca_n);
            std::vector<std::vector<double>> ref_dvs;
            for (const auto& a : ref_frame.atoms)
                if (!a.dv.empty()) ref_dvs.push_back(a.dv);
            pca.fit(ref_dvs);
            std::vector<std::vector<double>> dam_dvs;
            for (const auto& a : dam_frame.atoms)
                if (!a.dv.empty()) dam_dvs.push_back(a.dv);
            auto proj = pca.transform(dam_dvs);
            writePCACSV(pca_path, dam_frame, proj, cfg.pca_n);
            std::cout << "Written: " << pca_path << "\n";
        }

        std::cout << "\n=== Summary ===\n"
                  << "  Total atoms : " << dam_frame.atoms.size() << "\n"
                  << "  Lattice     : " << n_lattice << "\n"
                  << "  Defects     : " << n_defect  << "\n"
                  << "  Vacancies   : " << clusters.size() << "\n"
                  << "  chi k       : " << clf.chi_k << "\n"
                  << "  chi sigma   : " << clf.chi_sigma << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
