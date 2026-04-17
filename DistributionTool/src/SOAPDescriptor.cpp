#include "SOAPDescriptor.h"
#include "CellList.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

SOAPDescriptor::SOAPDescriptor(int n_max, int l_max, double r_cut, double sigma)
    : n_max_(n_max), l_max_(l_max)
    , n_pairs_(n_max * (n_max + 1) / 2)
    , rb_(n_max, r_cut, sigma)
    , sh_(l_max)
{}

void SOAPDescriptor::computeAll(Frame& frame) const {
    const int sh_sz = sh_.size();
    const int dv_sz = descriptorSize();
    const double pi = M_PI;

    // Pre-allocate all DV storage
    for (auto& a : frame.atoms) a.dv.assign(dv_sz, 0.0);

    CellList cl(frame, rb_.rcut());

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<double> c(n_max_ * sh_sz, 0.0);
        std::vector<double> rb_buf(n_max_);
        std::vector<double> sh_buf(sh_sz);

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 64)
        #endif
        for (int i = 0; i < static_cast<int>(frame.atoms.size()); ++i) {
            std::fill(c.begin(), c.end(), 0.0);

            for (const auto& nb : cl.neighbours(i)) {
                rb_.computeInto(nb.r, rb_buf.data());
                sh_.computeInto(nb.dx, nb.dy, nb.dz, sh_buf.data());
                for (int n = 0; n < n_max_; ++n) {
                    const double rb_n = rb_buf[n];
                    if (rb_n == 0.0) continue;
                    double* cp = c.data() + n * sh_sz;
                    for (int s = 0; s < sh_sz; ++s)
                        cp[s] += rb_n * sh_buf[s];
                }
            }

            // Power spectrum
            auto& dv = frame.atoms[i].dv;
            for (int l = 0; l <= l_max_; ++l) {
                const double factor = pi * std::sqrt(8.0 / (2.0*l + 1.0));
                const int l_base = l * l + l;  // index of m=0 in sh
                // sum over m = -l..l
                for (int n = 0; n < n_max_; ++n) {
                    const double* cn = c.data() + n * sh_sz;
                    for (int np = n; np < n_max_; ++np) {
                        const double* cnp = c.data() + np * sh_sz;
                        double pnnl = 0.0;
                        for (int m = -l; m <= l; ++m)
                            pnnl += cn[l_base + m] * cnp[l_base + m];
                        pnnl *= factor;
                        const int pair_idx = n*n_max_ - n*(n-1)/2 + (np - n);
                        dv[l * n_pairs_ + pair_idx] = pnnl;
                    }
                }
            }

            // Normalise
            double norm = 0.0;
            for (double v : dv) norm += v * v;
            if (norm > 0.0) {
                norm = 1.0 / std::sqrt(norm);
                for (double& v : dv) v *= norm;
            }
        }
    }
}

void SOAPDescriptor::saveDV(const Frame& frame, int atom_id,
                             const std::string& filename) const {
    for (const auto& a : frame.atoms) {
        if (a.id == atom_id) {
            std::ofstream f(filename);
            if (!f) throw std::runtime_error("Cannot open " + filename);
            for (double v : a.dv) f << v << "\n";
            return;
        }
    }
    throw std::runtime_error("Atom id " + std::to_string(atom_id) + " not found");
}

std::vector<double> SOAPDescriptor::loadDV(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open " + filename);
    std::vector<double> dv;
    double v;
    while (f >> v) dv.push_back(v);
    return dv;
}
