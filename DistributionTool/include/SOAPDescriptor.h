#pragma once
#include "AtomData.h"
#include "RadialBasis.h"
#include "SphericalHarmonics.h"
#include <vector>

class SOAPDescriptor {
public:
    // sigma=-1: auto (r_cut/(n_max+1))
    SOAPDescriptor(int n_max, int l_max, double r_cut, double sigma = -1.0);

    // Compute and store normalised SOAP descriptor in atom.dv for every atom.
    // Uses OpenMP if compiled with -fopenmp.
    void computeAll(Frame& frame) const;

    // Save the DV of a single atom (by atom id) to a file.
    void saveDV(const Frame& frame, int atom_id, const std::string& filename) const;

    // Load a DV from a file (for use as secondary reference).
    static std::vector<double> loadDV(const std::string& filename);

    int descriptorSize() const { return n_pairs_ * (l_max_ + 1); }
    int nmax() const { return n_max_; }
    int lmax() const { return l_max_; }
    double rcut() const { return rb_.rcut(); }

private:
    int    n_max_, l_max_, n_pairs_;
    RadialBasis        rb_;
    SphericalHarmonics sh_;
};
