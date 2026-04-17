#pragma once
#include "AtomData.h"
#include <array>
#include <cmath>
#include <vector>

// Linked-cell spatial index for O(N) neighbour search.
// Cell side >= r_cut, so a 27-cell search suffices.
class CellList {
public:
    struct Neighbour {
        int    idx;          // index into the atoms_ array
        double dx, dy, dz;  // displacement (i -> j), minimum-image PBC applied
        double r;
    };

    // Build from atom positions in frame using given cutoff.
    CellList(const Frame& frame, double r_cut) : box_(frame.box), r_cut_(r_cut) {
        atoms_ = &frame.atoms;
        const double lx = box_.lx(), ly = box_.ly(), lz = box_.lz();
        nc_[0] = std::max(1, static_cast<int>(lx / r_cut_));
        nc_[1] = std::max(1, static_cast<int>(ly / r_cut_));
        nc_[2] = std::max(1, static_cast<int>(lz / r_cut_));
        cs_[0] = lx / nc_[0];
        cs_[1] = ly / nc_[1];
        cs_[2] = lz / nc_[2];

        const int total_cells = nc_[0] * nc_[1] * nc_[2];
        head_.assign(total_cells, -1);
        next_.resize(atoms_->size(), -1);
        cell_of_.resize(atoms_->size());

        for (int i = 0; i < static_cast<int>(atoms_->size()); ++i) {
            int c = cellOf((*atoms_)[i].x, (*atoms_)[i].y, (*atoms_)[i].z);
            cell_of_[i] = c;
            next_[i] = head_[c];
            head_[c] = i;
        }
    }

    // All neighbours of atom i within r_cut (excluding i itself).
    std::vector<Neighbour> neighbours(int i) const {
        std::vector<Neighbour> result;
        const auto& ai = (*atoms_)[i];
        const int ci = cell_of_[i];
        const int ix = ci % nc_[0];
        const int iy = (ci / nc_[0]) % nc_[1];
        const int iz = ci / (nc_[0] * nc_[1]);
        const double r2cut = r_cut_ * r_cut_;

        for (int diz = -1; diz <= 1; ++diz)
        for (int diy = -1; diy <= 1; ++diy)
        for (int dix = -1; dix <= 1; ++dix) {
            int nx = ix + dix, ny = iy + diy, nz = iz + diz;
            // PBC wrapping for cell index
            double ox = 0, oy = 0, oz = 0; // image offsets
            if (nx < 0)      { nx += nc_[0]; ox = -box_.lx(); }
            else if (nx >= nc_[0]) { nx -= nc_[0]; ox =  box_.lx(); }
            if (ny < 0)      { ny += nc_[1]; oy = -box_.ly(); }
            else if (ny >= nc_[1]) { ny -= nc_[1]; oy =  box_.ly(); }
            if (nz < 0)      { nz += nc_[2]; oz = -box_.lz(); }
            else if (nz >= nc_[2]) { nz -= nc_[2]; oz =  box_.lz(); }

            // apply ox/oy/oz only if periodic in that direction
            if (!box_.periodic[0]) ox = 0;
            if (!box_.periodic[1]) oy = 0;
            if (!box_.periodic[2]) oz = 0;

            const int ncell = nz * nc_[1] * nc_[0] + ny * nc_[0] + nx;
            for (int j = head_[ncell]; j >= 0; j = next_[j]) {
                if (j == i) continue;
                const auto& aj = (*atoms_)[j];
                double dx = aj.x + ox - ai.x;
                double dy = aj.y + oy - ai.y;
                double dz = aj.z + oz - ai.z;
                // Apply minimum image (needed for non-boundary cells too)
                if (box_.periodic[0]) dx -= box_.lx() * std::round(dx / box_.lx());
                if (box_.periodic[1]) dy -= box_.ly() * std::round(dy / box_.ly());
                if (box_.periodic[2]) dz -= box_.lz() * std::round(dz / box_.lz());
                const double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < r2cut)
                    result.push_back({j, dx, dy, dz, std::sqrt(r2)});
            }
        }
        return result;
    }

    // Minimum squared distance from point (px,py,pz) to any atom.
    double nearestDist2FromPoint(double px, double py, double pz) const {
        const int ix = std::min(static_cast<int>((px - box_.xb[0]) / cs_[0]), nc_[0]-1);
        const int iy = std::min(static_cast<int>((py - box_.yb[0]) / cs_[1]), nc_[1]-1);
        const int iz = std::min(static_cast<int>((pz - box_.zb[0]) / cs_[2]), nc_[2]-1);
        double best = 1e300;

        for (int diz = -1; diz <= 1; ++diz)
        for (int diy = -1; diy <= 1; ++diy)
        for (int dix = -1; dix <= 1; ++dix) {
            int nx = ix + dix, ny = iy + diy, nz = iz + diz;
            double ox = 0, oy = 0, oz = 0;
            if (nx < 0)           { nx += nc_[0]; ox = -box_.lx(); }
            else if (nx >= nc_[0]){ nx -= nc_[0]; ox =  box_.lx(); }
            if (ny < 0)           { ny += nc_[1]; oy = -box_.ly(); }
            else if (ny >= nc_[1]){ ny -= nc_[1]; oy =  box_.ly(); }
            if (nz < 0)           { nz += nc_[2]; oz = -box_.lz(); }
            else if (nz >= nc_[2]){ nz -= nc_[2]; oz =  box_.lz(); }
            if (!box_.periodic[0]) ox = 0;
            if (!box_.periodic[1]) oy = 0;
            if (!box_.periodic[2]) oz = 0;

            const int ncell = nz*nc_[1]*nc_[0] + ny*nc_[0] + nx;
            for (int j = head_[ncell]; j >= 0; j = next_[j]) {
                const auto& aj = (*atoms_)[j];
                double dx = aj.x + ox - px;
                double dy = aj.y + oy - py;
                double dz = aj.z + oz - pz;
                if (box_.periodic[0]) dx -= box_.lx() * std::round(dx / box_.lx());
                if (box_.periodic[1]) dy -= box_.ly() * std::round(dy / box_.ly());
                if (box_.periodic[2]) dz -= box_.lz() * std::round(dz / box_.lz());
                best = std::min(best, dx*dx + dy*dy + dz*dz);
            }
        }
        return best;
    }

private:
    int cellOf(double x, double y, double z) const {
        int ix = std::min(static_cast<int>((x - box_.xb[0]) / cs_[0]), nc_[0]-1);
        int iy = std::min(static_cast<int>((y - box_.yb[0]) / cs_[1]), nc_[1]-1);
        int iz = std::min(static_cast<int>((z - box_.zb[0]) / cs_[2]), nc_[2]-1);
        if (ix < 0) ix = 0; if (iy < 0) iy = 0; if (iz < 0) iz = 0;
        return iz * nc_[1] * nc_[0] + iy * nc_[0] + ix;
    }

    const SimBox               box_;
    double                     r_cut_;
    const std::vector<Atom>*   atoms_;
    std::array<int,3>          nc_;
    std::array<double,3>       cs_;
    std::vector<int>           head_;
    std::vector<int>           next_;
    std::vector<int>           cell_of_;
};
