#include "LammpsDumpReader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace {

bool parseOneFrame(std::istream& in, Frame& frame) {
    std::string line;
    // ITEM: TIMESTEP
    while (std::getline(in, line)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) break;
    }
    if (!in) return false;
    if (!std::getline(in, line)) return false;
    frame.timestep = std::stoi(line);

    // ITEM: NUMBER OF ATOMS
    if (!std::getline(in, line)) return false; // ITEM: NUMBER OF ATOMS
    if (!std::getline(in, line)) return false;
    const int n_atoms = std::stoi(line);

    // ITEM: BOX BOUNDS
    if (!std::getline(in, line)) return false;
    // parse periodicity flags from "ITEM: BOX BOUNDS pp pp pp"
    {
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> toks;
        while (ss >> tok) toks.push_back(tok);
        // toks: ITEM: BOX BOUNDS [flags...]
        // after "BOUNDS" each token is x_flag y_flag z_flag
        int flag_start = 0;
        for (int k = 0; k < (int)toks.size(); ++k)
            if (toks[k] == "BOUNDS") { flag_start = k+1; break; }
        for (int d = 0; d < 3; ++d) {
            if (flag_start + d < (int)toks.size())
                frame.box.periodic[d] = (toks[flag_start + d].find('p') != std::string::npos);
            else
                frame.box.periodic[d] = true;
        }
    }
    // Read xlo xhi, ylo yhi, zlo zhi
    auto readBounds = [&](std::array<double,2>& b) {
        if (!std::getline(in, line)) return;
        std::istringstream ss(line);
        ss >> b[0] >> b[1];
    };
    readBounds(frame.box.xb);
    readBounds(frame.box.yb);
    readBounds(frame.box.zb);

    // ITEM: ATOMS header
    if (!std::getline(in, line)) return false;
    // Parse column names
    std::istringstream hss(line);
    std::string tok;
    std::vector<std::string> cols;
    while (hss >> tok) cols.push_back(tok);
    // Remove "ITEM:" and "ATOMS"
    cols.erase(std::remove(cols.begin(), cols.end(), "ITEM:"), cols.end());
    cols.erase(std::remove(cols.begin(), cols.end(), "ATOMS"), cols.end());

    auto colIdx = [&](const std::string& name) -> int {
        for (int k = 0; k < (int)cols.size(); ++k)
            if (cols[k] == name) return k;
        return -1;
    };
    const int ci_id   = colIdx("id");
    const int ci_type = colIdx("type");
    const int ci_x    = colIdx("x");
    const int ci_y    = colIdx("y");
    const int ci_z    = colIdx("z");

    if (ci_x < 0 || ci_y < 0 || ci_z < 0)
        throw std::runtime_error("LAMMPS dump missing x/y/z columns");

    frame.atoms.resize(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        if (!std::getline(in, line)) break;
        std::istringstream ss(line);
        std::vector<std::string> vals;
        while (ss >> tok) vals.push_back(tok);
        auto get = [&](int ci) -> double {
            if (ci < 0 || ci >= (int)vals.size()) return 0.0;
            return std::stod(vals[ci]);
        };
        Atom& a = frame.atoms[i];
        a.id   = (ci_id   >= 0) ? std::stoi(vals[ci_id])   : i+1;
        a.type = (ci_type >= 0) ? std::stoi(vals[ci_type]) : 1;
        a.x = get(ci_x);
        a.y = get(ci_y);
        a.z = get(ci_z);
    }
    return true;
}

} // anonymous namespace

Frame LammpsDumpReader::readFirstFrame(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open " + filename);
    Frame frame;
    if (!parseOneFrame(f, frame))
        throw std::runtime_error("No frame found in " + filename);
    return frame;
}

std::vector<Frame> LammpsDumpReader::readAllFrames(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open " + filename);
    std::vector<Frame> frames;
    Frame frame;
    while (parseOneFrame(f, frame)) {
        frames.push_back(std::move(frame));
        frame = Frame{};
    }
    return frames;
}
