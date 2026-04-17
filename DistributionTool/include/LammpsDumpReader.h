#pragma once
#include "AtomData.h"
#include <string>
#include <vector>

class LammpsDumpReader {
public:
    // Read the first (or only) frame from a LAMMPS dump file.
    // Supports columns: id type x y z (and ignores others).
    static Frame readFirstFrame(const std::string& filename);

    // Read all frames (for multi-timestep dumps).
    static std::vector<Frame> readAllFrames(const std::string& filename);
};
