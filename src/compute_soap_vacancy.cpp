/* compute_soap_vacancy.cpp
 *
 * Build: copy compute_soap_vacancy.h/.cpp into LAMMPS src/ and recompile.
 * No external dependencies (does NOT use the LAMMPS neighbor list).
 *
 * References:
 *   von Toussaint et al., FaVaD arXiv:2004.08184 (2020) — §2.3.2
 *   Domínguez-Gutiérrez & von Toussaint, NME 22, 100724 (2019)
 */

#include "compute_soap_vacancy.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include "utils.h"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <vector>

using namespace LAMMPS_NS;

/* ======================================================================
   Constructor / destructor
   ====================================================================== */

ComputeSOAPVacancy::ComputeSOAPVacancy(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg),
      vac_dist_(1.2), grid_spacing_(0.5), cluster_radius_(-1.0),
      has_file_(false)
{
  // Minimum: compute ID group soap_vacancy vac_dist
  if (narg < 4)
    error->all(FLERR,
               "Syntax: compute ID group soap_vacancy vac_dist "
               "[grid G] [cluster_radius R] [file FILE]");

  vac_dist_ = utils::numeric(FLERR, arg[3], false, lmp);
  if (vac_dist_ <= 0.0)
    error->all(FLERR, "compute soap_vacancy: vac_dist must be > 0");

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "grid") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute soap_vacancy: missing value after 'grid'");
      grid_spacing_ = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (grid_spacing_ <= 0.0)
        error->all(FLERR, "compute soap_vacancy: grid spacing must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "cluster_radius") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute soap_vacancy: missing value after 'cluster_radius'");
      cluster_radius_ = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (cluster_radius_ <= 0.0)
        error->all(FLERR, "compute soap_vacancy: cluster_radius must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "file") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute soap_vacancy: missing value after 'file'");
      out_file_  = std::string(arg[iarg+1]);
      has_file_  = true;
      iarg += 2;
    } else {
      error->all(FLERR, "compute soap_vacancy: unknown keyword");
    }
  }

  // Default cluster radius = vac_dist
  if (cluster_radius_ < 0.0) cluster_radius_ = vac_dist_;

  // Write CSV header (rank 0 only, truncate if exists)
  if (has_file_ && comm->me == 0) {
    std::ofstream f(out_file_, std::ios::trunc);
    if (!f)
      error->one(FLERR,
                 ("compute soap_vacancy: cannot open file: " + out_file_).c_str());
    f << "# timestep x y z d_near_max n_grid_pts\n";
  }

  // Global vector output: [n_vac_pts, void_volume, n_clusters]
  vector_flag      = 1;
  size_vector      = 3;
  extvector        = 0;  // do not normalise by natoms
  vector           = new double[3]{0.0, 0.0, 0.0};
}

/* ---------------------------------------------------------------------- */
ComputeSOAPVacancy::~ComputeSOAPVacancy()
{
  delete[] vector;
  vector = nullptr;
}

/* ---------------------------------------------------------------------- */
void ComputeSOAPVacancy::init()
{
  // No neighbor list required — we query atom->x directly.
  // Ghost atoms must be communicated before compute_vector() is called;
  // this is guaranteed by LAMMPS when the compute is invoked during a run.
}

/* ======================================================================
   Main entry point
   ====================================================================== */

void ComputeSOAPVacancy::compute_vector()
{
  invoked_vector = update->ntimestep;

  // 1. Build cell list from all atoms (local + ghost)
  VacCellList cl;
  buildCellList(cl);

  // 2. Scan local subdomain; collect vacant grid points
  std::vector<VacPt> local_pts = scanGrid(cl);

  // 3. MPI_Allgatherv: every rank gets the full list of vacant points
  const int local_n = static_cast<int>(local_pts.size());

  std::vector<int> counts(comm->nprocs);
  MPI_Allgather(&local_n, 1, MPI_INT,
                counts.data(), 1, MPI_INT, world);

  std::vector<int> displs(comm->nprocs, 0);
  for (int r = 1; r < comm->nprocs; r++)
    displs[r] = displs[r-1] + counts[r-1];
  const int total_n = displs[comm->nprocs-1] + counts[comm->nprocs-1];

  // Pack local VacPts into a flat double array (4 doubles each)
  std::vector<double> local_flat(local_n * 4);
  for (int i = 0; i < local_n; i++) {
    local_flat[4*i+0] = local_pts[i].x;
    local_flat[4*i+1] = local_pts[i].y;
    local_flat[4*i+2] = local_pts[i].z;
    local_flat[4*i+3] = local_pts[i].d_near;
  }

  // Scale counts and displs for the 4-double encoding
  std::vector<int> counts4(comm->nprocs), displs4(comm->nprocs);
  for (int r = 0; r < comm->nprocs; r++) {
    counts4[r] = counts[r] * 4;
    displs4[r] = displs[r] * 4;
  }

  std::vector<double> all_flat(total_n * 4);
  MPI_Allgatherv(local_flat.data(), local_n * 4, MPI_DOUBLE,
                 all_flat.data(), counts4.data(), displs4.data(),
                 MPI_DOUBLE, world);

  // Unpack
  std::vector<VacPt> all_pts(total_n);
  for (int i = 0; i < total_n; i++) {
    all_pts[i].x      = all_flat[4*i+0];
    all_pts[i].y      = all_flat[4*i+1];
    all_pts[i].z      = all_flat[4*i+2];
    all_pts[i].d_near = all_flat[4*i+3];
  }

  // 4. Cluster (identical result on all ranks)
  std::vector<Cluster> clusters;
  clusterPoints(all_pts, clusters);

  // 5. Populate output vector
  const double g3 = grid_spacing_ * grid_spacing_ * grid_spacing_;
  vector[0] = static_cast<double>(total_n);
  vector[1] = static_cast<double>(total_n) * g3;
  vector[2] = static_cast<double>(clusters.size());

  // 6. Write cluster file (rank 0 only, append one row per cluster)
  if (has_file_ && comm->me == 0)
    appendClusterFile(clusters);
}

/* ======================================================================
   buildCellList
   ====================================================================== */

void ComputeSOAPVacancy::buildCellList(VacCellList &cl) const
{
  double **x      = atom->x;
  const int natoms = atom->nlocal + atom->nghost;

  if (natoms == 0) return;

  // Bounding box of all atoms (local + ghost, which may be outside [boxlo,boxhi])
  double xmin = x[0][0], ymin = x[0][1], zmin = x[0][2];
  double xmax = x[0][0], ymax = x[0][1], zmax = x[0][2];
  for (int i = 1; i < natoms; i++) {
    xmin = std::min(xmin, x[i][0]); xmax = std::max(xmax, x[i][0]);
    ymin = std::min(ymin, x[i][1]); ymax = std::max(ymax, x[i][1]);
    zmin = std::min(zmin, x[i][2]); zmax = std::max(zmax, x[i][2]);
  }
  // Small margin to avoid boundary edge cases
  const double margin = 0.5;
  xmin -= margin; ymin -= margin; zmin -= margin;
  xmax += margin; ymax += margin; zmax += margin;

  // Cell size >= vac_dist_ so that a 3×3×3 search always suffices
  const double cs = std::max(vac_dist_, grid_spacing_);
  cl.build(x, natoms, xmin, ymin, zmin, xmax, ymax, zmax, cs);
}

/* ======================================================================
   scanGrid  –  find vacant grid points in this rank's subdomain
   ====================================================================== */

std::vector<ComputeSOAPVacancy::VacPt>
ComputeSOAPVacancy::scanGrid(const VacCellList &cl) const
{
  const double vd2  = vac_dist_ * vac_dist_;
  const double g    = grid_spacing_;

  // Global box
  const double xlo = domain->boxlo[0];
  const double ylo = domain->boxlo[1];
  const double zlo = domain->boxlo[2];
  const double xhi = domain->boxhi[0];
  const double yhi = domain->boxhi[1];
  const double zhi = domain->boxhi[2];

  // This rank's subdomain (half-open: [sublo, subhi))
  const double sxlo = domain->sublo[0], sxhi = domain->subhi[0];
  const double sylo = domain->sublo[1], syhi = domain->subhi[1];
  const double szlo = domain->sublo[2], szhi = domain->subhi[2];

  const int nx = static_cast<int>(std::ceil((xhi - xlo) / g));
  const int ny = static_cast<int>(std::ceil((yhi - ylo) / g));
  const int nz = static_cast<int>(std::ceil((zhi - zlo) / g));

  std::vector<VacPt> pts;

  for (int iz = 0; iz < nz; iz++) {
    const double pz = zlo + (iz + 0.5) * g;
    if (pz < szlo || pz >= szhi) continue;

    for (int iy = 0; iy < ny; iy++) {
      const double py = ylo + (iy + 0.5) * g;
      if (py < sylo || py >= syhi) continue;

      for (int ix = 0; ix < nx; ix++) {
        const double px = xlo + (ix + 0.5) * g;
        if (px < sxlo || px >= sxhi) continue;

        const double d2 = cl.nearestDist2(px, py, pz);
        if (d2 > vd2)
          pts.push_back({px, py, pz, std::sqrt(d2)});
      }
    }
  }
  return pts;
}

/* ======================================================================
   clusterPoints  –  greedy O(M²) clustering with minimum-image PBC
   ====================================================================== */

void ComputeSOAPVacancy::clusterPoints(
    const std::vector<VacPt> &pts,
    std::vector<Cluster> &clusters) const
{
  clusters.clear();
  if (pts.empty()) return;

  const double lx   = domain->xprd;
  const double ly   = domain->yprd;
  const double lz   = domain->zprd;
  const bool   pbcx = domain->periodicity[0];
  const bool   pbcy = domain->periodicity[1];
  const bool   pbcz = domain->periodicity[2];
  const double r2   = cluster_radius_ * cluster_radius_;

  // Sort by d_near descending (deepest void first → best cluster centres)
  std::vector<int> order(pts.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return pts[a].d_near > pts[b].d_near;
  });

  std::vector<bool> assigned(pts.size(), false);

  auto minImageDist2 = [&](const VacPt &a, const VacPt &b) -> double {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    double dz = b.z - a.z;
    if (pbcx) dx -= lx * std::round(dx / lx);
    if (pbcy) dy -= ly * std::round(dy / ly);
    if (pbcz) dz -= lz * std::round(dz / lz);
    return dx*dx + dy*dy + dz*dz;
  };

  for (int oi = 0; oi < static_cast<int>(order.size()); oi++) {
    const int seed = order[oi];
    if (assigned[seed]) continue;
    assigned[seed] = true;

    Cluster cl;
    cl.x          = pts[seed].x;
    cl.y          = pts[seed].y;
    cl.z          = pts[seed].z;
    cl.d_near_max = pts[seed].d_near;
    cl.n_pts      = 1;

    for (int oj = oi + 1; oj < static_cast<int>(order.size()); oj++) {
      const int j = order[oj];
      if (assigned[j]) continue;
      if (minImageDist2(pts[seed], pts[j]) <= r2) {
        assigned[j] = true;
        cl.n_pts++;
        // centre stays at seed (max d_near point)
      }
    }
    clusters.push_back(cl);
  }
}

/* ======================================================================
   appendClusterFile  –  write one row per cluster (called on rank 0 only)
   ====================================================================== */

void ComputeSOAPVacancy::appendClusterFile(
    const std::vector<Cluster> &clusters) const
{
  std::ofstream f(out_file_, std::ios::app);
  if (!f)
    error->one(FLERR,
               ("compute soap_vacancy: cannot append to file: " + out_file_).c_str());

  f << std::fixed << std::setprecision(6);
  const bigint ts = update->ntimestep;
  for (const auto &cl : clusters)
    f << ts    << "  "
      << cl.x  << "  " << cl.y << "  " << cl.z << "  "
      << cl.d_near_max << "  " << cl.n_pts << "\n";
}

/* ======================================================================
   VacCellList implementation
   ====================================================================== */

void ComputeSOAPVacancy::VacCellList::build(
    double **x, int natoms,
    double xlo_, double ylo_, double zlo_,
    double xhi_, double yhi_, double zhi_,
    double cell_size)
{
  xlo = xlo_; ylo = ylo_; zlo = zlo_;
  inv_cell = 1.0 / cell_size;

  nx = std::max(1, static_cast<int>((xhi_ - xlo_) * inv_cell) + 1);
  ny = std::max(1, static_cast<int>((yhi_ - ylo_) * inv_cell) + 1);
  nz = std::max(1, static_cast<int>((zhi_ - zlo_) * inv_cell) + 1);

  head.assign(nx * ny * nz, -1);
  next.assign(natoms, -1);
  ax.resize(natoms);
  ay.resize(natoms);
  az.resize(natoms);

  for (int i = 0; i < natoms; i++) {
    ax[i] = x[i][0];
    ay[i] = x[i][1];
    az[i] = x[i][2];

    const int cx = static_cast<int>((x[i][0] - xlo) * inv_cell);
    const int cy = static_cast<int>((x[i][1] - ylo) * inv_cell);
    const int cz = static_cast<int>((x[i][2] - zlo) * inv_cell);

    // Skip atoms outside the cell list bounds (should not happen with margin)
    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz)
      continue;

    const int cid = (cz * ny + cy) * nx + cx;
    next[i] = head[cid];
    head[cid] = i;
  }
}

/* ---------------------------------------------------------------------- */
double ComputeSOAPVacancy::VacCellList::nearestDist2(
    double px, double py, double pz) const
{
  const int cx0 = std::min(std::max(static_cast<int>((px - xlo) * inv_cell), 0), nx-1);
  const int cy0 = std::min(std::max(static_cast<int>((py - ylo) * inv_cell), 0), ny-1);
  const int cz0 = std::min(std::max(static_cast<int>((pz - zlo) * inv_cell), 0), nz-1);

  double min_d2 = 1e300;

  for (int dz = -1; dz <= 1; dz++)
  for (int dy = -1; dy <= 1; dy++)
  for (int dx = -1; dx <= 1; dx++) {
    const int cx = cx0 + dx;
    const int cy = cy0 + dy;
    const int cz = cz0 + dz;
    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;

    const int cid = (cz * ny + cy) * nx + cx;
    for (int i = head[cid]; i != -1; i = next[i]) {
      const double ddx = ax[i] - px;
      const double ddy = ay[i] - py;
      const double ddz = az[i] - pz;
      const double d2  = ddx*ddx + ddy*ddy + ddz*ddz;
      if (d2 < min_d2) min_d2 = d2;
    }
  }
  return min_d2;
}

/* ---------------------------------------------------------------------- */
double ComputeSOAPVacancy::memory_usage()
{
  // VacCellList is rebuilt per-call (temporary stack allocation).
  // Report the estimated per-call peak: 5 arrays × (nlocal + nghost) doubles.
  const int n = atom->nlocal + atom->nghost;
  return static_cast<double>(n) * 5 * sizeof(double);
}
