/* compute_local_freevol.cpp
 *
 * Local free-volume estimator + vacancy cluster identifier for LAMMPS.
 * See compute_local_freevol.h for full documentation.
 *
 * Build: copy .h and .cpp into LAMMPS src/ and recompile.
 */

#include "compute_local_freevol.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace LAMMPS_NS;

static constexpr double MY_4PI3 = 4.0 / 3.0 * M_PI;

/* ---------------------------------------------------------------------- */
ComputeLocalFreevol::ComputeLocalFreevol(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg)
{
  // Minimum: compute ID group local_freevol cutoff
  if (narg < 4)
    error->all(FLERR,
               "Syntax: compute ID group local_freevol cutoff "
               "[vref <v>] [thresh <t>] [vcut <c>]");

  double cutoff = utils::numeric(FLERR, arg[3], false, lmp);
  if (cutoff <= 0.0)
    error->all(FLERR, "compute local_freevol: cutoff must be positive");
  cutsq = cutoff * cutoff;

  // Defaults
  vref      = 0.0;
  thresh    = 0.30;   // 30% excess volume → vacancy candidate
  vcut_sq   = cutsq;  // default clustering cutoff = neighbour cutoff
  auto_vref = 1;

  // Parse optional keywords
  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "vref") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute local_freevol: missing value for 'vref'");
      vref = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (vref < 0.0)
        error->all(FLERR, "compute local_freevol: vref must be >= 0");
      auto_vref = (vref == 0.0) ? 1 : 0;
      iarg += 2;
    } else if (strcmp(arg[iarg], "thresh") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute local_freevol: missing value for 'thresh'");
      thresh = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "vcut") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute local_freevol: missing value for 'vcut'");
      double vc = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (vc <= 0.0)
        error->all(FLERR, "compute local_freevol: vcut must be positive");
      vcut_sq = vc * vc;
      iarg += 2;
    } else {
      error->all(FLERR, "compute local_freevol: unknown keyword");
    }
  }

  peratom_flag      = 1;
  size_peratom_cols = NCOLS;
  nmax              = 0;
}

/* ---------------------------------------------------------------------- */
ComputeLocalFreevol::~ComputeLocalFreevol()
{
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */
void ComputeLocalFreevol::init()
{
  if (!force->pair)
    error->all(FLERR, "compute local_freevol requires a pair style");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */
void ComputeLocalFreevol::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void ComputeLocalFreevol::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  int nlocal = atom->nlocal;

  // Grow output array if needed
  if (nlocal > nmax) {
    memory->destroy(array_atom);
    nmax = nlocal;
    memory->create(array_atom, nmax, NCOLS, "local_freevol:array_atom");
  }

  double **x       = atom->x;
  int    *numneigh = list->numneigh;
  int   **firstneigh = list->firstneigh;

  // ── Pass 1: compute V_local for every local atom ──────────────────────
  double V_sphere = MY_4PI3 * std::pow(std::sqrt(cutsq), 3.0);
  std::vector<double> vloc(nlocal);

  for (int i = 0; i < nlocal; i++) {
    int jnum  = numneigh[i];
    int *jlist = firstneigh[i];
    int nb = 0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      if (dx*dx + dy*dy + dz*dz < cutsq) nb++;
    }

    // Local atomic volume = sphere volume / (self + neighbours)
    vloc[i] = V_sphere / static_cast<double>(nb + 1);
  }

  // ── Pass 2: determine V_ref ────────────────────────────────────────────
  // Auto mode: compute median of V_local over all local atoms.
  // Median is more robust than mean under high defect concentration.
  if (auto_vref) {
    std::vector<double> sorted = vloc;
    std::sort(sorted.begin(), sorted.end());
    // Allreduce median across MPI ranks (use global median via gather)
    // For simplicity, use local median if only one rank; with MPI this
    // approximates the global median (good enough for reference tracking).
    int n = static_cast<int>(sorted.size());
    vref = (n % 2 == 0)
           ? 0.5 * (sorted[n/2 - 1] + sorted[n/2])
           : sorted[n/2];
    // In parallel runs, all ranks should agree on vref to get consistent
    // cluster IDs.  Broadcast rank-0 value.
    MPI_Bcast(&vref, 1, MPI_DOUBLE, 0, world);
  }

  // ── Pass 3: δV and vacancy flag ───────────────────────────────────────
  // Collect local atom indices of vacancy candidates
  std::vector<int> vac_idx;   // local atom indices
  vac_idx.reserve(nlocal / 20);

  for (int i = 0; i < nlocal; i++) {
    double dv = (vref > 1e-14) ? (vloc[i] - vref) / vref : 0.0;
    array_atom[i][0] = vloc[i];
    array_atom[i][1] = dv;
    array_atom[i][2] = (dv > thresh) ? 1.0 : 0.0;
    array_atom[i][3] = 0.0;   // cluster_id, filled below

    if (dv > thresh) vac_idx.push_back(i);
  }

  // ── Pass 4: cluster vacancy candidates ────────────────────────────────
  // Only local atoms are clustered here; ghost-atom connections are
  // handled by a global reduction via MPI (simplified: local clusters only,
  // suitable for most cases where clusters << box size).
  int nv = static_cast<int>(vac_idx.size());
  _uf_init(nv);

  for (int a = 0; a < nv; a++) {
    int ia = vac_idx[a];
    for (int b = a + 1; b < nv; b++) {
      int ib = vac_idx[b];
      double dx = x[ia][0] - x[ib][0];
      double dy = x[ia][1] - x[ib][1];
      double dz = x[ia][2] - x[ib][2];
      if (dx*dx + dy*dy + dz*dz < vcut_sq)
        _uf_union(a, b);
    }
  }

  // Assign compact cluster IDs (1-based) using canonical roots
  // Also include neighbour-based vacancy candidates from ghost atoms via
  // the full neighbour list: if a vacancy candidate has a ghost-atom
  // vacancy neighbour, they should be in the same cluster.  This is the
  // main limitation of local-only UF; a full parallel implementation
  // would require additional communication (left for future work).
  std::unordered_map<int,int> root_to_id;
  int next_id = 1;
  for (int a = 0; a < nv; a++) {
    int root = _uf_find(a);
    if (root_to_id.find(root) == root_to_id.end())
      root_to_id[root] = next_id++;
    array_atom[vac_idx[a]][3] = static_cast<double>(root_to_id[root]);
  }
}

/* ---------------------------------------------------------------------- */
/* Union-Find with path compression and union by rank                     */
/* ---------------------------------------------------------------------- */
void ComputeLocalFreevol::_uf_init(int n)
{
  _uf_parent.resize(n);
  _uf_rank.resize(n, 0);
  std::iota(_uf_parent.begin(), _uf_parent.end(), 0);
}

int ComputeLocalFreevol::_uf_find(int x)
{
  while (_uf_parent[x] != x) {
    _uf_parent[x] = _uf_parent[_uf_parent[x]];  // path halving
    x = _uf_parent[x];
  }
  return x;
}

void ComputeLocalFreevol::_uf_union(int a, int b)
{
  int ra = _uf_find(a);
  int rb = _uf_find(b);
  if (ra == rb) return;
  if (_uf_rank[ra] < _uf_rank[rb]) std::swap(ra, rb);
  _uf_parent[rb] = ra;
  if (_uf_rank[ra] == _uf_rank[rb]) _uf_rank[ra]++;
}

/* ---------------------------------------------------------------------- */
double ComputeLocalFreevol::memory_usage()
{
  return static_cast<double>(nmax) * NCOLS * sizeof(double);
}
