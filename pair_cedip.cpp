/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Sam McSweeney (Curtin University)
   email: sammy.mcsweeney@gmail.com

   Environment Dependent Interatomic Potential for Carbon
   References:
    1) N. A. Marks, Phys. Rev. B 63, 035401 (2000)

   Ported to modern LAMMPS (>= 29Aug2024) API.
   Changes vs. original 2014 version:
     - C-style <math.h> etc. replaced by <cmath>, <cstdio>, <cstdlib>, <cstring>
     - neigh_request.h removed (API merged into neighbor.h in modern LAMMPS)
     - neighbor->request(this) replaced by neighbor->add_request(this, NeighConst::REQ_FULL|NeighConst::REQ_GHOST)
     - bigint loop variable 'n' renamed to 'bn' to avoid shadowing class member 'n'
     - double **f and double **x declarations de-duplicated inside functions
       (kept only one declaration per function scope)
------------------------------------------------------------------------- */

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_cedip.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

double dot (vec, vec);
vec cross (vec, vec);
double norm (vec);
double norm2 (vec);

/* ---------------------------------------------------------------------- */

PairCEDIP::PairCEDIP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  one_coeff = 1;
  ghostneigh = 1;

  maxlocal = 0;
  num = nullptr;
  near = nullptr;
  ipage = nullptr;
  pgsize = oneatom = 0;

  manybody_flag = 1;

  z    = nullptr;
  Z    = nullptr;
  dz   = nullptr;
  p    = nullptr;
  dp   = nullptr;
  pi   = nullptr;
  pi2  = nullptr;
  pi3  = nullptr;
  dpi  = nullptr;
  dpi2 = nullptr;
  dpi3 = nullptr;

  Dzdx = nullptr;
  Dzdxx = nullptr;
  Dzdxx_data = nullptr;

  finiteforce = nullptr;
  finiteforce2 = nullptr;

  r = nullptr;
  N = nullptr;
  N_data = nullptr;

  zbl  = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairCEDIP::~PairCEDIP()
{
  memory->destroy(num);
  memory->sfree(near);
  delete [] ipage;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
    delete [] map;
  }

  deallocate_arrays();
}

/* ---------------------------------------------------------------------- */

void PairCEDIP::allocate_arrays()
{
  int allnum = list->inum + list->gnum;

  deallocate_arrays();

  memory->create(z, allnum, "cedip:z");
  memory->create(Z, allnum, "cedip:Z");
  memory->create(dz, allnum, maxn, "cedip:dz");
  memory->create(p,  allnum, maxn, "cedip:p");
  memory->create(dp, allnum, maxn, "cedip:dp");
  memory->create(pi,   allnum, "cedip:pi");
  memory->create(pi2,  allnum, "cedip:pi2");
  memory->create(pi3,  allnum, "cedip:pi3");
  memory->create(dpi,  allnum, "cedip:dpi");
  memory->create(dpi2, allnum, "cedip:dpi2");
  memory->create(dpi3, allnum, "cedip:dpi3");

  memory->create(finiteforce, maxn, "cedip:finiteforce");
  memory->create(finiteforce2, maxn, maxn, "cedip:finiteforce2");

  Dzdx       = (vec *)  memory->smalloc(maxn*sizeof(vec), "cedip:Dzdx");
  Dzdxx      = (vec **) memory->smalloc(maxn*sizeof(vec*), "cedip:Dzdxx");
  Dzdxx_data = (vec *)  memory->smalloc(maxn*maxn*sizeof(vec), "cedip:Dzdxx_data");

  memory->create(r, allnum, oneatom, "cedip:r");
  N      = (vec **) memory->smalloc(allnum*sizeof(vec*), "cedip:N");
  N_data = (vec *)  memory->smalloc(allnum*maxn*sizeof(vec), "cedip:N_data");

  // Set up Dzdxx's and N's 2D array structure
  // NOTE: renamed loop var to 'bn' (bigint-style counter) to avoid
  //       shadowing class member int 'n'
  bigint bn = 0;
  for (int i = 0; i < maxn; i++) {
    Dzdxx[i] = &Dzdxx_data[bn];
    bn += maxn;
  }

  bn = 0;
  for (int i = 0; i < allnum; i++) {
    N[i] = &N_data[bn];
    bn += maxn;
  }
}

void PairCEDIP::deallocate_arrays()
{
  memory->destroy(z);
  memory->destroy(Z);
  memory->destroy(dz);
  memory->destroy(p);
  memory->destroy(dp);
  memory->destroy(pi);
  memory->destroy(pi2);
  memory->destroy(pi3);
  memory->destroy(dpi);
  memory->destroy(dpi2);
  memory->destroy(dpi3);

  memory->destroy(finiteforce);
  memory->destroy(finiteforce2);

  memory->sfree(Dzdx);
  memory->sfree(Dzdxx);
  memory->sfree(Dzdxx_data);

  memory->destroy(r);
  memory->sfree(N);
  memory->sfree(N_data);
}

/* ----------------------------------------------------------------------
   create CEDIP neighbour list from main neighbour list
   CEDIP neighbour list stores neighbours of ghost atoms
   Patterned after REBO neighbour list
---------------------------------------------------------------------- */

void PairCEDIP::CEDIP_neigh()
{
  int i,j,ii,jj;
  int n_local,allnum,jnum,itype,jtype;
  vec Ri,Rj,Rij;
  double rijsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;

  if (atom->nmax > maxlocal) {
    maxlocal = atom->nmax;
    memory->destroy(num);
    memory->sfree(near);
    memory->create(num,maxlocal,"CEDIP:num");
    near = (int **) memory->smalloc(maxlocal*sizeof(int *), "CEDIP:near");
  }

  allnum = list->inum + list->gnum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  ipage->reset();

  maxn = 0;
  for (ii = 0; ii < allnum; ii++) {
    i = ilist[ii];

    n_local = 0;
    neighptr = ipage->vget();
    itype = map[type[i]];

    Ri.set(x[i][0], x[i][1], x[i][2]);

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = map[type[j]];

      Rj.set(x[j][0], x[j][1], x[j][2]);
      Rij = Rj - Ri;
      rijsq = norm2(Rij);

      if (rijsq < cutsq[itype][jtype]) {
          neighptr[n_local++] = j;
      }
    }

    near[i] = neighptr;
    num[i] = n_local;
    if (n_local > maxn)  maxn = n_local;
    ipage->vgot(n_local);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
}

void PairCEDIP::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  int *type       = atom->type;
  int nlocal      = atom->nlocal;

  int inum, allnum, *ilist;
  inum   = list->inum;
  allnum = list->inum + list->gnum;
  ilist  = list->ilist;

  CEDIP_neigh();
  allocate_arrays();
  calc_dists();

  for (ii = 0; ii < allnum; ii++) {
    i = ilist[ii];
    cutoff_fun(i);
    switch_fun(i);
  }

  u = 0.0;

  for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      F.set_zero();
      coordination(i);
      if (zbl == 0)  pair(i, evflag);
      else           pair_zbl(i, evflag);
      triple(i, evflag);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------- */

void PairCEDIP::pair(int i, int evflag)
{
  double **f      = atom->f;
  int nlocal      = atom->nlocal;
  int newton_pair = force->newton_pair;

  for (jj = 0; jj < num[i]; jj++) {

      j = near[i][jj];

      if (r[i][jj] >= a + aprime*Z[i] - 0.001)
          continue;

      u2 = calcU2(i, jj);
      u += u2;
      if (evflag) ev_tally(i, j, nlocal, newton_pair, u2, 0.0,
                           0.0, 0.0, 0.0, 0.0);

      for (kk = 0; kk < num[i]; kk++) {

          if ((jj != kk) && !finiteforce[kk])
              continue;

          k  = near[i][kk];

          F = calcF2(i, jj, kk);

          f[i][0] -= F.x;
          f[i][1] -= F.y;
          f[i][2] -= F.z;

          f[k][0] += F.x;
          f[k][1] += F.y;
          f[k][2] += F.z;

          fpair = norm(F);
          Fhat  = F / fpair;

          if (evflag) ev_tally(i, k, nlocal, newton_pair, 0.0, 0.0,
                               fpair, Fhat.x, Fhat.y, Fhat.z);

          for (ll = 0; ll < num[k]; ll++) {

              if (!finiteforce2[kk][ll])
                  continue;

              l = near[k][ll];

              F2 = Dzdxx[kk][ll] * f22;

              f[k][0] -= F2.x;
              f[k][1] -= F2.y;
              f[k][2] -= F2.z;

              f[l][0] += F2.x;
              f[l][1] += F2.y;
              f[l][2] += F2.z;

              fpair = norm(F2);
              Fhat  = F2 / fpair;
              if (evflag) ev_tally(k, l, nlocal, newton_pair, 0.0, 0.0,
                                   fpair, Fhat.x, Fhat.y, Fhat.z);

          }
      }
  }
}


/* -------------------------------- */

void PairCEDIP::pair_zbl(int i, int evflag)
{
  double **f      = atom->f;
  int nlocal      = atom->nlocal;
  int newton_pair = force->newton_pair;

  for (jj = 0; jj < num[i]; jj++) {

      j = near[i][jj];

      if (r[i][jj] >= a + aprime*Z[i] - 0.001)
          continue;

      zbl_phi = dzbl_phi = 0.0;
      for (int idx = 0; idx < 4; idx++) {
          zbl_tmp   = zbl_c[idx] * exp(-zbl_d[idx] * r[i][jj]);
          zbl_phi  += zbl_tmp;
          dzbl_phi -= zbl_tmp * zbl_d[idx];
      }
      zbl_phi  *= zbl_const;
      dzbl_phi *= zbl_const;

      uzbl  = zbl_phi / r[i][jj];
      uedip = calcU2(i, jj);

      zbl_exp  = exp((r[i][jj] - zbl_cut + zbl_shift) / zbl_skin);
      edip_exp = exp((r[i][jj] - zbl_cut - zbl_shift) / zbl_skin);
      zbl_sf   = 1.0/(zbl_exp  + 1.0);
      edip_sf  = 1.0 - 1.0/(edip_exp + 1.0);

      u2 = uedip*edip_sf + uzbl*zbl_sf;
      u += u2;
      if (evflag) ev_tally(i, j, nlocal, newton_pair, u2, 0.0,
                           0.0, 0.0, 0.0, 0.0);

      dzbl_sf  = -zbl_exp / (zbl_skin*(zbl_exp +1.0)*(zbl_exp +1.0));
      dedip_sf = edip_exp / (zbl_skin*(edip_exp+1.0)*(edip_exp+1.0));

      fzbl = (dzbl_phi - uzbl)*zbl_sf/r[i][jj] + dzbl_sf*uzbl;

      F2 = calcF2(i, jj, jj);
      F  = F2*edip_sf + Dxr*(uedip*dedip_sf + fzbl);

      f[i][0] -= F.x;
      f[i][1] -= F.y;
      f[i][2] -= F.z;

      f[j][0] += F.x;
      f[j][1] += F.y;
      f[j][2] += F.z;

      fpair = norm(F);
      Fhat  = F / fpair;

      if (evflag) ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0,
                           fpair, Fhat.x, Fhat.y, Fhat.z);

      for (kk = 0; kk < num[i]; kk++) {

          if (!finiteforce[kk])
              continue;

          k  = near[i][kk];

          if (k != j) {

              F = Dzdx[kk] * (f22*edip_sf);

              f[i][0] -= F.x;
              f[i][1] -= F.y;
              f[i][2] -= F.z;

              f[k][0] += F.x;
              f[k][1] += F.y;
              f[k][2] += F.z;

              fpair = norm(F);
              Fhat  = F / fpair;

              if (evflag) ev_tally(i, k, nlocal, newton_pair, 0.0, 0.0,
                                   fpair, Fhat.x, Fhat.y, Fhat.z);
          }

          for (ll = 0; ll < num[k]; ll++) {

              if (!finiteforce2[kk][ll])
                  continue;

              l = near[k][ll];

              F = Dzdxx[kk][ll] * (f22*edip_sf);

              f[k][0] -= F.x;
              f[k][1] -= F.y;
              f[k][2] -= F.z;

              f[l][0] += F.x;
              f[l][1] += F.y;
              f[l][2] += F.z;

              fpair = norm(F);
              Fhat  = F / fpair;

              if (evflag) ev_tally(k, l, nlocal, newton_pair, 0.0, 0.0,
                                   fpair, Fhat.x, Fhat.y, Fhat.z);
          }
      }
  }
}

/* ------------------------------- */

void PairCEDIP::triple(int i, int evflag)
{
  double **f      = atom->f;
  int nlocal      = atom->nlocal;
  int newton_pair = force->newton_pair;

  tau_fun(i);

  for (jj = 0; jj < num[i]-1; jj++) {

      j = near[i][jj];

      if (r[i][jj] >= a + aprime*Z[i] - 0.001)
          continue;

      for (kk = jj+1; kk < num[i]; kk++) {

          if (r[i][kk] >= a + aprime*Z[i] - 0.001)
              continue;

          u3 = calcU3(i, jj, kk);
          u += u3;
          if (evflag) ev_tally(i, j, nlocal, newton_pair, u3, 0.0,
                               0.0, 0.0, 0.0, 0.0);

          for (ll = 0; ll < num[i]; ll++) {

              if ((jj != ll) && (kk != ll) && !finiteforce[ll])
                  continue;

              l = near[i][ll];

              F = calcF3(i, jj, kk, ll);

              f[i][0] -= F.x;
              f[i][1] -= F.y;
              f[i][2] -= F.z;

              f[l][0] += F.x;
              f[l][1] += F.y;
              f[l][2] += F.z;

              fpair = norm(F);
              Fhat  = F / fpair;

              if (evflag) ev_tally(i, l, nlocal, newton_pair, 0.0, 0.0,
                                   fpair, Fhat.x, Fhat.y, Fhat.z);

              for (mm = 0; mm < num[l]; mm++) {

                  if (!finiteforce2[ll][mm])
                      continue;

                  m = near[l][mm];

                  F3 = Dzdxx[ll][mm] * f33;

                  f[l][0] -= F3.x;
                  f[l][1] -= F3.y;
                  f[l][2] -= F3.z;

                  f[m][0] += F3.x;
                  f[m][1] += F3.y;
                  f[m][2] += F3.z;

                  fpair = norm(F3);
                  Fhat  = F3 / fpair;
                  if (evflag) ev_tally(l, m, nlocal, newton_pair, 0.0, 0.0,
                                       fpair, Fhat.x, Fhat.y, Fhat.z);
              }
          }
      }
  }
}

/* ------------------------------------------------------- */

void PairCEDIP::coordination(int i)
{
    Z[i] = z[i];

    for (jj = 0; jj < num[i]; jj++) {
        j  = near[i][jj];

        if ((r[i][jj] > flow) && (r[i][jj] < fhigh)) finiteforce[jj] = true;
        else                                         finiteforce[jj] = false;

        Dzdx[jj] = N[i][jj] * dz[i][jj];

        for (kk = 0; kk < num[j]; kk++) {
            finiteforce2[jj][kk] = false;
            Dzdxx[jj][kk].set_zero();
        }
    }

    dihedral(i);
    repulsion(i);
    linear(i);
}

/* ------------------------------------------------------- */

void PairCEDIP::dihedral(int i)
{
    if (z[i] <= 4.0) {
    for (jj = 0; jj < num[i]; jj++) {
        j  = near[i][jj];

        if (z[j] >= 4.0 || r[i][jj] >= phigh)
            continue;

        for (kk = 0; kk < num[i]; kk++) {

            if (kk == jj || r[i][kk] >= phigh)
                continue;

            for (ll = kk+1; ll < num[i]; ll++) {

                if (ll == jj || r[i][ll] >= phigh)
                    continue;

                for (mm = 0; mm < num[j]; mm++) {

                    m = near[j][mm];
                    if (m == i || r[j][mm] >= phigh)
                        continue;

                    finiteforce[jj] = true;
                    finiteforce[kk] = true;
                    finiteforce[ll] = true;
                    finiteforce2[jj][mm] = true;

                    Rx[0]  = N[i][kk];
                    Rx[1]  = N[i][ll];
                    Rx[2]  = N[j][mm];

                    rr[0]  = r[i][kk];
                    rr[1]  = r[i][ll];
                    rr[2]  = r[j][mm];

                    pp[0]  = p[i][kk];
                    pp[1]  = p[i][ll];
                    pp[2]  = p[j][mm];
                    pp4    = pp[0] * pp[1] * pp[2] * p[i][jj];

                    dpp[0] = dp[i][kk];
                    dpp[1] = dp[i][ll];
                    dpp[2] = dp[j][mm];

                    gg     = pi3[i] * pi3[j];
                    dgg[0] = dz[i][kk] * dpi3[i] *  pi3[j];
                    dgg[1] = dz[i][ll] * dpi3[i] *  pi3[j];
                    dgg[2] = dz[j][mm] *  pi3[i] * dpi3[j];

                    for (n1 = 0; n1 <= 2; n1++) {
                        n2 = (n1 + 1) % 3;
                        n3 = (n1 + 2) % 3;

                        Xk   = cross(Rx[n2], Rx[n3]);
                        dotp =   dot(Rx[n1], Xk);

                        dz0  = pp[n2] * pp[n3] * p[i][jj];
                        dz1  = dotp*dotp * dz0 * (dgg[n1]*pp[n1] + gg*dpp[n1]);
                        dz2  = gg * pp4 * 2.0/rr[n1] * dotp;

                        Dzr[n1] = Rx[n1]*(dz1 - dotp*dz2) + Xk*dz2;
                    }

                    Z[i] += Zdih * (gg * dotp*dotp * pp4);

                    Dzdx[kk]      += Dzr[0] * Zdih;
                    Dzdx[ll]      += Dzr[1] * Zdih;
                    Dzdxx[jj][mm] += Dzr[2] * Zdih;

                    double bit1 = (pi3[i]*dpi3[j] + dpi3[i]*pi3[j]) * p[i][jj] * dz[i][jj];
                    double bit2 =  pi3[i]* pi3[j] * dp[i][jj];
                    double fac_local = Zdih * dotp*dotp * pp[0]*pp[1]*pp[2] * (bit1 + bit2);

                    Dzdx[jj] += N[i][jj]*fac_local;

                    for (nni = 0; nni < num[i]; nni++) {

                        if (nni == jj || nni == kk || nni == ll || r[i][nni] >= fhigh)
                            continue;

                        finiteforce[nni] = true;

                        Dzdx[nni] += N[i][nni] * (Zdih * dpi3[i] * dz[i][nni] * pi3[j] *
                                                    dotp*dotp * pp4);
                    }

                    for (nnj = 0; nnj < num[j]; nnj++) {

                        n = near[j][nnj];
                        if (n == i || nnj == mm || r[j][nnj] >= fhigh)
                            continue;

                        finiteforce2[jj][nnj] = true;

                        Dzdxx[jj][nnj] += N[j][nnj] * (Zdih * dpi3[j] * dz[j][nnj] *
                                                       pi3[i] * dotp*dotp * pp4);
                    }

                } // mm
            } // ll
        } // kk
    } // jj
    } // if z[i] <= 4.0
}

/* ------------------------------------------------------- */

void PairCEDIP::repulsion(int i)
{
    if (z[i] > 4.0)  return;

    for (jj = 0; jj < num[i]; jj++) {

        j  = near[i][jj];

        if (z[j] >= 4.0 || r[i][jj] <= plow || r[i][jj] >= c0)
            continue;

        for (kk = 0; kk < num[i]; kk++) {

            if (kk == jj || r[i][kk] >= phigh)
                continue;

            for (ll = kk+1; ll < num[i]; ll++) {

                if (ll == jj || r[i][ll] >= phigh)
                    continue;

                finiteforce[jj] = true;
                finiteforce[kk] = true;
                finiteforce[ll] = true;

                Rx[0]  = N[i][jj];
                Rx[1]  = N[i][kk];
                Rx[2]  = N[i][ll];

                rr[0]  = r[i][jj];
                rr[1]  = r[i][kk];
                rr[2]  = r[i][ll];
                cc     = rr[0] - c0;

                pp[0]  = (1.0 - p[i][jj])*cc*cc;
                pp[1]  =        p[i][kk];
                pp[2]  =        p[i][ll];
                pp4    = pp[0] * pp[1] * pp[2];

                dpp[0] = -dp[i][jj]*cc*cc + (1.0-p[i][jj])*2.0*cc;
                dpp[1] =  dp[i][kk];
                dpp[2] =  dp[i][ll];

                gg     = pi3[i] * pi[j];
                dgg[0] = dz[i][jj] * (dpi3[i]*pi[j] + dpi[j]*pi3[i]);
                dgg[1] = dz[i][kk] *  dpi3[i]*pi[j];
                dgg[2] = dz[i][ll] *  dpi3[i]*pi[j];

                for (n1 = 0; n1 <= 2; n1++) {
                    n2 = (n1 + 1) % 3;
                    n3 = (n1 + 2) % 3;

                    Xk   = cross(Rx[n2], Rx[n3]);
                    dotp =   dot(Rx[n1], Xk);

                    dz1  = dotp*dotp * pp[n2]*pp[n3] * (dgg[n1]*pp[n1] + gg*dpp[n1]);
                    dz2  = gg * pp4 * 2.0/rr[n1] * dotp;

                    Dzr[n1] = Rx[n1]*(dz1 - dotp*dz2) + Xk*dz2;
                }

                Z[i] += Zrep * gg * dotp*dotp * pp4;

                Dzdx[jj] += Dzr[0] * Zrep;
                Dzdx[kk] += Dzr[1] * Zrep;
                Dzdx[ll] += Dzr[2] * Zrep;

                for (nni = 0; nni < num[i]; nni++) {

                    if (nni == jj || nni == kk || nni == ll || r[i][nni] >= fhigh)
                        continue;

                    finiteforce[nni] = true;

                    Dzdx[nni] += N[i][nni] * (Zrep * dpi3[i]*dz[i][nni] * pi[j] *
                                                dotp*dotp * pp4);
                }

                for (nnj = 0; nnj < num[j]; nnj++) {

                    n  = near[j][nnj];
                    if (n == i || r[j][nnj] >= fhigh)
                        continue;

                    finiteforce2[jj][nnj] = true;

                    Dzdxx[jj][nnj] += N[j][nnj] * (Zrep * dpi[j]*dz[j][nnj] * pi3[i] *
                                                   dotp*dotp * pp4);
                }
            }
        }
    }
}

/* ------------------------------------------------------- */

void PairCEDIP::linear(int i)
{
    for (jj = 0; jj < num[i]; jj++) {

        j = near[i][jj];

        if (z[j] >= 4.0 || r[i][jj] <= plow || r[i][jj] >= c0)
            continue;

        for (kk = 0; kk < num[i]; kk++) {

            if (kk == jj || r[i][kk] >= phigh)
                continue;

            finiteforce[jj] = true;
            finiteforce[kk] = true;

            Rx[0]  = N[i][jj];
            Rx[1]  = N[i][kk];

            rr[0]  = r[i][jj];
            rr[1]  = r[i][kk];
            cc     = rr[0] - c0;

            pp[0]  = (1.0 - p[i][jj]) * cc * cc;
            pp[1]  =        p[i][kk];
            pp2    = pp[0] * pp[1];

            dpp[0] = -dp[i][jj]*cc*cc + (1.0-p[i][jj])*2.0*cc;
            dpp[1] =  dp[i][kk];

            gg     = pi2[i] * pi[j];
            dgg[0] = dz[i][jj] * (dpi2[i]*pi[j] + pi2[i]*dpi[j]);
            dgg[1] = dz[i][kk] *  dpi2[i]*pi[j];

            dotp  = dot(N[i][jj], N[i][kk]);

            for (n1 = 0; n1 <= 1; n1++) {
                n2 = 1 - n1;

                Dzr[n1] = Rx[n1] * ((1.0-dotp*dotp) * pp[n2] * (dgg[n1]*pp[n1] + gg*dpp[n1])) -
                          (Rx[n2] - Rx[n1]*dotp) * (pp2 * gg * 2.0 * dotp / rr[n1]);
            }

            Z[i] += Zrep * gg * (1 - dotp*dotp) * pp2;

            Dzdx[jj] += Dzr[0]*Zrep;
            Dzdx[kk] += Dzr[1]*Zrep;

            for (nni = 0; nni < num[i]; nni++) {

                if (nni == jj || nni == kk || r[i][nni] >= fhigh)
                    continue;

                finiteforce[nni] = true;

                Dzdx[nni] += N[i][nni] * (Zrep * dpi2[i] * dz[i][nni] *
                             pi[j] * (1.0-dotp*dotp) * pp2);
            }

            for (nnj = 0; nnj < num[j]; nnj++) {

                n = near[j][nnj];
                if (n == i || r[j][nnj] >= fhigh)
                    continue;

                finiteforce2[jj][nnj] = true;

                Dzdxx[jj][nnj] += N[j][nnj] * (Zrep * dpi[j] * dz[j][nnj] *
                                               pi2[i] * (1.0-dotp*dotp) * pp2);
            }
        }
    }
}


/* ---------------------------------------------------------------------- */

double PairCEDIP::calcU2(int i, int jj)
{
    arg   = 1.0 / (r[i][jj] - a - aprime*Z[i]);
    bond  = exp(-beta*Z[i]*Z[i]);
    r2    = 1.0 / (r[i][jj] * r[i][jj]);
    r4    = r2*r2;
    r5    = r4 / r[i][jj];

    part1 = BB*r4 - bond;
    part2 = eps * exp(sigma*arg);

    f22 = part2 * (2.0*beta*Z[i]*bond + part1*sigma*arg*arg*aprime);

    return (part1 * part2);
}

vec PairCEDIP::calcF2(int i, int jj, int kk)
{
    Dxr   = N[i][kk] * kron(jj,kk);
    Part3 = Dxr*(-4.0*BB*r5) + Dzdx[kk]*(2.0*beta*Z[i]*bond);
    Part4 = (Dxr - Dzdx[kk]*aprime) * (-sigma*arg*arg);
    Res   = (Part3 + Part4*part1) * part2;

    return Res;
}

double PairCEDIP::calcU3(int i, int jj, int kk)
{
    cosi    = dot(N[i][jj], N[i][kk]);
    Dcosj   = (N[i][kk] - N[i][jj]*cosi) / r[i][jj];
    Dcosk   = (N[i][jj] - N[i][kk]*cosi) / r[i][kk];

    arg1    = 1.0/(r[i][jj] - a - aprime*Z[i]);
    arg2    = 1.0/(r[i][kk] - a - aprime*Z[i]);

    Zi0     = Z[i] - Z0;
    zexpgam = lambda0 * exp(gamma*(arg1+arg2) - lambdaprime*Zi0*Zi0);

    ct      = cosi + tau;
    expcos  = exp(-q*ct*ct);
    theta   = 1.0 - expcos;
    dtheta  = 2.0*q*ct*expcos;

    u3term  = zexpgam * theta;

    arg11   =  u3term * gamma*arg1*arg1;
    arg22   =  u3term * gamma*arg2*arg2;
    argz    = -u3term * 2.0*lambdaprime*Zi0;
    f33     =  zexpgam*dtheta*dtau + aprime*(arg11+arg22) + argz;

    return u3term;
}

vec PairCEDIP::calcF3(int i, int jj, int kk, int ll)
{
    Dri = (Dzdx[ll]*aprime - N[i][jj]*kron(jj,ll)) * arg11;
    Drj = (Dzdx[ll]*aprime - N[i][kk]*kron(kk,ll)) * arg22;
    Drz = Dri + Drj + Dzdx[ll]*argz;

    Cosjk = Dcosj*kron(jj,ll) + Dcosk*kron(kk,ll);

    Res = Drz + (Cosjk + Dzdx[ll]*dtau)*(zexpgam*dtheta);

    return Res;
}

/*******************************
*  Calculate cutoff functions  *
*******************************/
void PairCEDIP::cutoff_fun(int i)
{
  z[i] = 0.0;

  double f_cut;

  double low[2]  = {flow,  plow};
  double high[2] = {fhigh, phigh};

  double *val[2];
  double *dval[2];

  double  x,  invx,  invx3,  frac;
  double dx, dinvx, dinvx3, dfrac;

  for (n1 = 0; n1 < 2; n1++) {

      for (jj = 0; jj < num[i]; jj++) {

          val[0]  = &f_cut;
          val[1]  = &p[i][jj];

          dval[0] = &dz[i][jj];
          dval[1] = &dp[i][jj];

          if (r[i][jj] < low[n1] + 0.001) {
              *val[n1]  = 1.0;
              *dval[n1] = 0.0;
          }
          else if (r[i][jj] > high[n1] - 0.001) {
              *val[n1]  = 0.0;
              *dval[n1] = 0.0;
          }
          else {
              dx    = 1.0 / (high[n1] - low[n1]);
              x     = (r[i][jj] - low[n1]) * dx;
              invx  = 1.0 / x;
              invx3 = invx*invx*invx;
              frac  = 1.0 / (1.0-invx3);

              dinvx  =    -invx*invx * dx;
              dinvx3 = 3.0*invx*invx * dinvx;
              dfrac  =     frac*frac * dinvx3;

              *val[n1]  = exp(alpha*frac);
              *dval[n1] = *val[n1] * alpha * dfrac;
          }

          if (n1 == 0)  z[i] += f_cut;
      }
  }
}


/**********************************
*  Calculate switching functions  *
**********************************/
void PairCEDIP::switch_fun(int i)
{
  double zi = z[i];

  double a2 = zi - 2.0;
  double a3 = zi - 3.0;

  double a2a = a2*a2 - 1.0;
  double a3a = a3*a3 - 1.0;

  if (zi > 4.0) {
    pi2[i] = 0.0;              dpi2[i] = 0.0;
    pi3[i] = 0.0;              dpi3[i] = 0.0;
    pi[i]  = 0.0;              dpi[i]  = 0.0;
  }
  else if (zi > 3.0) {
    pi2[i] = 0.0;              dpi2[i] = 0.0;
    pi3[i] = a3a * a3a;        dpi3[i] = 4.0 * a3 * a3a;
    pi[i]  = pi3[i];           dpi[i]  = dpi3[i];
  }
  else if (zi > 2.0) {
    pi2[i] = a2a * a2a;        dpi2[i] = 4.0 * a2 * a2a;
    pi3[i] = a3a * a3a;        dpi3[i] = 4.0 * a3 * a3a;
    pi[i]  = 1.0;              dpi[i]  = 0.0;
  }
  else if (zi > 1.0) {
    pi2[i] = a2a * a2a;        dpi2[i] = 4.0 * a2 * a2a;
    pi3[i] = 0.0;              dpi3[i] = 0.0;
    pi[i]  = 1.0;              dpi[i]  = 0.0;
  }
  else {
    pi2[i] = 0.0;              dpi2[i] = 0.0;
    pi3[i] = 0.0;              dpi3[i] = 0.0;
    pi[i]  = 1.0;              dpi[i]  = 0.0;
  }
}

/*************************************
*  Calculate tau and dtau from Z[i]  *
*************************************/
void PairCEDIP::tau_fun(int i)
{
    Zi   = Z[i];
    tt   = t1+t2*Zi;
    fac  = 1.0/12.0 * (1.0 + tanh(tt));
    ch   = cosh(tt);

    tau  = 1.0 - Zi*fac;
    dtau = -Zi*t2/(12.0*ch*ch) - fac;
}

/* ---------------------------------------------------------------------- */

void PairCEDIP::calc_dists()
{
  int i,ii,j,jj;

  int allnum, *ilist;
  double **x = atom->x;
  allnum     = list->inum + list->gnum;
  ilist      = list->ilist;

  vec Ri, Rj, Rij;

  for (ii = 0; ii < allnum; ii++) {
    i = ilist[ii];

    Ri.set(x[i][0], x[i][1], x[i][2]);

    for (jj = 0; jj < num[i]; jj++) {
        j = near[i][jj];

        Rj.set(x[j][0], x[j][1], x[j][2]);

        Rij      = Ri - Rj;
        r[i][jj] = norm(Rij);
        N[i][jj] = Rij / r[i][jj];
    }
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCEDIP::settings(int narg, char **arg)
{
  if (narg >= 2) error->all(FLERR,"Illegal pair_style command");
  else if (narg == 1) {
    if (strcmp(arg[0],"zbl") == 0) zbl = 1;
    else error->all(FLERR,"Unknown CEDIP style in pair_style command");
  }
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCEDIP::coeff(int narg, char **arg)
{
  if (!allocated) {
    allocated = 1;
    n = atom->ntypes;

    memory->create(setflag,n+1,n+1,"CEDIP:setflag");
    memory->create(cutsq,n+1,n+1,"CEDIP:cutsq");
    memory->create(cutghost,n+1,n+1,"CEDIP:cutghost");
    map = new int[n+1];
  }

  if (narg != 2 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  for (i = 2; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-1] = -1;
      continue;
    }
    else {
      map[i-1] = 1;
    }
  }

  setup();

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairCEDIP::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style CEDIP requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style CEDIP requires newton pair on");

  // Modern LAMMPS API: neighbor->add_request() replaces neighbor->request()
  // REQ_FULL  = full neighbour list (not half)
  // REQ_GHOST = include ghost atoms
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  // local CEDIP neighbour list pages
  int create = 0;
  if (ipage == nullptr) create = 1;
  if (pgsize != neighbor->pgsize) create = 1;
  if (oneatom != neighbor->oneatom) create = 1;

  if (create) {
    delete [] ipage;
    pgsize = neighbor->pgsize;
    oneatom = neighbor->oneatom;

    int nmypage = comm->nthreads;
    ipage = new MyPage<int>[nmypage];
    for (int i = 0; i < nmypage; i++) {
      ipage[i].init(oneatom,pgsize,1);
    }
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCEDIP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cutmax = c0+0.2;
  cutsq[i][j] = cutmax*cutmax;
  cutghost[i][j] = 2.0*cutmax;
  cutghost[j][i] = cutghost[i][j];

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairCEDIP::setup()
{
  eps    = 20.0853862863495;
  BB     = 0.827599951322299;
  beta   = 0.0490161172713279;
  sigma  = 1.25714643580808;
  a      = 1.89225338775144;
  aprime = 0.169794491000172;

  Z0          = 3.615;
  lambda0     = 19.86394896867;
  lambdaprime = 0.30;
  gamma       = 1.35419222406125;
  q           = 3.5;

  flow  = 1.547;
  fhigh = 2.270;
  plow  = 1.48 ;
  phigh = 2.000;
  alpha = 1.544;
  Zdih  = 0.30;
  Zrep  = 0.06;
  c0    = 3.2;

  t1 = -6.0*2.5;
  t2 =  6.0;

  zbl_c[0] = 0.02817;
  zbl_c[1] = 0.28022;
  zbl_c[2] = 0.50986;
  zbl_c[3] = 0.18175;

  zbl_cut   = 0.75;
  zbl_skin  = 0.07;
  zbl_shift = 0.03;

  zbl_1 = 6.0;
  zbl_2 = 6.0;
  zbl_au = 0.46850 / (pow(zbl_1,(double)0.23) + pow(zbl_2,(double)0.23));
  zbl_d[0] = 0.20162 / zbl_au;
  zbl_d[1] = 0.40290 / zbl_au;
  zbl_d[2] = 0.94229 / zbl_au;
  zbl_d[3] = 3.19980 / zbl_au;

  zbl_const = 0.5*(zbl_1 * zbl_2 * 1.602e-19 * 1.0e10) / (4 * M_PI * 8.85e-12);
}

/* ----------------------------- */

vec vec::operator- ()
{
  vec result(-x,-y,-z);
  return result;
}

vec vec::operator+ (const vec& v1)
{
  vec result(this->x + v1.x, this->y + v1.y, this->z + v1.z);
  return result;
}

vec vec::operator- (const vec& v1)
{
  vec result(this->x - v1.x, this->y - v1.y, this->z - v1.z);
  return result;
}

vec vec::operator+= (const vec& v1)
{
  this->x += v1.x;
  this->y += v1.y;
  this->z += v1.z;
  return *this;
}

vec vec::operator-= (const vec& v1)
{
  this->x -= v1.x;
  this->y -= v1.y;
  this->z -= v1.z;
  return *this;
}

vec vec::operator* (const double& d)
{
  vec result(x*d, y*d, z*d);
  return result;
}

vec vec::operator/ (const double& d)
{
  vec result(x/d, y/d, z/d);
  return result;
}

void vec::set(double newx, double newy, double newz)
{
  this->x = newx;
  this->y = newy;
  this->z = newz;
}

void vec::set_zero()
{
  this->x = 0.0;
  this->y = 0.0;
  this->z = 0.0;
}

double dot (vec v1, vec v2) {
  return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

vec cross (vec v1, vec v2) {
  vec result(v1.y*v2.z - v1.z*v2.y,
             v1.z*v2.x - v1.x*v2.z,
             v1.x*v2.y - v1.y*v2.x);
  return result;
}

double norm (vec v1) {
  return sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
}

double norm2 (vec v1) {
  return (v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
}
