/* -*- c++ -*- ----------------------------------------------------------
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
   Key changes vs. original 2014 version:
     - neighbor->requests[] replaced by neighbor->add_request() API
     - neigh_request.h no longer exists; include neighbor.h only
     - C-style includes replaced by C++ headers (<cmath>, <cstdio>, etc.)
     - bigint loop variable declared locally to avoid shadowing
     - sys/time.h / timestamp_t removed (unused in implementation)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(cedip,PairCEDIP);
// clang-format on
#else

#ifndef LMP_PAIR_CEDIP_H
#define LMP_PAIR_CEDIP_H

#include "pair.h"
#include "my_page.h"

namespace LAMMPS_NS {

class vec {
  public:
    double x, y, z;

    vec() : x(0), y(0), z(0) { }
    vec(double a, double b, double c) : x(a), y(b), z(c) { }
    ~vec() { }

    void set(double, double, double);
    void set_zero();

    vec operator-  ();
    vec operator+  (const vec& v1);
    vec operator-  (const vec& v1);
    vec operator+= (const vec& v1);
    vec operator-= (const vec& v1);
    vec operator*  (const double& d);
    vec operator/  (const double& d);
};


class PairCEDIP : public Pair {
 public:
  PairCEDIP(class LAMMPS *);
  ~PairCEDIP() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:

  void allocate_arrays();
  void deallocate_arrays();
  void CEDIP_neigh();

  // Parameters for Carbon EDIP

  /*************************************************************
  * Variable naming convention:                                *
  *   scalars - first letter lower case                        *
  *   vectors - first letter upper case                        *
  *   Exceptions: Z = generalised coordination (scalar);       *
  *               model parameters (as shown in Marks (2000)). *
  *************************************************************/

  // Two-body parameters:
  double eps;
  double BB;
  double beta;
  double sigma;
  double a;
  double aprime;

  // Three-body parameters:
  double Z0;
  double lambda0;
  double lambdaprime;
  double gamma;
  double q;

  // Coordination parameters
  double flow;
  double fhigh;
  double plow;
  double phigh;
  double alpha;
  double Zdih;
  double Zrep;
  double c0;

  // "tau" variables
  double tau, dtau;
  double t1, t2;
  double fac, ch, tt, Zi;

  // Position vectors
  double  **r;
  vec     **N;
  vec     *N_data;

  // Energy and force terms
  double u2,u3,u;
  double uedip,uzbl;
  vec    F, F2, F3, Fhat;
  double fpair;

  // Coordination and cutoff functions
  double *z;
  double ztmp;
  double **dz;

  double **p;
  double **dp;

  double *Z;
  vec *Dzdx;
  vec **Dzdxx;
  vec *Dzdxx_data;

  bool *finiteforce;
  bool **finiteforce2;

  // Switching functions
  double  *pi,  *pi2,  *pi3;
  double *dpi, *dpi2, *dpi3;

  // Temp variables for calculating pair energy & force
  double arg, bond, r2, r4, r5, part1, part2, f22;
  vec    Dxr, Part3, Part4, Res;

  // Temp variables for calculating triple energy & force
  double cosi, arg1, arg2, Zi0, zexpgam, ct, expcos;
  double theta, dtheta, u3term;
  double arg11, arg22, argz, f33;
  vec    Dcosj, Dcosk, Cosjk;
  vec    Dri, Drj, Drz;

  // Temp variables for calculating generalised coordination Z
  vec    Xk;
  vec    Rx[3], Dzr[3];
  double cc, pp2, pp4, gg, dotp;
  double dz0, dz1, dz2;
  double rr[3], pp[3], dpp[3], dgg[3];
  int    n1, n2, n3;

  // Generic atom-counting variables
  int i,j,k,l,m,n;
  int ii,jj,kk,ll,mm,nni,nnj;

  double cutmax;
  int *map;

  int maxlocal;
  int pgsize;
  int oneatom;
  int maxn;
  MyPage<int> *ipage;
  int *num;
  int **near;

  /* ZBL parameters */
  int    zbl;
  double zbl_cut, zbl_skin, zbl_shift;
  double zbl_1, zbl_2, zbl_au;
  double zbl_c[4], zbl_d[4], zbl_tmp;
  double zbl_const, zbl_phi, dzbl_phi;
  double zbl_exp, edip_exp;
  double zbl_sf, edip_sf;
  double dzbl_sf, dedip_sf;
  double fzbl;

  void setup();

  void calc_dists();
  void cutoff_fun(int);
  void switch_fun(int);
  void tau_fun(int);

  void pair(int, int);
  void pair_zbl(int, int);
  void triple(int, int);

  void coordination(int);
  void dihedral(int);
  void repulsion(int);
  void linear(int);

  double calcU2(int, int);
  vec    calcF2(int, int, int);

  double calcU3(int, int, int);
  vec    calcF3(int, int, int, int);

  inline double kron(const int a, const int b) const {
    return (a == b) ? 1.0 : 0.0;
  };

};

} // end namespace LAMMPS_NS

#endif
#endif
