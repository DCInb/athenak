// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
// Numerical oracle for native multigrid. The reference operator and PCG are
// host implementations, independent of the device hierarchy and its kernels.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "diffusion/native_multigrid.hpp"
#include "mesh/restriction.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
int rank = 0, ranks = 1;
double Sum(double value) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return value;
}
double Dot(const std::vector<double> &x, const std::vector<double> &y) {
  return Sum(std::inner_product(x.begin(), x.end(), y.begin(), 0.0));
}
void Check(bool success, const char *message) {
  if (Sum(success ? 0.0 : 1.0) != 0) {
    std::cerr << message << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::exit(1);
  }
}
void Upload(const std::vector<double> &host, const DvceArray1D<double> &device) {
  auto mirror = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < host.size(); ++i) mirror(i) = host[i];
  Kokkos::deep_copy(device, mirror);
}
std::vector<double> Download(const DvceArray1D<double> &device) {
  auto mirror = Kokkos::create_mirror_view_and_copy(HostMemSpace(), device);
  std::vector<double> host(mirror.size());
  for (std::size_t i = 0; i < host.size(); ++i) host[i] = mirror(i);
  return host;
}
void CheckConstantRestriction() {
  DvceArray5D<Real> field("constant restriction donors", 1, 1, 2, 2, 2);
  DvceArray1D<Real> restricted("constant restriction", 3);
  for (const Real value : {0.9653823089344712, 0.003461769106552886,
                           1.e-160, 1.e250}) {
    Kokkos::deep_copy(field, value);
    Kokkos::parallel_for("oracle_constant_restriction",
        Kokkos::RangePolicy<DevExeSpace>(0, 3), KOKKOS_LAMBDA(int dimension) {
      restricted(dimension) = RestrictLinearAverage(
          field, 0, 0, 0, 0, 0, dimension > 0, dimension > 1);
    });
    auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), restricted);
    for (int d = 0; d < 3; ++d) {
      Check(host(d) == value, "Scalar AMR restriction changed a constant mode");
    }
  }
}

void CheckIsolatedCorrections(bool device_mpi) {
  for (const bool stars : {true, false}) {
    const int total = stars ? 256 : 257;
    const int first = total*rank/ranks, end = total*(rank+1)/ranks, n = end-first;
    std::vector<int> rows{0}, columns;
    for (int i = first; i < end; ++i) {
      for (int j = 0; j < total; ++j) {
        const bool edge = stars ? (i%2 == j%2 && i != j && (i < 2 || j < 2)) :
                                  (i != j && (i == 0 || j == 0));
        if (edge) columns.push_back(j);
      }
      rows.push_back(columns.size());
    }
    diffusion::NativeMultigrid mg;
    mg.Configure(rows, columns, first, total, device_mpi);
    DvceArray1D<double> dg("isolated conductance", columns.size());
    DvceArray1D<double> dm("isolated mass", n), db("isolated rhs", n), dx("isolated x", n);
    std::vector<double> mass(n, 1.0), rhs(n, 0.0), conductance(columns.size());
    if (first == 0 && n) rhs[0] = 1.0;
    Upload(mass, dm);
    Upload(rhs, db);
    for (const double weak : {0.0, 1.e-80, 1.e-20}) {
      if (stars && weak != 0.0) continue;
      for (int i = 0; i < n; ++i) for (int q = rows[i]; q < rows[i+1]; ++q) {
        conductance[q] = stars || std::max(first+i, columns[q]) <= 128 ? 1.e4 : weak;
      }
      Upload(conductance, dg);
      mg.Setup(dg, dm);
      mg.Apply(db, dx);
      const auto x = Download(dx);
      double contamination = 0.0;
      for (int i = 0; i < n; ++i) if (stars ? (first+i)%2 == 1 : first+i > 128) {
        contamination = std::max(contamination, std::abs(x[i]));
      }
      contamination = Sum(contamination);
      if (rank == 0) std::cout << "isolation stars=" << stars << " weak=" << weak
          << " contamination=" << contamination << std::endl;
      Check(contamination <= 1.e4*weak,
            "Coarse correction mixed disconnected or mass-dominated cells");
    }
    if (!stars) {
      // Vary mass alone at fixed topology/conductance. This changes which
      // cells need coarse correction; a reused hierarchy must match fresh setup.
      for (int i = 0; i < n; ++i) for (int q = rows[i]; q < rows[i+1]; ++q) {
        conductance[q] = std::max(first+i, columns[q]) <= 128 ? 1.e4 : 1.0;
      }
      Upload(conductance, dg);
      for (const double weak_mass : {1.e12, 1.e-4, 1.e12}) {
        for (int i = 0; i < n; ++i) mass[i] = first+i > 128 ? weak_mass : 1.0;
        Upload(mass, dm);
        mg.Setup(dg, dm);
        mg.Apply(db, dx);
        auto reused = Download(dx);
        diffusion::NativeMultigrid fresh;
        fresh.Configure(rows, columns, first, total, device_mpi);
        fresh.Setup(dg, dm);
        fresh.Apply(db, dx);
        auto reference = Download(dx);
        double error = 0.0, norm = 0.0;
        for (int i = 0; i < n; ++i) {
          error += (reused[i]-reference[i])*(reused[i]-reference[i]);
          norm += reference[i]*reference[i];
        }
        error = std::sqrt(Sum(error)/Sum(norm));
        if (rank == 0) std::cout << "mass-reuse mass=" << weak_mass
            << " relative_error=" << error << std::endl;
        Check(error < 2.e-12, "Mass-only update retained stale coarse membership");
      }
    }
  }
}

void CheckStiffDifference(bool device_mpi) {
  // On a complete graph with unit mass, every zero-sum vector is an exact
  // eigenvector with eigenvalue 1+n*g. Opposite integer RHS entries make the
  // reference independent of a floating-point dense factorization.
  for (const int total : {2, 63}) {
    const int first = total*rank/ranks, end = total*(rank+1)/ranks, n = end-first;
    std::vector<int> rows{0}, columns;
    for (int i = first; i < end; ++i) {
      for (int j = 0; j < total; ++j) if (i != j) columns.push_back(j);
      rows.push_back(columns.size());
    }
    diffusion::NativeMultigrid mg;
    mg.Configure(rows, columns, first, total, device_mpi);
    DvceArray1D<double> dg("difference g", columns.size()), dm("difference m", n);
    DvceArray1D<double> db("difference b", n), dx("difference x", n);
    std::vector<double> mass(n, 1.0), rhs(n);
    for (int i = 0; i < n; ++i) rhs[i] = 2*(first+i)-(total-1);
    Upload(mass, dm); Upload(rhs, db);
    for (double g : {1.e8, 1.e12, 1.e14}) {
      Upload(std::vector<double>(columns.size(), g), dg);
      mg.Setup(dg, dm); mg.Apply(db, dx);
      const auto x = Download(dx);
      double error = 0.0, norm = 0.0;
      for (int i = 0; i < n; ++i) {
        error += std::pow(x[i]*(1.0+total*g)-rhs[i], 2);
        norm += rhs[i]*rhs[i];
      }
      error = std::sqrt(Sum(error)/Sum(norm));
      if (rank == 0) std::cout << "stiff-difference cells=" << total << " g=" << g
          << " relative_error=" << error << std::endl;
      Check(error < 2.e-12, "Coarse inverse lost stiff difference modes");
    }
  }
}

void CheckStiffRoot(bool device_mpi) {
  // A constant has exactly zero graph flux, so A*1=mass independently of
  // diffusion strength. Check the direct coarse inverse itself: Krylov
  // refinement would hide a loss of its small mass term during factorization.
  for (const int total : {2, 63}) {
    const int first = total*rank/ranks, end = total*(rank+1)/ranks, n = end-first;
    std::vector<int> rows{0}, columns;
    for (int i = first; i < end; ++i) {
      for (int j = 0; j < total; ++j) {
        if (std::abs(i-j) == 1 || std::abs(i-j) == 3) columns.push_back(j);
      }
      rows.push_back(columns.size());
    }
    diffusion::NativeMultigrid mg;
    mg.Configure(rows, columns, first, total, device_mpi);
    DvceArray1D<double> dg("stiff root conductance", columns.size());
    DvceArray1D<double> dm("stiff root mass", n), dx("stiff root solution", n);
    std::vector<double> mass(n), conductance(columns.size());
    for (int i = 0; i < n; ++i) mass[i] = 0.6+0.2*std::sin(first+i);
    Upload(mass, dm);
    for (const double stiffness : {1.e8, 1.e12, 1.e14}) {
      for (int i = 0; i < n; ++i) for (int q = rows[i]; q < rows[i+1]; ++q) {
        const int a = std::min(first+i, columns[q]);
        conductance[q] = stiffness*(1.0+0.25*std::sin(a));
      }
      Upload(conductance, dg);
      mg.Setup(dg, dm);
      mg.Apply(dm, dx);
      const auto x = Download(dx);
      double error = 0.0, balance = 0.0, total_mass = 0.0;
      for (int i = 0; i < n; ++i) {
        error += (x[i]-1.0)*(x[i]-1.0);
        balance += mass[i]*(x[i]-1.0);
        total_mass += mass[i];
      }
      error = std::sqrt(Sum(error)/total);
      balance = std::abs(Sum(balance))/Sum(total_mass);
      if (rank == 0) std::cout << "stiff-root cells=" << total
          << " stiffness=" << stiffness << " constant_error=" << error
          << " conservation_error=" << balance << std::endl;
      Check(error < 2.e-12 && balance < 2.e-12,
            "Coarse inverse lost the stiff constant mode");
    }
  }
}

void CheckStiffSchur(bool device_mpi) {
  // Eight independent K_{3,509} graphs. Interior leaves are eliminated exactly,
  // leaving 24 hubs for the direct root. The exact solution of A*x=c is x=c.
  const int total = 4096;
  const int first = total*rank/ranks, end = total*(rank+1)/ranks, n = end-first;
  std::vector<int> rows{0}, columns;
  for (int i = first; i < end; ++i) {
    const int start = (i/512)*512;
    for (int j = start; j < start+512; ++j)
      if ((i-start < 3) != (j-start < 3)) columns.push_back(j);
    rows.push_back(columns.size());
  }
  diffusion::NativeMultigrid mg;
  mg.Configure(rows, columns, first, total, device_mpi);
  DvceArray1D<double> dg("Schur g", columns.size()), dm("Schur m", n);
  DvceArray1D<double> db("Schur b", n), dx("Schur x", n);
  std::vector<double> mass(n, 1.0), conductance(columns.size());
  Upload(mass, dm);
  for (double g : {1.e8, 1.e12, 1.e14}) {
    for (int i = 0; i < n; ++i) for (int q = rows[i]; q < rows[i+1]; ++q)
      conductance[q] = g*(1.+.25*std::sin(std::min(first+i, columns[q])));
    Upload(conductance, dg);
    mg.Setup(dg, dm);
    for (double c : {1., 0.9653823089344712}) {
      Upload(std::vector<double>(n,c), db); mg.Apply(db,dx);
      const auto x = Download(dx);
      std::vector<double> all(total);
#if MPI_PARALLEL_ENABLED
      std::vector<int> counts(ranks), starts(ranks);
      for (int r=0; r<ranks; ++r) { starts[r]=total*r/ranks; counts[r]=total*(r+1)/ranks-starts[r]; }
      MPI_Allgatherv(x.data(),n,MPI_DOUBLE,all.data(),counts.data(),starts.data(),MPI_DOUBLE,MPI_COMM_WORLD);
#else
      all=x;
#endif
      double error=0., residual=0.;
      for (int i=0; i<n; ++i) {
        error += (x[i]-c)*(x[i]-c);
        double ax=x[i];
        for (int q=rows[i];q<rows[i+1];++q) ax += conductance[q]*(x[i]-all[columns[q]]);
        residual += (ax-c)*(ax-c);
      }
      error=std::sqrt(Sum(error)/total)/c;
      residual=std::sqrt(Sum(residual)/total)/c;
      if (rank==0) std::cout << "stiff-schur g=" << g << " c=" << c
          << " error=" << error << " residual=" << residual << std::endl;
      Check(error < 2.e-12 && residual < 2.e-12,
            "Exact Schur interpolation lost the stiff constant mode");
    }
  }
  // In the opposite, mass-dominated limit, leaves must retain their tiny
  // physical response to the three bright hubs. Anchoring those leaves to a
  // hub would subtract away this response. This exact bipartite formula does
  // not use the hierarchy or a floating-point dense solve as its reference.
  for (double weak : {0.0, 1.e-80, 1.e-20}) {
    Upload(std::vector<double>(columns.size(), weak), dg);
    std::vector<double> rhs(n);
    for (int i = 0; i < n; ++i) {
      const int local = (first+i)%512;
      rhs[i] = local < 3 ? local+1 : 0.0;
    }
    Upload(rhs, db); mg.Setup(dg, dm); mg.Apply(db, dx);
    const auto x = Download(dx);
    const double leaf = 6.0*weak/(1.0+512.0*weak);
    double error = 0.0;
    for (int i = 0; i < n; ++i) {
      const int local = (first+i)%512;
      const double expected = local < 3 ?
          (local+1+509.0*weak*leaf)/(1.0+509.0*weak) : leaf;
      const double difference = expected == 0.0 ? std::abs(x[i]) :
          std::abs((x[i]-expected)/expected);
      error = std::max(error, difference);
    }
    error = Sum(error);
    if (rank == 0) std::cout << "weak-schur g=" << weak
        << " relative_error=" << error << std::endl;
    Check(error < 2.e-12, "Schur interpolation erased weak physical coupling");
  }
}

void Run(int nx, int ny, int nz, bool device_mpi, bool bad_mass,
         bool empty_rank = false, bool disconnected = false, bool switch_strength = false,
         bool cross_rank_pairs = false) {
  const int total = nx*ny*nz;
  const int skipped = empty_rank && ranks > 1 ? 1 : 0;
  const int owners = ranks-skipped;
  const int first = total*std::max(0, rank-skipped)/owners;
  const int end = total*std::max(0, rank+1-skipped)/owners, n = end-first;
  std::vector<int> rows{0}, columns, counts(ranks), starts(ranks);
  for (int r = 0; r < ranks; ++r) {
    starts[r] = total*std::max(0, r-skipped)/owners;
    counts[r] = total*std::max(0, r+1-skipped)/owners-starts[r];
  }
  const int stride[3] = {1, nx, nx*ny}, extent[3] = {nx, ny, nz};
  for (int i = first; i < end; ++i) {
    std::vector<int> neighbors;
    for (int d = 0; d < 3; ++d) if (extent[d] > 1) {
      const int coordinate = (i/stride[d])%extent[d];
      if (coordinate > 0) neighbors.push_back(i-stride[d]);
      if (coordinate+1 < extent[d]) neighbors.push_back(i+stride[d]);
    }
    // Number strongly coupled pairs so that their two endpoints live on
    // different MPI ranks. Local aggregation then has no admissible merge.
    if (cross_rank_pairs) neighbors = {(i+total/2)%total};
    std::sort(neighbors.begin(), neighbors.end());
    if (disconnected) {
      neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
          [i](int j) { return i/8 != j/8; }), neighbors.end());
    }
    columns.insert(columns.end(), neighbors.begin(), neighbors.end());
    rows.push_back(columns.size());
  }
  diffusion::NativeMultigrid mg;
  mg.Configure(rows, columns, first, total, device_mpi);
  DvceArray1D<double> dg("oracle conductance", columns.size()), dm("oracle mass", n);
  DvceArray1D<double> dr("oracle rhs", n), dz("oracle correction", n);
  std::vector<double> mass(n), conductance(columns.size()), global(total);
  const auto apply = [&](const std::vector<double> &x) {
    Upload(x, dr);
    mg.Apply(dr, dz);
    return Download(dz);
  };
  const auto matvec = [&](const std::vector<double> &x) {
#if MPI_PARALLEL_ENABLED
    MPI_Allgatherv(x.data(), n, MPI_DOUBLE, global.data(), counts.data(), starts.data(),
                  MPI_DOUBLE, MPI_COMM_WORLD);
#else
    global = x;
#endif
    std::vector<double> ax(n);
    for (int i = 0; i < n; ++i) {
      ax[i] = mass[i]*x[i];
      for (int q = rows[i]; q < rows[i+1]; ++q) {
        ax[i] += conductance[q]*(x[i]-global[columns[q]]);
      }
    }
    return ax;
  };
  // Reuse a hierarchy while changing both heat capacity and diffusion; then
  // force a symbolic rebuild, just as a mesh redistribution/regrid does.
  for (int update = 0; update < 3; ++update) {
    if (update == 2) mg.Configure(rows, columns, first, total, device_mpi);
    for (int i = 0; i < n; ++i) {
      mass[i] = 0.6+0.2*std::sin(first+i)+0.3*update;
      if (cross_rank_pairs) mass[i] = 1.0+0.3*update;
      if (bad_mass && rank == 0 && i == 0) mass[i] = -1;
      for (int q = rows[i]; q < rows[i+1]; ++q) {
        const int a = std::min(first+i, columns[q]), b = std::max(first+i, columns[q]);
        const double left = a < total/2 ? 1.0 : 1.0e-6;
        const double right = b < total/2 ? 1.0 : 1.0e-6;
        conductance[q] = (update == 0 ? 1.0 : 3.0)*80.0*2*left*right/(left+right);
        if (!disconnected && a%31 == 0 && b%7 == 0) conductance[q] = 0.0;
        if (cross_rank_pairs) {
          conductance[q] = (update == 0 ? 1.0 : 3.0)*std::pow(10.0, 3.0+6.0*(a%31)/30.0);
        }
        if (switch_strength) {
          // Rotate stiff diffusion from x to y without changing the CSR graph.
          // The second setup must detect that the old aggregate bonds are weak.
          conductance[q] = b-a == (update == 0 ? 1 : nx) ? 1e5 : 1e-5;
        }
      }
    }
    Upload(conductance, dg);
    Upload(mass, dm);
    mg.Setup(dg, dm);
    std::vector<double> v(n), w(n), exact(n);
    for (int i = 0; i < n; ++i) {
      v[i] = std::sin(0.27*(first+i));
      w[i] = std::cos(0.19*(first+i));
      exact[i] = cross_rank_pairs ?
          0.4+std::sin(0.27*((first+i)%(total/2))) : 0.4+v[i]+0.2*w[i];
    }
    auto bv = apply(v), bw = apply(w);
    const double v_bw = Dot(v, bw), w_bv = Dot(w, bv);
    Check(std::abs(v_bw-w_bv) < 2e-12*(1+std::abs(v_bw)+std::abs(w_bv)),
          "V-cycle is not symmetric");
    Check(Dot(v, bv) > 0, "V-cycle is not positive definite");
    const auto zero = apply(std::vector<double>(n, 0));
    Check(Dot(zero, zero) == 0, "V-cycle retained a previous initial guess");
    const auto again = apply(v);
    for (int i = 0; i < n; ++i) bv[i] -= again[i];
    Check(Dot(bv, bv) < 1e-26*(1+Dot(again, again)), "V-cycle is not repeatable");
    const auto b = matvec(exact);
    auto r = b;
    std::vector<double> x(n, 0), z = apply(r), p = z;
    double rz = Dot(r, z), norm = Dot(b, b);
    int iterations = 0;
    // The anisotropic oracle has condition numbers above 1e6. Resolve the
    // residual more tightly than the solution-error guard requires; changing
    // a valid preconditioner must not make early residual stopping dominate it.
    for (; iterations < 400 && Dot(r, r) > 1e-28*norm; ++iterations) {
      auto ap = matvec(p);
      const double pap = Dot(p, ap);
      Check(pap > 0 && rz > 0, "PCG lost definiteness");
      const double alpha = rz/pap;
      for (int i = 0; i < n; ++i) { x[i] += alpha*p[i]; r[i] -= alpha*ap[i]; }
      z = apply(r);
      const double next = Dot(r, z), beta = next/rz;
      for (int i = 0; i < n; ++i) p[i] = z[i]+beta*p[i];
      rz = next;
    }
    auto residual = matvec(x);
    auto error = x;
    for (int i = 0; i < n; ++i) { residual[i] -= b[i]; error[i] -= exact[i]; }
    const double relative = std::sqrt(Dot(residual, residual)/norm);
    const double solution_error = std::sqrt(Dot(error, error)/Dot(exact, exact));
    if (rank == 0) std::cout << "oracle-check cells=" << total << " update=" << update
        << " iterations=" << iterations << " equation_error=" << relative
        << " solution_error=" << solution_error << std::endl;
    if (cross_rank_pairs) {
      // The independent pair constants should be represented on a coarse
      // graph even when no pair can be aggregated within an MPI rank.
      Check(iterations < 16, "Rank partition destroyed the stiff coarse correction");
    }
    Check(relative < 2e-11 && iterations < 400, "True equation residual failed");
    Check(solution_error < 2e-9,
          "Manufactured solution failed");
    if (rank == 0) std::cout << "native-mg cells=" << total << " update=" << update
        << " levels=" << mg.levels() << " iterations=" << iterations
        << " residual=" << relative << " bytes=" << mg.bytes() << std::endl;
  }
}
}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#endif
  const bool device_mpi = argc > 1 && std::string(argv[1]) == "--device-mpi";
  const bool bad_mass = argc > 1 && std::string(argv[1]) == "--bad-mass";
  Kokkos::initialize(argc, argv);
  {
    CheckConstantRestriction();
    Run(2, 1, 1, device_mpi, bad_mass);  // Includes empty MPI ranks with -np 4.
    CheckStiffRoot(device_mpi);
    CheckStiffSchur(device_mpi);
    CheckStiffDifference(device_mpi);
    CheckIsolatedCorrections(device_mpi);
    Run(4096, 1, 1, device_mpi, false, false, false, false, true);
    Run(257, 1, 1, device_mpi, false);
    Run(16, 16, 8, device_mpi, false);
    Run(257, 1, 1, device_mpi, false, true);  // Empty rank during graph coarsening.
    Run(1024, 1, 1, device_mpi, false, false, true);  // Zero retained coarse edges.
    Run(16, 16, 16, device_mpi, false, false, false, true);
    Run(32, 32, 32, device_mpi, false, false, false, true);
    Run(32, 32, 16, device_mpi, false);  // Distributed level above the replica cutoff.
    Run(4097, 1, 1, device_mpi, false, true);  // Large hierarchy with an empty rank.
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return 0;
}
