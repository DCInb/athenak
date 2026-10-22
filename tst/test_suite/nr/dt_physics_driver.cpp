// GPU/CPU identities for the D-T reaction and Maxwellian alpha collision operators.
#include <cmath>
#include <iostream>
#include "nuclear/dt_cell.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    using namespace nuclear;
    DvceArray1D<int> checks("checks", 1);
    DvceArray1D<double> result("extent", 1);
    materials::MaterialMixtureDevice mixture;
    mixture.nmaterials = 3;
    mixture.species = DvceArray1D<materials::SpeciesProperties>("species",3);
    auto host = Kokkos::create_mirror_view(mixture.species);
    host(0).abar = 2; host(0).zbar = 1;
    host(1).abar = 4; host(1).zbar = 2;
    host(2).abar = 3; host(2).zbar = 1;
    Kokkos::deep_copy(mixture.species,host);
    Kokkos::parallel_for("dt_physics", Kokkos::RangePolicy<DevExeSpace>(0,1),
    KOKKOS_LAMBDA(int) {
      int bad = 0;
      // Published Bosch-Hale tabulation (MIT MFE Formulary), cm^3/s.
      const double temperatures[5] = {1,2,5,10,20};
      const double expected[5] = {6.857e-21,2.977e-19,1.366e-17,1.136e-16,4.330e-16};
      for (int n = 0; n < 5; ++n) {
        if (fabs(DTReactivity(temperatures[n])/expected[n]-1.0) > 5.0e-4) ++bad;
      }
      if (DTReactivity(0.0) != 0.0 || DTReactivity(101.0) >= 0.0) ++bad;
      const double exposure[5] = {1.0e-40,1.0e-25,1.0e-23,1.0,1.0e100};
      for (double kdt : exposure) {
        const double a = 1.0e24, b = 2.0e24;
        const double same = BurnExtent(a,a,kdt);
        const double exact = a/(1.0+1.0/(a*kdt));
        if (fabs(same/exact-1.0) > 1.0e-13) ++bad;
        const double x = BurnExtent(a,b,kdt);
        if (!(x >= 0.0 && x <= a)) ++bad;
        if (x != BurnExtent(b,a,kdt)) ++bad;
        if (kdt*a < 1.0e-10 && fabs(x/(a*b*kdt)-1.0) > 1.0e-10) ++bad;
      }
      // Independent cold-target Rutherford normalization and asymptotic powers.
      constexpr double charge = 4.803204712570263e-10;
      const double v = sqrt(2.0*kAlphaEnergy/kAlphaMass);
      const auto cold = AlphaTargetCollision(kAlphaEnergy,1e24,2*kAmu,1,0,7);
      const double analytic = 16*kPi*pow(charge,4)*1e24*7/(2*kAmu*v);
      if (fabs(cold.loss/analytic-1.0) > 2e-14) ++bad;
      const auto faster = AlphaTargetCollision(4*kAlphaEnergy,1e24,2*kAmu,1,0,7);
      if (fabs(faster.loss/cold.loss-0.5) > 1e-14 ||
          fabs(faster.deflection/cold.deflection-0.125) > 1e-14) ++bad;
      const auto warm = AlphaTargetCollision(kAlphaEnergy,1e24,kElectronMass,1,1e8,7);
      if (!(warm.loss > 0.0 && warm.loss < analytic*2*kAmu/kElectronMass &&
            warm.deflection > 0.0)) ++bad;
      // Small-x evaluation must remain positive and finite across its series boundary.
      for (double t = 1e7; t < 1e14; t *= 2) {
        const auto r = AlphaTargetCollision(1e6*kKev,1e24,kElectronMass,1,t,7);
        if (!Kokkos::isfinite(r.loss) || !(r.deflection > 0.0)) ++bad;
      }
      BurnCell cell;
      cell.mixture = mixture; cell.d=0; cell.a=1; cell.t=2;
      cell.rho=10; cell.partial[0]=4; cell.partial[1]=0; cell.partial[2]=6;
      cell.density_unit=1; cell.pressure_unit=1e18; cell.multiplier=1;
      cell.ei = 1.5*(4.0/kAmu)*5*kKev/1e18; cell.ee=cell.ei;
      double extent;
      if (!cell.Integrate(1e-10,1e-8,extent)) ++bad;
      result(0)=extent;
      if (!cell.Integrate(1e-6,1e-6,extent) || extent < 0.99*(2.0/kAmu) ||
          extent > 2.0/kAmu) ++bad;
      checks(0)=bad;
    });
    auto check = Kokkos::create_mirror_view_and_copy(HostMemSpace(),checks);
    auto extent = Kokkos::create_mirror_view_and_copy(HostMemSpace(),result);
    failures += check(0);
    // Independent RK4 time integration of the ideal, isochoric Q-free burn equations.
    const double number = 2.0/kAmu;
    const auto rhs = [number](double y) {
      return number*(1-y)*(1-y)*DTReactivity(5.0/(1.0-0.5*y));
    };
    double y=0, h=1e-10/10000;
    for (int n=0;n<10000;++n) {
      double a=rhs(y), b=rhs(y+h*a/2), c=rhs(y+h*b/2), d=rhs(y+h*c);
      y += h*(a+2*b+2*c+d)/6;
    }
    if (fabs(extent(0)/number-y) > 2e-8) ++failures;
    std::cout << "D-T physics failures=" << failures << '\n';
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
