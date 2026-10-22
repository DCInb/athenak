// Query the production CH/He EOS and independently integrate a homogeneous
// electron-ion energy-exchange ODE. Uses the collision case's code units.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"
#include "materials/material_mixture.hpp"

namespace {
constexpr double kb = 1.3806488e-16;
constexpr double mu = 1.660538921e-24;
constexpr double me = 9.1093837015e-28;
constexpr double qe = 4.803204712570263e-10;
constexpr double pi = 3.14159265358979323846;
constexpr double ev_kelvin = 11604.518121550082;
constexpr double tk = 1.e16*(13./9.)*mu/kb;
constexpr double density_unit = 1.1;

struct Cell {
  double rho, ti, te, y, ei, ee, cvi, cve, ne, zeff, zbar, abar;
  double tau, kappa, chi;
  int flags;
};

struct TableLogToStderr {
  std::streambuf *saved = std::cout.rdbuf(std::cerr.rdbuf());
  ~TableLogToStderr() { std::cout.rdbuf(saved); }
};

struct Model {
  materials::IonmixTwoTemperatureTableOptions options;
  materials::IonmixTwoTemperatureTable ch, he;
  materials::MaterialMixtureDevice mix;

  static materials::IonmixTwoTemperatureTableOptions Options() {
    materials::IonmixTwoTemperatureTableOptions o;
    o.bounds_policy = materials::IonmixBoundsPolicy::flash_extrapolate;
    o.geometric_interpolation = true;
    o.density_to_cgs = density_unit;
    o.temperature_to_kelvin = tk;
    o.pressure_from_cgs = 1./(density_unit*1.e16);
    o.specific_energy_from_cgs = 1.e-16;
    return o;
  }

  Model(const std::string &ch_file, const std::string &he_file)
      : options(Options()), ch(ch_file, options), he(he_file, options) {
    mix.nmaterials = 2;
    mix.species = DvceArray1D<materials::SpeciesProperties>("audit-species", 2);
    mix.Species(0).abar = 6.5;
    mix.Species(0).zbar = 3.5;
    mix.Species(0).zeff = 37./7.;
    mix.Species(1).abar = 4.002602;
    mix.Species(1).zbar = 2.;
    mix.Species(1).zeff = 2.;
    std::vector<materials::IonmixTwoTemperatureTableDevice> tables{
        ch.DeviceData(), he.DeviceData()};
    const auto bytes = tables.size()*sizeof(tables[0]);
    HostArray1D<unsigned char> storage("audit-tables-host", bytes);
    std::memcpy(storage.data(), tables.data(), bytes);
    mix.material_table_storage = DvceArray1D<unsigned char>("audit-tables", bytes);
    Kokkos::deep_copy(mix.material_table_storage, storage);
    mix.material_tables = reinterpret_cast<const materials::IonmixTwoTemperatureTableDevice *>(
        mix.material_table_storage.data());
    mix.use_tabular_eos = true;
    mix.density_to_cgs = density_unit;
    mix.temperature_to_kelvin = tk;
  }

  double Capacity(materials::IonmixComponent component, double rho,
                  double temperature, double y) const {
    const double lo = std::max(temperature*std::exp(-.001),
                              mix.MinimumTransportTemperature(y));
    const double hi = temperature*std::exp(.001);
    const auto low = mix.StateFromRhoTemperaturesNoSound(rho, lo, lo, y);
    const auto high = mix.StateFromRhoTemperaturesNoSound(rho, hi, hi, y);
    return component == materials::IonmixComponent::ion
        ? (high.ion_specific_internal_energy-low.ion_specific_internal_energy)/(hi-lo)
        : (high.electron_specific_internal_energy-low.electron_specific_internal_energy)/(hi-lo);
  }

  Cell AtTemperature(double rho, double ti, double te, double y) const {
    const auto s = mix.StateFromRhoTemperaturesNoSound(rho, ti, te, y);
    Cell c{};
    c.rho = rho; c.ti = s.ion_temperature; c.te = s.electron_temperature;
    c.y = y; c.ei = s.ion_specific_internal_energy;
    c.ee = s.electron_specific_internal_energy;
    c.cvi = Capacity(materials::IonmixComponent::ion, rho, c.ti, y);
    c.cve = Capacity(materials::IonmixComponent::electron, rho, c.te, y);
    c.ne = s.electron_number_density_cgs; c.zeff = s.effective_charge;
    c.zbar = s.mean_ionization; c.abar = mix.MeanAtomicMass(y);
    c.flags = s.query_flags;
    // Classical Landau-Spitzer electron temperature relaxation time, in ns.
    const double mi = c.abar*mu;
    c.tau = 1.e9*3.*me*mi*std::pow(kb*tk*(c.te/me+c.ti/mi), 1.5)/
        (8.*std::sqrt(2.*pi)*c.ne*c.zeff*std::pow(qe, 4)*10.);
    // FLASH SpitzerHighZ conductivity, without saturation, in erg/(s cm K).
    const double coefficient = std::pow(8./pi, 1.5)*std::pow(kb, 3.5)/
                               (std::pow(qe, 4)*std::sqrt(me));
    c.kappa = coefficient*std::pow(c.te*tk, 2.5)/((c.zbar+3.3)*10.);
    const double volumetric_capacity = rho*density_unit*c.cve*1.e16/tk;
    c.chi = c.kappa/volumetric_capacity;
    if (!(c.cvi > 0. && c.cve > 0. && c.tau > 0.) || !std::isfinite(c.chi)) {
      throw std::runtime_error("Nonphysical EOS query or transport coefficient");
    }
    return c;
  }

  Cell AtEnergy(double rho, double ei, double ee, double y) const {
    const auto s = mix.StateFromRhoSpecificEnergiesNoSound(rho, ei, ee, y);
    return AtTemperature(rho, s.ion_temperature, s.electron_temperature, y);
  }

  double Rate(double rho, double energy, double ee, double y) const {
    const Cell c = AtEnergy(rho, energy-ee, ee, y);
    return c.cve*(c.ti-c.te)/c.tau;
  }
};

void Header() {
  std::cout << "rho_code,ti_code,te_code,y_ch,ei_code,ee_code,cvi_code,cve_code,"
               "ne_cgs,zeff,zbar,abar,tau_ei_ns,tau_difference_ns,kappa_cgs,chi_cgs,"
               "query_flags,exchange_fallback,exchange_energy_error,exchange_difference_error\n";
}

void PrintCell(const Model &model, const Cell &c) {
  const double decay = std::exp(-(1.+c.cve/c.cvi)*2.e-4/c.tau);
  const double difference = (c.te-c.ti)*decay;
  const auto check = model.mix.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
      c.rho, c.ei, c.ee, c.ti, c.te, difference, c.y);
  std::cout << c.rho << ',' << c.ti << ',' << c.te << ',' << c.y << ','
            << c.ei << ',' << c.ee << ',' << c.cvi << ',' << c.cve << ','
            << c.ne << ',' << c.zeff << ',' << c.zbar << ',' << c.abar << ','
            << c.tau << ',' << c.tau/(1.+c.cve/c.cvi) << ',' << c.kappa << ','
            << c.chi << ',' << c.flags << ',' << check.used_fallback << ','
            << check.energy_residual << ',' << check.temperature_difference_residual << '\n';
}

void Integrate(const Model &model, double rho, double ti, double te, double y,
               double end, double max_step) {
  Cell c = model.AtTemperature(rho, ti, te, y);
  const double total = c.ei+c.ee;
  const double initial_delta = std::abs(c.ti-c.te);
  double time = 0., ee = c.ee;
  std::cout << "time_ns,ti_ev,te_ev,ei_code,ee_code,tau_ei_ns\n";
  auto write = [&]() {
    c = model.AtEnergy(rho, total-ee, ee, y);
    std::cout << time << ',' << c.ti*tk/ev_kelvin << ',' << c.te*tk/ev_kelvin
              << ',' << total-ee << ',' << ee << ',' << c.tau << '\n';
  };
  write();
  for (int step = 0; time < end; ++step) {
    if (step > 1000000) throw std::runtime_error("Reference integration step limit");
    c = model.AtEnergy(rho, total-ee, ee, y);
    if (std::abs(c.ti-c.te) <= 1.e-11*std::max(initial_delta, c.ti+c.te)) {
      time = end;
      write();
      break;
    }
    const double h = std::min({end-time, max_step, .01*c.tau/(1.+c.cve/c.cvi)});
    const double k1 = model.Rate(rho, total, ee, y);
    const double k2 = model.Rate(rho, total, ee+.5*h*k1, y);
    const double k3 = model.Rate(rho, total, ee+.5*h*k2, y);
    const double k4 = model.Rate(rho, total, ee+h*k3, y);
    ee += h*(k1+2.*k2+2.*k3+k4)/6.;
    time += h;
    write();
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: thermal_cell_query CH_TABLE HE_TABLE temperature|energy|relax "
                 "[rho_code Ti_code Te_code Y_CH duration_ns max_step_ns]\n"
                 "Query modes read rows rho_code Ti/ei_code Te/ee_code Y_CH from stdin.\n";
    return 2;
  }
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    std::unique_ptr<Model> storage;
    {
      TableLogToStderr redirect;
      storage = std::make_unique<Model>(argv[1], argv[2]);
    }
    const Model &model = *storage;
    std::cout << std::setprecision(17);
    const std::string mode = argv[3];
    if (mode == "relax") {
      if (argc != 10) throw std::runtime_error("relax requires six numeric arguments");
      Integrate(model, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]),
                std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9]));
    } else if (mode == "temperature" || mode == "energy") {
      Header();
      double rho, first, second, y;
      while (std::cin >> rho >> first >> second >> y) {
        const auto c = mode == "temperature" ? model.AtTemperature(rho, first, second, y)
                                             : model.AtEnergy(rho, first, second, y);
        PrintCell(model, c);
      }
      if (!std::cin.eof()) throw std::runtime_error("Invalid query row");
    } else {
      throw std::runtime_error("Unknown query mode");
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
  return status;
}
