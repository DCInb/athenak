#ifndef TWO_TEMPERATURE_OPACITY_TABLE_HPP_
#define TWO_TEMPERATURE_OPACITY_TABLE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.hpp
//! \brief Density- and electron-temperature-dependent multigroup opacity tables.

#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "materials/material_mixture.hpp"

class ParameterInput;

namespace two_temperature {

enum OpacityKind {
  opacity_transport = 0,
  opacity_absorption = 1,
  opacity_emission = 2
};

//! Sentinel stored in `log_values` for table entries that have no logarithm.  Any real
//! log(v) for a positive double is >= log(DBL_MIN) ~= -708, so -DBL_MAX/2 is
//! unambiguous and a single comparison separates the two branches.
constexpr Real kNonPositiveLog = -8.988465674311579e+307;

struct OpacityTableLocation {
  int id0 = 0;
  int it0 = 0;
  Real fd = 0.0;
  Real ft = 0.0;
};

//----------------------------------------------------------------------------------------
//! \struct OpacityTableDevice
//! \brief Lightweight device-copyable view of a tabulated opacity model.

struct OpacityTableDevice {
  DvceArray1D<Real> density;
  DvceArray1D<Real> temperature;
  // Axis logarithms, evaluated once on the device at load time.  The axes are constant
  // per table, so recomputing log(density(lower)) and log(density(upper)) inside every
  // lookup costs four of the six axis transcendentals for nothing.  They are filled by a
  // device kernel rather than on the host so the stored value is bit-identical to the
  // one the inline log() produced, which keeps this a byte-exact change.
  DvceArray1D<Real> log_density;
  DvceArray1D<Real> log_temperature;
  DvceArray4D<Real> values;
  // log(values), with kNonPositiveLog wherever the entry is not positive.
  DvceArray4D<Real> log_values;
  int ndensity = 0;
  int ntemperature = 0;
  bool log_interpolation = false;
  bool geometric_interpolation = false;
  bool log_coordinates = false;
  Real density_scale = 1.0;
  Real temperature_scale = 1.0;
  Real transport_scale = 1.0;
  Real absorption_scale = 1.0;
  Real emission_scale = 1.0;

  KOKKOS_INLINE_FUNCTION
  OpacityTableLocation Locate(Real code_density, Real code_temperature) const {
    Real d = code_density*density_scale;
    Real t = code_temperature*temperature_scale;

    OpacityTableLocation location;
    if (ndensity > 1) {
      if (d >= density(ndensity-1)) {
        location.id0 = ndensity-2;
        location.fd = 1.0;
      } else if (d > density(0)) {
        int lower = 0;
        int upper = ndensity-1;
        while (upper-lower > 1) {
          int middle = (lower+upper)/2;
          if (density(middle) > d) {
            upper = middle;
          } else {
            lower = middle;
          }
        }
        location.id0 = lower;
        if (log_coordinates) {
          location.fd = (log(d)-log_density(lower))/
                        (log_density(upper)-log_density(lower));
        } else {
          location.fd = (d-density(lower))/(density(upper)-density(lower));
        }
      }
    }

    if (ntemperature > 1) {
      if (t >= temperature(ntemperature-1)) {
        location.it0 = ntemperature-2;
        location.ft = 1.0;
      } else if (t > temperature(0)) {
        int lower = 0;
        int upper = ntemperature-1;
        while (upper-lower > 1) {
          int middle = (lower+upper)/2;
          if (temperature(middle) > t) {
            upper = middle;
          } else {
            lower = middle;
          }
        }
        location.it0 = lower;
        if (log_coordinates) {
          location.ft = (log(t)-log_temperature(lower))/
                        (log_temperature(upper)-log_temperature(lower));
        } else {
          location.ft = (t-temperature(lower))/(temperature(upper)-temperature(lower));
        }
      }
    }
    return location;
  }

  KOKKOS_INLINE_FUNCTION
  Real Get(int kind, int group, const OpacityTableLocation &location) const {
    int id1 = (ndensity > 1) ? location.id0+1 : location.id0;
    int it1 = (ntemperature > 1) ? location.it0+1 : location.it0;
    Real result;
    // Geometric interpolation used to evaluate log() on all four corners of every
    // (group, kind) lookup.  The deck runs 20 groups against a two-material table and
    // Couple performs two kinds, so that was ~320 log plus 80 exp per cell, repeated on
    // every face by AddFluxes and again by the transport limiter.  The corner logarithms
    // are properties of the table, so they are stored once; only the final exp is left.
    // This mirrors what the IONMIX table already does with its own log_values.
    if (geometric_interpolation) {
      const Real l00 = log_values(kind, group, location.id0, location.it0);
      const Real l01 = log_values(kind, group, location.id0, it1);
      const Real l10 = log_values(kind, group, id1, location.it0);
      const Real l11 = log_values(kind, group, id1, it1);
      // A non-positive table entry has no logarithm; the loader marks those corners
      // with a sentinel so the zero-safe linear branch is selected exactly as before,
      // without having to load the linear values just to test their sign.
      if (l00 > kNonPositiveLog && l01 > kNonPositiveLog &&
          l10 > kNonPositiveLog && l11 > kNonPositiveLog) {
        const Real lower = (1.0-location.ft)*l00+location.ft*l01;
        const Real upper = (1.0-location.ft)*l10+location.ft*l11;
        result = exp((1.0-location.fd)*lower+location.fd*upper);
        if (kind == opacity_transport) return result*transport_scale;
        if (kind == opacity_absorption) return result*absorption_scale;
        return result*emission_scale;
      }
    }
    const Real v00 = values(kind, group, location.id0, location.it0);
    const Real v01 = values(kind, group, location.id0, it1);
    const Real v10 = values(kind, group, id1, location.it0);
    const Real v11 = values(kind, group, id1, it1);
    const Real lower = (1.0-location.ft)*v00+location.ft*v01;
    const Real upper = (1.0-location.ft)*v10+location.ft*v11;
    result = (1.0-location.fd)*lower+location.fd*upper;
    if (log_interpolation) result = exp(result);

    if (kind == opacity_transport) return result*transport_scale;
    if (kind == opacity_absorption) return result*absorption_scale;
    return result*emission_scale;
  }

  KOKKOS_INLINE_FUNCTION
  Real Get(int kind, int group, Real code_density, Real code_temperature) const {
    return Get(kind, group, Locate(code_density, code_temperature));
  }
};

//----------------------------------------------------------------------------------------
//! \class OpacityTable
//! \brief Reads and owns a FLASH-style multigroup opacity lookup table.

class OpacityTable {
 public:
  OpacityTable(ParameterInput *pin, int expected_groups,
               const DualArray1D<Real> &expected_group_bounds);
  OpacityTable(ParameterInput *pin, int expected_groups,
               const DualArray1D<Real> &expected_group_bounds,
               const std::string &input_block, const std::string &parameter_prefix);
  ~OpacityTable() = default;

  OpacityTableDevice DeviceData() const;

  void BuildLogAxes();

 private:
  int ndensity_;
  int ntemperature_;
  int ngroups_;
  bool log_interpolation_;
  bool geometric_interpolation_;
  bool log_coordinates_;
  Real density_scale_;
  Real temperature_scale_;
  Real transport_scale_;
  Real absorption_scale_;
  Real emission_scale_;

  DualArray1D<Real> density_;
  DualArray1D<Real> temperature_;
  DualArray1D<Real> log_density_;
  DualArray1D<Real> log_temperature_;
  DualArray4D<Real> values_;
  DualArray4D<Real> log_values_;
};

//----------------------------------------------------------------------------------------
//! \struct MixedOpacityTableDevice
//! \brief Partial-density, mass-weighted additive multi-material opacity closure.
//!
//! kappa_mix = sum_s Y_s kappa_s(rho Y_s, Te), which recovers every pure-material limit.

struct MixedOpacityTableLocation {
  OpacityTableLocation material[materials::kMaxMaterials];
  Real mass_fraction[materials::kMaxMaterials] = {};
  int count = 2;
};

struct MixedOpacityTableDevice {
  // The first two remain named for the two-material call sites; index 2 onward follow.
  OpacityTableDevice material0;
  OpacityTableDevice material1;
  OpacityTableDevice extra_material[materials::kMaxMaterials-2];
  int nmaterials = 2;

  KOKKOS_INLINE_FUNCTION
  const OpacityTableDevice &Material(const int index) const {
    if (index == 0) return material0;
    if (index == 1) return material1;
    return extra_material[index-2];
  }

  KOKKOS_INLINE_FUNCTION
  MixedOpacityTableLocation Locate(
      Real density, Real temperature,
      const materials::MaterialComposition &mix) const {
    MixedOpacityTableLocation location;
    location.count = mix.count;
    for (int n = 0; n < mix.count; ++n) {
      location.mass_fraction[n] = mix.y[n];
      if (mix.y[n] > 0.0) {
        location.material[n] = Material(n).Locate(density*mix.y[n], temperature);
      }
    }
    return location;
  }

  KOKKOS_INLINE_FUNCTION
  MixedOpacityTableLocation Locate(
      Real density, Real temperature, Real y0_in) const {
    MixedOpacityTableLocation location;
    location.count = 2;
    const Real y0 = fmin(fmax(y0_in, 0.0), 1.0);
    location.mass_fraction[0] = y0;
    location.mass_fraction[1] = 1.0-y0;
    if (y0 > 0.0) {
      location.material[0] = material0.Locate(density*y0, temperature);
    }
    if (y0 < 1.0) {
      location.material[1] = material1.Locate(density*(1.0-y0), temperature);
    }
    return location;
  }

  KOKKOS_INLINE_FUNCTION
  Real Get(int kind, int group, const MixedOpacityTableLocation &location) const {
    Real result = 0.0;
    for (int n = 0; n < location.count; ++n) {
      if (location.mass_fraction[n] > 0.0) {
        result += location.mass_fraction[n]*
                  Material(n).Get(kind, group, location.material[n]);
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real Get(int kind, int group, Real density, Real temperature, Real y0_in) const {
    return Get(kind, group, Locate(density, temperature, y0_in));
  }

  KOKKOS_INLINE_FUNCTION
  Real Get(int kind, int group, Real density, Real temperature,
           const materials::MaterialComposition &mix) const {
    return Get(kind, group, Locate(density, temperature, mix));
  }
};

class MixedOpacityTable {
 public:
  MixedOpacityTable(ParameterInput *pin, int expected_groups,
                    const DualArray1D<Real> &expected_group_bounds,
                    int nmaterials = 2);
  ~MixedOpacityTable() = default;

  MixedOpacityTableDevice DeviceData() const;

 private:
  int nmaterials_ = 2;
  std::vector<std::unique_ptr<OpacityTable>> tables_;
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_OPACITY_TABLE_HPP_
