#ifndef TWO_TEMPERATURE_OPACITY_TABLE_HPP_
#define TWO_TEMPERATURE_OPACITY_TABLE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.hpp
//! \brief Density- and electron-temperature-dependent multigroup opacity tables.

#include "athena.hpp"

#include <string>

class ParameterInput;

namespace two_temperature {

enum OpacityKind {
  opacity_transport = 0,
  opacity_absorption = 1,
  opacity_emission = 2
};

//----------------------------------------------------------------------------------------
//! \struct OpacityTableDevice
//! \brief Lightweight device-copyable view of a tabulated opacity model.

struct OpacityTableDevice {
  DvceArray1D<Real> density;
  DvceArray1D<Real> temperature;
  DvceArray4D<Real> values;
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
  Real Get(int kind, int group, Real code_density, Real code_temperature) const {
    Real d = code_density*density_scale;
    Real t = code_temperature*temperature_scale;

    int id0 = 0;
    Real fd = 0.0;
    if (ndensity > 1) {
      if (d >= density(ndensity-1)) {
        id0 = ndensity-2;
        fd = 1.0;
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
        id0 = lower;
        if (log_coordinates) {
          fd = (log(d)-log(density(lower)))/
               (log(density(upper))-log(density(lower)));
        } else {
          fd = (d-density(lower))/(density(upper)-density(lower));
        }
      }
    }

    int it0 = 0;
    Real ft = 0.0;
    if (ntemperature > 1) {
      if (t >= temperature(ntemperature-1)) {
        it0 = ntemperature-2;
        ft = 1.0;
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
        it0 = lower;
        if (log_coordinates) {
          ft = (log(t)-log(temperature(lower)))/
               (log(temperature(upper))-log(temperature(lower)));
        } else {
          ft = (t-temperature(lower))/(temperature(upper)-temperature(lower));
        }
      }
    }

    int id1 = (ndensity > 1) ? id0+1 : id0;
    int it1 = (ntemperature > 1) ? it0+1 : it0;
    const Real v00 = values(kind, group, id0, it0);
    const Real v01 = values(kind, group, id0, it1);
    const Real v10 = values(kind, group, id1, it0);
    const Real v11 = values(kind, group, id1, it1);
    Real result;
    if (geometric_interpolation && v00 > 0.0 && v01 > 0.0 &&
        v10 > 0.0 && v11 > 0.0) {
      const Real lower = (1.0-ft)*log(v00)+ft*log(v01);
      const Real upper = (1.0-ft)*log(v10)+ft*log(v11);
      result = exp((1.0-fd)*lower+fd*upper);
    } else {
      const Real lower = (1.0-ft)*v00+ft*v01;
      const Real upper = (1.0-ft)*v10+ft*v11;
      result = (1.0-fd)*lower+fd*upper;
      if (log_interpolation) result = exp(result);
    }

    if (kind == opacity_transport) return result*transport_scale;
    if (kind == opacity_absorption) return result*absorption_scale;
    return result*emission_scale;
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
  DualArray4D<Real> values_;
};

//----------------------------------------------------------------------------------------
//! \struct MixedOpacityTableDevice
//! \brief Partial-density, mass-weighted additive two-material opacity closure.

struct MixedOpacityTableDevice {
  OpacityTableDevice material0;
  OpacityTableDevice material1;

  KOKKOS_INLINE_FUNCTION
  Real Get(int kind, int group, Real density, Real temperature, Real y0_in) const {
    const Real y0 = fmin(fmax(y0_in, 0.0), 1.0);
    Real result = 0.0;
    if (y0 > 0.0) result += y0*material0.Get(kind, group, density*y0, temperature);
    if (y0 < 1.0) {
      result += (1.0-y0)*material1.Get(
          kind, group, density*(1.0-y0), temperature);
    }
    return result;
  }
};

class MixedOpacityTable {
 public:
  MixedOpacityTable(ParameterInput *pin, int expected_groups,
                    const DualArray1D<Real> &expected_group_bounds);
  ~MixedOpacityTable() = default;

  MixedOpacityTableDevice DeviceData() const;

 private:
  OpacityTable material0_;
  OpacityTable material1_;
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_OPACITY_TABLE_HPP_
