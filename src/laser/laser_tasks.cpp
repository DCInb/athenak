//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_tasks.cpp
//! \brief Laser task assembly and 2T source coupling.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "laser/laser.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "tasklist/task_list.hpp"

namespace laser {

void Laser::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  // Insert between the carrier fluid's dual-energy and source-term tasks.
  TaskID location = use_mhd_fluid_ ? pmy_pack_->pmhd->id.srctrms
                                   : pmy_pack_->phydro->id.srctrms;
  TaskID dependency = use_mhd_fluid_ ? pmy_pack_->pmhd->id.duale
                                     : pmy_pack_->phydro->id.duale;
  id.initialize = tl["stagen"]->InsertTask(
      &Laser::InitializeStep, this, dependency, location);
  dependency = id.initialize;
  id.trace = tl["stagen"]->InsertTask(
      &Laser::TraceAndDeposit, this, dependency, location);
  dependency = id.trace;
  id.apply = tl["stagen"]->InsertTask(
      &Laser::ApplySource, this, dependency, location);
  dependency = id.apply;
  id.clear = tl["stagen"]->InsertTask(
      &Laser::ClearBuffers, this, dependency, location);
}

TaskStatus Laser::InitializeStep(Driver *pdrive, int stage) {
  stage_has_power_ = UpdateBeamPowers(pmy_pack_->pmesh->time, pmy_pack_->pmesh->dt);
  diagnostics_ = LaserDiagnostics();
  actual_transport_iterations_ = 0;
  if (!stage_has_power_) {
    transport_state_ = LaserTransportState::finished;
    if (!instantaneous_data_zero_) {
      ClearInstantaneousData(false);
      instantaneous_data_zero_ = true;
    }
    return TaskStatus::complete;
  }

  instantaneous_data_zero_ = false;
  RefreshGlobalBlockInfo();
  ClearInstantaneousData(stage == 1);
  Kokkos::deep_copy(device_diagnostics_, 0.0);
  Kokkos::deep_copy(device_counters_, 0);
  transport_state_ = LaserTransportState::trace_local;
  mpi_wave_ = 0;
  InitializeRays();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Clear instantaneous diagnostics while retaining cumulative deposited energy. At the
//! first stage, save the old cumulative field for the low-storage RK recurrence.

void Laser::ClearInstantaneousData(bool capture_cumulative_start) {
  auto data = cell_data;
  auto energy_start = cumulative_energy_start_;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  par_for("laser_clear_stage", DevExeSpace(), 0, nmb1, 0, ncell_data-1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    if (n != 1) data(m, n, k, j, i) = 0.0;
    if (capture_cumulative_start && n == 1) {
      energy_start(m, 0, k, j, i) = data(m, 1, k, j, i);
    }
  });
}

TaskStatus Laser::TraceAndDeposit(Driver *pdrive, int stage) {
  if (!stage_has_power_) return TaskStatus::complete;
  return AdvanceDistributedTransport();
}

TaskStatus Laser::ApplySource(Driver *pdrive, int stage) {
  if (!stage_has_power_) return TaskStatus::complete;
  Real beta_dt = pdrive->beta[stage-1]*pmy_pack_->pmesh->dt;
  Real gam0 = pdrive->gam0[stage-1];
  Real gam1 = pdrive->gam1[stage-1];
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int iele = electron_index_;
  bool heat_electrons = deposition_target_ == DepositionTarget::electron;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto u0 = FluidCons();
  auto data = cell_data;
  auto energy_start = cumulative_energy_start_;
  par_for("laser_apply_source", DevExeSpace(), 0, nmb1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real deposited_energy = beta_dt*data(m, 0, k, j, i);
    u0(m, IEN, k, j, i) += deposited_energy;
    if (heat_electrons) u0(m, iele, k, j, i) += deposited_energy;
    data(m, 1, k, j, i) =
        gam0*data(m, 1, k, j, i) +
        gam1*energy_start(m, 0, k, j, i) + deposited_energy;
  });
  return TaskStatus::complete;
}

TaskStatus Laser::ClearBuffers(Driver *pdrive, int stage) {
  return TaskStatus::complete;
}

} // namespace laser
