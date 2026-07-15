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
#include "laser/laser.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "tasklist/task_list.hpp"

namespace laser {

void Laser::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID location = pmy_pack_->pmhd->id.srctrms;
  TaskID dependency = pmy_pack_->pmhd->id.duale;
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
  // Keep cumulative deposited energy (component 1), but clear per-stage diagnostics.
  auto data = cell_data;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  par_for("laser_clear_stage", DevExeSpace(), 0, nmb1, 0, 3,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    int component = (n == 0) ? 0 : n + 1;
    data(m, component, k, j, i) = 0.0;
  });
  Kokkos::deep_copy(device_diagnostics_, 0.0);
  Kokkos::deep_copy(device_counters_, 0);
  diagnostics_ = LaserDiagnostics();
  return TaskStatus::complete;
}

TaskStatus Laser::TraceAndDeposit(Driver *pdrive, int stage) {
  // Straight-ray transport is supplied by laser_trace.cpp in the next gated layer.
  return TaskStatus::complete;
}

TaskStatus Laser::ApplySource(Driver *pdrive, int stage) {
  // The skeleton intentionally performs no update; zero-power runs are exactly inert.
  return TaskStatus::complete;
}

TaskStatus Laser::ClearBuffers(Driver *pdrive, int stage) {
  return TaskStatus::complete;
}

} // namespace laser
