"""CPU device test for unequal-grid tabular material mixing and LLF pressure."""

from pathlib import Path
import shutil

import pytest

import test_suite.testutils as testutils


def test_mixed_inverse_interval_energy_cache_source():
    local_source_root = Path(__file__).resolve().parent.parents[2]
    material_source = (
        local_source_root / "src/materials/material_mixture.hpp").read_text(
            encoding="utf-8")
    ionmix_source = (
        local_source_root / "src/materials/ionmix_two_temperature_table.hpp"
    ).read_text(encoding="utf-8")

    assert "struct IonmixEnergyIntervalCache" in ionmix_source
    assert "Kokkos::fma(fraction, upper" in ionmix_source
    cached_inverse = material_source.split(
        "ComponentAtTemperature MixtureComponentFromRhoSpecificEnergyCached(",
        1)[1].split(
            "ComponentAtTemperature MixtureComponentFromRhoSpecificEnergy(",
            1)[0]
    assert "MixedEnergyIntervalCache energy_cache;" in cached_inverse
    assert cached_inverse.count(
        "MixtureComponentEnergyFromCachedDensity(") == 1


test_directory = Path(__file__).resolve().parent
source_root = test_directory.parents[2]
driver = test_directory / "tabular_material_mixture_driver.cpp"
ch_table = test_directory / "tabular_material_mixture_ch_fixture.dat"
he_table = test_directory / "tabular_material_mixture_he_fixture.dat"
unit_build = Path("tabular_material_mixture_closure_unit_build")


def test_cached_flux_and_exchange_sources():
    flux_source = (source_root / "src/mhd/mhd_fluxes.cpp").read_text(
        encoding="utf-8")
    llf_source = (source_root / "src/mhd/rsolvers/llf_mhd.hpp").read_text(
        encoding="utf-8")
    exchange_source = (
        source_root / "src/two_temperature/two_temperature.cpp").read_text(
            encoding="utf-8")
    material_source = (
        source_root / "src/materials/material_mixture.hpp").read_text(
            encoding="utf-8")
    ionmix_source = (
        source_root / "src/materials/ionmix_two_temperature_table.hpp").read_text(
            encoding="utf-8")
    dual_energy_source = (
        source_root / "src/mhd/mhd_dual_energy.cpp").read_text(
            encoding="utf-8")
    radiation_source = (
        source_root / "src/two_temperature/thermal_radiation.cpp").read_text(
            encoding="utf-8")
    output_source = (source_root / "src/outputs/basetype_output.cpp").read_text(
        encoding="utf-8")
    assert "ptwo_temp->thermodynamics" in flux_source
    assert "ReconstructMaterialThermodynamicsX1" in flux_source
    assert "ReconstructMaterialThermodynamicsX2" in flux_source
    assert "ReconstructMaterialThermodynamicsX3" in flux_source
    assert "StateFromRhoSpecificEnergies" not in llf_source
    assert "StateTemperaturesFromRhoTotalEnergyTemperatureDifference" in (
        exchange_source)
    assert "ElectronHeatCapacityFraction(" in exchange_source
    assert "1.0+local_fe/local_fi" in exchange_source
    assert "if (exchange.used_fallback == 2)" in exchange_source
    assert exchange_source.count("materials::ionmix_energy_below_table") >= 6
    assert "iteration < 32" not in exchange_source
    initialize_source = exchange_source.split(
        "void TwoTemperature::Initialize", 1)[1].split(
            "void TwoTemperature::Sync", 1)[0]
    # Initialization consumes only floor component energies and query flags before
    # its final full state reconstruction.
    assert initialize_source.count("MinimumPressureEnergyState(") == 1
    assert initialize_source.count("StateFromRhoSpecificEnergies(") == 1
    assert "MinimumStateNoSound(" not in initialize_source
    sync_source = exchange_source.split(
        "void TwoTemperature::Sync", 1)[1].split(
            "void TwoTemperature::Exchange", 1)[0]
    # Sync needs only floor energies and the advected pressure partition from its
    # intermediate states. Its final state must remain the full cache-producing query.
    assert sync_source.count("MinimumPressureEnergyState(") == 1
    assert sync_source.count("PressureEnergyFromRhoSpecificEnergies(") == 1
    assert sync_source.count("StateFromRhoSpecificEnergies(") == 1
    assert "MinimumStateNoSound(" not in sync_source
    assert "StateFromRhoSpecificEnergiesNoSound(" not in sync_source
    # Dual-energy updates consume only pressure and floor energy.  Keep their hot
    # tabular paths on the reduced queries; Sync fills the complete thermodynamic cache.
    assert dual_energy_source.count(
        "PressureEnergyFromRhoSpecificEnergies(") == 2
    assert dual_energy_source.count("MinimumPressureEnergyState(") == 2
    assert "StateFromRhoSpecificEnergiesNoSound(" not in dual_energy_source
    assert "MinimumStateNoSound(" not in dual_energy_source
    assert "StateFromRhoSpecificEnergies(" not in dual_energy_source
    assert "MinimumState(" not in dual_energy_source
    material_exchange_source = exchange_source.split(
        "void TwoTemperature::Exchange", 1)[1].split(
            "void TwoTemperature::RefreshMaterialThermodynamics", 1)[0]
    # Exchange likewise uses its floor only to bound the two component energies.
    assert material_exchange_source.count("MinimumPressureEnergyState(") == 1
    assert "MinimumStateNoSound(" not in material_exchange_source
    assert material_exchange_source.count(
        "materials::MaterialTransientExchangeState exchange") == 1
    assert material_exchange_source.count(
        "StateTemperaturesFromRhoTotalEnergyTemperatureDifference(") == 1
    assert "StateFromRhoTotalEnergyTemperatureDifference(" not in (
        material_exchange_source)
    exchange_selection = material_exchange_source.split(
        "const materials::MaterialTransientExchangeState exchange =", 1)[1].split(
            "query_flags |= exchange.temperatures.query_flags", 1)[0]
    assert exchange_selection.count(
        "mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(") == 1
    assert "radiation_refreshes_cache" not in exchange_selection
    assert material_exchange_source.count("StateFromRhoTemperatures(") == 1
    assert material_exchange_source.count(
        "StoreMaterialTemperaturesAndFlags(") == 1
    assert material_exchange_source.count("StoreMaterialThermodynamics(") == 1
    store_selection = material_exchange_source.split(
        "if (!radiation_refreshes_cache) {", 1)[1].split(
            "const Real local_fe = mixture.ElectronHeatCapacityFraction(y0);",
            1)[0]
    full_store_branch, radiation_store_branch = store_selection.split(
        "return;", 1)
    assert full_store_branch.count("StateFromRhoTemperatures(") == 1
    assert full_store_branch.count("StoreMaterialThermodynamics(") == 1
    assert "StoreMaterialTemperaturesAndFlags(" not in full_store_branch
    assert "StateFromRhoTemperatures(" not in radiation_store_branch
    assert "StoreMaterialThermodynamics(" not in radiation_store_branch
    assert radiation_store_branch.count(
        "StoreMaterialTemperaturesAndFlags(") == 1
    bounded_recovery_source = material_exchange_source.split(
        "if (eion_new != density*exchange.ion_specific_internal_energy) {",
        1)[1].split("query_flags |= state.query_flags;", 1)[0]
    assert bounded_recovery_source.count(
        "const materials::MaterialThermodynamicState bounded_state") == 1
    assert bounded_recovery_source.count(
        "mixture.StateFromRhoSpecificEnergiesNoSound(") == 1
    assert "PressureEnergyFromRhoSpecificEnergies(" not in bounded_recovery_source
    table_temperature_source = ionmix_source.split(
        "IonmixTemperatureState TemperatureFromRhoTemperature(", 1)[1].split(
            "IonmixComponentState ComponentFromRhoTemperature(", 1)[0]
    assert "EvaluateWithLocations(" not in table_temperature_source
    assert "StateAtLocations(" not in table_temperature_source
    assert table_temperature_source.count(
        "const AxisLocation density = Locate(") == 1
    assert table_temperature_source.count(
        "const AxisLocation temperature = Locate(") == 1
    assert table_temperature_source.index(
        "const AxisLocation density = Locate(") < table_temperature_source.index(
            "const AxisLocation temperature = Locate(")
    table_pressure_energy_source = ionmix_source.split(
        "IonmixPressureEnergyState PressureEnergyFromRhoTemperature(",
        1)[1].split(
            "IonmixTemperatureState TemperatureFromRhoTemperature(", 1)[0]
    assert table_pressure_energy_source.count(
        "const AxisLocation density = Locate(") == 1
    assert table_pressure_energy_source.count(
        "const AxisLocation temperature = Locate(") == 1
    assert table_pressure_energy_source.count("EvaluateWithLocations(") == 4
    assert "bounded_log_coordinate" not in table_pressure_energy_source
    table_pressure_energy_order = [
        "const AxisLocation density = Locate(",
        "const AxisLocation temperature = Locate(",
        "EvaluateWithLocations(ion_pressure",
        "EvaluateWithLocations(ion_specific_internal_energy",
        "EvaluateWithLocations(electron_pressure",
        "EvaluateWithLocations(electron_specific_internal_energy",
    ]
    assert [table_pressure_energy_source.index(token)
            for token in table_pressure_energy_order] == sorted(
                table_pressure_energy_source.index(token)
                for token in table_pressure_energy_order)
    native_pressure_energy_source = ionmix_source.split(
        "IonmixPressureEnergyState PressureEnergyFromRhoMinimumTemperature(",
        1)[1].split(
            "IonmixPressureEnergyState PressureEnergyFromRhoTemperature(", 1)[0]
    assert native_pressure_energy_source.count(
        "const AxisLocation density = Locate(") == 1
    assert "if (minimum_temperature_round_trips_exactly == 0)" in (
        native_pressure_energy_source)
    assert "PressureEnergyFromRhoTemperature(" in native_pressure_energy_source
    assert "const AxisLocation temperature = Locate(" not in (
        native_pressure_energy_source)
    assert "temperature.lower = 0" in native_pressure_energy_source
    assert "temperature.fraction = 0.0" in native_pressure_energy_source
    assert "temperature.bounded_log_coordinate = log_temperature_kelvin(0)" in (
        native_pressure_energy_source)
    assert "temperature.query_flags = ionmix_query_in_bounds" in (
        native_pressure_energy_source)
    native_pressure_energy_order = [
        "const AxisLocation density = Locate(",
        "EvaluateWithLocations(ion_pressure",
        "EvaluateWithLocations(ion_specific_internal_energy",
        "EvaluateWithLocations(electron_pressure",
        "EvaluateWithLocations(electron_specific_internal_energy",
    ]
    assert [native_pressure_energy_source.index(token)
            for token in native_pressure_energy_order] == sorted(
                native_pressure_energy_source.index(token)
                for token in native_pressure_energy_order)
    material_temperature_source = material_source.split(
        "ComponentTemperatureState SpeciesTemperatureFromRhoTemperature(",
        1)[1].split(
            "MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperatures(",
            1)[0]
    assert "MixtureComponentFromRhoTemperature(" not in material_temperature_source
    assert ".ComponentFromRhoTemperature(" not in material_temperature_source
    transient_location_source = material_source.split(
        "const ComponentTemperatureState canonical_ion =", 1)[1].split(
            "} else {", 1)[0]
    assert transient_location_source.count(
        "MixtureTemperatureFromRhoTemperature(") == 2
    assert "MixtureComponentFromRhoTemperature(" not in transient_location_source
    # The mixed inverse reuses each material's prepared density location across all
    # bisection probes and then across the paired ion/electron inversions.
    assert "struct SpeciesDensityCache" in material_source
    assert "struct MixedDensityCache" in material_source
    prepared_table_source = ionmix_source.split(
        "IonmixComponentState ComponentFromPreparedDensityTemperature(",
        1)[1].split("IonmixTemperatureState TemperatureFromRhoTemperature(", 1)[0]
    assert "const IonmixDensityLocation &prepared_density" in (
        prepared_table_source)
    assert "const Real density_fraction" not in prepared_table_source

    species_cached_source = material_source.split(
        "ComponentAtTemperature SpeciesComponentFromCachedDensity(",
        1)[1].split(
            "ComponentAtTemperature MixtureComponentFromCachedDensity(",
            1)[0]
    assert species_cached_source.count("PrepareDensityLocation(") == 1
    assert species_cached_source.count("if (cache.status == 0)") == 1
    assert species_cached_source.count("if (cache.status == 1)") == 1
    assert species_cached_source.count("if (cache.status == 3)") == 2
    assert species_cached_source.count(
        "table.ComponentFromPreparedDensityTemperature(") == 1

    mixture_cached_source = material_source.split(
        "ComponentAtTemperature MixtureComponentFromCachedDensity(",
        1)[1].split(
            "ComponentTemperatureState SpeciesTemperatureFromRhoTemperature(",
            1)[0]
    assert mixture_cached_source.count(
        "SpeciesComponentFromCachedDensity(") == 2
    assert mixture_cached_source.index("material0_table") < (
        mixture_cached_source.index("material1_table"))

    cached_inverse_source = material_source.split(
        "ComponentAtTemperature MixtureComponentFromRhoSpecificEnergyCached(",
        1)[1].split(
            "ComponentAtTemperature MixtureComponentFromRhoSpecificEnergy(",
            1)[0]
    cached_signature = cached_inverse_source.split("{", 1)[0]
    assert "MixedDensityCache &cache" in cached_signature
    assert cached_inverse_source.count(
        "MixtureComponentFromCachedDensity(") == 3
    assert cached_inverse_source.count(
        "MixtureComponentEnergyFromCachedDensity(") == 1
    assert "MixedEnergyIntervalCache energy_cache;" in cached_inverse_source
    assert "iteration < 48; ++iteration" in cached_inverse_source

    single_inverse_source = material_source.split(
        "ComponentAtTemperature MixtureComponentFromRhoSpecificEnergy(",
        1)[1].split(
            "int MixtureComponentSpecificEnergyQueryFlags(",
            1)[0]
    assert single_inverse_source.count("MixedDensityCache cache;") == 1
    assert single_inverse_source.count(
        "MixtureComponentFromRhoSpecificEnergyCached(") == 1

    reduced_flag_source = material_source.split(
        "int MixtureComponentSpecificEnergyQueryFlags(", 1)[1].split(
            "ComponentPairAtTemperature MixtureComponentsFromRhoSpecificEnergies(",
            1)[0]
    assert reduced_flag_source.count("MixedDensityCache cache;") == 1
    assert "for (int iteration = 0;" not in reduced_flag_source

    pair_inverse_source = material_source.split(
        "ComponentPairAtTemperature MixtureComponentsFromRhoSpecificEnergies(",
        1)[1].split("Real TabularElectronNumberPerAtomicMass(", 1)[0]
    assert pair_inverse_source.count("MixedDensityCache cache;") == 1
    assert pair_inverse_source.count(
        "MixtureComponentFromRhoSpecificEnergyCached(") == 2
    assert pair_inverse_source.index("IonmixComponent::ion") < (
        pair_inverse_source.index("IonmixComponent::electron"))
    floor_pressure_energy_source = material_source.split(
        "MaterialPressureEnergyState MinimumPressureEnergyState(",
        1)[1].split("MaterialThermodynamicState MinimumState(", 1)[0]
    assert floor_pressure_energy_source.count(
        "TabularPressureEnergyFromRhoTemperature(") == 3
    assert floor_pressure_energy_source.count(
        "TabularPressureEnergyFromRhoNativeMinimum(") == 1
    assert "TabularPressureEnergyFromRhoTemperatures(" not in (
        floor_pressure_energy_source)
    assert floor_pressure_energy_source.index(
        "TabularPressureEnergyFromRhoNativeMinimum(") < (
            floor_pressure_energy_source.index(
                "TabularPressureEnergyFromRhoTemperature("))
    paired_mixture_source = material_source.split(
        "MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperature(",
        1)[1].split(
            "MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperatures(",
            1)[0]
    assert paired_mixture_source.count(
        "SpeciesPressureEnergyFromRhoTemperature(") == 2
    assert paired_mixture_source.index("material0_table") < (
        paired_mixture_source.index("material1_table"))
    paired_species_source = material_source.split(
        "MaterialPressureEnergyState SpeciesPressureEnergyFromRhoTemperature(",
        1)[1].split(
            "MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperature(",
            1)[0]
    assert paired_species_source.count("pressure_scale") == 3
    assert paired_species_source.count("ionmix_density_below_table") == 1
    assert paired_species_source.count(
        "table.PressureEnergyFromRhoTemperature(") == 1
    native_species_source = material_source.split(
        "MaterialPressureEnergyState "
        "SpeciesPressureEnergyFromRhoMinimumTemperature(", 1)[1].split(
            "MaterialPressureEnergyState "
            "TabularPressureEnergyFromRhoNativeMinimum(", 1)[0]
    assert native_species_source.count("pressure_scale") == 3
    assert native_species_source.count("ionmix_density_below_table") == 1
    assert native_species_source.count(
        "table.PressureEnergyFromRhoMinimumTemperature(") == 1
    native_mixture_source = material_source.split(
        "MaterialPressureEnergyState TabularPressureEnergyFromRhoNativeMinimum(",
        1)[1].split("MaterialPressureEnergyState IdealPressureEnergy", 1)[0]
    assert native_mixture_source.count(
        "SpeciesPressureEnergyFromRhoMinimumTemperature(") == 2
    assert native_mixture_source.count(
        "SpeciesPressureEnergyFromRhoTemperature(") == 2
    assert native_mixture_source.index("material0_table") < (
        native_mixture_source.index("material1_table"))
    # The prepared token is forward-only; pure endpoints retain the direct inverse API.
    assert "PreparedDensitySpecificEnergy" not in ionmix_source
    assert "SpecificEnergyFromPreparedDensity" not in ionmix_source
    assert material_exchange_source.count("pradiation->Couple(") == 1
    coupling_tail = material_exchange_source.split("pradiation->Couple", 1)[1]
    assert coupling_tail.count("RefreshMaterialThermodynamics(") == 1
    refresh_source = exchange_source.split(
        "void TwoTemperature::RefreshMaterialThermodynamics", 1)[1].split(
            "void TwoTemperature::CloseBiermannStage", 1)[0]
    assert refresh_source.count("StateFromRhoSpecificEnergies(") == 1
    # Radiation coupling consumes only the electron floor energy.
    assert radiation_source.count("mixture.MinimumPressureEnergyState(") == 1
    assert "mixture.MinimumStateNoSound(" not in radiation_source
    assert "density, y0, pressure_floor, temperature_floor" in radiation_source
    assert "eele_old-eele_floor-negative" in radiation_source
    assert '"eos_flags"' in output_source
    assert "TwoTemperature::eos_query_flags" in output_source


def build_driver():
    athena_build = Path.cwd().parent
    kokkos_package = athena_build / "cmake_packages" / "Kokkos"
    cmake_source = unit_build / "source"
    cmake_source.mkdir(parents=True)
    cmake_text = f"""
cmake_minimum_required(VERSION 3.16)
project(tabular_material_mixture_closure_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Kokkos CONFIG REQUIRED
             PATHS \"{kokkos_package}\" NO_DEFAULT_PATH)
add_executable(tabular_material_mixture_driver
  \"{driver}\"
  \"{source_root / 'src/materials/ionmix_two_temperature_table.cpp'}\")
target_include_directories(tabular_material_mixture_driver PRIVATE
  \"{source_root / 'src'}\" \"{athena_build}\")
target_compile_definitions(tabular_material_mixture_driver PRIVATE
  KOKKOS_DEPENDENCE)
target_link_libraries(tabular_material_mixture_driver PRIVATE Kokkos::kokkos)
"""
    (cmake_source / "CMakeLists.txt").write_text(cmake_text, encoding="ascii")
    assert testutils.run_command([
        "cmake", "-S", str(cmake_source), "-B", str(unit_build / "build")])
    assert testutils.run_command([
        "cmake", "--build", str(unit_build / "build"), "--parallel", "2"])
    return unit_build / "build" / "tabular_material_mixture_driver"


def test_run():
    try:
        executable = build_driver()
        assert testutils.run_command([
            str(executable), str(ch_table), str(he_table)], timeout=60.0)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        shutil.rmtree(unit_build, ignore_errors=True)
