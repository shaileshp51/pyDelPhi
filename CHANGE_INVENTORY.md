# pyDelPhi v0.2.0 to v0.3.0 change inventory

## Summary

| File | Change summary |
|---|---:|
| `PKG-INFO` | `185 +-` |
| `README.md` | `172 +-` |
| `pydelphi/__init__.py` | `2 +-` |
| `pydelphi/app/delphi.py` | `1663 +++----------` |
| `pydelphi/config/logging_config.py` | `4 +-` |
| `pydelphi/constants/__init__.py` | `3 +` |
| `pydelphi/constants/application.py` | `37 +` |
| `pydelphi/data/references.bib` | `30 +` |
| `pydelphi/energy/calculator.py` | `57 +-` |
| `pydelphi/energy/coulombic.py` | `9 +-` |
| `pydelphi/energy/coulombic_nl.py` | `2 +-` |
| `pydelphi/energy/energy_models.py` | `2 +` |
| `pydelphi/energy/lj.py` | `2 +-` |
| `pydelphi/energy/nonpolar.py` | `741 +-----` |
| `pydelphi/energy/reactionfield_iscm.py` | `11 +-` |
| `pydelphi/examples/5tif/README.md` | `132 -` |
| `pydelphi/foundation/context.py` | `626 ++++-` |
| `pydelphi/foundation/enums.py` | `35 +` |
| `pydelphi/foundation/platforms.py` | `2 +-` |
| `pydelphi/scripts/pydelphi_help.py` | `316 ++-` |
| `pydelphi/scripts/pydelphi_static.py` | `2 +-` |
| `pydelphi/site/writesite.py` | `900 +++++--` |
| `pydelphi/solver/pb/common_pb.py` | `28 +-` |
| `pydelphi/solver/pb/nwt/nonlinear_pb.py` | `127 +-` |
| `pydelphi/solver/pb/sor/nonlinear_pb.py` | `307 ++-` |
| `pydelphi/solver/rpb/common_rpb.py` | `75 -` |
| `pydelphi/solver/rpb/sor/linear_rpb.py` | `83 +-` |
| `pydelphi/solver/shared/sor/base.py` | `243 +-` |
| `pydelphi/solver/solver.py` | `13 +` |
| `pydelphi/space/__init__.py` | `9 +` |
| `pydelphi/space/core/__init__.py` | `9 +` |
| `pydelphi/space/core/gaussian.py` | `6 +-` |
| `pydelphi/space/core/grid_charger.py` | `9 +` |
| `pydelphi/space/core/vdw/__init__.py` | `9 +` |
| `pydelphi/space/core/vdw/cuber.py` | `9 +` |
| `pydelphi/space/core/vdw/helper.py` | `55 +-` |
| `pydelphi/space/core/vdw/helper_cpu.py` | `9 +` |
| `pydelphi/space/core/vdw/helper_cuda.py` | `33 +-` |
| `pydelphi/space/core/vdw/vertex_indexer.py` | `9 +` |
| `pydelphi/space/core/vdw_eps_initizer.py` | `89 +-` |
| `pydelphi/space/core/voxelizer.py` | `8 +` |
| `pydelphi/space/gcs_surface.py` | `8 +` |
| `pydelphi/space/space.py` | `72 +-` |
| `pydelphi/space/surface.py` | `27 +-` |
| `pydelphi/space/vdwms.py` | `2581 +++-----------------` |
| `pydelphi/utils/cuda/double.py` | `9 -` |
| `pydelphi/utils/cuda/single.py` | `8 -` |
| `pydelphi/utils/io/format/assorted/custom_reader.py` | `788 +++++-` |
| `pydelphi/utils/io/format/assorted/custom_writer.py` | `453 +++-` |
| `pydelphi/utils/io/format/pdb_pqr.py` | `491 +++-` |
| `pydelphi/utils/io/inproc.py` | `1395 +++++++++--` |
| `.../param_definitions/calculation_params.py` | `84 +-` |
| `.../param_definitions/infile_params.py` | `338 ++-` |
| `.../param_definitions/outfile_params.py` | `254 +-` |
| `.../inproc_helpers/param_definitions/parameters.py` | `414 +++-` |
| `.../param_definitions/solvent_params.py` | `68 +` |
| `pydelphi/utils/io/inproc_helpers/props_assigner.py` | `1201 ++++++++-` |
| `pydelphi/utils/io/readers.py` | `356 +--` |
| `pydelphi/utils/io/writers.py` | `14 +-` |
| `pyproject.toml` | `59 +-` |

**Overall:** 60 files changed, 8,554 insertions, and 6,129 deletions.

## Changed files

| Status | File |
|:---:|---|
| `M` | `PKG-INFO` |
| `M` | `README.md` |
| `M` | `pydelphi/__init__.py` |
| `M` | `pydelphi/app/delphi.py` |
| `M` | `pydelphi/config/logging_config.py` |
| `M` | `pydelphi/constants/__init__.py` |
| `M` | `pydelphi/constants/application.py` |
| `M` | `pydelphi/data/references.bib` |
| `M` | `pydelphi/energy/calculator.py` |
| `M` | `pydelphi/energy/coulombic.py` |
| `M` | `pydelphi/energy/coulombic_nl.py` |
| `M` | `pydelphi/energy/energy_models.py` |
| `M` | `pydelphi/energy/lj.py` |
| `M` | `pydelphi/energy/nonpolar.py` |
| `M` | `pydelphi/energy/reactionfield_iscm.py` |
| `M` | `pydelphi/examples/5tif/README.md` |
| `M` | `pydelphi/foundation/context.py` |
| `M` | `pydelphi/foundation/enums.py` |
| `M` | `pydelphi/foundation/platforms.py` |
| `M` | `pydelphi/scripts/pydelphi_help.py` |
| `M` | `pydelphi/scripts/pydelphi_static.py` |
| `M` | `pydelphi/site/writesite.py` |
| `M` | `pydelphi/solver/pb/common_pb.py` |
| `M` | `pydelphi/solver/pb/nwt/nonlinear_pb.py` |
| `M` | `pydelphi/solver/pb/sor/nonlinear_pb.py` |
| `M` | `pydelphi/solver/rpb/common_rpb.py` |
| `M` | `pydelphi/solver/rpb/sor/linear_rpb.py` |
| `M` | `pydelphi/solver/shared/sor/base.py` |
| `M` | `pydelphi/solver/solver.py` |
| `M` | `pydelphi/space/__init__.py` |
| `M` | `pydelphi/space/core/__init__.py` |
| `M` | `pydelphi/space/core/gaussian.py` |
| `M` | `pydelphi/space/core/grid_charger.py` |
| `M` | `pydelphi/space/core/vdw/__init__.py` |
| `M` | `pydelphi/space/core/vdw/cuber.py` |
| `M` | `pydelphi/space/core/vdw/helper.py` |
| `M` | `pydelphi/space/core/vdw/helper_cpu.py` |
| `M` | `pydelphi/space/core/vdw/helper_cuda.py` |
| `M` | `pydelphi/space/core/vdw/vertex_indexer.py` |
| `M` | `pydelphi/space/core/vdw_eps_initizer.py` |
| `M` | `pydelphi/space/core/voxelizer.py` |
| `M` | `pydelphi/space/gcs_surface.py` |
| `M` | `pydelphi/space/space.py` |
| `M` | `pydelphi/space/surface.py` |
| `M` | `pydelphi/space/vdwms.py` |
| `M` | `pydelphi/utils/cuda/double.py` |
| `M` | `pydelphi/utils/cuda/single.py` |
| `M` | `pydelphi/utils/io/format/assorted/custom_reader.py` |
| `M` | `pydelphi/utils/io/format/assorted/custom_writer.py` |
| `M` | `pydelphi/utils/io/format/pdb_pqr.py` |
| `M` | `pydelphi/utils/io/inproc.py` |
| `M` | `pydelphi/utils/io/inproc_helpers/param_definitions/calculation_params.py` |
| `M` | `pydelphi/utils/io/inproc_helpers/param_definitions/infile_params.py` |
| `M` | `pydelphi/utils/io/inproc_helpers/param_definitions/outfile_params.py` |
| `M` | `pydelphi/utils/io/inproc_helpers/param_definitions/parameters.py` |
| `M` | `pydelphi/utils/io/inproc_helpers/param_definitions/solvent_params.py` |
| `M` | `pydelphi/utils/io/inproc_helpers/props_assigner.py` |
| `M` | `pydelphi/utils/io/readers.py` |
| `M` | `pydelphi/utils/io/writers.py` |
| `M` | `pyproject.toml` |

## Untracked files

| File |
|---|
| `CHANGE_INVENTORY.md` |
| `environment-traj.yml` |
| `environment.yml` |
| `pydelphi/app/core/atom_materializer.py` |
| `pydelphi/app/core/cuda_utils.py` |
| `pydelphi/app/core/focusing_prep.py` |
| `pydelphi/app/core/frc_writer.py` |
| `pydelphi/app/core/maps_memory.py` |
| `pydelphi/app/core/output_maps.py` |
| `pydelphi/app/core/pbe_runner.py` |
| `pydelphi/app/core/policy.py` |
| `pydelphi/app/core/rpbe_runner.py` |
| `pydelphi/app/core/space_factory.py` |
| `pydelphi/app/core/summary.py` |
| `pydelphi/app/delphi_trajectory.py` |
| `pydelphi/data/crgsiz/amber-ff14sb-mbondi-set.crg` |
| `pydelphi/data/crgsiz/amber-ff14sb-mbondi-set.siz` |
| `pydelphi/data/crgsiz/amber-ff14sb-mbondi2-set.crg` |
| `pydelphi/data/crgsiz/amber-ff14sb-mbondi2-set.siz` |
| `pydelphi/data/crgsiz/amber-ff14sb-mbondi3-set.crg` |
| `pydelphi/data/crgsiz/amber-ff14sb-mbondi3-set.siz` |
| `pydelphi/data/crgsiz/amber-ff19sb-mbondi-set.crg` |
| `pydelphi/data/crgsiz/amber-ff19sb-mbondi-set.siz` |
| `pydelphi/data/crgsiz/amber-ff19sb-mbondi2-set.crg` |
| `pydelphi/data/crgsiz/amber-ff19sb-mbondi2-set.siz` |
| `pydelphi/data/crgsiz/amber-ff19sb-mbondi3-set.crg` |
| `pydelphi/data/crgsiz/amber-ff19sb-mbondi3-set.siz` |
| `pydelphi/data/crgsiz/amber-ff99sb-mbondi-set.crg` |
| `pydelphi/data/crgsiz/amber-ff99sb-mbondi-set.siz` |
| `pydelphi/data/crgsiz/amber-ff99sb-mbondi2-set.crg` |
| `pydelphi/data/crgsiz/amber-ff99sb-mbondi2-set.siz` |
| `pydelphi/data/crgsiz/amber-ff99sb-mbondi3-set.crg` |
| `pydelphi/data/crgsiz/amber-ff99sb-mbondi3-set.siz` |
| `pydelphi/data/crgsiz/amber-legacy.crg` |
| `pydelphi/data/crgsiz/amber-legacy.siz` |
| `pydelphi/data/crgsiz/charmm-c36m-prot-na-pbeq-set.crg` |
| `pydelphi/data/crgsiz/charmm-c36m-prot-na-pbeq-set.siz` |
| `pydelphi/data/crgsiz/charmm-legacy.crg` |
| `pydelphi/data/crgsiz/charmm-legacy.siz` |
| `pydelphi/data/test_cases/1he8/1he8.pqr` |
| `pydelphi/data/test_cases/1he8/param_1he8_py.prm` |
| `pydelphi/data/test_cases/5tif/5tif.pqr` |
| `pydelphi/data/test_cases/5tif/param_5tif_py.prm` |
| `pydelphi/data/test_cases/arg/amber99sb_sig-eps-gamma.vdw` |
| `pydelphi/data/test_cases/arg/arg.pqr` |
| `pydelphi/data/test_cases/arg/arg_surface-3A.zeta` |
| `pydelphi/data/test_cases/arg/param_arg_py.prm` |
| `pydelphi/data/test_cases/barnase/amber.crg` |
| `pydelphi/data/test_cases/barnase/amber.siz` |
| `pydelphi/data/test_cases/barnase/amber99sb_sig-eps-gamma-1.vdw` |
| `pydelphi/data/test_cases/barnase/barnase.pdb` |
| `pydelphi/data/test_cases/example-results-delphicpp-8_5_0.tsv` |
| `pydelphi/data/test_cases/focus_barnase/barnase-plus-probe.pqr` |
| `pydelphi/data/test_cases/focus_barnase/focus.ref.frc` |
| `pydelphi/data/test_cases/focus_barnase/param_child.prm` |
| `pydelphi/data/test_cases/focus_barnase/param_parent.prm` |
| `pydelphi/data/test_cases/frc_analytical/README.md` |
| `pydelphi/data/test_cases/frc_analytical/custom.crg` |
| `pydelphi/data/test_cases/frc_analytical/custom.siz` |
| `pydelphi/data/test_cases/frc_analytical/param_pdb_crgsiz.prm` |
| `pydelphi/data/test_cases/frc_analytical/param_pqr.prm` |
| `pydelphi/data/test_cases/frc_analytical/pdb_crgsiz.ref.frc` |
| `pydelphi/data/test_cases/frc_analytical/pqr.ref.frc` |
| `pydelphi/data/test_cases/frc_analytical/system.pdb` |
| `pydelphi/data/test_cases/frc_analytical/system.pqr` |
| `pydelphi/data/test_cases/nonlinear/1brs.pdb` |
| `pydelphi/data/test_cases/nonlinear/amber.crg` |
| `pydelphi/data/test_cases/nonlinear/amber.siz` |
| `pydelphi/data/test_cases/nonlinear/amber_times_5.crg` |
| `pydelphi/data/test_cases/sphere/amber99sb_sig-eps-gamma-1.vdw` |
| `pydelphi/data/test_cases/sphere/param_sph_py.prm` |
| `pydelphi/data/test_cases/sphere/sphere.crg` |
| `pydelphi/data/test_cases/sphere/sphere.pdb` |
| `pydelphi/data/test_cases/sphere/sphere.siz` |
| `pydelphi/data/test_cases/twoatoms/param_two-atoms_gauss_py.prm` |
| `pydelphi/data/test_cases/twoatoms/param_two-atoms_trad_py.prm` |
| `pydelphi/data/test_cases/twoatoms/two-atoms.pqr` |
| `pydelphi/data/test_cases/zeta_lysozyme/lysozyme.pqr` |
| `pydelphi/data/test_cases/zeta_lysozyme/param_zeta.prm` |
| `pydelphi/data/test_cases/zeta_lysozyme/param_zeta2.prm` |
| `pydelphi/data/test_cases/zeta_lysozyme/pydelphi_traj_regression_test_report.tsv` |
| `pydelphi/data/test_cases/zeta_lysozyme/zeta_4A.ref.zphi` |
| `pydelphi/data/test_cases/zeta_lysozyme/zeta_4A.zphi` |
| `pydelphi/data/test_cases/zeta_sphere/param_zeta.prm` |
| `pydelphi/data/test_cases/zeta_sphere/param_zeta2.prm` |
| `pydelphi/data/test_cases/zeta_sphere/sphere.pqr` |
| `pydelphi/data/test_cases/zeta_sphere/zeta_8A.ref.zphi` |
| `pydelphi/data/test_cases/zeta_sphere/zeta_8A.zphi` |
| `pydelphi/data/test_traj/amber-ff14sb-mbondi3-set.crg` |
| `pydelphi/data/test_traj/amber-ff14sb-mbondi3-set.siz` |
| `pydelphi/data/test_traj/peptide-nframes-10.dcd` |
| `pydelphi/data/test_traj/peptide-nframes-10.nc` |
| `pydelphi/data/test_traj/peptide-nframes-10.trr` |
| `pydelphi/data/test_traj/peptide.pdb` |
| `pydelphi/data/test_traj/peptide.pqr` |
| `pydelphi/data/test_traj/peptide.prmtop` |
| `pydelphi/data/test_traj/peptide.psf` |
| `pydelphi/data/test_traj/pydelphi_traj_regression_test_report.tsv` |
| `pydelphi/data/test_traj/test_traj.tsv` |
| `pydelphi/data/test_traj/traj_ref_values.tsv` |
| `pydelphi/energy/electrfe_ff.py` |
| `pydelphi/examples/focus_barnase/barnase-plus-probe.pqr` |
| `pydelphi/examples/focus_barnase/focus.out.frc` |
| `pydelphi/examples/focus_barnase/focus.out32.frc` |
| `pydelphi/examples/focus_barnase/focus.ref.frc` |
| `pydelphi/examples/focus_barnase/focus_parent.frc` |
| `pydelphi/examples/focus_barnase/param_child.prm` |
| `pydelphi/examples/focus_barnase/param_parent.prm` |
| `pydelphi/examples/frc_analytical/README.md` |
| `pydelphi/examples/frc_analytical/custom.crg` |
| `pydelphi/examples/frc_analytical/custom.siz` |
| `pydelphi/examples/frc_analytical/pdb_custom_ignore.frc` |
| `pydelphi/examples/frc_analytical/pdb_custom_ignore.prm` |
| `pydelphi/examples/frc_analytical/pdb_custom_source.pqr` |
| `pydelphi/examples/frc_analytical/pdb_custom_target.pqr` |
| `pydelphi/examples/frc_analytical/pqr_custom_ignore.frc` |
| `pydelphi/examples/frc_analytical/pqr_custom_ignore.ref.frc` |
| `pydelphi/examples/frc_analytical/pqr_custom_source.pqr` |
| `pydelphi/examples/frc_analytical/pqr_custom_target.frc` |
| `pydelphi/examples/frc_analytical/pqr_custom_target.pqr` |
| `pydelphi/examples/frc_analytical/pqr_ignore.prm` |
| `pydelphi/examples/frc_analytical/pqr_target.prm` |
| `pydelphi/examples/frc_analytical/pqr_target_out_source.pqr` |
| `pydelphi/examples/frc_analytical/pqr_targte_out_target.pqr` |
| `pydelphi/examples/frc_analytical/source.pqr` |
| `pydelphi/examples/frc_analytical/system.pdb` |
| `pydelphi/examples/frc_analytical/system.pqr` |
| `pydelphi/examples/frc_analytical/target.pqr` |
| `pydelphi/examples/test_traj/amber-ff14sb-mbondi3-set.crg` |
| `pydelphi/examples/test_traj/amber-ff14sb-mbondi3-set.siz` |
| `pydelphi/examples/test_traj/param-pqr.prm` |
| `pydelphi/examples/test_traj/param-traj.prm` |
| `pydelphi/examples/test_traj/param.prm` |
| `pydelphi/examples/test_traj/peptide-nframes-10.dcd` |
| `pydelphi/examples/test_traj/peptide-nframes-10.nc` |
| `pydelphi/examples/test_traj/peptide-nframes-10.trr` |
| `pydelphi/examples/test_traj/peptide.pdb` |
| `pydelphi/examples/test_traj/peptide.pqr` |
| `pydelphi/examples/test_traj/peptide.prmtop` |
| `pydelphi/examples/test_traj/peptide.psf` |
| `pydelphi/foundation/models.py` |
| `pydelphi/geometry/__init__.py` |
| `pydelphi/geometry/gaussian_overlap.py` |
| `pydelphi/geometry/nonpolar.py` |
| `pydelphi/scripts/pydelphi_traj.py` |
| `pydelphi/space/core/adjacency_builder_csr.py` |
| `pydelphi/space/core/vdw/init_boundary_finder.py` |
| `pydelphi/space/core/vdw/internals.py` |
| `pydelphi/space/core/vdw/process_bgp/__init__.py` |
| `pydelphi/space/core/vdw/process_bgp/cuda_iter_gpu.py` |
| `pydelphi/space/core/vdw/process_bgp/cuda_kernels.py` |
| `pydelphi/space/core/vdw/process_bgp/helpers_cpu.py` |
| `pydelphi/space/core/vdw/process_bgp/helpers_cuda.py` |
| `pydelphi/space/core/vdw/process_bgp/helpers_parallel.py` |
| `pydelphi/space/core/vdw/process_bgp/hybrid_colored_iter_cpu.py` |
| `pydelphi/space/core/vdw/process_bgp/parallel_iter_cpu.py` |
| `pydelphi/space/core/vdw/process_bgp/serial_colored_iter_cpu.py` |
| `pydelphi/space/core/vdw/process_bgp/serial_iter_cpu.py` |
| `pydelphi/space/core/vdw/sas_builder/__init__.py` |
| `pydelphi/space/core/vdw/sas_builder/molecular_surface.py` |
| `pydelphi/space/core/vdw/sas_builder/sas_parallel.py` |
| `pydelphi/space/core/vdw/scale_bgp/__init__.py` |
| `pydelphi/space/core/vdw/scale_bgp/initial_bgp.py` |
| `pydelphi/space/core/vdw/scale_bgp/reentrant_bgp.py` |
| `pydelphi/space/core/vdw/scale_bgp/scale_boundary.py` |
| `pydelphi/tests/test_energy_regression.py` |
| `pydelphi/tests/test_traj_regression.py` |
| `pydelphi/utils/io/atomkey_fields.py` |
| `pydelphi/utils/io/format/assorted/__init__.py` |
| `pydelphi/utils/io/format/assorted/frc_reader.py` |
| `pydelphi/utils/io/format/assorted/nparray_writer.py` |
| `pydelphi/utils/io/format/assorted/trajectory_logger.py` |
| `pydelphi/utils/io/format/format_resolver.py` |
| `pydelphi/utils/io/inproc_helpers/param_definitions/miscfunc_params.py` |
| `pydelphi/utils/io/lite/ensemble.py` |
| `pydelphi/utils/io/lite/topology_lite.py` |
| `pydelphi/utils/io/lite/trajectory_lite.py` |
| `pydelphi/utils/io/topology/__init__.py` |
| `pydelphi/utils/io/topology/pdb_reader_lite.py` |
| `pydelphi/utils/io/topology/pqr_reader_lite.py` |
| `pydelphi/utils/io/topology/prmtop_reader_lite.py` |
| `pydelphi/utils/io/topology/psf_reader_lite.py` |
| `pydelphi/utils/io/trajectory/__init__.py` |
| `pydelphi/utils/io/trajectory/dcd_reader_lite.py` |
| `pydelphi/utils/io/trajectory/nc_reader_lite.py` |
| `pydelphi/utils/io/trajectory/trr_reader_lite.py` |
| `pydelphi/utils/select_handler.py` |
| `pydelphi/utils/select_lang.py` |
| `pydelphi/utils/splash.py` |
