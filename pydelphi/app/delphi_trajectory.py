#!/usr/bin/env python
# coding: utf-8

# This file is part of pyDelPhi.
# Copyright (C) 2025 The pyDelPhi Project and contributors.
#
# pyDelPhi is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with pyDelPhi. If not, see <https://www.gnu.org/licenses/>.


import os
import time
from datetime import datetime
import numpy as np

from pydelphi.foundation.enums import (
    Precision,
    BoundaryCondition,
    BioModel,
)

from pydelphi.config.global_runtime import (
    set_precision,
    vprint,
)

from pydelphi.utils.io.inproc import Inputs
import pydelphi.utils.io.writers as wrt
import pydelphi.utils.io.readers as rdr

from pydelphi.constants import (
    ConstDelPhiFloats as ConstDelPhi,
)

from pydelphi.config.logging_config import (
    NOTICE,
    INFO,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)

MODULE_NAME = __name__

from pydelphi.app.core.summary import (
    summary_str,
    summarize_parentrun_str,
)

from pydelphi.app.core.policy import TrajPolicy, enforce_traj_policy


class DelphiTrajApp:
    """
    Main application class for Delphi calculations.

    This class manages the setup, execution, and output of Delphi electrostatic calculations,
    supporting both vacuum and water phases. It handles parameter input, runtime context/state management,
    RPBE solving, spatial map writing, and timing operations.
    """

    def __init__(self, prmfile, platform, user_inputs=None):
        """
        Initializes DelphiApp instance.

        Args:
            prmfile (str): Path to the parameter file.
            platform (Platform): Platform configuration object (CPU or CUDA).
            user_inputs (Inputs, optional): User-provided input parameters. Defaults to None,
                                             in which case inputs are parsed from the prmfile.
        """
        self.traj_policy = TrajPolicy()
        self.energy_settings = None
        self.prmfile = prmfile
        self.platform = platform
        self.inp = user_inputs
        self._ctx = None  # RuntimeContext to hold calculation data/state
        self.timings = {}  # Dictionary to store timing of different calculation stages
        set_precision(platform.precision)  # Set calculation precision based on platform

        from pydelphi.config.global_runtime import (
            PRECISION,
            delphi_bool,
            delphi_int,
            delphi_real,
        )

        if PRECISION.int_value == Precision.SINGLE.int_value:
            from pydelphi.utils.prec.single import set_atom_grid_coords
        elif PRECISION.int_value == Precision.DOUBLE.int_value:
            from pydelphi.utils.prec.double import set_atom_grid_coords

        import pydelphi.foundation.context as context

        from pydelphi.space import space
        from pydelphi.solver.solver import PBESolver
        from pydelphi.energy.calculator import calculate_all_energies

        # **Store imports as instance attributes**
        self.PRECISION = PRECISION
        # Get the effective verbosity level for this specific module
        # This will consider the global setting and any specific override for MODULE_NAME
        self._VERBOSITY = get_effective_verbosity(MODULE_NAME)
        self.delphi_bool = delphi_bool
        self.delphi_int = delphi_int
        self.delphi_real = delphi_real
        # print(
        #     platform.precision,
        #     self._VERBOSITY,
        #     self.delphi_bool,
        #     self.delphi_int,
        #     self.delphi_real,
        # )

        self.space = space  # the self.space is imported space module

        self.RuntimeContext = (
            context.RuntimeContext
        )  # Note: self.RuntimeContext is class type RuntimeContext, while .ctx is its object
        self.PBESolver = PBESolver

        # Following are references to functions imported from other modules
        self.set_atom_grid_coords = set_atom_grid_coords
        self.calculate_all_energies = calculate_all_energies

        if self.inp is None:
            from pydelphi.utils.io.inproc import Inputs

            self.inp = Inputs(app_mode="trajectory", strict=True)
            self.inp.parse_inputs(self.prmfile)

    @property
    def ctx(self):
        """Returns the current RuntimeContext instance."""
        return self._ctx

    @ctx.setter
    def ctx(self, value):
        """Sets the current RuntimeContext instance."""
        self._ctx = value

    def run(
        self,
        energy_outfile,
        traj_logfile,
        run_label,
        overwrite,
    ):
        tic_start = time.perf_counter()
        progmsg = " pyDelphi started on: {} ".format(
            datetime.now().strftime("%b-%d-%Y %H:%M:%S")
        )
        vprint(INFO, self._VERBOSITY, f"\n\n{progmsg:*^90s}")

        from pydelphi.utils.splash import format_splash, format_citation

        vprint(INFO, self._VERBOSITY, format_splash())

        # Read the inputs for the run from the parametrs file
        if self.inp is None:
            self.inp = Inputs(app_mode="trajectory", strict=True)
            self.inp.parse_inputs(self.prmfile)
            print("reading parameters from file: ", os.path.abspath(self.prmfile))
        print()

        # Update RuntimeContext (dc): with the input params for setting up further calculations
        self.ctx = self.RuntimeContext(
            self.inp.get_param_value("temperature"),
            self.inp.get_param_value("exdi"),
            self.inp.get_param_value("gapdi"),
            self.inp.get_param_value("indi"),
            self.inp.get_param_value("probe_radius"),
            self.inp.get_param_value("radius_offset"),
            self.inp.get_param_value("pressure_coefficient"),
            precision=self.platform.precision,
            dtype_int=self.delphi_int,
            dtype_real=self.delphi_real,
        )
        prm_acenter = self.inp.get_param("acenter")
        self.ctx.enforce_acenter = prm_acenter.issupplied
        # self.ctx.acenter[:] = self.inp.gridbox_center.astype(self.delphi_real)[:]
        # enforce_traj_policy(self.inp,self.traj_policy,traj,state,io)

        from pydelphi.energy.energy_models import EnergySettings

        energy_settings = EnergySettings()
        energy_settings.platform = self.platform
        energy_settings.pb_approximation = self.inp.get_param_value("pb_approximation")
        energy_settings.dielectric_model = self.inp.get_param_value("dielectric_model")
        energy_settings.surface_method = self.inp.get_param_value("surface_method")

        calc_energy_param = self.inp.get_param("calculate_energies")
        if calc_energy_param.is_attribute_inuse("coulombic"):
            energy_settings.calculate_coulombic_energy = True
        if calc_energy_param.is_attribute_inuse("lj"):
            energy_settings.calculate_lj = True
        if calc_energy_param.is_attribute_inuse("np"):
            energy_settings.calculate_nonpolar = True
        if calc_energy_param.is_attribute_inuse("polar"):
            energy_settings.calculate_reactionfield = True
        if calc_energy_param.is_attribute_inuse("grid"):
            energy_settings.calculate_grid_energy = True

        # Freeze to mark finalized state of energy_settings configuration.
        energy_settings.freeze()

        self.energy_settings = energy_settings

        from pydelphi.utils.energy_terms import (
            ENERGY_TERM_ABBREVIATIONS,
        )
        from pydelphi.utils.io.format.assorted.custom_writer import (
            write_energies_to_tsv,
        )

        # Update RuntimeContext (ctx): Setup debylength
        if self.inp.ensemble is not None and len(self.inp.ensemble) > 0:
            from pydelphi.app.core.atom_materializer import (
                build_atoms_from_top_and_frame0,
                update_atoms_coords_inplace,
            )
            from pydelphi.utils.io.format.assorted.trajectory_logger import (
                TrajectoryLogger,
            )

            trj_log = TrajectoryLogger(
                fp=traj_logfile, platform=self.platform, inp=self.inp
            )

            for ens_label, ens_obj in self.inp.ensemble.items():
                ens_top = ens_obj.top
                ens_traj = ens_obj.traj
                ens_start = ens_obj.start
                ens_stop = ens_obj.stop
                ens_stride = ens_obj.stride
                # print("ens_obj:", ens_obj)
                print(f"ens(start={ens_start}, stop={ens_stop}, stride={ens_stride})")

                (final_rms, final_dphi, total_iters, convergence_status) = (
                    None,
                    None,
                    0,
                    "Unknown",
                )

                atom_data = None
                atoms_keys = None
                atoms = {}
                is_firstframe = True
                frame_index = ens_start

                for frame in ens_traj.iter_xyz(
                    start=ens_start, stop=ens_stop, stride=ens_stride
                ):
                    frame_index, frame_xyz = frame

                    tic_frame = time.perf_counter()
                    if is_firstframe:
                        self.inp.objects = ["is a molecule  0", " "]
                        (atoms_keys, atom_data, atom_serial, resid0, resSeq) = (
                            build_atoms_from_top_and_frame0(
                                top=ens_top,
                                frame_xyz=frame_xyz,
                                delphi_real=self.delphi_real,
                            )
                        )
                        # print("atoms_keys=", atoms_keys)
                        # print("atom_serial=", atom_serial)
                        # print("resid0=", resid0)
                        # print("resSeq=", resSeq)
                        # print("self.inp.atoms=", self.inp.atoms)
                        if len(atoms_keys) == atom_data.shape[0]:
                            for atom_idx in range(atom_data.shape[0]):
                                a_key = atoms_keys[atom_idx]
                                a_data = atom_data[atom_idx, :]
                                atoms[a_key] = a_data

                            self.inp.atoms = atoms
                            # print("atoms=", atoms)
                            # print("self.inp.atoms=", self.inp.atoms)
                            self.ctx.atoms_init(
                                self.inp.atoms, self.inp.objects, is_focusing=False
                            )
                    else:
                        update_atoms_coords_inplace(
                            traj=ens_traj,
                            frame_xyz=frame_xyz,
                            atoms_data=self.ctx.atoms_data,
                            delphi_real=self.delphi_real,
                        )
                    # frame_index += ens_stride

                    self.ctx.summarize_atoms_data(
                        extremas_rule=self.inp.get_param_value("solute_extrema"),
                        acenter=self.ctx.acenter,
                        enforce_acenter=self.ctx.enforce_acenter,
                        max_atom_radius=self.ctx.max_atom_radius,
                    )
                    self.ctx.set_debyelength(
                        self.inp.get_param_value("salt_concentration"),
                        self.inp.get_param_value("temperature"),
                        self.inp.get_param_value("exdi"),
                    )

                    # Update RuntimeContext (ctx): Setup gridbox parameters
                    gridbox_margin = 0
                    if self.inp.get_param("gridbox_margin").active:
                        gridbox_margin = self.inp.get_param_value("gridbox_margin")

                    self.ctx.grid_params(
                        scale=self.inp.get_param_value("scale"),
                        perfil=self.inp.get_param_value("percent_fill"),
                        gridbox_margin=gridbox_margin,
                        gridbox_size=self.inp.get_param_value("grid_size"),
                        gridbox_type=self.inp.get_param_value("gridbox_type"),
                        grid_offset=self.ctx.grid_offset,
                    )
                    # Abort if DIPOLAR boundary condition is requested for systems with no charges.
                    if (
                        self.inp.get_param_value("boundary_condition").int_value
                        == BoundaryCondition.DIPOLAR.int_value
                    ):
                        if (
                            self.ctx.num_negative_charge == 0
                            and self.ctx.num_positive_charge == 0
                        ):
                            charges_missing = (
                                "-ve" if self.ctx.num_negative_charge == 0 else "+ve"
                            )
                            msg = (
                                f"INPUT ERROR: System has none charged atoms. Dipolar boundary conditions requires at-least one. \n"
                                "Try COULOMBIC boundary condition instead."
                            )
                            raise ValueError(msg)

                    if (
                        self.inp.get_param_value("biomodel").int_value
                        == BioModel.PBE.int_value
                    ):
                        from pydelphi.app.core.focusing_prep import (
                            prepare_focusing_if_needed,
                        )

                        prepare_focusing_if_needed(
                            inp=self.inp,
                            ctx=self.ctx,
                            rdr=rdr,
                            delphi_real=self.delphi_real,
                        )

                    # Update RuntimeContext (ctx): the gridbox origin
                    self.ctx.grid_origin = self.ctx.setup_gridmap_3d(
                        self.ctx.grid_center,
                        self.ctx.grid_shape,
                        self.ctx.scale,
                    )

                    # Update RuntimeContext (ctx): add input atoms information
                    # NOTE: for focusing runs some grid_indices may be beyond its valid boundary
                    # and should be checked and processed accordingly in space module
                    for ia, atom_data in enumerate(self.ctx.atoms_data):
                        self.set_atom_grid_coords(
                            atom_data,
                            self.ctx.grid_origin,
                            self.ctx.grid_spacing,
                        )

                    # Update RuntimeContext (ctx): update the gridbox shape
                    self.ctx.grid_shape = self.ctx.gridbox_size_to_shape_array()

                    # Print out parameters summary for the run
                    if self._VERBOSITY <= INFO:
                        vprint(
                            INFO,
                            self._VERBOSITY,
                            summary_str(
                                platform=self.platform,
                                inp=self.inp,
                                ctx=self.ctx,
                                indent_spaces=4,
                                field_width=44,
                                format_specifier="s",
                            ),
                        )

                    if (
                        self.inp.get_param_value("biomodel").int_value
                        == BioModel.PBE.int_value
                    ):
                        from pydelphi.app.core.pbe_runner import run_pbe

                        static_meta = dict()
                        if is_firstframe:
                            static_meta = {
                                "q_total": self.ctx.total_charge,
                                "q_pos": self.ctx.positive_charge,
                                "q_neg": self.ctx.negative_charge,
                                "n_pos": self.ctx.num_positive_charge,
                                "n_neg": self.ctx.num_negative_charge,
                            }

                        (final_rms, final_dphi, total_iters, convergence_status) = (
                            run_pbe(
                                inp=self.inp,
                                ctx=self.ctx,
                                platform=self.platform,
                                space_module=self.space,
                                verbosity=self._VERBOSITY,
                                lvl_debug=DEBUG,
                                lvl_info=INFO,
                                lvl_trace=TRACE,
                                approx_zero=self.delphi_real(
                                    ConstDelPhi.ApproxZero.value
                                ),
                                timings=self.timings,
                                erg_settings=self.energy_settings,
                                calculate_all_energies=self.calculate_all_energies,
                                PBESolverCtor=self.PBESolver,
                            )
                        )

                    prm_frc = self.inp.get_param("frc")

                    if prm_frc.issupplied:
                        from pydelphi.app.core.frc_writer import write_frc_if_requested

                        output_frc_file = None
                        if prm_frc.issupplied:
                            output_frc_file = prm_frc.get_attribute("outfile")

                        write_frc_if_requested(
                            inp=self.inp,
                            ctx=self.ctx,
                            rdr=rdr,
                            frc_outfile=output_frc_file,
                            frc_target_atoms=self.frc_target_atoms,
                            delphi_real=self.delphi_real,
                        )

                    # Write pqr file if requested
                    prm_out_modpdb4 = self.inp.get_param("out__modpdb4")
                    if prm_out_modpdb4.issupplied:
                        out_file = prm_out_modpdb4.get_attribute("file")
                        out_file_frm = out_file.replace(
                            ".pqr", f"_frame{frame_index}.pqr"
                        )
                        # print(
                        #     "atom_keys=",
                        #     len(atoms_keys),
                        #     "atom_data=",
                        #     len(self.ctx.atoms_data),
                        # )
                        # print("atom_keys=", atoms_keys)
                        wrt.write_atoms(
                            out_file_frm,
                            atoms=None,
                            objects=dict(),
                            atom_keys=atoms_keys,
                            atom_data=self.ctx.atoms_data,
                            sort=False,
                        )

                    toc_frame = time.perf_counter()
                    frame_wall_s = toc_frame - tic_frame
                    self.timings["Time taken frame processing"] = "{:0.3f}".format(
                        toc_frame - tic_frame
                    )

                    timing_message, energy_message = (
                        self.ctx.energy_results.generate_energy_report_strings(
                            indent_spaces=4, field_width=50, format_specifier="s"
                        )
                    )

                    if self._VERBOSITY <= INFO:
                        vprint(INFO, self._VERBOSITY, "")
                        for kt, vt in self.timings.items():
                            vprint(
                                INFO,
                                self._VERBOSITY,
                                f"    Time> {kt:<44s} : {vt:>13s} s",
                            )

                    vprint(INFO, self._VERBOSITY, timing_message)

                    vprint(NOTICE, self._VERBOSITY, energy_message)

                    energies = self.ctx.energy_results.energies

                    # Write results
                    ordered_keys = None
                    if is_firstframe:
                        trj_log.start(static_meta)
                        header_meta = {
                            "scale:": self.ctx.scale,
                            "dielectric_model:": self.inp.get_param_value(
                                "dielectric_model"
                            ),
                            "bc:": self.inp.get_param_value("bc"),
                            "solver:": self.inp.get_param_value("solver"),
                            "pb_approximation": self.inp.get_param_value(
                                "pb_approximation"
                            ),
                            "indi:": self.inp.get_param_value("indi"),
                            "exdi:": self.inp.get_param_value("indi"),
                            "gridbox_type:": self.inp.get_param_value("gridbox_type"),
                            "probe_radius:": self.inp.get_param_value("probe_radius"),
                            "radius_offset:": self.inp.get_param_value("probe_radius"),
                            "pressure_coefficient:": self.inp.get_param_value(
                                "pressure_coefficient"
                            ),
                            "salt": self.inp.get_param_value("salt_concentration"),
                            "ions_radii:": self.inp.get_param_value("ions_radii"),
                            "absolute_temperature:": self.inp.get_param_value(
                                "absolute_temperature"
                            ),
                        }
                        if overwrite:
                            try:
                                os.remove(energy_outfile)
                            except Exception as e:
                                pass

                    ordered_keys = write_energies_to_tsv(
                        energies=energies,
                        energy_outfile=energy_outfile,
                        run_label=f"{run_label}:{ens_label}",
                        key_mapping=ENERGY_TERM_ABBREVIATIONS,
                        frame=frame_index,
                        only_phase=True,
                        write_header=is_firstframe,
                        header_meta=header_meta,
                        ordered_keys=ordered_keys,
                    )
                    is_firstframe = False

                    frame_data = {
                        "frame": frame_index,
                        "percent_fill": self.ctx.perfil,
                        "gridbox_margin": self.ctx.gridbox_margin,
                        "nx": self.ctx.grid_shape[0],
                        "ny": self.ctx.grid_shape[1],
                        "nz": self.ctx.grid_shape[2],
                        "orig_x": self.ctx.grid_origin[0],
                        "orig_y": self.ctx.grid_origin[1],
                        "orig_z": self.ctx.grid_origin[2],
                        "range_x": self.ctx.solute_range[0],
                        "range_y": self.ctx.solute_range[1],
                        "range_z": self.ctx.solute_range[2],
                        "ctr_nq_x": self.ctx.centroid_negative_charge[0],
                        "ctr_nq_y": self.ctx.centroid_negative_charge[1],
                        "ctr_nq_z": self.ctx.centroid_negative_charge[2],
                        "ctr_pq_x": self.ctx.centroid_positive_charge[0],
                        "ctr_pq_y": self.ctx.centroid_positive_charge[1],
                        "ctr_pq_z": self.ctx.centroid_positive_charge[2],
                        "ctr_x": self.ctx.centroid[0],
                        "ctr_y": self.ctx.centroid[1],
                        "ctr_z": self.ctx.centroid[2],
                        "final_rms": final_rms,
                        "final_dphi": final_dphi,
                        "total_iters": total_iters,
                        "status": convergence_status,
                        "wall_s": frame_wall_s,
                    }
                    trj_log.write_frame(frame_data)

                    self.ctx._reset_maps()

        toc_final = time.perf_counter()
        total_exec_time = "{:.3f}".format(toc_final - tic_start)
        self.timings["Total time taken"] = total_exec_time

        vprint(
            INFO,
            self._VERBOSITY,
            f"    Time> {'Total time taken':<44s} : {total_exec_time:>13s} s",
        )
        vprint(INFO, self._VERBOSITY, "")
        print()
        vprint(INFO, self._VERBOSITY, format_citation())
        if self._VERBOSITY <= INFO:
            print("{:^90s}".format("*"))
            print("{:^90s}".format("***"))
            print(
                "{:*^90s}".format(
                    "Calcuation finished at: {}".format(
                        datetime.now().strftime("%b-%d-%Y %H:%M:%S")
                    )
                )
            )
            print("\n\n")

        return energies
