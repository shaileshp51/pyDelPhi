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
import sys
import time
from datetime import datetime
from copy import deepcopy
import numpy as np

from pydelphi.foundation.enums import (
    Precision,
    BoundaryCondition,
    BioModel,
    DielectricModel,
)

from pydelphi.config.global_runtime import (
    set_precision,
    vprint,
    delphi_real,
)

from pydelphi.utils.io.inproc import Inputs
import pydelphi.utils.io.writers as wrt
import pydelphi.utils.io.readers as rdr

from pydelphi.constants import (
    LEN_ATOMFIELDS,
    ATOMFIELD_CHARGE,
    ATOMFIELD_MEDIA_ID,
    ConstDelPhiFloats as ConstDelPhi,
    ATOMFIELD_X,
    ATOMFIELD_Z,
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


class DelphiApp:
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

        self.energy_settings = None
        self.prmfile = prmfile
        self.platform = platform
        self.inp = user_inputs
        self.frc_source_atoms = None
        self.frc_target_atoms = None
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

        from pydelphi.solver.rpb.sor.linear_rpb import RPBESolver
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

        self.context = context
        self.space = space  # the self.space is imported space module
        self.set_atom_grid_coords = set_atom_grid_coords

        self.RuntimeContext = (
            context.RuntimeContext
        )  # Note: self.RuntimeContext is class type RuntimeContext, while .ctx is its object
        self.RPBESolver = RPBESolver
        self.PBESolver = PBESolver

        # Following are references to functions imported from other modules
        self.calculate_all_energies = calculate_all_energies

        if self.inp is None:
            from pydelphi.utils.io.inproc import Inputs

            self.inp = Inputs()
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
        run_label,
        overwrite,
    ):
        tic_prep = time.perf_counter()
        progmsg = " pyDelphi started on: {} ".format(
            datetime.now().strftime("%b-%d-%Y %H:%M:%S")
        )
        vprint(INFO, self._VERBOSITY, f"\n\n{progmsg:*^90s}")

        from pydelphi.utils.splash import format_splash, format_citation

        vprint(INFO, self._VERBOSITY, format_splash())

        # Read the inputs for the run from the parametrs file
        if self.inp is None:
            self.inp = Inputs()
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

        inp_atoms_dict = dict(self.inp.atoms)
        atoms_keys = list(inp_atoms_dict.keys())

        self.ctx.grid_offset[:] = self.inp.gridbox_offset.astype(self.delphi_real)[:]

        self.ctx.selections_spec = dict(self.inp.selections_spec)  # strings, cheap
        if len(self.ctx.selections_spec) > 0:
            from pydelphi.utils.select_handler import materialize_selections

            # atoms_values = list(inp_atoms_dict.values())
            self.ctx.selections_idx = materialize_selections(
                self.ctx.selections_spec, atoms_keys
            )

        atoms_dict = inp_atoms_dict
        prm_frc = self.inp.get_param("frc")

        if prm_frc.issupplied:
            source = prm_frc.get_attribute("source")
            target = prm_frc.get_attribute("target")
            target_file = prm_frc.get_attribute("target_file")
            tmode = prm_frc.get_attribute("target_mode")

            target_file_provided = target_file != ""
            target_provided = target != ""
            use_target_selection = target_provided and not target_file_provided

            # ------------------------------------------------------------
            # Validate source selection.
            # ------------------------------------------------------------
            if source == "":
                print(
                    "frc(...): source selection is required. "
                    'Define a source using select(...), then pass source="NAME".',
                    file=sys.stderr,
                )
                sys.exit(1)

            if source not in self.ctx.selections_idx:
                print(
                    f"frc(...): source selection '{source}' is not defined. "
                    "Define it first using select(...).",
                    file=sys.stderr,
                )
                sys.exit(1)

            source_idx = np.asarray(self.ctx.selections_idx[source], dtype=int)

            if len(source_idx) == 0:
                print(
                    f"frc(...): source selection '{source}' is empty.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # ------------------------------------------------------------
            # Validate target source.
            # target_file overrides target selection.
            # ------------------------------------------------------------
            if not target_file_provided and not target_provided:
                print(
                    "frc(...): target selection is required when target_file is empty. "
                    'Provide either target="NAME" or target_file="path".',
                    file=sys.stderr,
                )
                sys.exit(1)

            if target_file_provided and target_provided:
                print(
                    "frc(...): both target and target_file were supplied; "
                    "target_file overrides target selection.",
                    file=sys.stderr,
                )

            target_idx = None
            target_atoms = {}

            if use_target_selection:
                if target not in self.ctx.selections_idx:
                    print(
                        f"frc(...): target selection '{target}' is not defined. "
                        "Define it first using select(...).",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                target_idx = np.asarray(self.ctx.selections_idx[target], dtype=int)

                if len(target_idx) == 0:
                    print(
                        f"frc(...): target selection '{target}' is empty.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                mask_common = np.isin(target_idx, source_idx)
                if mask_common.any():
                    common_atoms = target_idx[mask_common]
                    print(
                        "frc(...): expected source and target to be mutually exclusive "
                        "selections, but common atom indices were found: "
                        f"{common_atoms.tolist()}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            # ------------------------------------------------------------
            # Build source_atoms from the selected source/target mode.
            # ------------------------------------------------------------
            source_atoms = {}

            if tmode == "uncharge":
                if use_target_selection:
                    union_idx = np.union1d(source_idx, target_idx)

                    # This invariant only applies when both source and target are
                    # selections from the loaded system. It does not apply to target_file.
                    if len(union_idx) != len(atoms_keys):
                        print(
                            "frc(...): expected source ∪ target to cover the full system "
                            "in uncharge mode, but union has "
                            f"{len(union_idx)} atoms while system has {len(atoms_keys)}. "
                            "Define selections so that source and target partition or "
                            "cover the solute.",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                else:
                    union_idx = source_idx

                for idx in union_idx:
                    atom_k = atoms_keys[idx]
                    source_atoms[atom_k] = inp_atoms_dict[atom_k].copy()

                if use_target_selection:
                    for idx in target_idx:
                        atom_k = atoms_keys[idx]
                        source_atoms[atom_k][ATOMFIELD_CHARGE] = 0.0
                        target_atoms[atom_k] = inp_atoms_dict[atom_k].copy()

            elif tmode == "ignore":
                for idx in source_idx:
                    atom_k = atoms_keys[idx]
                    source_atoms[atom_k] = inp_atoms_dict[atom_k].copy()

                if use_target_selection:
                    for idx in target_idx:
                        atom_k = atoms_keys[idx]
                        target_atoms[atom_k] = inp_atoms_dict[atom_k].copy()

            else:
                print(
                    f"frc(...): unsupported target_mode '{tmode}'. "
                    "Expected one of: uncharge, ignore.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # ------------------------------------------------------------
            # Read target atoms/evaluation points from target_file, if supplied.
            # target_file overrides target selection.
            # ------------------------------------------------------------
            if target_file_provided:
                ext_of_target_file = (
                    os.path.splitext(target_file)[1].lower().lstrip(".")
                )

                if ext_of_target_file not in {"pdb", "pqr", "frc"}:
                    print(
                        f"frc(...): unsupported target_file format '{ext_of_target_file}' "
                        f"for file '{target_file}'. Supported formats are: pdb, pqr, frc.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                try:
                    target_atoms = rdr.read_frc(
                        target_file,
                        format=ext_of_target_file,
                    )
                except FileNotFoundError:
                    print(
                        f"frc(...): target_file not found: {target_file}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                except Exception as exc:
                    print(
                        f"frc(...): failed to read target_file '{target_file}': {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                if target_atoms is None or len(target_atoms) == 0:
                    print(
                        f"frc(...): target_file '{target_file}' did not provide any "
                        "evaluation atoms or points.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            atoms_dict = source_atoms

            self.frc_source_atoms = source_atoms
            self.frc_target_atoms = target_atoms

        prm_acenter = self.inp.get_param("acenter")
        prm_in_frc = self.inp.get_param("in__frc")
        scale = self.inp.get_param_value("scale")
        self.ctx.enforce_acenter = prm_acenter.issupplied or prm_in_frc.issupplied

        grid_center = self.context.get_gridbox_center_override(
            acenter=prm_acenter,
            in_frc=prm_in_frc,
            scale=scale,
            atoms_keys=atoms_keys,
            atoms_dict=inp_atoms_dict,
            selections_idx=self.ctx.selections_idx,
            acenter_center_from_file=lambda ac_frc_file, sc: rdr.calculate_center_of_frc_atoms(
                ac_frc_file, self.ctx.grid_offset, sc
            ),
            frc_center_from_file=lambda frc_file, sc: rdr.calculate_center_of_frc_atoms(
                frc_file, self.ctx.grid_offset, sc
            ),
        )

        # Update RuntimeContext (ctx): Setup debylength
        self.ctx.atoms_init_and_summary(
            atoms=atoms_dict,
            objects=self.inp.objects,
            extremas_rule=self.inp.get_param_value("solute_extrema"),
            acenter=grid_center,
            enforce_acenter=self.ctx.enforce_acenter,
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
        shift_angs = self.ctx.grid_offset / scale  # (3,) Å
        self.ctx.grid_origin[:] = self.ctx.grid_origin - shift_angs
        self.ctx.grid_center[:] = self.ctx.grid_center - shift_angs

        # acenter(x,y,z), acenter(sel=...), and acenter(file=...) define an
        # absolute real-space center in Angstroms.  Preserve that explicit
        # center after grid parameter setup so focusing child boxes do not
        # silently fall back to a shifted/zero center.
        if self.ctx.enforce_acenter:
            self.ctx.acenter[:] = grid_center[:]
            self.ctx.grid_center[:] = grid_center[:]

        # Abort if DIPOLAR boundary condition is requested for systems with no charges.
        if (
            self.inp.get_param_value("boundary_condition").int_value
            == BoundaryCondition.DIPOLAR.int_value
        ):
            if self.ctx.num_negative_charge == 0 and self.ctx.num_positive_charge == 0:
                charges_missing = "-ve" if self.ctx.num_negative_charge == 0 else "+ve"
                msg = (
                    f"INPUT ERROR: System has none charged atoms. Dipolar boundary conditions requires at-least one. \n"
                    "Try COULOMBIC boundary condition instead."
                )
                print(msg)
                sys.exit(1)

        # Finalize RuntimeContext (ctx): current/focused grid geometry before
        # focusing preparation.  prepare_focusing_if_needed() expects the child
        # grid center, shape, origin, and scale to be finalized.
        self.ctx.grid_shape = self.ctx.gridbox_size_to_shape_array()
        self.ctx.grid_origin = self.ctx.setup_gridmap_3d(
            self.ctx.grid_center,
            self.ctx.grid_shape,
            self.ctx.scale,
        )

        if self.inp.get_param_value("biomodel").int_value == BioModel.PBE.int_value:
            from pydelphi.app.core.focusing_prep import prepare_focusing_if_needed

            prepare_focusing_if_needed(
                inp=self.inp,
                ctx=self.ctx,
                rdr=rdr,
                frc_target_atoms=self.frc_target_atoms,
                delphi_real=self.delphi_real,
            )

        # Update RuntimeContext (ctx): add atom grid coordinates after focusing
        # preparation.  In focusing mode, ctx.prepare_focusing() filters the
        # source atoms to the portion that belongs in/near the child box.  FRC
        # target atoms remain separate and are passed only to the FRC writer.
        for ia, atom_data in enumerate(self.ctx.atoms_data):
            self.set_atom_grid_coords(
                atom_data,
                self.ctx.grid_origin,
                self.ctx.grid_spacing,
            )
            # print(atom_data[ATOMFIELD_CHARGE], atom_data[3:6])

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
            vprint(INFO, self._VERBOSITY, "")
            if (
                self.inp.get_param_value("boundary_condition")
                == BoundaryCondition.FOCUSING
            ):
                vprint(
                    INFO,
                    self._VERBOSITY,
                    summarize_parentrun_str(
                        ctx=self.ctx,
                        indent_spaces=4,
                        field_width=44,
                        format_specifier="s",
                    ),
                )
            vprint(INFO, self._VERBOSITY, "")
            vprint(INFO, self._VERBOSITY, "=" * 90)
            vprint(INFO, self._VERBOSITY, "")

        toc_prep = time.perf_counter()
        self.timings["Setting up the grid"] = "{:0.3f}".format(toc_prep - tic_prep)

        final_rms, final_dphi, total_iters, convergence_status = (
            -1.0,
            -1.0,
            0,
            "UNKNOWN",
        )

        if self.inp.get_param_value("biomodel").int_value == BioModel.RPBE.int_value:
            from pydelphi.app.core.rpbe_runner import run_rpbe

            run_rpbe(
                inp=self.inp,
                ctx=self.ctx,
                platform=self.platform,
                space_module=self.space,
                verbosity=self._VERBOSITY,
                lvl_debug=DEBUG,
                lvl_info=INFO,
                approx_zero=self.delphi_real(ConstDelPhi.ApproxZero.value),
                timings=self.timings,
                erg_settings=self.energy_settings,
                calculate_all_energies=self.calculate_all_energies,
                RPBESolverCtor=self.RPBESolver,
            )
        elif self.inp.get_param_value("biomodel").int_value == BioModel.PBE.int_value:
            from pydelphi.app.core.pbe_runner import run_pbe

            (final_rms, final_dphi, total_iters, convergence_status) = run_pbe(
                inp=self.inp,
                ctx=self.ctx,
                platform=self.platform,
                space_module=self.space,
                verbosity=self._VERBOSITY,
                lvl_debug=DEBUG,
                lvl_info=INFO,
                lvl_trace=TRACE,
                approx_zero=self.delphi_real(ConstDelPhi.ApproxZero.value),
                timings=self.timings,
                erg_settings=self.energy_settings,
                calculate_all_energies=self.calculate_all_energies,
                PBESolverCtor=self.PBESolver,
            )

        # prm_out_frc = self.inp.get_param("out__frc")
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
                total_iters=total_iters,
                final_rms=final_rms,
                final_dphi=final_dphi,
                convergence_status=convergence_status,
                frc_outfile=output_frc_file,
                frc_target_atoms=self.frc_target_atoms,
                delphi_real=self.delphi_real,
            )

        timing_message, energy_message = (
            self.ctx.energy_results.generate_energy_report_strings(
                indent_spaces=4, field_width=50, format_specifier="s"
            )
        )

        toc_final = time.perf_counter()

        if self._VERBOSITY <= INFO:
            vprint(INFO, self._VERBOSITY, "")
            for kt, vt in self.timings.items():
                vprint(INFO, self._VERBOSITY, f"    Time> {kt:<44s} : {vt:>13s} s")

        vprint(INFO, self._VERBOSITY, timing_message)
        total_exec_time = "{:.3f}".format(toc_final - tic_prep)
        self.timings["Total time taken"] = total_exec_time

        vprint(
            INFO,
            self._VERBOSITY,
            f"    Time> {'Total time taken':<44s} : {total_exec_time:>13s} s",
        )
        vprint(INFO, self._VERBOSITY, "")

        vprint(NOTICE, self._VERBOSITY, energy_message)

        energies = self.ctx.energy_results.energies

        # Write results
        # print(energies)
        from pydelphi.utils.energy_terms import (
            ENERGY_TERM_ABBREVIATIONS,
        )
        from pydelphi.utils.io.format.assorted.custom_writer import (
            write_energies_to_tsv,
        )

        if overwrite:
            try:
                os.remove(energy_outfile)
            except Exception as e:
                pass
        write_energies_to_tsv(
            energies,
            energy_outfile,
            run_label,
            ENERGY_TERM_ABBREVIATIONS,
            write_header=True,
        )

        # Write pqr file if requested
        prm_out_modpdb4 = self.inp.get_param("out__modpdb4")
        if prm_out_modpdb4.issupplied:
            out_file = prm_out_modpdb4.get_attribute("file")
            out_fmt = prm_out_modpdb4.get_attribute("format")
            wrt.write_atoms(out_file, self.inp.atoms, objects=dict(), format=out_fmt)

        # Write selections to file if requested
        for sel_name, sel_call in self.inp.out_sel_calls.items():
            wrt.write_selection(
                filename=sel_call["file"],
                format=sel_call["format"],
                all_atoms_keys=atoms_keys,
                all_atoms_dict=inp_atoms_dict,
                sel_atoms_key_indices=self.ctx.selections_idx[sel_name],
            )

        self.ctx._reset_maps()
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
