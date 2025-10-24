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

#
# pyDelPhi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#


import os
import time
from datetime import datetime
import numpy as np

from pydelphi.foundation.enums import (
    Precision,
    BoundaryCondition,
    BioModel,
    DielectricModel,
    SurfaceMethod,
    MemoryState,
    PBApproximation,
    IonExclusionRegion,
)

from pydelphi.config.global_runtime import (
    set_precision,
    vprint,
)

from pydelphi.utils.io.inproc import Inputs
import pydelphi.utils.io.writers as wrt
import pydelphi.utils.io.readers as rdr

from pydelphi.constants import (
    LEN_ATOMFIELDS,
    ATOMFIELD_CHARGE,
    ATOMFIELD_MEDIA_ID,
    ConstDelPhiFloats as ConstDelPhi,
)

from pydelphi.config.logging_config import (
    CRITICAL,
    ERROR,
    WARNING,
    NOTICE,
    INFO,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)

MODULE_NAME = __name__


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

        self.space = space  # the self.space is imported space module
        self.set_atom_grid_coords = set_atom_grid_coords

        self.RuntimeContext = (
            context.RuntimeContext
        )  # Note: self.RuntimeContext is class type RuntimeContext, while .rc is its object
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

    def _format_coords(self, coords):
        """Formats coordinates for output."""
        return ", ".join([f"{s:12.4f}" for s in coords])

    def _format_grid_size(self, grid_shape):
        """Formats grid size for output."""
        return ", ".join([f"{s:12d}" for s in grid_shape])

    def summary(self, indent_spaces, field_width, format_specifier="s"):
        """
        Generates a summary of the DelphiApp instance and its settings.

        This method prints a formatted summary to the console, including platform details,
        input parameters, and calculated runtime context/state properties if available.

        Args:
            indent_spaces (int): Number of spaces for indentation.
            field_width (int): Width of the field names in the summary.
            format_specifier (str): Format specifier for field names (e.g., 's' for string).
        """
        indent = " " * indent_spaces
        field_format = f"{{:{field_width}{format_specifier}}}"
        summary_lines = []

        summary_lines.append(
            f"{indent}{field_format.format('platform')} = {self.platform.active:10s}"
        )

        if self.platform.active == "cpu":
            summary_lines.append(
                f"{indent}{field_format.format('num_threads')} = {self.platform.names['cpu']['num_threads']:d}"
            )
        elif self.platform.active == "cuda":
            selected_dev_id = self.platform.names["cuda"]["selected_id"]
            summary_lines.append(
                f"{indent}{field_format.format('num_cpus')} = {self.platform.names['cpu']['num_threads']:d}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('device_index')} = {selected_dev_id:d}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('device_identity')} = {self.platform.names['cuda']['device'][selected_dev_id]['device_identity']}"
            )

        summary_lines.append(
            f"{indent}{field_format.format('precision')} = {self.platform.precision}"
        )
        inp_str = self.inp.info_str(
            include_statements=True,
            include_functions=False,
            indent_spaces=indent_spaces,
            field_width=field_width,
            format_specifier=format_specifier,
        )
        summary_lines.append(f"{indent}{inp_str}")  # Input parameter summary

        if self.ctx:
            # runtime context/state properties summary
            if (
                self.inp.get_param_value("boundary_condition").int_value
                != BoundaryCondition.FOCUSING.int_value
            ):
                summary_lines.append(
                    f"{indent}{field_format.format('percent_fill')} = {self.ctx.perfil:12.04f}"
                )
                # if self.inp.get_param("gridbox_margin").active:
                summary_lines.append(
                    f"{indent}{field_format.format('gridbox_margin')} = {self.ctx.gridbox_margin:12.04f}"
                )
            summary_lines.append(
                f"{indent}{field_format.format('scale')} = {self.ctx.scale:12.04f}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('grid_size: (nX, nY, nZ)')} = {self._format_grid_size(self.ctx.grid_shape)}"
            )

            summary_lines.append(
                f"{indent}{field_format.format('Sum of (-)-ve charges')} = {self.ctx.negative_charge:12.6g}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Sum of (+)-ve charges')} = {self.ctx.positive_charge:12.6g}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Sum of all charges')} = {self.ctx.total_charge:12.6g}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Number of (-)-ve charged atoms')} = {self.ctx.num_negative_charge:12.6g}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Number of (+)-ve charged atoms')} = {self.ctx.num_positive_charge:12.6g}"
            )

            if self.ctx.extremas_rule:
                summary_lines.append(
                    f"{indent}{field_format.format('Minimum solute coords: (X, Y, Z)')} = {self._format_coords(self.ctx.coords_by_axis_min)}"
                )
                summary_lines.append(
                    f"{indent}{field_format.format('Maximum solute coords: (X, Y, Z)')} = {self._format_coords(self.ctx.coords_by_axis_max)}"
                )
                summary_lines.append(
                    f"{indent}{field_format.format('Range of solute coords: (X, Y, Z)')} = {self._format_coords(self.ctx.solute_range)}"
                )
            else:
                summary_lines.append(
                    f"{indent}{field_format.format('Minimum solute coords (-) radius: (X, Y, Z)')} = {self._format_coords(self.ctx.boundary_min)}"
                )
                summary_lines.append(
                    f"{indent}{field_format.format('Maximum solute coords (+) radius: (X, Y, Z)')} = {self._format_coords(self.ctx.boundary_max)}"
                )
                summary_lines.append(
                    f"{indent}{field_format.format('Range of solute coords (-/+) radius: (X, Y, Z)')} = {self._format_coords(self.ctx.solute_range)}"
                )

            summary_lines.append(
                f"{indent}{field_format.format('Centroid of (-)-ve charges: (X, Y, Z)')} = {self._format_coords(self.ctx.centroid_negative_charge)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Centroid of (+)-ve charges: (X, Y, Z)')} = {self._format_coords(self.ctx.centroid_positive_charge)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Centroid of all the atoms: (X, Y, Z)')} = {self._format_coords(self.ctx.centroid)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Origin of gridbox: (X, Y, Z)')} = {self._format_coords(self.ctx.grid_origin)}"
            )

            center_coords = [
                s + self.ctx.grid_spacing * (nd // 2)
                for s, nd in zip(self.ctx.grid_origin, self.ctx.grid_shape)
            ]
            summary_lines.append(
                f"{indent}{field_format.format('Center of gridbox: (X, Y, Z)')} = {self._format_coords(center_coords)}"
            )

        return "\n".join(summary_lines)

    def summarize_parentrun(self, indent_spaces, field_width, format_specifier="s"):
        indent = " " * indent_spaces
        field_format = f"{{:{field_width}{format_specifier}}}"
        summary_lines = []

        summary_lines.append(
            f"{indent}{field_format.format('Scale of parent-run')} = {self.ctx.scale_parentrun:12.04f}"
        )
        # summary_lines.append(
        #     f"{indent}{field_format.format('Origin of parent-run gridbox: (X, Y, Z)')} = {self._format_coords(self.dc.grid_origin_parentrun)}"
        # )
        summary_lines.append(
            f"{indent}{field_format.format('Center of parent-run gridbox: (X, Y, Z)')} = {self._format_coords(self.ctx.grid_center_parentrun)}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('grid_size parent-run: (nX, nY, nZ)')} = {self._format_grid_size(self.ctx.grid_shape_parentrun)}"
        )

        return "\n".join(summary_lines)

    def _write_cube_and_verify(
        self,
        filename,
        scale,
        grid_center,
        grid_shape,
        data_1d,
        content,
        format_type,
        binary_precision,
        is_4d=False,
        verify=True,
    ):
        """
        Writes a cube file (or 4D cube) and optionally verifies its content after reading.

        Args:
            filename (str): Path to the cube file.
            scale (float): Grid scaling factor.
            grid_center (numpy.ndarray): Center of the grid.
            grid_shape (tuple): Shape of the grid (or 4D shape).
            data_1d (numpy.ndarray): 1D data array to write.
            content (str): Comment string for the cube file.
            format_type (str): Cube file format ('cube' or 'phi').
            binary_precision (type): Binary format output data precision.
            is_4d (bool): True for writing 4D cube, False for 3D.
            verify (bool): True to verify file content after writing.

        Returns:
            bool: True if writing and verification (if enabled) are successful, False otherwise.
        """
        if is_4d:
            # Write 4D cube file
            wrt.write_cube_4d(
                filename,
                scale,
                grid_center,
                grid_shape,
                data_1d,
                content,
                binary_precision=binary_precision,
                format=format_type,
            )
            # 4D verification is not implemented yet.
            vprint(
                DEBUG,
                self._VERBOSITY,
                f"4D cube written to {filename}. Verification not implemented.",
            )
            return True  # Assuming success for 4D until verification is added

        # Write 3D cube file
        wrt.write_cube(
            filename,
            scale,
            grid_center,
            grid_shape,
            data_1d,
            binary_precision=binary_precision,
            content=content,
            format=format_type,
        )

        if not verify:
            vprint(DEBUG, f"Verification skipped for {filename}")
            return True

        # Read back the written cube file for verification
        (
            read_scale,
            read_grid_center,
            read_grid_shape,
            read_data_3d,
            read_origin_bohr,
            read_vectors_bohr,
            read_comment,
            read_data_type_comment,
            read_endianness,
            read_marker,
        ) = rdr.read_cube(filename, format=format_type)

        # Verification checks
        if not np.isclose(scale, read_scale):
            vprint(
                ERROR,
                self._VERBOSITY,
                f"ERROR: Scale mismatch in {filename}: {scale} != {read_scale}",
            )
            return False
        if not np.allclose(grid_center, read_grid_center):
            vprint(
                ERROR,
                self._VERBOSITY,
                f"ERROR: Grid center mismatch in {filename}: {grid_center} != {read_grid_center}",
            )
            return False
        if not np.array_equal(grid_shape, read_grid_shape):
            vprint(
                ERROR,
                self._VERBOSITY,
                f"ERROR: Grid shape mismatch in {filename}: {grid_shape} != {read_grid_shape}",
            )
            return False
        if data_1d.ndim == 1:
            read_data_1d_ravelled = read_data_3d.ravel()
            if not np.allclose(data_1d, read_data_1d_ravelled):
                vprint(ERROR, self._VERBOSITY, f"ERROR: Data mismatch in {filename}")
                max_diff = np.max(np.abs(data_1d - read_data_1d_ravelled))
                vprint(WARNING, self._VERBOSITY, f"Max difference: {max_diff}")
                return False
        elif data_1d.ndim == 3:
            if not np.allclose(data_1d, read_data_3d):
                vprint(ERROR, self._VERBOSITY, f"ERROR: Data mismatch in {filename}")
                max_diff = np.max(np.abs(data_1d - read_data_3d))
                vprint(WARNING, self._VERBOSITY, f"Max difference: {max_diff}")
                return False

        binary_data_type_str = (
            "binary float64 (Fortran unformatted, little-endian)"
            if data_1d.dtype == np.float64
            else "binary float32 (Fortran unformatted, little-endian)"
        )
        data_type_comment = (
            f"Data type: {data_1d.dtype}"
            if format_type != "phi"
            else f"Data type: {binary_data_type_str}"
        )
        written_data_type_comment = f"Gaussian cube {content} ({'binary format Fortran unformatted, little-endian' if format_type == 'phi' else 'text format'}, {data_type_comment})"
        # if written_data_type_comment != read_data_type_comment:
        #     vprint(
        #         PRINT_MANDATORY,
        #         f"ERROR: Comment mismatch in {filename}: {written_data_type_comment} != {read_data_type_comment}",
        #     )
        #     return False

        vprint(DEBUG, self._VERBOSITY, f"Verification successful for {filename}")
        return True

    def _write_spatial_maps(self):
        """
        Writes spatial maps (density and surface) to cube files and verifies them.

        This method handles writing of Gaussian density and surface maps as cube files,
        based on user-defined output parameters and dielectric model settings.
        It calls `_write_cube_and_verify` to perform the actual writing and verification.
        """
        out_density = self.inp.get_param("out_density")
        dielectric_model = self.inp.get_param_value("dielectric_model")
        out_surf = self.inp.get_param("out_surf")

        # Write Gaussian density map if requested and dielectric model is Gaussian
        if (
            dielectric_model.int_value == DielectricModel.GAUSSIAN.int_value
            and out_density
            and out_density.active
        ):
            density_format = out_density.get_attribute("format")
            density_bin_precision = (
                np.float32
                if out_density.get_attribute("precision") == "single"
                else np.float64
            )
            if out_density.get_attribute("point") in ("both", "grid"):
                content = f"Gaussian density map: m={self.inp.get_param_value('gaussian_exponent')}, sigma={self.inp.get_param_value('sigma')}"
                if not self._write_cube_and_verify(
                    out_density.get_attribute("file"),
                    self.ctx.scale,
                    self.ctx.grid_center,
                    self.ctx.grid_shape,
                    self.ctx.gauss_density_map_1d,
                    content,
                    format_type=density_format,
                    binary_precision=density_bin_precision,
                ):
                    print("Verification failed for density map.")

            if out_density.get_attribute("point") in ("both", "mid"):
                datafile = self._get_phase_specific_filename(
                    out_density.get_attribute("file"), "midpoint"
                )
                content = f"Gaussian density map at midpoints: m={self.inp.get_param_value('gaussian_exponent')}, sigma={self.inp.get_param_value('sigma')}"
                data_shape_4d = np.array(list(self.ctx.grid_shape) + [3], dtype=int)
                if not self._write_cube_and_verify(
                    datafile,
                    self.ctx.scale,
                    self.ctx.grid_center,
                    data_shape_4d,
                    self.ctx.gauss_density_map_midpoints_1d,
                    content,
                    format_type=density_format,
                    binary_precision=density_bin_precision,
                    is_4d=True,
                    verify=False,  # Verification for 4D cube is skipped
                ):
                    print("Verification failed for density map.")

        # Write surface map if requested
        if out_surf and out_surf.active:
            surf_format = out_surf.get_attribute("format")
            surf_bin_precision = (
                np.float32
                if out_surf.get_attribute("precision") == "single"
                else np.float64
            )
            surface_method = self.inp.get_param_value("surface_method")
            if surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
                content = (
                    f"Surface map: srfexp={self.inp.get_param_value('surface_density_exponent')}, "
                    "m={self.inp.get_param_value('gaussian_exponent')}, sigma={self.inp.get_param_value('sigma')}"
                )
                data_1d = self.ctx.surface_map_1d
            elif surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
                content = "Surface map also known as idebmap generated with GAUSSIONCUTOFF method."
                data_1d = self.ctx.ion_exclusion_map_1d
            elif surface_method.int_value == SurfaceMethod.VDW.int_value:
                content = "Surface map also known as idebmap generated with VDW method."
                data_1d = self.ctx.dielectric_boundary_map_1d
            else:
                return  # Exit if surface_method is not recognized

            if not self._write_cube_and_verify(
                out_surf.get_attribute("file"),
                self.ctx.scale,
                self.ctx.grid_center,
                self.ctx.grid_shape,
                data_1d,
                content,
                format_type=surf_format,
                binary_precision=surf_bin_precision,
            ):
                vprint(DEBUG, self._VERBOSITY, "Verification failed for surface map.")

    def _write_phase_dependent_maps(self, isvacuum):
        """
        Writes phase-dependent maps (epsilon, phi, and zeta-phi) to files and verifies them.

        This method writes epsilon maps (grid point and midpoint), phi maps, and zeta-phi maps
        to cube or other format files based on user-defined output parameters. It distinguishes
        between vacuum and water phase maps using the `isvacuum` flag and calls `_write_cube_and_verify`
        for cube file writing and verification. Zeta-phi map writing is handled separately.

        Args:
            isvacuum (bool): True if writing vacuum phase maps, False for water phase maps.
        """
        out_eps = self.inp.get_param("out_eps")
        out_phi = self.inp.get_param("out_phi")
        out_zphi = self.inp.get_param("out_zphi")
        surface_method = self.inp.get_param_value("surface_method")
        srfexp = self.inp.get_param_value("surface_density_exponent")
        m = self.inp.get_param_value("gaussian_exponent")
        sigma = self.inp.get_param_value("sigma")
        is_zeta_on = self.inp.get_param_value("zeta_potential")

        # Handle epsilon map output
        if out_eps and out_eps.active:
            point_type = out_eps.get_attribute("point")
            media_type = out_eps.get_attribute("media")
            eps_format = out_eps.get_attribute("format")
            eps_bin_precision = (
                np.float32
                if out_eps.get_attribute("precision") == "single"
                else np.float64
            )

            # Write grid point epsilon map if requested
            if point_type in ("both", "grid"):
                datalabel = "vacuum" if isvacuum else "water"
                datafile = out_eps.get_attribute("file")
                if media_type == "both":
                    datafile = self._get_phase_specific_filename(
                        datafile, datalabel
                    )  # Generate phase-specific filename
                datamap = (
                    self.ctx.epsilon_map_1d
                )  # Epsilon map is same for both phases, using dc.epsilon_map_1d

                epsmap_title = self._construct_eps_map_title(
                    surface_method, srfexp, m, sigma, datalabel
                )  # Construct title string

                if datamap is not None:
                    self._write_cube_and_verify(
                        datafile,
                        self.ctx.scale,
                        self.ctx.grid_center,
                        self.ctx.grid_shape,
                        datamap,
                        epsmap_title,
                        format_type=eps_format,
                        binary_precision=eps_bin_precision,
                    )

            # Write midpoint epsilon map if requested (4D cube)
            if point_type in ("both", "mid"):
                datalabel = "vacuum_mids" if isvacuum else "water_mids"
                datafile = out_eps.get_attribute("file")

                if media_type == "both":
                    datafile = self._get_phase_specific_filename(
                        datafile, datalabel
                    )  # Generate phase-specific filename
                datamap = (
                    self.ctx.epsilon_map_midpoints_vacuum_1d  # Correctly select phase-specific midpoint map
                    if isvacuum
                    else self.ctx.epsilon_map_midpoints_water_1d
                )

                epsmap_title = self._construct_eps_map_title(
                    surface_method, srfexp, m, sigma, datalabel
                )  # Construct title string

                if datamap is not None:
                    data_shape_4d = np.array(
                        list(self.ctx.grid_shape) + [3], dtype=int
                    )  # Shape for 4D cube
                    self._write_cube_and_verify(
                        datafile,
                        self.ctx.scale,
                        self.ctx.grid_center,
                        data_shape_4d,
                        datamap,
                        epsmap_title,
                        format_type=eps_format,
                        binary_precision=eps_bin_precision,
                        is_4d=True,
                        verify=False,  # Verification for 4D cube is skipped
                    )

        # Handle phi map output
        if out_phi and out_phi.active:
            datalabel = "vacuum" if isvacuum else "water"
            datafile = out_phi.get_attribute("file")
            phi_format = out_phi.get_attribute("format")
            phi_bin_precision = (
                np.float32
                if out_phi.get_attribute("precision") == "single"
                else np.float64
            )

            if out_phi.get_attribute("media") == "both":
                datafile = self._get_phase_specific_filename(
                    datafile, datalabel
                )  # Generate phase-specific filename
            datamap = (
                self.ctx.phimap_in_vacuum if isvacuum else self.ctx.phimap_in_water
            )  # Correctly select phase-specific phimap

            phimap_title = self._construct_phi_map_title(
                srfexp, m, sigma, datalabel, surface_method
            )  # Construct phi map title

            if datamap is not None:
                self._write_cube_and_verify(
                    datafile,
                    self.ctx.scale,
                    self.ctx.grid_center,
                    self.ctx.grid_shape,
                    datamap,
                    phimap_title,
                    format_type=phi_format,
                    binary_precision=phi_bin_precision,
                )

        # Handle zeta-phi output
        if is_zeta_on and (not isvacuum) and (out_zphi and out_zphi.active):
            datafile = out_zphi.get_attribute("file")
            wrt.write_zeta_phi(
                datafile,
                self.ctx.grid_center,
                self.ctx.zeta_surf_grid_coords,
                self.ctx.zeta_surf_grid_indices,
                self.ctx.num_zeta_surf_grid_coords,
                self.ctx.phimap_in_water,  # Note: zeta-phi is written only for water phase in original code
            )
            vprint(
                DEBUG,
                self._VERBOSITY,
                f"Zeta-phi written to {datafile}. Verification not implemented.",
            )  # Verification not implemented

    def _get_phase_specific_filename(self, datafile, datalabel):
        """
        Generates a phase-specific filename by inserting phase label before the extension.

        Args:
            datafile (str): Original filename.
            datalabel (str): Phase label to insert (e.g., "vacuum", "water").

        Returns:
            str: Phase-specific filename.
        """
        return (
            datafile[: datafile.rindex(".")]
            + f"_{datalabel}"
            + datafile[datafile.rindex(".") :]
        )

    def _construct_eps_map_title(self, surface_method, srfexp, m, sigma, datalabel):
        """
        Constructs a title string for epsilon map cube files based on surface method.

        Args:
            surface_method (SurfaceMethod): Surface method used.
            srfexp (float): Surface density exponent.
            m (float): Gaussian exponent.
            sigma (float): Sigma value for Gaussian.
            datalabel (str): Data label for the phase.

        Returns:
            str: Epsilon map title string.
        """
        if surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
            return f"gridpoint-epsmap({datalabel}): srfexp={srfexp} m={m} sigma={sigma}"
        else:
            return f"gridpoint-epsmap({datalabel})"

    def _construct_phi_map_title(self, srfexp, m, sigma, datalabel, surface_method):
        """
        Constructs a title string for phi map cube files based on surface method.

        Args:
            srfexp (float): Surface density exponent.
            m (float): Gaussian exponent.
            sigma (float): Sigma value for Gaussian.
            datalabel (str): Data label for the phase.
            surface_method (SurfaceMethod): Surface method used.

        Returns:
            str: Phi map title string.
        """
        if surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
            return f"phimap({datalabel}): srfexp={srfexp} m={m} sigma={sigma}"
        else:
            return f"phimap({datalabel})"

    def _log_epsilon_map_shapes(self, phase_name, epsilon_map, epsilon_map_midpoints):
        """
        Logs the shapes of epsilon maps if verbosity level is sufficient.

        Args:
            phase_name (str): Name of the phase (e.g., "vacuum", "water").
            epsilon_map (numpy.ndarray): Epsilon map array.
            epsilon_map_midpoints (numpy.ndarray): Epsilon map at midpoints array.
        """
        # Check verbosity level to decide if logging is needed
        if self._VERBOSITY <= DEBUG:
            if epsilon_map is None:
                print(f"self.dc.epsilon_map_1d= None")
            else:
                print(f"self.dc.epsilon_map_1d.shape= {epsilon_map.shape}")

            epsilon_map_midpoints_attr_name = f"epsilon_map_midpoints_{phase_name}_1d"
            if epsilon_map_midpoints is None:
                print(f"self.dc.{epsilon_map_midpoints_attr_name}= None")
            else:
                print(
                    f"self.dc.{epsilon_map_midpoints_attr_name}.shape= {epsilon_map_midpoints.shape}"
                )

    def _determine_cuda_thread_count(self):
        """Determine the optimal thread count based on grid points."""
        n_grid_points = (
            self.ctx.grid_shape[0] * self.ctx.grid_shape[1] * self.ctx.grid_shape[2]
        )
        n_threads = 512
        n_blocks = (n_grid_points + 2 * n_threads - 1) // (2 * n_threads)
        while n_blocks < n_threads:
            n_threads = n_threads // 2
            n_blocks = (n_grid_points + 2 * n_threads - 1) // (2 * n_threads)
        if n_threads == 0:
            n_threads = 1
        return n_threads

    def _initialize_space_obj(self):
        """Initialize the space object with parameters from input and runtime context/state."""
        is_focusing = (
            self.inp.get_param_value("boundary_condition").int_value
            == BoundaryCondition.FOCUSING.int_value
        )
        return self.space.Space(
            platform=self.platform,
            is_surf_midpoints=self.inp.get_param_value("midpoint_dielectric_gaussian"),
            scale=self.ctx.scale,
            exdi=self.inp.get_param_value("exdi"),
            gapdi=self.inp.get_param_value("gapdi"),
            indi=self.inp.get_param_value("indi"),
            media_epsilon=self.ctx.media_epsilon,
            probe_radius=self.inp.get_param_value("probe_radius"),
            probe_radius2=self.inp.get_param_value("probe_radius2"),
            debye_length=self.ctx.debye_length,
            salt_radius=self.inp.get_param_value("ions_radii"),
            gaussian_sigma=self.inp.get_param_value("gaussian_sigma"),
            gaussian_exponent=self.inp.get_param_value("gaussian_exponent"),
            max_atom_radius=self.ctx.max_atom_radius,
            verbosity=self._VERBOSITY,
            dielectric_model=self.inp.get_param("dielectric_model"),
            surface_method=self.inp.get_param_value("surface_method"),
            surface_density_exponent=self.inp.get_param_value(
                "surface_density_exponent"
            ),
            surface_offset=self.inp.get_param_value("surface_offset"),
            grid_shape=self.ctx.grid_shape,
            grid_origin=self.ctx.grid_origin,
            atoms_data=self.ctx.atoms_data,
            objects_data=self.ctx.objects_data,
            is_focusing=is_focusing,
            grid_shape_parentrun=self.ctx.grid_shape_parentrun,
            grid_origin_parentrun=self.ctx.grid_origin_parentrun,
            acenter=self.inp.gridbox_center,
            num_objects=len(self.ctx.objects_data) // 2,
            num_molecules=1,
            use_zeta_surf=self.inp.get_param_value("zeta_potential"),
            zeta_distance=self.inp.get_param_value("zeta_distance"),
            coords_by_axis_min=self.ctx.coords_by_axis_min,
            coords_by_axis_max=self.ctx.coords_by_axis_max,
            enabled_nonpolar_energy=self.inp.get_param(
                "calculate_energies"
            ).is_attribute_inuse("np"),
        )

    def _reset_phase_dependent_maps(self, vacuum):
        """Resets phase-dependent maps in RuntimeContext to None to save memory."""
        if vacuum:
            self.ctx.epsilon_map_midpoints_vacuum_1d = None
            self.ctx.grad_eps_dot_gad_coul_vacuum_1d = None
            self.ctx.grad_epsgauss_map_vacuum_1d = None
            self.ctx.grad_epsilon_map_vacuum_1d = None
        else:
            self.ctx.epsilon_map_midpoints_water_1d = None
            self.ctx.grad_eps_dot_gad_coul_water_1d = None
            self.ctx.grad_epsgauss_map_water_1d = None
            self.ctx.grad_epsilon_map_water_1d = None

    def _process_phase_rpbe(self, vacuum, n_threads, space_obj):
        """Processes RPBE calculation for a given phase (vacuum or water)."""
        phase_name = "vacuum" if vacuum else "water"

        tic_epscalc = time.perf_counter()
        space_obj.calc_phase_spatial_epsilon_map_midpoints(
            is_regularized=True, vacuum=vacuum
        )
        self.ctx.epsilon_map_1d = space_obj.epsilon_map_1d
        if vacuum:
            self.ctx.epsilon_map_midpoints_vacuum_1d = (
                space_obj.epsilon_map_midpoints_1d
            )
        else:
            self.ctx.epsilon_map_midpoints_water_1d = space_obj.epsilon_map_midpoints_1d
        self.timings.update(space_obj.timings)
        self._write_spatial_maps()

        if self._VERBOSITY <= DEBUG:
            eps_map_name = (
                "self.dc.epsilon_map_midpoints_vacuum_1d"
                if vacuum
                else "self.dc.epsilon_map_midpoints_water_1d"
            )
            eps_map = (
                self.ctx.epsilon_map_midpoints_vacuum_1d
                if vacuum
                else self.ctx.epsilon_map_midpoints_water_1d
            )
            if self.ctx.epsilon_map_1d is None:
                print("self.dc.epsilon_map_1d=", None)
            else:
                print("self.dc.epsilon_map_1d.shape=", self.ctx.epsilon_map_1d.shape)
            if eps_map is None:
                print(f"{eps_map_name}=", None)
            else:
                print(
                    f"{eps_map_name}.shape=",
                    eps_map.shape,
                )
        toc_epscalc = time.perf_counter()
        self.timings[f"Calculating epsilon map in {phase_name}"] = "{:0.3f}".format(
            toc_epscalc - tic_epscalc
        )

        rpbe_solver = self.RPBESolver(
            self.platform,
            self._VERBOSITY,
            n_threads,
            self.ctx.grid_shape,
            self.ctx.coulomb_map_1d,
            self.ctx.grad_coulomb_map_1d,
        )
        # Solve RPBE: for the current phase (vacuum or water)
        if self._VERBOSITY <= INFO:
            print(f"\n    RPBE> run is starting for solute in {phase_name} phase.")

        non_zero_salt = self.inp.get_param_value("salt") != 0.0
        ion_exclusion_map_1d_args = self.ctx.surface_map_1d  # Default surface map
        surface_method = self.inp.get_param_value("surface_method")
        if surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
            ion_exclusion_map_1d_args = (
                self.ctx.ion_exclusion_map_1d
            )  # Use ion exclusion map for GAUSSIANCUTOFF
        output_phimap = rpbe_solver.run(
            vacuum=vacuum,
            non_zero_salt=non_zero_salt,
            bound_cond=BoundaryCondition.COULOMBIC,
            ion_exclusion_method=IonExclusionRegion.SOLUTESURFACE,
            gaussian_exponent=self.inp.get_param_value("gaussian_exponent"),
            itr_block_size=self.inp.get_param_value("iteration_block_size"),
            max_linear_iters=self.inp.get_param_value("linit"),
            scale=self.ctx.scale,
            exdi=self.inp.get_param_value("exdi"),
            gapdi=self.inp.get_param_value("gapdi"),
            indi=self.inp.get_param_value("indi"),
            debye_length=self.ctx.debye_length,
            total_pve_charge=self.ctx.positive_charge,
            total_nve_charge=self.ctx.negative_charge,
            max_rms=self.inp.get_param_value("max_rmsd"),
            max_dphi=self.inp.get_param_value("max_delta_phi"),
            check_dphi=self.inp.get_param("max_delta_phi").active,
            epkt=self.ctx.epkt,
            approx_zero=self.delphi_real(ConstDelPhi.ApproxZero.value),
            grid_shape=self.ctx.grid_shape,
            grid_origin=self.ctx.grid_origin,
            atoms_data=self.ctx.atoms_data,
            density_map_1d=self.ctx.gauss_density_map_1d,
            solute_surface_map_1d=self.ctx.surface_map_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d_args,
            epsilon_map_1d=self.ctx.epsilon_map_1d,
            epsmap_midpoints_1d=(
                self.ctx.epsilon_map_midpoints_vacuum_1d
                if vacuum
                else self.ctx.epsilon_map_midpoints_water_1d
            ),  # Correct epsilon map based on phase
            centroid_pve_charge=self.ctx.centroid_positive_charge,
            centroid_nve_charge=self.ctx.centroid_negative_charge,
            grad_surface_map_1d=self.ctx.grad_surface_map_1d,
        )

        self.timings.update(rpbe_solver.timings)

        if vacuum:
            self.ctx.phimap_in_vacuum = output_phimap
        else:
            self.ctx.phimap_in_water = output_phimap

        if vacuum:
            self.ctx.coulomb_map_1d = rpbe_solver.coulomb_map_1d
            self.ctx.grad_coulomb_map_1d = rpbe_solver.grad_coulomb_map_1d

        self.ctx.grad_epsgauss_map_vacuum_1d = rpbe_solver.grad_epsin_map_1d
        self.ctx.grad_epsilon_map_vacuum_1d = rpbe_solver.grad_epsmap_1d
        self.ctx.grad_eps_dot_gad_coul_vacuum_1d = rpbe_solver.eps_dot_coul_map_1d

        toc_rpb_phase = time.perf_counter()
        self.timings[f"Solving RPBE in {phase_name}"] = "{:0.3f}".format(
            toc_rpb_phase - toc_epscalc
        )

        self._write_phase_dependent_maps(vacuum)
        if (
            self.inp.get_param_value("memory_state").int_value
            == MemoryState.MINIMAL.int_value
        ):
            self._reset_phase_dependent_maps(
                vacuum
            )  # Use reset function to clear phase-dependent maps

    def _run_rpbe(self):
        if self.inp.get_param_value("biomodel").int_value == BioModel.RPBE.int_value:
            vacuum = True
            tic_gauss = time.perf_counter()
            n_threads = self._determine_cuda_thread_count()
            space_obj = self._initialize_space_obj()
            space_obj.run(n_threads, self.ctx)
            space_obj.update_runtime_context(ctx=self.ctx)

            self._process_phase_rpbe(
                vacuum, n_threads, space_obj
            )  # Process vacuum phase RPBE

            self.calculate_all_energies(
                vacuum=True,
                final=False,
                ctx=self.ctx,
                erg_settings=self.energy_settings,
            )

            # Let's solve RPBE for the water phase
            # Following steps are performed in _process_phase_rpbe function:
            # 1. Calculate the epsilon_map in water-phase at grid points and midpoints
            # 2. Solve RPBE with vacuum=False
            vacuum = False
            self._process_phase_rpbe(
                vacuum, n_threads, space_obj
            )  # Process water phase RPBE

            self.calculate_all_energies(
                vacuum=False,
                final=True,
                ctx=self.ctx,
                erg_settings=self.energy_settings,
            )

            self._write_phase_dependent_maps(
                vacuum
            )  # vacuum is False here, so writes water phase maps
            if (
                self.inp.get_param_value("memory_state").int_value
                == MemoryState.MINIMAL.int_value
            ):
                self._reset_phase_dependent_maps(vacuum)  # Reset water phase maps

    def _process_phase_pbe(self, vacuum, n_threads, space_obj):
        """Processes PBE calculation for a given phase (vacuum or water)."""
        phase_name = "vacuum" if vacuum else "water"

        tic_epscalc = time.perf_counter()
        has_dencut = self.inp.get_param("density_cutoff").issupplied
        has_srfcut = self.inp.get_param("surface_cutoff").issupplied
        gaussian_density_cutoff = 0.0
        gaussian_epsilon_cutoff = self.inp.get_param_value("indi")
        if has_dencut or not (has_srfcut or has_srfcut):
            gaussian_density_cutoff = self.inp.get_param_value("density_cutoff")
        if has_srfcut:
            gaussian_epsilon_cutoff = self.inp.get_param_value("surface_cutoff")
        space_obj.calc_phase_spatial_epsilon_map_midpoints(
            is_regularized=False,
            vacuum=vacuum,
            gaussian_density_cutoff=gaussian_density_cutoff,
            gaussian_epsilon_cutoff=gaussian_epsilon_cutoff,
        )
        self.ctx.epsilon_map_1d = space_obj.epsilon_map_1d
        if vacuum:
            self.ctx.epsilon_map_midpoints_vacuum_1d = (
                space_obj.epsilon_map_midpoints_1d
            )
        else:
            self.ctx.epsilon_map_midpoints_water_1d = space_obj.epsilon_map_midpoints_1d
        self.timings.update(space_obj.timings)
        if not vacuum:
            self._write_spatial_maps()

        if self._VERBOSITY <= TRACE:
            eps_map_name = (
                "self.dc.epsilon_map_midpoints_vacuum_1d"
                if vacuum
                else "self.dc.epsilon_map_midpoints_water_1d"
            )
            eps_map = (
                self.ctx.epsilon_map_midpoints_vacuum_1d
                if vacuum
                else self.ctx.epsilon_map_midpoints_water_1d
            )
            if self.ctx.epsilon_map_1d is None:
                print("self.dc.epsilon_map_1d=", None)
            else:
                print("self.dc.epsilon_map_1d.shape=", self.ctx.epsilon_map_1d.shape)
            if eps_map is None:
                print(f"{eps_map_name}=", None)
            else:
                print(
                    f"{eps_map_name}.shape=",
                    eps_map.shape,
                )
        toc_epscalc = time.perf_counter()
        self.timings[f"Calculating epsilon map in {phase_name}"] = "{:0.3f}".format(
            toc_epscalc - tic_epscalc
        )

        pbe_solver = self.PBESolver(
            self.platform,
            self._VERBOSITY,
            n_threads,
            self.inp.get_param_value("solver").name.lower(),
            self.ctx.grid_shape,
        )

        # Solve PBE: for the current phase (vacuum or water)
        vprint(
            INFO,
            self._VERBOSITY,
            f"\n    PBE> run is starting for solute in {phase_name} phase.",
        )

        ion_exclusion_map_1d_args = self.ctx.surface_map_1d  # Default surface map
        surface_method = self.inp.get_param_value("surface_method")
        if surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
            ion_exclusion_map_1d_args = self.ctx.ion_exclusion_map_1d
        elif surface_method.int_value == SurfaceMethod.VDW.int_value:
            ion_exclusion_map_1d_args = (
                self.ctx.dielectric_boundary_map_1d == False
            )  # Use dielectric boundary map for VDW

        if self._VERBOSITY <= DEBUG:
            grid_sz = self.ctx.grid_shape
            vprint(DEBUG, self._VERBOSITY,
                "ion_exclusion_map_1d_args(mid-x-slice).shape=",
                ion_exclusion_map_1d_args.shape,
            )
            if self._VERBOSITY <= TRACE:
                begin = grid_sz[0] * grid_sz[1] * grid_sz[2] // 2
                for i in range(grid_sz[1]):
                    end = begin + grid_sz[2]
                    print("(")
                    for j in range(grid_sz[2]):
                        print(ion_exclusion_map_1d_args[begin + j], ", ", end="")
                    print(")")
                    begin = end
                print("\n\n\n")

        non_zero_salt = self.inp.get_param_value("salt") != 0.0
        nonlinear_coupling_steps = 0
        pb_approximation = self.inp.get_param_value("pb_approximation")
        if pb_approximation.int_value == PBApproximation.NONLINEAR.int_value:
            nonlinear_coupling_steps = self.inp.get_param_value(
                "nonlinear_coupling_steps"
            )
        output_phimap_1d = pbe_solver.solve_pbe(
            vacuum=vacuum,
            bound_cond=self.inp.get_param_value("boundary_condition"),
            dielectric_model=self.inp.get_param_value("dielectric_model"),
            gaussian_exponent=self.inp.get_param_value("gaussian_exponent"),
            itr_block_size=self.inp.get_param_value("iteration_block_size"),
            max_linear_iters=self.inp.get_param_value("linit"),
            max_nonlinear_iters=self.inp.get_param_value("nonlinit"),
            max_nonlinear_coupling_dphi=self.inp.get_param_value(
                "max_nonlinear_coupling_delta_phi"
            ),
            coupling_steps=nonlinear_coupling_steps,
            manual_relaxation_value=self.inp.get_param_value("nlrelpar"),
            scale=self.ctx.scale,
            scale_parentrun=self.ctx.scale_parentrun,
            exdi=self.inp.get_param_value("exdi"),
            indi=self.inp.get_param_value("indi"),
            debye_length=self.ctx.debye_length,
            non_zero_salt=non_zero_salt,
            total_pve_charge=self.ctx.positive_charge,
            total_nve_charge=self.ctx.negative_charge,
            max_rms=self.inp.get_param_value("max_rmsd"),
            max_dphi=self.inp.get_param_value("max_delta_phi"),
            check_dphi=self.inp.get_param("max_delta_phi").active,
            epkt=self.ctx.epkt,
            approx_zero=self.delphi_real(ConstDelPhi.ApproxZero.value),
            grid_shape=self.ctx.grid_shape,
            grid_origin=self.ctx.grid_origin,
            grid_shape_parentrun=self.ctx.grid_shape_parentrun,
            grid_origin_parentrun=self.ctx.grid_origin_parentrun,
            atoms_data=self.ctx.atoms_data,
            density_map_1d=self.ctx.gauss_density_map_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d_args,
            epsilon_map_1d=self.ctx.epsilon_map_1d,
            epsmap_midpoints_1d=(
                self.ctx.epsilon_map_midpoints_vacuum_1d
                if vacuum
                else self.ctx.epsilon_map_midpoints_water_1d
            ),  # Correct epsilon map based on phase
            centroid_pve_charge=self.ctx.centroid_positive_charge,
            centroid_nve_charge=self.ctx.centroid_negative_charge,
            charged_gridpoints_1d=self.ctx.charged_gridpoints_1d,
            phimap_parentrun=self.ctx.phimap_parentrun,
        )
        # Note: solver returned phimap as 1d array, so must be reshaped to 3d for further use
        output_phimap_3d = output_phimap_1d.reshape(self.ctx.grid_shape)
        self.timings.update(pbe_solver.timings)

        if vacuum:
            self.ctx.phimap_in_vacuum = output_phimap_3d
        else:
            self.ctx.phimap_in_water = output_phimap_3d
            # In water phase the ion_exclusion_map_1d with Gaussian salt depends on epsilon distribution and
            # updated in solver. Thus, we must update the ion_exclusion_map_1d used in solver in ctx for
            # consistency for non-polar energy calculation
            if surface_method.int_value in {
                SurfaceMethod.GAUSSIAN.int_value,
                SurfaceMethod.GCS.int_value,
            }:
                self.ctx.ion_exclusion_map_1d = 1.0 - ion_exclusion_map_1d_args

        toc_pb_phase = time.perf_counter()
        self.timings[f"Solving PBE in {phase_name}"] = "{:0.3f}".format(
            toc_pb_phase - toc_epscalc
        )

    def _run_pbe(self):
        if self.inp.get_param_value("biomodel").int_value == BioModel.PBE.int_value:
            n_cuda_threads = self._determine_cuda_thread_count()
            tic_space_init = time.perf_counter()
            space_obj = self._initialize_space_obj()
            toc_space_init = time.perf_counter()
            self.timings[f"space initialization"] = "{:0.3f}".format(
                toc_space_init - tic_space_init
            )
            space_obj.run(n_cuda_threads, self.ctx)
            space_obj.update_runtime_context(ctx=self.ctx)
            toc_space_run = time.perf_counter()
            self.timings[f"space running"] = "{:0.3f}".format(
                toc_space_run - toc_space_init
            )
            vacuum = True  # Process vacuum phase first for PBE
            energy_grid_vacuum = 0.0
            energy_rxn_vacuum = 0.0
            time_energy_vac = 0.0
            dielectrc_model_value = self.inp.get_param_value("dielectric_model")
            if (
                dielectrc_model_value.int_value
                == DielectricModel.TWODIELECTRIC.int_value
            ):
                self.ctx.induced_surf_charge_positions = (
                    space_obj.induced_surf_charge_positions
                )
                self.ctx.dielectric_boundary_grids = space_obj.dielectric_boundary_grids

            if (
                dielectrc_model_value.int_value
                != DielectricModel.TWODIELECTRIC.int_value
            ):
                tic_pb_vacuum = time.perf_counter()
                self._process_phase_pbe(
                    vacuum, n_cuda_threads, space_obj
                )  # Process vacuum phase PBE
                toc_pb_vacuum = time.perf_counter()
                self.timings["Solving PBE in vacuum"] = "{:0.3f}".format(
                    toc_pb_vacuum - tic_pb_vacuum
                )
                self.calculate_all_energies(
                    vacuum=vacuum,
                    final=False,
                    ctx=self.ctx,
                    erg_settings=self.energy_settings,
                )

                self._write_phase_dependent_maps(vacuum)
                if (
                    self.inp.get_param_value("memory_state").int_value
                    == MemoryState.MINIMAL.int_value
                ):
                    self._reset_phase_dependent_maps(vacuum)

            vacuum = False  # Process water phase
            tic_pb_water = time.perf_counter()
            self._process_phase_pbe(
                vacuum, n_cuda_threads, space_obj
            )  # Process water phase PBE
            toc_pb_water = time.perf_counter()

            self.calculate_all_energies(
                vacuum=vacuum,
                final=True,
                ctx=self.ctx,
                erg_settings=self.energy_settings,
            )

            tic_pb_out_maps = time.perf_counter()
            self._write_phase_dependent_maps(
                vacuum
            )  # vacuum is False here, so writes water phase maps
            if (
                self.inp.get_param_value("memory_state").int_value
                == MemoryState.MINIMAL.int_value
            ):
                self._reset_phase_dependent_maps(vacuum)  # Reset water phase maps

            toc_pb_out_maps = time.perf_counter()
            self.timings["Writing solvent-phase output maps"] = "{:0.3f}".format(
                time_energy_vac + (toc_pb_out_maps - tic_pb_out_maps)
            )

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
            precision=self.platform.precision,
            dtype_int=self.delphi_int,
            dtype_real=self.delphi_real,
        )
        prm_acenter = self.inp.get_param("acenter")
        self.ctx.enforce_acenter = prm_acenter.issupplied
        self.ctx.acenter[:] = self.inp.gridbox_center.astype(self.delphi_real)[:]

        # Update RuntimeContext (ctx): Setup debylength
        self.ctx.atoms_summary(
            atoms=self.inp.atoms,
            objects=self.inp.objects,
            extremas_rule=self.inp.get_param_value("solute_extrema"),
            acenter=self.ctx.acenter,
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
        )
        # Abort if DIPOLAR boundary condition is requested for systems with exclusively +ve or -ve charges.
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
                raise ValueError(msg)

        # Setup for FOCUSING run if requested
        if self.inp.get_param_value("biomodel").int_value == BioModel.PBE.int_value:
            num_cuda_threads = self._determine_cuda_thread_count()
            if (
                self.inp.get_param_value("boundary_condition").int_value
                == BoundaryCondition.FOCUSING.int_value
            ):
                if (
                    self.inp.get_param_value("dielectric_model")
                    == DielectricModel.TWODIELECTRIC
                ):
                    in_phi = self.inp.get_param("in_phi")
                    if in_phi.issupplied:
                        self.ctx.prepare_focusing(
                            self.ctx.scale,
                            self.ctx.num_atoms,
                            self.ctx.num_objects,
                            self.ctx.grid_shape,
                            self.ctx.acenter,
                            self.ctx.atoms_data,
                        )
                        atoms_to_focus = {}
                        for this_atom in self.ctx.atoms_data:
                            atom_k = self.ctx.atoms_index_to_keys[
                                int(this_atom[LEN_ATOMFIELDS])
                            ]
                            atoms_to_focus[atom_k] = this_atom
                        (
                            self.ctx.scale_parentrun,
                            self.ctx.grid_center_parentrun,
                            self.ctx.grid_shape_parentrun,
                            self.ctx.phimap_parentrun,
                            read_origin_bohr,
                            read_vectors_bohr,
                            self.ctx.phimap_comment_parentrun,
                            read_data_type_comment,
                            self.ctx.phimap_endianness_parentrun,
                            self.ctx.phimap_marker_parentrun,
                        ) = rdr.read_cube(
                            in_phi.get_attribute("file"),
                            format=in_phi.get_attribute("format"),
                        )
                        self.ctx.grid_origin_parentrun = (
                            self.ctx.grid_center_parentrun
                            - (self.ctx.grid_shape_parentrun // 2)
                            * (1.0 / self.ctx.scale_parentrun)
                        )
                        # Note for focusing there may be a subset of all atoms, so re-summarize considering the change.
                        self.ctx.atoms_summary(
                            atoms_to_focus,
                            objects=self.inp.objects,
                            extremas_rule=self.inp.get_param_value("solute_extrema"),
                            acenter=self.ctx.acenter,
                            enforce_acenter=self.ctx.enforce_acenter,
                            is_focusing=True,
                        )
                    else:
                        raise ValueError(
                            "Parentrun phimap is not provided but required for FOCUSING."
                        )
                else:
                    raise ValueError(
                        "FOCUSING boundary condition is compatible with only TWODIELECTRIC dielectric model"
                    )
            else:
                # Let's make parentrun parameters same as current-run for non-focusing runs (unused here)
                # but parameters need to be passed to respect the common interface signature.
                self.ctx.scale_parentrun = self.ctx.scale
                self.ctx.grid_center_parentrun = self.ctx.grid_center
                self.ctx.grid_shape_parentrun = self.ctx.grid_shape
                # NOTE: dc.grid_origin is not set yet, first will be set then assigned to dc.grid_origin_parentrun
                self.ctx.grid_origin = self.ctx.setup_gridmap_3d(
                    self.ctx.grid_center,
                    self.ctx.grid_shape,
                    self.ctx.scale,
                )
                self.ctx.grid_origin_parentrun = self.ctx.grid_origin
                self.ctx.phimap_parentrun = np.zeros(
                    (3, 3, 3), dtype=self.delphi_real
                )  # dummy param for non-focusing

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
            # print(atom_data[ATOMFIELD_CHARGE], atom_data[3:6])

        # Update RuntimeContext (ctx): update the gridbox shape
        self.ctx.grid_shape = self.ctx.gridbox_size_to_shape_array()

        # Print out parameters summary for the run
        if self._VERBOSITY <= INFO:
            vprint(
                INFO,
                self._VERBOSITY,
                self.summary(indent_spaces=4, field_width=44, format_specifier="s"),
            )
            vprint(INFO, self._VERBOSITY, "")
            if (
                self.inp.get_param_value("boundary_condition")
                == BoundaryCondition.FOCUSING
            ):
                vprint(
                    INFO,
                    self._VERBOSITY,
                    self.summarize_parentrun(
                        indent_spaces=4, field_width=44, format_specifier="s"
                    ),
                )
            vprint(INFO, self._VERBOSITY, "")
            vprint(INFO, self._VERBOSITY, "=" * 90)
            vprint(INFO, self._VERBOSITY, "")

        toc_prep = time.perf_counter()
        self.timings["Setting up the grid"] = "{:0.3f}".format(toc_prep - tic_prep)

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

        if self.inp.get_param_value("biomodel").int_value == BioModel.RPBE.int_value:
            self._run_rpbe()
        elif self.inp.get_param_value("biomodel").int_value == BioModel.PBE.int_value:
            self._run_pbe()

        prm_out_frc = self.inp.get_param("out_frc")
        if prm_out_frc.issupplied:
            import pydelphi.site.writesite as wrts

            prm_in_frc = self.inp.get_param("in_frc")
            atoms_frc = rdr.read_frc(prm_in_frc.get_attribute("file"))

            wrts.write_frc_file(
                output_frc_file=prm_out_frc.get_attribute("file"),
                frc_atoms_dict=atoms_frc,
                grid_shape=self.ctx.grid_shape,
                percentage_fill=self.ctx.perfil,
                external_dielectric=self.ctx.external_dielectric_scaled * self.ctx.epkt,
                media_eps=self.ctx.media_epsilon,
                epkt=self.ctx.epkt,
                ion_strength=self.inp.get_param_value("salt"),
                ion_radius=self.inp.get_param_value("ions_radii"),
                linear_iteration_num=self.inp.get_param_value("linit"),
                non_linear_iteration_num=self.inp.get_param_value("nonlinit"),
                boundary_type=self.inp.get_param_value("boundary_condition"),
                file_map_record="frc map",
                probe_radii=self.inp.get_param_value("probe_radius"),
                box_center=self.ctx.grid_center,
                grid_offset=np.zeros(3, dtype=self.delphi_real),
                scale_factor=self.ctx.scale,
                potential_map=self.ctx.phimap_in_water,
                potential_upper_bond=np.max(self.ctx.phimap_in_water),
                dielectric_map_bool=self.ctx.dielectric_boundary_map_1d,
                surface_charge_pos_array=self.ctx.induced_surf_charge_positions,
                surface_charge_e_array=self.ctx.induced_surf_charges[::4],
                boundary_grid_array=self.ctx.dielectric_boundary_grids,
                charge_grid_num=self.ctx.charged_gridpoints_1d,
                charge_pos_array=self.ctx.atoms_data[:, 0:3],
                atomic_charge_list=self.ctx.atoms_data[:, ATOMFIELD_CHARGE],
                atom_eps_array=self.ctx.atoms_data[:, ATOMFIELD_MEDIA_ID],
                residue_num=0,
                out_atom_desc=self.inp.get_param("site").is_attribute_inuse("atom"),
                out_atom_coords=self.inp.get_param("site").is_attribute_inuse(
                    "coordinates"
                ),
                out_charge=self.inp.get_param("site").is_attribute_inuse("charge"),
                out_grid_pot=self.inp.get_param("site").is_attribute_inuse("potential"),
                out_field=self.inp.get_param("site").is_attribute_inuse("field"),
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
        prm_out_modpdb4 = self.inp.get_param("out_modpdb4")
        if prm_out_modpdb4.issupplied:
            out_file = prm_out_modpdb4.get_attribute("file")
            wrt.write_pqr(out_file, self.inp.atoms, objects=dict())

        self.ctx._reset_maps()
        print()
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
