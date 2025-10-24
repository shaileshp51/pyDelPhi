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

from typing import List, Tuple, Optional

import numpy as np, struct

from pydelphi.config.global_runtime import (
    delphi_real,
)

from pydelphi.constants import (
    ConstDelPhiInts,
)

RES_NUMBER_UNKNOWN = ConstDelPhiInts.ResidueNumberUnknown.value


def write_cube(
    filepath,
    scale_factor,
    grid_center,
    grid_shape,  # Expected as [nx, ny, nz]
    data_array,  # Expected shape (nx, ny, nz) C-order or 1D C-ravelled
    format="cube",
    binary_precision=np.float32,
    marker_size=8,
    content="scalar_data",
    allow_non_c_contiguous=False,
):
    """
    Writes a 3D scalar data array to a Gaussian cube file in specified format and precision.

    **Complies with H5cube-spec data block order (z fastest, y middle, x slowest).**

    **Input Data Array:**
    - Must be a NumPy array representing a grid with shape (nx, ny, nz).
    - Access is expected to be data_array[ix, iy, iz]. This corresponds to a **C-order** array
      in NumPy with shape (nx, ny, nz).
    - If 1D, it is assumed to be C-order ravelled from shape (nx, ny, nz).

    Args:
        filepath (str): Path to the output file.
        scale_factor (float): Scaling factor for grid spacing (grid points per Angstrom).
        grid_center (np.ndarray): 1D numpy array of size 3, representing grid center [x, y, z] in Angstroms.
        grid_shape (np.ndarray): 1D numpy array of size 3 (integers), [nx, ny, nz] grid dimensions.
        data_array (np.ndarray): NumPy array containing scalar data values.
                                  Expected shape (nx, ny, nz) (C-order) or 1D C-ravelled.
        format (str, optional):  'cube' (text) or 'phi' (binary). Defaults to 'cube'.
        binary_precision (dtype, optional): Data type for binary output ('phi' format only, np.float64 or np.float32).
                                 Defaults to np.float32 for phi format.
        marker_size (int, optional): Size of record markers in bytes for 'phi' format (4 or 8). Defaults to 8.
        content (str, optional): Description of the data content (e.g., 'potential', 'density').
                                 Included in the header comment. Defaults to 'scalar_data'.
        allow_non_c_contiguous (bool): If True, no-warning is emitted when non-C-contiguous 3D-arrays found.

    Raises:
        ValueError: If `data_array` is invalid, or invalid `format`, `binary_precision`, or `marker_size`.
        AssertionError: If the shape of the provided 3D `data_array` is NOT `(nx, ny, nz)` as given by `grid_shape`.
    """
    format = format.lower()
    if format not in ["cube", "phi"]:
        raise ValueError(f"Invalid format '{format}'. Must be 'cube' or 'phi'.")
    if format == "phi":
        if binary_precision not in [np.float64, np.float32]:
            raise ValueError(
                f"Invalid dtype '{binary_precision}' for phi format. Must be np.float64 or np.float32."
            )
        if marker_size not in [4, 8]:
            raise ValueError(
                f"Invalid marker_size '{marker_size}' for phi format. Must be 4 or 8."
            )
        binary_output = True
        binary_precision = binary_precision
    elif format == "cube":
        binary_output = False
        binary_precision = np.float64  # Dummy
        marker_size = 8  # Dummy

    _write_cube_gaussian_format(
        filepath,
        scale_factor,
        grid_center,
        grid_shape,
        data_array,
        binary_output,
        binary_precision,
        marker_size,
        content,
        allow_non_c_contiguous,
    )


def _write_cube_gaussian_format(
    filepath,
    scale_factor,
    grid_center,
    grid_shape,  # Expected as [nx, ny, nz]
    data_array,  # Expected shape (nx, ny, nz) C-order or 1D C-ravelled
    binary_output,
    binary_precision,
    marker_size,
    content,
    allow_non_c_contiguous,
):
    """
    Internal function to write a 3D scalar data array to Gaussian cube file format (cube or phi).
    Handles data reordering for standard cube format (Z-fastest).
    """
    angstrom_to_bohr = 0.5291772108
    step_size_angstrom = 1.0 / scale_factor
    origin_angstrom = grid_center - step_size_angstrom * (grid_shape - 1) / 2.0
    origin_bohr = origin_angstrom / angstrom_to_bohr
    step_size_bohr = step_size_angstrom / angstrom_to_bohr

    nx, ny, nz = grid_shape

    # Input data handling and reshape to C-order (nx, ny, nz)
    if not isinstance(data_array, np.ndarray):
        raise ValueError("Input data_array must be a NumPy array.")

    if data_array.ndim == 1:
        expected_1d_size = nx * ny * nz
        if data_array.size != expected_1d_size:
            raise ValueError(
                f"1D input array size ({data_array.size}) does not match expected size ({expected_1d_size}) for grid shape {grid_shape}."
            )
        # Assume 1D input is C-order ravelled, reshape to (nx, ny, nz) C-order
        data_3d = data_array.reshape((nx, ny, nz), order="C")
    elif data_array.ndim == 3:
        expected_data_array_shape = (nx, ny, nz)
        assert data_array.shape == expected_data_array_shape, (
            f"Shape mismatch! Expected 3D data_array shape (nx, ny, nz) = {expected_data_array_shape} "
            f"matching grid_shape {grid_shape}, but got shape {data_array.shape}."
        )
        # We assume the user intends this 3D array to be accessed [ix, iy, iz], which implies C-order for this shape.
        if not data_array.flags["C_CONTIGUOUS"]:
            if not allow_non_c_contiguous:
                print(
                    "Warning: Input 3D data_array is not C-contiguous. Writing will involve potential data copy for performance."
                )
        data_3d = data_array  # Use 3D input directly
    else:
        raise ValueError(
            f"Input data_array must be either 1D or 3D, but got {data_array.ndim} dimensions."
        )

    # Define orthogonal axis-aligned vectors in Bohr
    vectors_bohr = np.array(
        [
            [step_size_bohr, 0.0, 0.0],
            [0.0, step_size_bohr, 0.0],
            [0.0, 0.0, step_size_bohr],
        ]
    )

    with open(filepath, "wb" if binary_output else "w") as cube_file:
        _write_cube_header(
            cube_file,
            scale_factor,  # Pass Angstrom scale_factor for line 1
            grid_shape,
            grid_center,  # Pass Angstrom grid_center for line 1
            origin_bohr,  # Pass Bohr origin for third header line
            vectors_bohr,  # Pass Bohr vectors for lines 4-6
            binary_output,
            data_3d.dtype,  # Infer datatype from the C-order 3D array
            binary_precision,
            marker_size,
            content,
        )

        if binary_output:
            # Binary data order (C-order bytes for Z-fastest file)
            data_bytes = (
                data_3d.astype(binary_precision)
                .astype(np.dtype(binary_precision).newbyteorder("<"))  # Little-endian
                .tobytes(
                    order="C"
                )  # Get bytes in C-order, which matches Z-fastest file layout for (nx, ny, nz) C-array
            )
            record_size = len(data_bytes)
            marker_type_code = "q" if marker_size == 8 else "i"

            cube_file.write(
                struct.pack("<" + marker_type_code, record_size)
            )  # Write start marker
            cube_file.write(data_bytes)  # Write binary data block
            cube_file.write(
                struct.pack("<" + marker_type_code, record_size)
            )  # Write end marker
        else:
            # Text data loop order for Z-fastest file
            values_on_line = 0
            # Iterate x (slowest), then y (middle), then z (fastest) to write in Z-fastest file order
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        # Access the C-order (nx, ny, nz) array at data_3d[ix, iy, iz]
                        cube_file.write(f"{data_3d[ix, iy, iz]:13.5e}")
                        values_on_line += 1
                        if values_on_line == 6:
                            cube_file.write("\n")
                            values_on_line = 0
                    # Ensure a newline after each row of the fastest changing dimension (z)
                    if values_on_line > 0:
                        cube_file.write("\n")
                        values_on_line = 0  # Reset for the next line (y-row)


def _write_cube_header(
    cube_file,
    scale_factor_angstrom,  # Use Angstrom for line 1
    grid_shape,  # [nx, ny, nz]
    grid_center_angstrom,  # Use Angstrom for line 1
    origin_bohr,  # Use Bohr for line 3
    vectors_bohr,  # Use Bohr for lines 4-6
    binary_output,
    data_array_dtype,  # Original numpy dtype
    binary_precision,  # dtype for binary output
    marker_size,
    content,
):
    """Writes the header section of the cube file (internal function)."""
    nx, ny, nz = grid_shape

    binary_data_type_str = (
        "binary float64 (Fortran unformatted, little-endian)"
        if binary_precision == np.float64
        else "binary float32 (Fortran unformatted, little-endian)"
    )
    data_type_comment = (
        f"Data type: {data_array_dtype}"
        if not binary_output
        else f"Data type: {binary_data_type_str}"
    )

    # Define the header lines as strings
    # Line 1: scale_factor nx grid_center_x grid_center_y grid_center_z
    line1 = f"{scale_factor_angstrom:10.6f} {nx:6d} {grid_center_angstrom[0]:10.6f} {grid_center_angstrom[1]:10.6f} {grid_center_angstrom[2]:10.6f}\n"
    # Line 2: Comment
    line2 = f"Gaussian cube {content} ({'binary format Fortran unformatted, little-endian' if binary_output else 'text format'}, {data_type_comment})\n"
    # Line 3: Natoms origin_x origin_y origin_z (Natoms = 1 hardcoded as in user's original writer)
    line3 = f"{1:>5d} {origin_bohr[0]:>14.6f} {origin_bohr[1]:>14.6f} {origin_bohr[2]:>14.6f}\n"
    # Line 4: Nx vecX_x vecX_y vecX_z (Nx is positive, vectors are in Bohr)
    line4 = f"{nx:>5d} {vectors_bohr[0][0]:>14.6f} {vectors_bohr[0][1]:>14.6f} {vectors_bohr[0][2]:>14.6f}\n"
    # Line 5: Ny vecY_x vecY_y vecY_z
    line5 = f"{ny:>5d} {vectors_bohr[1][0]:>14.6f} {vectors_bohr[1][1]:>14.6f} {vectors_bohr[1][2]:>14.6f}\n"
    # Line 6: Nz vecZ_x vecZ_y vecZ_z
    line6 = f"{nz:>5d} {vectors_bohr[2][0]:>14.6f} {vectors_bohr[2][1]:>14.6f} {vectors_bohr[2][2]:>14.6f}\n"
    # Line 7: Atom data (1 atom hardcoded, Atomic_number charge pos_x pos_y pos_z in Bohr)
    # The charge is often 0.0 in cube files. The position (0.0, 0.0, 0.0) corresponds to the origin.
    line7 = f"{1:>5d} {0.0:>14.6f} {0.0:>14.6f} {0.0:>14.6f} {0.0:>14.6f}\n"  # Hardcoded dummy atom at origin

    header_lines = [line1, line2, line3, line4, line5, line6, line7]

    if binary_output:
        # Write header lines as binary records
        marker_type_code = "q" if marker_size == 8 else "i"
        little_endian = "<"
        for line in header_lines:
            line_bytes = line.encode("ascii")
            record_size = len(line_bytes)
            cube_file.write(
                struct.pack(little_endian + marker_type_code, record_size)
            )  # Start marker
            cube_file.write(line_bytes)  # Data
            cube_file.write(
                struct.pack(little_endian + marker_type_code, record_size)
            )  # End marker
    else:
        # Write header lines as text
        for line in header_lines:
            cube_file.write(line)


def write_cube_4d(
    filename_prefix: str,
    scale: float,
    grid_center: Tuple[float, float, float],
    grid_dimensions: Tuple[int, int, int, int],
    gridmap4d: np.ndarray[delphi_real],
    maptitle: str,
    binary_precision: type,
    dim4labels: Optional[List[str]] = None,
    format: str = "cube",
) -> None:
    """
    Writes a 4D grid map into multiple 3D files in the Gaussian Cube format.

    Args:
        filename_prefix: Prefix for the output filenames.
        scale: Scaling factor for grid spacing.
        grid_center: Origin of the grid in atomic units (x, y, z).
        grid_dimensions: Dimensions of the grid (nx, ny, nz, nw).
        gridmap4d: 4D array of grid values.
        maptitle: Title of the map.
        binary_precision: Precision of binary file output.
        dim4labels: Labels for the 4th dimension.
        format: File format (default is 'cube').
    """
    if filename_prefix.endswith(format):
        filename_prefix = filename_prefix[: -len(format) - 1]

    gridmap4d_local = gridmap4d.copy()
    if not (len(gridmap4d_local.shape) == 4 or len(gridmap4d_local.shape) == 1):
        raise ValueError("Input array must have 4 dimensions (nx, ny, nz, nw).")
    elif len(gridmap4d_local.shape) == 1:
        gridmap4d_local = np.reshape(gridmap4d_local, grid_dimensions)

    if dim4labels is not None and len(dim4labels) != grid_dimensions[3]:
        raise ValueError(
            "Provide labels for all 4th dimensions or set dim4labels to None."
        )

    for m in range(grid_dimensions[3]):
        dim_lbl = dim4labels[m] if dim4labels else f"4th_dim{m}"
        fname = f"{filename_prefix}_{dim_lbl}.{format}"
        maplbl = f"{maptitle}_{dim_lbl}"
        write_cube(
            filepath=fname,
            scale_factor=scale,
            grid_center=grid_center,
            grid_shape=grid_dimensions[:3],
            data_array=gridmap4d_local[:, :, :, m],
            content=maplbl,
            format=format,
            binary_precision=binary_precision,
            allow_non_c_contiguous=True,
        )


def read_cube(filepath, format):
    """
    Reads volumetric data and header information from a Gaussian cube (.cube)
    or phi format (.phi) file.

    **Complies with H5cube-spec data block order (z fastest, y middle, x slowest).**
    **Output data array is (nx, ny, nz) shape, C-order, intended for access data[ix, iy, iz].**

    Args:
        filepath (str): Path to the input file.
        format (str): 'cube' (text) or 'phi' (binary).

    Returns:
        tuple: A tuple containing:
               - scale_factor (float): Scaling factor for grid spacing (grid points per Angstrom).
               - grid_center (np.ndarray): 1D numpy array of size 3, grid center [x, y, z] in Angstroms.
               - grid_shape (np.ndarray): 1D numpy array of size 3, [nx, ny, nz] grid dimensions.
               - data_array_3d (np.ndarray): The 3D numpy array of volumetric data (nx, ny, nz), C-order.
               - origin (np.ndarray): The origin of the grid [ox, oy, oz] in Bohr.
               - vectors (np.ndarray): The grid vectors [[vx_x, ...], [vy_x, ...], [vz_x, ...]] in Bohr.
                                          vectors[0] for X, [1] for Y, [2] for Z.
               - comments (list): A list of header comment strings (usually 2).
               - data_type_comment (str): Description of the data type read.
               - endianness_detected (str): Endianness of binary data ("little-endian").
               - marker_size_detected (int or None): Marker size for binary data (4, 8, or None for text).

    Raises:
        ValueError: If the format is invalid.
        IOError: If the file cannot be read or parsing fails.
        Exception: For other unexpected errors during reading.
    """
    file_format = format.lower()
    if file_format not in ["cube", "phi"]:
        raise ValueError(f"Invalid format '{format}'. Must be 'cube' or 'phi'.")

    try:
        if file_format == "cube":
            # Open in text mode for consistent line reading
            with open(filepath, "r") as f_text:
                # _read_cube_text now returns expanded tuple
                (
                    scale_factor,
                    grid_center,
                    grid_shape,
                    data_array_3d,
                    origin,
                    vectors,
                    comments,
                    data_type_comment,
                ) = _read_cube_text(f_text)
                endianness_detected = "N/A"  # Not applicable for text
                marker_size_detected = None  # Not applicable for text
                return (
                    scale_factor,
                    grid_center,
                    grid_shape,
                    data_array_3d,
                    origin,
                    vectors,
                    comments,
                    data_type_comment,
                    endianness_detected,
                    marker_size_detected,
                )

        elif file_format == "phi":
            # Open in binary mode
            with open(filepath, "rb") as f_binary:
                (
                    scale_factor,
                    grid_center,
                    grid_shape,
                    data_array_3d,
                    origin,
                    vectors,
                    comments,
                    data_type_comment,
                    marker_size_detected,
                ) = _read_cube_binary(f_binary)
                endianness_detected = (
                    "little-endian"  # Hardcoded as only little-endian is supported
                )
                return (
                    scale_factor,
                    grid_center,
                    grid_shape,
                    data_array_3d,
                    origin,
                    vectors,
                    comments,
                    data_type_comment,
                    endianness_detected,
                    marker_size_detected,
                )
        return None

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        # Catch other potential errors during file operations
        raise IOError(f"Error reading file {filepath}: {e}")


# Helper for binary header peeking (still needed within _read_cube_binary)
def _peek_dims_from_binary_header(f, marker_size):
    """
    Tries to read dimensions (nx, ny, nz) from a binary header without reading all data.
    Used for marker size auto-detection fallback *within* _read_cube_binary.
    """
    little_endian = "<"
    marker_format = "q" if marker_size == 8 else "i"
    marker_byte_size = struct.calcsize(little_endian + marker_format)

    def read_record_size(file_handle, marker_fmt, marker_byte_size_val):
        size_bytes = file_handle.read(marker_byte_size_val)
        if len(size_bytes) != marker_byte_size_val:
            raise IOError("Premature end of file while reading record size marker.")
        record_size = struct.unpack(little_endian + marker_fmt, size_bytes)[0]
        if record_size < 0:
            raise IOError(f"Invalid (negative) record size read: {record_size}")
        return record_size

    # Skip records until we get to lines 4, 5, 6 which contain dimensions
    # We expect 7 header records before the data block. Dims are in 4, 5, 6.
    try:
        # Skip record 1
        rec1_size = read_record_size(f, marker_format, marker_byte_size)
        f.seek(f.tell() + rec1_size + marker_byte_size)  # Skip data + end marker

        # Skip record 2
        rec2_size = read_record_size(f, marker_format, marker_byte_size)
        f.seek(f.tell() + rec2_size + marker_byte_size)  # Skip data + end marker

        # Skip record 3 (Natoms, Origin)
        rec3_size = read_record_size(f, marker_format, marker_byte_size)
        f.seek(f.tell() + rec3_size + marker_byte_size)  # Skip data + end marker

        # Read record 4 (Nx, vecX)
        rec4_size = read_record_size(f, marker_format, marker_byte_size)
        rec4_data = f.read(rec4_size)
        f.seek(f.tell() + marker_byte_size)  # Skip end marker
        line4_str = rec4_data.decode("ascii").strip()
        line4_parts = line4_str.split()
        nx = int(round(abs(float(line4_parts[0]))))

        # Read record 5 (Ny, vecY)
        rec5_size = read_record_size(f, marker_format, marker_byte_size)
        rec5_data = f.read(rec5_size)
        f.seek(f.tell() + marker_byte_size)  # Skip end marker
        line5_str = rec5_data.decode("ascii").strip()
        line5_parts = line5_str.split()
        ny = int(round(abs(float(line5_parts[0]))))

        # Read record 6 (Nz, vecZ)
        rec6_size = read_record_size(f, marker_format, marker_byte_size)
        rec6_data = f.read(rec6_size)
        f.seek(f.tell() + marker_byte_size)  # Skip end marker
        line6_str = rec6_data.decode("ascii").strip()
        line6_parts = line6_str.split()
        nz = int(round(abs(float(line6_parts[0]))))

        return nx, ny, nz

    except (IOError, struct.error, ValueError, IndexError) as e:
        # print(f"Failed to peek dimensions with marker size {marker_size}: {e}") # Debugging
        raise IOError(f"Failed to peek dimensions with marker size {marker_size}: {e}")


# Helper for reading text cube (used by read_cube)
def _read_cube_text(f_text):
    """Reads data and header from a text-based Gaussian cube file (internal function)."""
    try:
        # Read line 1: scale_factor nx grid_center_x grid_center_y grid_center_z
        line1 = list(map(float, f_text.readline().split()))
        scale_factor = line1[0]  # Grid points per Angstrom
        # The second value on line 1 is often Natoms, but the writer puts nx.
        # We'll rely on line 4 for the actual nx used for data dimensions.
        # The next three values are grid center in Angstroms.
        grid_center = np.array(line1[2:5])

        # Read lines 2: Comments
        comments = [f_text.readline().strip()]

        # Read line 3: Natoms origin_x origin_y origin_z
        line3 = list(map(float, f_text.readline().split()))
        natoms = int(line3[0])
        origin = np.array(line3[1:4])  # Origin in Bohr

        # Read lines 4-6: Npoints_axis vecX_x vecX_y vecX_z etc.
        line4 = list(map(float, f_text.readline().split()))
        nx = int(
            round(abs(line4[0]))
        )  # Use abs and round for safety, Npoints_axis is positive
        vec_x = np.array(line4[1:4])  # Vector for X axis in Bohr

        line5 = list(map(float, f_text.readline().split()))
        ny = int(round(abs(line5[0])))
        vec_y = np.array(line5[1:4])  # Vector for Y axis in Bohr

        line6 = list(map(float, f_text.readline().split()))
        nz = int(round(abs(line6[0])))
        vec_z = np.array(line6[1:4])  # Vector for Z axis in Bohr

        vectors = np.array([vec_x, vec_y, vec_z])
        grid_shape = np.array([nx, ny, nz], dtype=np.int32)  # Derive grid_shape

    except (ValueError, IndexError) as e:
        raise IOError(f"Error parsing cube header lines: {e}")
    except Exception as e:
        # Catch other unexpected errors during header parsing
        raise IOError(f"Unexpected error during header parsing: {e}")

    # Read atom data (if any)
    atom_data = []
    try:
        for i in range(natoms):
            atom_line = f_text.readline().split()
            if len(atom_line) >= 5:
                atom_data.append(
                    [
                        int(atom_line[0]),
                        float(atom_line[1]),
                        float(atom_line[2]),
                        float(atom_line[3]),
                        float(atom_line[4]),
                    ]
                )
            else:
                print(
                    f"Warning: Skipping incomplete atom line {i+1}: {' '.join(atom_line)}"
                )
    except (ValueError, IndexError) as e:
        print(
            f"Warning: Error parsing atom data lines: {e}. Attempting to continue assuming data starts next."
        )

    # Read volumetric data (Corrected loop to read exactly expected_size values)
    flat_data = []
    read_count = 0  # Keep track of how many values read so far
    expected_size = nx * ny * nz  # Already calculated

    try:
        # Iterate through lines from the current position in the file
        # (which is after the header and atom data)
        for i, line in enumerate(f_text):
            # Stop reading if we've already read enough data
            if read_count >= expected_size:
                break

            words = line.split()
            if not words:
                continue  # Skip empty lines

            try:
                values = [float(v) for v in words]
            except ValueError as e:
                # Handle non-float words, maybe a trailing comment or unexpected data
                print(
                    f"Warning: Skipping invalid data values on line {7+i+1} (file line number): {e} - '{line.strip()}'"
                )
                continue  # Skip this line

            # Take only necessary values if adding the whole line would exceed expected_size
            remaining_to_read = expected_size - read_count
            if len(values) > remaining_to_read:
                values = values[:remaining_to_read]

            flat_data.extend(values)
            read_count += len(values)

            # Stop reading once we have exactly the expected number of values
            if read_count == expected_size:
                break

    except Exception as e:
        # Catch other unexpected errors during data block reading
        raise IOError(f"Error reading data block: {e}")

    # Check if we read exactly the expected size (in case the file ended prematurely)
    if read_count != expected_size:
        raise ValueError(
            f"Data size mismatch: Expected {expected_size} data points ({nx}x{ny}x{nz}), but only read {read_count}. File may be truncated."
        )

    # flat_data is now guaranteed (if no exception) to have exactly expected_size values
    # Reshape the 1D data array into a 3D array of shape (nx, ny, nz)
    try:
        data_array_3d = np.array(flat_data, dtype=np.float64).reshape(
            (nx, ny, nz), order="C"
        )
    except ValueError as e:
        raise IOError(
            f"Error reshaping data to dimensions ({nx}, {ny}, {nz}) with order 'C': {e}"
        )

    # Determine data type comment for return
    data_type_comment = f"Data type: {data_array_3d.dtype} (from text file)"

    # Return the expanded set of information
    return (
        scale_factor,
        grid_center,
        grid_shape,
        data_array_3d,
        origin,
        vectors,
        comments,
        data_type_comment,
    )


# Helper for reading binary cube (used by read_cube)
def _read_cube_binary(f):
    """Reads data and header from a binary (phi) format Gaussian cube file (internal function).
    Marker size is auto-detected."""
    little_endian = "<"
    # Auto-detect marker size
    f.seek(0)
    try:
        marker_bytes_8byte = f.read(8)
        f.seek(0)
        if len(marker_bytes_8byte) == 8:
            potential_marker_8byte = struct.unpack("<q", marker_bytes_8byte)[0]
            if 0 <= potential_marker_8byte < 4096:
                marker_size = 8
            else:
                marker_bytes_4byte = f.read(4)
                f.seek(0)
                if len(marker_bytes_4byte) == 4:
                    potential_marker_4byte = struct.unpack("<i", marker_bytes_4byte)[0]
                    if 0 <= potential_marker_4byte < 4096:
                        marker_size = 4
                    else:
                        # Fallback: try reading header with 8, then 4 byte markers
                        try:
                            f.seek(0)
                            _peek_dims_from_binary_header(
                                f, 8
                            )  # Just check if header reads
                            marker_size = 8
                        except Exception:
                            try:
                                f.seek(0)
                                _peek_dims_from_binary_header(
                                    f, 4
                                )  # Just check if header reads
                                marker_size = 4
                            except Exception:
                                raise IOError(
                                    "Could not determine binary marker size (4 or 8 bytes) or read dimensions from header."
                                )
                else:
                    raise IOError("File too small to contain 4-byte binary marker.")
        else:
            # Not enough for 8 bytes, try 4
            marker_bytes_4byte = f.read(4)
            f.seek(0)
            if len(marker_bytes_4byte) == 4:
                potential_marker_4byte = struct.unpack("<i", marker_bytes_4byte)[0]
                if 0 <= potential_marker_4byte < 4096:
                    marker_size = 4
                else:
                    try:
                        f.seek(0)
                        _peek_dims_from_binary_header(f, 4)
                        marker_size = 4
                    except Exception:
                        try:
                            f.seek(0)
                            _peek_dims_from_binary_header(
                                f, 8
                            )  # Less likely but possible - Typo here.
                            _peek_dims_from_binary_header(
                                f, 8
                            )  # Less likely but possible
                            marker_size = 8
                        except Exception:
                            raise IOError(
                                "Could not determine binary marker size (4 or 8 bytes) or read dimensions from header."
                            )
            else:
                raise IOError("File too small to contain binary marker.")
    except Exception as e:
        raise IOError(f"Failed to determine phi format details or read header: {e}")

    marker_format = "q" if marker_size == 8 else "i"
    marker_byte_size = struct.calcsize(little_endian + marker_format)

    f.seek(0)  # Ensure file pointer is at the beginning for reading records

    def read_record(file_handle, marker_fmt, marker_byte_size_val):
        """Reads a single Fortran unformatted binary record."""
        try:
            # Read start marker
            size_bytes_start = file_handle.read(marker_byte_size_val)
            if len(size_bytes_start) != marker_byte_size_val:
                raise IOError(
                    "Premature end of file while reading start record size marker."
                )
            record_size = struct.unpack(little_endian + marker_fmt, size_bytes_start)[0]
            if record_size < 0:
                raise IOError(f"Invalid (negative) record size read: {record_size}")

            # Read data block
            data_bytes = file_handle.read(record_size)
            if len(data_bytes) != record_size:
                raise IOError(
                    f"Premature end of file while reading data block of size {record_size}."
                )

            # Read end marker
            size_bytes_end = file_handle.read(marker_byte_size_val)
            if len(size_bytes_end) != marker_byte_size_val:
                raise IOError(
                    "Premature end of file while reading end record size marker."
                )
            end_record_size = struct.unpack(little_endian + marker_fmt, size_bytes_end)[
                0
            ]

            if record_size != end_record_size:
                print(
                    f"Warning: Mismatched record sizes (start: {record_size}, end: {end_record_size})"
                )

            return data_bytes

        except struct.error as e:
            raise IOError(f"Struct error during binary record reading: {e}")
        except Exception as e:
            raise IOError(f"Error during binary record reading: {e}")

    # Read header records (expecting 7 records based on the writer's structure)
    header_records = []
    try:
        for i in range(7):
            header_records.append(read_record(f, marker_format, marker_byte_size))
    except IOError as e:
        raise IOError(f"Error reading binary header record {i+1}: {e}")

    # Decode and parse header records
    scale_factor = None
    grid_center = None
    comments = []
    origin = None
    vectors = None
    nx, ny, nz = None, None, None

    try:
        # --- NEW: Decode and parse Record 1 (scale_factor, grid_center, nx) ---
        line1_str = header_records[0].decode("ascii").strip()
        line1_parts = line1_str.split()
        if len(line1_parts) >= 5:  # Expect at least 5 values on line 1
            scale_factor = float(line1_parts[0])  # Grid points per Angstrom
            # The second value is often Natoms, but writer puts nx.
            # We'll use nx from record 4 for dims.
            grid_center = np.array(
                list(map(float, line1_parts[2:5]))
            )  # Grid center in Angstroms
        else:
            print(
                f"Warning: Header record 1 (line 1) has unexpected format: '{line1_str}'. Scale factor and grid center may not be parsed."
            )
        # --- END NEW: Decode and parse Record 1 ---

        # Record 2: Comment
        comments.append(header_records[1].decode("ascii").strip())

        # Record 3: Natoms origin_x origin_y origin_z
        line3_str = header_records[2].decode("ascii").strip()
        line3_parts = line3_str.split()
        natoms = int(line3_parts[0])
        origin = np.array(list(map(float, line3_parts[1:4])))  # Origin in Bohr

        # Record 4: Nx vecX_x vecX_y vecX_z
        line4_str = header_records[3].decode("ascii").strip()
        line4_parts = line4_str.split()
        nx = int(round(abs(float(line4_parts[0]))))
        vec_x = np.array(list(map(float, line4_parts[1:4])))  # Vector X in Bohr

        # Record 5: Ny vecY_x vecY_y vecY_z
        line5_str = header_records[4].decode("ascii").strip()
        line5_parts = line5_str.split()
        ny = int(round(abs(float(line5_parts[0]))))
        vec_y = np.array(list(map(float, line5_parts[1:4])))  # Vector Y in Bohr

        # Record 6: Nz vecZ_x vecZ_y vecZ_z
        line6_str = header_records[5].decode("ascii").strip()
        line6_parts = line6_str.split()
        nz = int(round(abs(float(line6_parts[0]))))
        vec_z = np.array(list(map(float, line6_parts[1:4])))  # Vector Z in Bohr

        vectors = np.array([vec_x, vec_y, vec_z])
        grid_shape = np.array([nx, ny, nz], dtype=np.int32)  # Derive grid_shape

        # Record 7: Atom data (consumed)
        # atom_data_record_str = header_records[6].decode('ascii').strip()

    except Exception as e:
        raise IOError(f"Error decoding or parsing binary header records: {e}")

    # Check if necessary header info was parsed
    if nx is None or ny is None or nz is None or origin is None or vectors is None:
        raise IOError("Failed to parse essential grid information from binary header.")

    # Read volumetric data record
    try:
        data_bytes = read_record(f, marker_format, marker_byte_size)
    except IOError as e:
        raise IOError(f"Error reading binary data record: {e}")

    # Infer data type from record size and dimensions
    expected_total_values = nx * ny * nz
    actual_data_bytes = len(data_bytes)

    dtype_read = None
    if expected_total_values > 0:
        if actual_data_bytes == expected_total_values * struct.calcsize("<f"):
            dtype_read = np.float32
        elif actual_data_bytes == expected_total_values * struct.calcsize("<d"):
            dtype_read = np.float64
        else:
            raise IOError(
                f"Data block size ({actual_data_bytes} bytes) does not match expected size for float32 or float64 based on dimensions ({nx}x{ny}x{nz})."
            )

    elif actual_data_bytes > 0:
        raise IOError(
            f"Data block contains {actual_data_bytes} bytes, but grid dimensions are {nx}x{ny}x{nz}. Data size mismatch."
        )
    else:
        dtype_read = np.float64  # Default for empty data block

    # Convert binary data to numpy array and reshape to (nx, ny, nz) C-order
    try:
        flat_data = np.frombuffer(data_bytes, dtype=dtype_read)
    except Exception as e:
        raise IOError(f"Error converting binary data bytes to numpy array: {e}")

    if len(flat_data) != expected_total_values:
        raise IOError(
            f"Data size mismatch after binary read: Expected {expected_total_values} values, but got {len(flat_data)}. File may be corrupt."
        )

    try:
        data_array_3d = flat_data.reshape((nx, ny, nz), order="C")
    except ValueError as e:
        raise IOError(
            f"Error reshaping binary data to dimensions ({nx}, {ny}, {nz}) with order 'C': {e}"
        )

    # Determine data type comment for return
    data_type_comment = f"Data type: {data_array_3d.dtype}"

    # Return the expanded set of information
    # Corrected return order to match the user's requested signature
    return (
        scale_factor,
        grid_center,
        grid_shape,
        data_array_3d,
        origin,
        vectors,
        comments,
        data_type_comment,
        marker_size,
    )
