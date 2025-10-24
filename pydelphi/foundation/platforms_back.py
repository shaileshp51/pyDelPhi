#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


"""
Module for platform detection and management, including CPU and CUDA support.

This module provides the `Platform` class, responsible for detecting and
managing the computational platform on which pydelphi is running.

Key functionalities include:
    - CPU information detection (number of threads).
    - CUDA device detection and availability check (if numba is installed).
    - Selection and activation of a platform (CPU or CUDA).
    - Setting calculation precision (using enums defined in the 'enums' module).

CUDA detection is performed conditionally, and the 'numba' library is only
imported within this module when CUDA-related checks are necessary. This
ensures that modules using other parts of pydelphi (e.g., just the enums)
do not require 'numba' as a dependency.

The module also includes a CUDA usability test kernel to ensure that detected
CUDA devices are functional for computation.
"""

import os
import re
import warnings
import numpy as np

# Attempt Numba import for CUDA functionality
try:
    from numba import cuda, jit
    from numba.core.errors import NumbaPerformanceWarning

    _numba_imported = True
except ImportError:
    cuda = None
    jit = None
    NumbaPerformanceWarning = None
    _numba_imported = False
    # No print/warning here, check happens later if CUDA is requested

# --- CUDA Test Kernel (only used if Numba is imported) ---
if _numba_imported:

    @cuda.jit(fastmath=False)  # fastmath potentially relevant for compute kernels
    def _test_cuda_kernel(test_array, test_array_square):
        """Simple kernel to test basic CUDA functionality."""
        i = cuda.grid(1)
        if i < test_array.size:
            test_array_square[i] = test_array[i] * test_array[i]


_PRINT_INFO = False


class Platform:
    """
    Manages CPU and CUDA platform detection, selection, and properties silently.

    Detects available CPU cores and CUDA devices, stores their properties,
    and allows activating a specific platform and device. Focuses on setting
    internal state for application logic checks, minimizing console output
    during detection.
    """

    MAX_KERNEL_SHARED_MEM_THREADS = 1024
    _is_cuda_initialized = None  # Cache for basic Numba CUDA check

    def __init__(self, debug: bool = False) -> None:
        """Initializes the Platform class, silently detecting capabilities."""
        self.debug = debug
        if self.debug:
            print("DEBUG: Platform initialization started.")
        self.names = {}
        self.active = ""
        from pydelphi.foundation.enums import Precision

        self.precision = Precision.DOUBLE

        # --- Initialize CPU Info (Default 1, check env/os) ---
        cpu_threads = 1
        try:
            omp_threads = os.environ.get("OMP_NUM_THREADS")
            numba_threads = os.environ.get("NUMBA_NUM_THREADS")
            if omp_threads:
                cpu_threads = int(omp_threads)
            elif numba_threads:
                cpu_threads = int(numba_threads)
            else:
                cpu_count = os.cpu_count()
                if cpu_count:
                    cpu_threads = cpu_count
        except:
            # Silently ignore errors parsing env vars or cpu_count, keep default 1
            pass
        self.names["cpu"] = {"available": True, "num_threads": max(1, cpu_threads)}

        # --- Initialize CUDA Info ---
        self.names["cuda"] = {"available": False, "selected_id": None, "device": {}}
        try:
            if _numba_imported:
                if self.debug:
                    print(
                        f"DEBUG: Numba is imported. Checking cuda.is_available(): {self.is_cuda_available()}"
                    )
                if self.is_cuda_available():
                    if self.debug:
                        print("DEBUG: Calling _detect_cuda_multiple()...")
                    self._detect_cuda_multiple()
                else:
                    if self.debug:
                        print(
                            "DEBUG: cuda.is_available() returned False, skipping _detect_cuda_multiple."
                        )
            else:
                if self.debug:
                    print("DEBUG: Numba not imported, skipping CUDA detection.")
        except Exception as e:
            # This is a critical error, so it's printed regardless of debug flag
            print(
                f"ERROR: Critical error during CUDA detection in __init__: {e}",
                flush=True,
            )
            self.names["cuda"]["available"] = False
        if self.debug:
            print(
                f"DEBUG: Final CUDA availability status: {self.names['cuda']['available']}"
            )

    def _detect_cuda_multiple(self):
        """Silently detects multiple CUDA devices and their properties."""
        if self.debug:
            print("DEBUG: Entering _detect_cuda_multiple.")

        if not _numba_imported or not cuda.is_available():
            self.names["cuda"]["available"] = False
            if self.debug:
                print(
                    "DEBUG: Numba not imported or cuda not available within _detect_cuda_multiple."
                )
            return
        initial_device_id = -1
        try:
            initial_device_id = cuda.get_current_device().id
        except Exception:
            pass  # Ignore if getting initial device fails

        for gpu in cuda.gpus:
            device_id = gpu.id
            # Initialize props; default usable=False unless test passes
            device_props = {"usable": False, "device_identity": f"Device {device_id}"}
            try:
                cuda.select_device(device_id)
                device = cuda.get_current_device()

                # --- Get Device Identity (Silently) ---
                try:
                    device_identity_raw = str(device.get_device_identity)
                    device_identity = re.sub(
                        r"^\<.+\<", "", device_identity_raw
                    ).replace(">>", "")
                    device_identity = re.sub(
                        r"CUDA device.+'b", "", device_identity
                    ).replace("''", "'")
                    device_props["device_identity"] = device_identity.strip("' ")
                except:
                    pass  # Ignore parsing errors

                # --- Get Memory Info (Silently) ---
                try:
                    meminfo = cuda.current_context().get_memory_info()
                    device_props["memory_free(MiB)"] = meminfo.free // (1024 * 1024)
                    device_props["memory_total(MiB)"] = meminfo.total // (1024 * 1024)
                except:
                    pass  # Ignore memory info errors

                # --- Capture All Uppercase Attributes (Silently) ---
                attribs = [s for s in dir(device) if s.isupper()]
                for attr in attribs:
                    try:
                        device_props[attr] = getattr(device, attr)
                    except:
                        pass  # Ignore attribute errors

                # --- Test Usability (Silently) ---
                try:
                    # _test_cuda now returns only True/False
                    if self._test_cuda():
                        device_props["usable"] = True
                        overall_cuda_available = (
                            True  # Mark CUDA available if any device is usable
                        )
                        if self.debug:
                            print(
                                f"DEBUG: CUDA usability test PASSED for device {device_id}."
                            )
                    else:
                        if self.debug:
                            print(
                                f"DEBUG: CUDA usability test FAILED for device {device_id}."
                            )

                except Exception as e_test:
                    # Keep this error visible even if not in full debug mode
                    print(
                        f"ERROR: Exception during CUDA usability test for device {device_id}: {e_test}",
                        flush=True,
                    )
                    device_props["usable"] = False

                self.names["cuda"]["device"][device_id] = device_props
                if self.debug:
                    print(
                        f"DEBUG: Device {device_id} properties after processing: {device_props}"
                    )

            except Exception as e:
                # Log critical error processing a specific device if desired
                if device_id not in self.names["cuda"]["device"]:
                    self.names["cuda"]["device"][device_id] = {
                        "usable": False,
                        "device_identity": f"Device {device_id} (Error)",
                    }

        self.names["cuda"]["available"] = overall_cuda_available
        if self.debug:
            print(
                f"DEBUG: _detect_cuda_multiple finished. Overall CUDA available: {overall_cuda_available}"
            )

        # Restore original device selection silently
        if initial_device_id != -1:
            try:
                cuda.select_device(initial_device_id)
                if self.debug:
                    print(
                        f"DEBUG: Restored original CUDA device selection to ID: {initial_device_id}."
                    )
            except Exception as e:
                if (
                    self.debug
                ):  # Only warn if in debug mode, as it's not critical for functionality
                    print(
                        f"WARNING: Failed to restore initial CUDA device {initial_device_id}: {e}",
                        flush=True,
                    )

    def _test_cuda(self):
        """Silently tests CUDA usability by running a simple kernel. Returns True/False."""
        if not _numba_imported:
            return False

        # Silence NumbaPerformanceWarning locally if needed/possible
        if (
            NumbaPerformanceWarning is not None
        ):  # Check if NumbaPerformanceWarning is available (numba imported)
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

        try:
            test_array = np.arange(100, dtype=np.float64)
            expected_result = np.square(test_array)
            result_cu = np.zeros_like(test_array)
            test_array_dev = cuda.to_device(test_array)
            result_cu_dev = cuda.device_array_like(result_cu)
            threads_per_block = 32
            blocks_per_grid = (
                test_array.size + (threads_per_block - 1)
            ) // threads_per_block
            # Ensure kernel is available
            if "_test_cuda_kernel" in globals():
                _test_cuda_kernel[blocks_per_grid, threads_per_block](
                    test_array_dev, result_cu_dev
                )
                cuda.synchronize()
                result_cu_dev.copy_to_host(result_cu)

                if self.debug:
                    print(f"DEBUG: Expected result (first 10): {expected_result[:10]}")
                    print(f"DEBUG: Actual GPU result (first 10): {result_cu[:10]}")

                test_passed = np.allclose(expected_result, result_cu)
                if self.debug:
                    print(f"DEBUG: _test_cuda: Kernel result comparison: {test_passed}")
                return test_passed
            else:
                if self.debug:
                    print(
                        "DEBUG: _test_cuda: _test_cuda_kernel not in globals, returning False."
                    )
                return False
        except Exception as e:
            print(f"ERROR: _test_cuda: CUDA test kernel exception: {e}", flush=True)
            return False

    @classmethod
    def is_cuda_available(cls):
        """Silently checks if Numba's CUDA is potentially available."""
        if not _numba_imported:
            return False
        if cls._is_cuda_initialized is None:
            try:
                if cuda and cuda.is_available() and len(cuda.gpus) > 0:
                    cls._is_cuda_initialized = True
                else:
                    cls._is_cuda_initialized = False
            except:  # Catch any exception during check
                cls._is_cuda_initialized = False
        return cls._is_cuda_initialized

    def get_available_platforms(self) -> list:
        """Returns a list of names of platforms marked as available."""
        return [name for name, props in self.names.items() if props.get("available")]

    def get_usable_cuda_device_ids(self) -> list:
        """Returns a list of IDs of detected CUDA devices marked as usable."""
        if not self.names.get("cuda", {}).get("available"):
            return []
        return sorted(
            [
                id
                for id, props in self.names["cuda"]["device"].items()
                if props.get("usable")
            ]
        )

    def available(self):
        """Prints names of available platforms (for simple interactive check)."""
        print("Available platforms:", self.get_available_platforms())

    def activate(self, name, n_cpus=None, device_id=None):
        """Activates a platform (CPU or CUDA) for computation.

        If activating CUDA and device_id is None, selects the first available and usable device.
        Updates the platform's CPU thread count if n_cpus is provided. Minimal console output.

        Args:
            name (str): Platform name to activate ('cpu' or 'cuda').
            n_cpus (int, optional): Number of CPU threads to use. Defaults to None (keep existing).
            device_id (int, optional): CUDA device ID to select. Defaults to None (auto-select).

        Raises:
            ValueError: if the platform name is invalid or unavailable.
            RuntimeError: if CUDA requested but no usable device found or ID invalid/unusable.
        """
        name_ = name.lower().strip()
        available_platforms = self.get_available_platforms()
        if name_ not in available_platforms:
            raise ValueError(
                f"Invalid or unavailable platform name: '{name}'. Available platforms: {available_platforms}"
            )

        if n_cpus is not None:
            try:
                self.names["cpu"]["num_threads"] = max(1, int(n_cpus))
            except:
                pass  # Silently ignore bad n_cpus input, keep existing

        self.active = name_

        if name_ == "cuda":
            usable_ids = self.get_usable_cuda_device_ids()
            if (
                not usable_ids
            ):  # Should have been caught by availability check, but good practice
                raise RuntimeError(
                    "CUDA platform requested, but no usable CUDA devices found."
                )

            selected_device_id = None
            if device_id is not None:
                if device_id not in usable_ids:
                    raise RuntimeError(
                        f"Specified CUDA device ID {device_id} not found or is not usable. Usable IDs: {usable_ids}"
                    )
                selected_device_id = device_id
            else:
                selected_device_id = usable_ids[0]  # Auto-select first usable
                # Optional: print info only on auto-selection, or use logging
                # print(f"Info: No CUDA device specified, activating first usable device: ID {selected_device_id}")

            self.names["cuda"]["selected_id"] = selected_device_id
            try:
                cuda.select_device(selected_device_id)
            except Exception as e:
                # This is a critical error, should probably raise
                raise RuntimeError(
                    f"Failed to select CUDA device {selected_device_id}: {e}"
                )
        else:  # CPU activated
            if "cuda" in self.names:
                self.names["cuda"]["selected_id"] = None

        # Minimal confirmation message
        if _PRINT_INFO:
            print(f"Info: Platform '{self.active}' activated.", end="")
            if self.active == "cuda":
                print(f" [Device ID: {self.names['cuda']['selected_id']}]")
            else:
                print()

    def set_precision(self, precision):
        """Sets the calculation precision."""
        # (Implementation remains the same as before)
        from pydelphi.foundation.enums import Precision

        if isinstance(precision, Precision):
            self.precision = precision
        else:
            try:
                value_str = str(precision).upper().split(".")[-1]
                self.precision = Precision[value_str]
            except:
                raise ValueError(
                    f"Invalid precision: '{precision}'. Options: {Precision.list()}"
                )
        # Optional: print confirmation
        # print(f"Info: Calculation precision set to {self.precision.name}")

    def __repr__(self):
        """Returns a detailed string representation for debugging/info."""
        # (Implementation remains the same as before - detailed output is useful here)
        outs = ["Platform Configuration Summary", "=" * 100, "CPU Platform:"]
        # CPU Info
        if self.names.get("cpu", {}).get("available"):
            outs.append(f"  - Status: Available")
            outs.append(
                f"  - Host Threads: {self.names['cpu'].get('num_threads', 'N/A')}"
            )
        else:
            outs.append("  - Status: Unavailable/Error")
        outs.append("-" * 100)
        # CUDA Info
        outs.append("CUDA Platform:")
        cuda_info = self.names.get("cuda", {})
        if cuda_info.get("available"):
            outs.append(
                f"  - Status: Available (Found {len(cuda_info.get('device', {}))} device(s), {len(self.get_usable_cuda_device_ids())} usable)"
            )
            outs.append(
                f"  - Selected Device ID: {cuda_info.get('selected_id', 'None')}"
            )
            outs.append("  - Detected Devices:")
            for dev_id, props in sorted(cuda_info.get("device", {}).items()):
                identity = props.get("device_identity", f"Device {dev_id}")
                usable_str = "Usable" if props.get("usable") else "Not Usable"
                mem = props.get("memory_total(MiB)", "N/A")
                max_t = props.get("MAX_THREADS_PER_BLOCK", "N/A")
                cc = f"{props.get('COMPUTE_CAPABILITY_MAJOR', '?')}.{props.get('COMPUTE_CAPABILITY_MINOR', '?')}"
                outs.append(
                    f"    - ID {dev_id}: {identity} ({usable_str}) CC: {cc}, Mem: {mem} MiB, Max Thr/Blk: {max_t}"
                )
        elif not _numba_imported:
            outs.append("  - Status: Unavailable (Numba not installed)")
        else:
            outs.append("  - Status: Unavailable (No usable devices found/Error)")
        outs.append("-" * 100)
        # Active Platform Info
        active_platform = self.active if self.active else "None"
        outs.append(f"Active Platform: {active_platform}")
        if self.active:
            outs.append(f"Calculation Precision: {self.precision.name}")
        outs.append("=" * 100)
        return "\n".join(outs)

    def __str__(self):
        """Returns the detailed string representation."""
        return self.__repr__()

    def properties(self):
        """Returns properties of the currently active platform/device."""
        # (Implementation remains the same as before, keeping necessary warnings/errors)
        if not self.active:
            # Keep this warning as it indicates incorrect usage
            print("Warning: No platform activated. Call platform.activate() first.")
            return None
        elif self.active == "cuda":
            cuda_info = self.names.get("cuda", {})
            selected_id = cuda_info.get("selected_id", None)
            if selected_id is None:
                # Keep this warning
                print(
                    "Warning: CUDA platform is active, but no specific device is selected/available."
                )
                return None
            device_props_dict = cuda_info.get("device", {})
            if selected_id not in device_props_dict:
                # Keep this error message
                print(
                    f"Error: Selected CUDA device ID {selected_id} not found in detected devices."
                )
                return None
            selected_device_props = device_props_dict[selected_id].copy()
            selected_device_props["cpu_threads_for_host"] = self.names.get(
                "cpu", {}
            ).get("num_threads", 1)
            return selected_device_props
        else:  # CPU
            return self.names.get("cpu", {}).copy()
