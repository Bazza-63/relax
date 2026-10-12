# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""ROCm-owned TIRx modules."""

import os
import warnings
from pathlib import Path

from tvm_ffi.libinfo import load_lib_ctypes

from tvm.base import _LOADED_LIBS


def _detect_target_from_device(dev):
    from tvm.target import Target  # pylint: disable=import-outside-toplevel

    return Target(
        {
            "kind": "rocm",
            "mtriple": "amdgcn-amd-amdhsa",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def _preload_hip_runtime_windows():
    """Windows: load the HIP runtime from the ROCm install by full path.

    The AMD display driver ships its own amdhip64_*.dll in System32, and the standard DLL
    search order prefers it over PATH. Once the ROCm copy is loaded, later by-name loads
    (libtvm_runtime_rocm, libtvm_runtime_extra, hipBLAS) bind to it. The C++ delay-load
    hook (src/backend/rocm/runtime/rocm_dll_loader_win.cc) does the same for non-Python
    hosts.
    """
    if os.name != "nt":
        return
    root = next((os.environ[v] for v in ("ROCM_PATH", "HIP_PATH") if os.environ.get(v)), None)
    if not root:
        return
    bin_dir = Path(root) / "bin"
    dlls = sorted(bin_dir.glob("amdhip64*.dll"))
    if not dlls:
        return
    import ctypes  # pylint: disable=import-outside-toplevel

    # LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: dependencies
    # resolve from the ROCm bin directory first.
    try:
        _LOADED_LIBS["amdhip64"] = ctypes.WinDLL(str(dlls[-1]), winmode=0x1100)
    except OSError as err:
        warnings.warn(f"Failed to load the ROCm HIP runtime {dlls[-1]}: {err}")


def register_backend():
    """Register ROCm-owned Python semantics."""
    from tvm.target.detect_target import register_device_target_detector

    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    _preload_hip_runtime_windows()
    try:
        # Runtime sidecars only need registration side effects; libtvm_runtime is global.
        _LOADED_LIBS["tvm_runtime_rocm"] = load_lib_ctypes(
            package="tvm",
            target_name="tvm_runtime_rocm",
            extra_lib_paths=[runtime_dir],
            mode="RTLD_LOCAL",
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass
    register_device_target_detector("rocm", _detect_target_from_device)
    return None


__all__ = ["register_backend"]
