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
"""Utility for ROCm backend"""

import os
import re
import shutil
import subprocess
import warnings
from os.path import exists, isdir, join

import tvm_ffi

import tvm.runtime
import tvm.target

from . import utils


def find_lld(required=True):
    """Find ld.lld in system.

    Parameters
    ----------
    required : bool
        Whether it is required,
        runtime error will be raised if the compiler is required.

    Returns
    -------
    valid_list : list of str
        List of possible paths.

    Note
    ----
    The ROCm install's own linker comes first (``LLVM_PATH``, ``<root>/lib/llvm/bin``,
    ``<root>/llvm/bin``), then ld.lld on PATH, preferring the major LLVM version TVM was
    built with. Works on Windows (``ld.lld.exe``).
    """
    names = []
    major = tvm.target.codegen.llvm_version_major(allow_none=True)
    if major is not None:
        names += [f"ld.lld-{major}.0", f"ld.lld-{major}"]
    names += ["ld.lld"]

    dirs = []
    llvm_path = os.environ.get("LLVM_PATH")
    if llvm_path:
        dirs.append(join(llvm_path, "bin"))
    try:
        rocm_path = find_rocm_path()
        dirs += [join(rocm_path, "lib", "llvm", "bin"), join(rocm_path, "llvm", "bin")]
    except RuntimeError:
        pass

    candidates = [join(d, n) for d in dirs for n in ["ld.lld"]] + names
    valid_list = []
    for cand in candidates:
        path = shutil.which(cand)
        if path and path not in valid_list:
            valid_list.append(path)
    if not valid_list and required:
        raise RuntimeError("cannot find ld.lld, candidates are: " + str(candidates))
    return valid_list


def rocm_link(in_file, out_file, lld=None):
    """Link relocatable ELF object to shared ELF object using lld

    Parameters
    ----------
    in_file : str
        Input file name (relocatable ELF object file)

    out_file : str
        Output file name (shared ELF object file)

    lld : str, optional
        The lld linker, if not specified,
        we will try to guess the matched clang version.
    """

    # if our result has undefined symbols, it will fail to load
    # (hipModuleLoad/hipModuleLoadData), but with a somewhat opaque message
    # so we have ld.lld check this here.
    # If you get a complaint about missing symbols you might want to check the
    # list of bitcode files below.
    args = [
        lld if lld is not None else find_lld()[0],
        "--no-undefined",
        "-shared",
        in_file,
        "-o",
        out_file,
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Linking error using ld.lld:\n"
        msg += out.decode("utf-8", errors="replace")
        raise RuntimeError(msg)


@tvm_ffi.register_global_func("tvm_callback_rocm_link")
def callback_rocm_link(obj_bin):
    """Links object file generated from LLVM to HSA Code Object

    Parameters
    ----------
    obj_bin : bytearray
        The object file

    Return
    ------
    cobj_bin : bytearray
        The HSA Code Object
    """
    tmp_dir = utils.tempdir()
    tmp_obj = tmp_dir.relpath("rocm_kernel.o")
    tmp_cobj = tmp_dir.relpath("rocm_kernel.co")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(obj_bin))
    rocm_link(tmp_obj, tmp_cobj)
    cobj_bin = bytearray(open(tmp_cobj, "rb").read())
    return cobj_bin


def find_rocm_bitcode_dir(rocm_path=None):
    """Find the directory holding the ROCm device library bitcode (ocml.bc, oclc_*.bc).

    Checks ``HIP_DEVICE_LIB_PATH``, then ``<root>/lib/llvm/amdgcn/bitcode`` (ROCm 7+ and
    TheRock), ``<root>/amdgcn/bitcode`` (ROCm 3.9 to 6.x),
    ``<root>/lib/llvm/lib/clang/*/lib/amdgcn/bitcode`` and ``<root>/lib`` (ROCm <= 3.8).
    """
    env_dir = os.environ.get("HIP_DEVICE_LIB_PATH")
    candidates = [env_dir] if env_dir else []
    if rocm_path is None:
        rocm_path = find_rocm_path()
    candidates += [
        join(rocm_path, "lib", "llvm", "amdgcn", "bitcode"),
        join(rocm_path, "amdgcn", "bitcode"),
    ]
    clang_dir = join(rocm_path, "lib", "llvm", "lib", "clang")
    if isdir(clang_dir):
        for ver in sorted(os.listdir(clang_dir), reverse=True):
            candidates.append(join(clang_dir, ver, "lib", "amdgcn", "bitcode"))
    candidates.append(join(rocm_path, "lib"))
    for cand in candidates:
        if exists(join(cand, "ocml.bc")) or exists(join(cand, "ocml.amdgcn.bc")):
            return cand
    raise RuntimeError(
        "Cannot find the ROCm device library bitcode (ocml.bc). Looked in: " + str(candidates)
    )


@tvm_ffi.register_global_func("tvm_callback_rocm_bitcode_path")
def callback_rocm_bitcode_path(arch=None, wavefront_size=None, rocdl_dir=None):
    """Utility function to find the ROCm device library bitcodes to link

    Parameters
    ----------
    arch : str, optional
        The target GPU, e.g. "gfx1101" (feature suffixes like ":xnack-" are ignored).
        Defaults to the detected GPU.

    wavefront_size : int, optional
        32 or 64. Defaults to 32 on gfx10 and newer, 64 otherwise.

    rocdl_dir : str, optional
        The bitcode directory. Defaults to :py:func:`find_rocm_bitcode_dir`.

    Returns
    -------
    bitcode_files : list of str
        ocml plus exactly one oclc_* control library per setting. The control libraries
        define the same constants, so linking both the on and off variants would let the
        first one silently win.
    """
    if rocdl_dir is None:
        rocdl_dir = find_rocm_bitcode_dir()
    if not arch:
        arch = get_rocm_arch()
    arch = str(arch).split(":")[0]
    isa = arch[3:] if arch.startswith("gfx") else arch
    if wavefront_size is None:
        wavefront_size = 32 if _gfx_major(isa) >= 10 else 64
    wavefront_size = int(wavefront_size)

    def find(name):
        for fname in (name + ".bc", name + ".amdgcn.bc"):  # rocm >= 3.9, rocm <= 3.8
            path = join(rocdl_dir, fname)
            if exists(path):
                return path
        return None

    bitcode_files = []
    ocml = find("ocml")
    if ocml is None:
        raise RuntimeError("could not find bitcode ocml in " + rocdl_dir)
    bitcode_files.append(ocml)

    # Control libraries. ROCm 7+ dropped oclc_daz_opt_* and oclc_correctly_rounded_sqrt_*
    # (denormal/sqrt behaviour now comes from function attributes), so those are optional.
    wave = "on" if wavefront_size == 64 else "off"
    for name in (
        "oclc_daz_opt_on",
        "oclc_correctly_rounded_sqrt_off",
        "oclc_finite_only_off",
        "oclc_unsafe_math_off",
        f"oclc_wavefrontsize64_{wave}",
    ):
        path = find(name)
        if path:
            bitcode_files.append(path)

    isa_file = find(f"oclc_isa_version_{isa}")
    if isa_file:
        bitcode_files.append(isa_file)
    else:
        warnings.warn(
            f"No ROCm device library for gfx{isa} (oclc_isa_version_{isa}) in {rocdl_dir}"
        )

    # Code object ABI: v5 is LLVM's default; fall back to the newest available.
    abi = find("oclc_abi_version_500")
    if abi is None:
        abis = sorted(f for f in os.listdir(rocdl_dir) if re.match(r"oclc_abi_version_\d+\.bc$", f))
        abi = join(rocdl_dir, abis[-1]) if abis else None
    if abi:
        bitcode_files.append(abi)

    return tvm.runtime.convert(bitcode_files)


def _gfx_major(isa):
    """GFX major version from the part after "gfx": "1101" -> 11, "90a" -> 9."""
    digits = isa if len(isa) <= 2 else (isa[:2] if len(isa) >= 4 else isa[:1])
    match = re.match(r"\d+", digits)
    return int(match.group(0)) if match else 0


def parse_compute_version(compute_version):
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error: " + str(err))


def have_matrixcore(compute_version=None):
    """Either MatrixCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    Returns
    -------
    have_matrixcore : bool
        True if MatrixCore support is provided, False otherwise
    """
    if compute_version is None:
        if tvm.rocm(0).exist:
            compute_version = tvm.rocm(0).compute_version
        else:
            raise RuntimeError("No ROCm runtime found")
    major, _ = parse_compute_version(compute_version)
    # matrix core first introduced in 8.0
    if major >= 8:
        return True

    return False


@tvm_ffi.register_global_func("tvm_callback_rocm_get_arch")
def get_rocm_arch(rocm_path=None):
    """Utility function to get the AMD GPU architecture

    Parameters
    ----------
    rocm_path : str
        The path to rocm installation directory

    Returns
    -------
    gpu_arch : str
        The AMD GPU architecture, e.g. "gfx1101". Queried from the HIP runtime, then
        amdgpu-arch / offload-arch, then rocminfo (Linux). Falls back to "gfx900".
    """
    gpu_arch = "gfx900"
    try:
        dev = tvm.rocm(0)
        if dev.exist:
            arch = dev._GetDeviceAttr(dev.dlpack_device_type(), dev.index, 10)  # kGcnArch
            if arch:
                return str(arch).split(":")[0]
    except Exception:  # pylint: disable=broad-except
        pass

    if rocm_path is None:
        try:
            rocm_path = find_rocm_path()
        except RuntimeError:
            rocm_path = None
    tools = []
    for tool in ("amdgpu-arch", "offload-arch"):
        if rocm_path:
            tools.append(join(rocm_path, "lib", "llvm", "bin", tool))
            tools.append(join(rocm_path, "llvm", "bin", tool))
        tools.append(tool)
    if rocm_path:
        tools.append(join(rocm_path, "bin", "rocminfo"))
    for tool in tools:
        exe = shutil.which(tool)
        if not exe:
            continue
        try:
            out = subprocess.check_output([exe], stderr=subprocess.STDOUT, timeout=60)
        except (subprocess.SubprocessError, OSError):
            continue
        match = re.search(r"(gfx[0-9a-f]+)", out.decode("utf-8", errors="replace"))
        if match:
            return match.group(1)
    print(f"ROCm GPU architecture not detected, using default {gpu_arch}")
    return gpu_arch


def find_rocm_path():
    """Utility function to find ROCm path

    Returns
    -------
    path : str
        Path to ROCm root: ``ROCM_PATH``, then ``HIP_PATH`` (empty values count as unset),
        then the root of hipconfig / amdclang found on PATH, then ``/opt/rocm`` (not on
        Windows).
    """
    for var in ("ROCM_PATH", "HIP_PATH"):
        value = os.environ.get(var)
        if value and isdir(value):
            return value
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        return os.path.dirname(os.path.dirname(os.path.realpath(hipconfig)))
    amdclang = shutil.which("amdclang")
    if amdclang:
        # <root>/lib/llvm/bin/amdclang
        return os.path.realpath(join(os.path.dirname(amdclang), "..", "..", ".."))
    if os.name != "nt" and exists("/opt/rocm"):
        return "/opt/rocm"
    raise RuntimeError("Cannot find ROCm path: set ROCM_PATH or HIP_PATH")
