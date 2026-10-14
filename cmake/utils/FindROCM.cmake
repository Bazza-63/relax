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

#######################################################
# Enhanced version of find rocm.
#
# Usage:
#   find_rocm(${USE_ROCM})
#
# - When USE_ROCM=/path/to/rocm-sdk-path, use the sdk
# - When USE_ROCM=ON, use the first of: $ENV{ROCM_PATH}, $ENV{HIP_PATH} (empty values count
#   as unset), the root of hipconfig/amdclang found on PATH, /opt/rocm (not on Windows)
#
# Provide variables:
#
# - ROCM_FOUND
# - ROCM_ROOT
# - ROCM_INCLUDE_DIRS
# - ROCM_HIPHCC_LIBRARY
# - ROCM_HIPBLAS_LIBRARY, ROCM_HIPBLASLT_LIBRARY (if found)
# - ROCM_HIPBLASLT_TARGET (roc::hipblaslt, if only hipBLASLt's CMake package was found)
# - ROCM_HIP_DLL_NAMES, ROCM_HIPBLAS_DLL_NAMES (Windows only: DLL file names in <root>/bin)
#

macro(find_rocm use_rocm)
  set(__use_rocm ${use_rocm})
  set(__rocm_sdk "")
  if(IS_DIRECTORY "${__use_rocm}")
    set(__rocm_sdk "${__use_rocm}")
    message(STATUS "Custom ROCM SDK PATH=" ${__use_rocm})
  elseif(NOT "$ENV{ROCM_PATH}" STREQUAL "" AND IS_DIRECTORY "$ENV{ROCM_PATH}")
    set(__rocm_sdk "$ENV{ROCM_PATH}")
  elseif(NOT "$ENV{HIP_PATH}" STREQUAL "" AND IS_DIRECTORY "$ENV{HIP_PATH}")
    set(__rocm_sdk "$ENV{HIP_PATH}")
  else()
    # Derive the root from tools on PATH: <root>/bin/hipconfig or <root>/lib/llvm/bin/amdclang.
    find_program(__rocm_hipconfig NAMES hipconfig)
    find_program(__rocm_amdclang NAMES amdclang)
    if(__rocm_hipconfig)
      get_filename_component(__rocm_sdk "${__rocm_hipconfig}" DIRECTORY)
      get_filename_component(__rocm_sdk "${__rocm_sdk}" DIRECTORY)
    elseif(__rocm_amdclang)
      get_filename_component(__rocm_sdk "${__rocm_amdclang}" DIRECTORY)
      get_filename_component(__rocm_sdk "${__rocm_sdk}/../../.." ABSOLUTE)
    elseif(NOT WIN32 AND IS_DIRECTORY /opt/rocm)
      set(__rocm_sdk /opt/rocm)
    endif()
    unset(__rocm_hipconfig CACHE)
    unset(__rocm_amdclang CACHE)
  endif()

  if(__rocm_sdk)
    file(TO_CMAKE_PATH "${__rocm_sdk}" __rocm_sdk)
    set(ROCM_ROOT "${__rocm_sdk}")
    set(ROCM_INCLUDE_DIRS ${__rocm_sdk}/include)
    set(__rocm_lib_dirs ${__rocm_sdk}/lib ${__rocm_sdk}/lib64)
    find_library(ROCM_HIPHCC_LIBRARY amdhip64 PATHS ${__rocm_lib_dirs} NO_DEFAULT_PATH)
    find_library(ROCM_HIPBLAS_LIBRARY hipblas PATHS ${__rocm_lib_dirs} NO_DEFAULT_PATH)
    find_library(ROCM_HIPBLASLT_LIBRARY NAMES hipblaslt libhipblaslt PATHS ${__rocm_lib_dirs}
                 PATH_SUFFIXES "" hipblaslt NO_DEFAULT_PATH)
    if(NOT ROCM_HIPBLASLT_LIBRARY)
      # Fall back to hipBLASLt's CMake package (roc::hipblaslt carries the import library).
      find_package(hipblaslt CONFIG QUIET PATHS ${__rocm_sdk} NO_DEFAULT_PATH)
      if(TARGET roc::hipblaslt)
        set(ROCM_HIPBLASLT_TARGET roc::hipblaslt)
      endif()
    endif()

    if(WIN32)
      # Import libraries name the DLLs (e.g. amdhip64.lib -> amdhip64_7.dll). They are
      # delay-loaded from <root>/bin; see src/backend/rocm/runtime/rocm_dll_loader_win.cc.
      file(GLOB __rocm_hip_dlls RELATIVE "${__rocm_sdk}/bin" "${__rocm_sdk}/bin/amdhip64*.dll")
      set(ROCM_HIP_DLL_NAMES ${__rocm_hip_dlls})
      file(GLOB __rocm_hipblas_dlls RELATIVE "${__rocm_sdk}/bin"
           "${__rocm_sdk}/bin/hipblas.dll" "${__rocm_sdk}/bin/hipblas_*.dll"
           "${__rocm_sdk}/bin/libhipblaslt.dll" "${__rocm_sdk}/bin/hipblaslt*.dll")
      set(ROCM_HIPBLAS_DLL_NAMES ${__rocm_hipblas_dlls})
    endif()

    if(ROCM_HIPHCC_LIBRARY)
      set(ROCM_FOUND TRUE)
    endif()
  endif(__rocm_sdk)
  if(ROCM_FOUND)
    message(STATUS "Found ROCM_ROOT=" ${ROCM_ROOT})
    message(STATUS "Found ROCM_INCLUDE_DIRS=" ${ROCM_INCLUDE_DIRS})
    message(STATUS "Found ROCM_HIPHCC_LIBRARY=" ${ROCM_HIPHCC_LIBRARY})
    message(STATUS "Found ROCM_HIPBLAS_LIBRARY=" ${ROCM_HIPBLAS_LIBRARY})
    message(STATUS "Found ROCM_HIPBLASLT_LIBRARY=" ${ROCM_HIPBLASLT_LIBRARY})
    if(ROCM_HIPBLASLT_TARGET)
      message(STATUS "Found ROCM_HIPBLASLT_TARGET=" ${ROCM_HIPBLASLT_TARGET})
    endif()
    if(WIN32)
      message(STATUS "Found ROCM_HIP_DLL_NAMES=" "${ROCM_HIP_DLL_NAMES}")
      message(STATUS "Found ROCM_HIPBLAS_DLL_NAMES=" "${ROCM_HIPBLAS_DLL_NAMES}")
    endif()
  endif(ROCM_FOUND)
endmacro(find_rocm)
