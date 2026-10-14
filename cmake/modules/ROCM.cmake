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

# ROCM Module
find_rocm(${USE_ROCM})

if(ROCM_FOUND)
  # always set the includedir
  # avoid global retrigger of CMake
  include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
  add_definitions(-D__HIP_PLATFORM_AMD__=1)
endif(ROCM_FOUND)

# Windows: the display driver ships its own amdhip64_*.dll in System32, which the standard
# DLL search order prefers over PATH. Delay-load the ROCm DLLs and resolve them by full path
# from the ROCm install instead (src/backend/rocm/runtime/rocm_dll_loader_win.cc).
set(TVM_ROCM_DLL_LOADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/rocm/runtime/rocm_dll_loader_win.cc)
function(tvm_rocm_windows_delayload objs_target link_target)
  if(NOT WIN32)
    return()
  endif()
  if(NOT (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
    message(WARNING "ROCm on Windows needs an MSVC-compatible linker for /DELAYLOAD; "
                    "the HIP runtime may be loaded from System32")
    return()
  endif()
  set(_dlls ${ARGN})
  if(NOT _dlls)
    message(FATAL_ERROR "No ROCm DLLs found in ${ROCM_ROOT}/bin to delay-load")
  endif()
  target_compile_definitions(${objs_target} PRIVATE TVM_ROCM_CONFIGURED_ROOT="${ROCM_ROOT}")
  foreach(_dll IN LISTS _dlls)
    target_link_options(${link_target} PRIVATE "LINKER:/DELAYLOAD:${_dll}")
  endforeach()
  target_link_libraries(${link_target} PRIVATE delayimp)
endfunction()

if(USE_ROCM)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCM, USE_ROCM=" ${USE_ROCM})
  endif()
  message(STATUS "Build rocm device runtime")

  # Includes rocm_dll_loader_win.cc, which is empty outside Windows.
  tvm_file_glob(GLOB RUNTIME_ROCM_SRCS src/backend/rocm/runtime/*.cc)

  set(_rocm_libs ${ROCM_HIPHCC_LIBRARY})

  add_library(tvm_runtime_rocm_objs OBJECT ${RUNTIME_ROCM_SRCS})
  target_link_libraries(tvm_runtime_rocm_objs PUBLIC tvm_ffi_header)
  set_target_properties(tvm_runtime_rocm_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TVM_VISIBILITY_FLAG)
    target_compile_options(tvm_runtime_rocm_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
  endif()
  add_library(tvm_runtime_rocm SHARED $<TARGET_OBJECTS:tvm_runtime_rocm_objs>)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_rocm)
  target_link_libraries(tvm_runtime_rocm PUBLIC tvm_runtime ${_rocm_libs})
  tvm_rocm_windows_delayload(tvm_runtime_rocm_objs tvm_runtime_rocm ${ROCM_HIP_DLL_NAMES})
  tvm_configure_target_library(tvm_runtime_rocm RUNTIME_MODULE)
endif(USE_ROCM)

# HIPBLAS contrib goes into libtvm_runtime_extra.
if(USE_ROCM AND USE_HIPBLAS)
  message(STATUS "Build with HIPBLAS support")
  tvm_file_glob(GLOB HIPBLAS_CONTRIB_SRC src/relax/backend/contrib/hipblas/*.cc)
  list(APPEND COMPILER_SRCS ${HIPBLAS_CONTRIB_SRC})
  tvm_file_glob(GLOB HIPBLAS_CONTRIB_SRCS src/runtime/extra/contrib/hipblas/*.cc)
  add_library(tvm_hipblas_objs OBJECT ${HIPBLAS_CONTRIB_SRCS})
  target_link_libraries(tvm_hipblas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_hipblas_objs ${ROCM_HIPBLAS_LIBRARY}
                        ${ROCM_HIPHCC_LIBRARY})
  # The hipBLAS contrib calls hipBLASLt directly, so it must be linked.
  if(ROCM_HIPBLASLT_LIBRARY)
    target_link_libraries(tvm_runtime_extra PRIVATE ${ROCM_HIPBLASLT_LIBRARY})
  elseif(ROCM_HIPBLASLT_TARGET)
    target_link_libraries(tvm_runtime_extra PRIVATE ${ROCM_HIPBLASLT_TARGET})
  else()
    message(FATAL_ERROR "USE_HIPBLAS needs hipBLASLt, but neither its library nor its CMake "
                        "package was found under ${ROCM_ROOT}")
  endif()
endif(USE_ROCM AND USE_HIPBLAS)

if(USE_ROCM AND USE_THRUST)
  message(STATUS "Build with rocThrust support")
  # rocThrust must be compiled by a HIP-capable compiler. On Windows use amdclang++:
  # hipcc can fail to find the C++ standard headers there.
  if(${CMAKE_CXX_COMPILER} MATCHES "(hipcc|amdclang\\+\\+)(\\.exe)?$")
    message(STATUS "Using ${CMAKE_CXX_COMPILER} to compile rocthrust code.")
  else()
    message(FATAL_ERROR "Set CXX=hipcc or CXX=amdclang++ to compile rocthrust code.")
  endif()

  find_package(rocprim REQUIRED)
  find_package(rocthrust REQUIRED)
  set_source_files_properties(src/runtime/extra/contrib/thrust/thrust.cu PROPERTIES LANGUAGE CXX)
  add_library(tvm_rocthrust_objs OBJECT src/runtime/extra/contrib/thrust/thrust.cu)
  target_link_libraries(tvm_rocthrust_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_rocthrust_objs roc::rocthrust)
endif(USE_ROCM AND USE_THRUST)

# Windows: libtvm_runtime_extra loads the ROCm DLLs (HIP runtime, hipBLAS) through the same
# delay-load hook as libtvm_runtime_rocm.
if(WIN32 AND USE_ROCM AND (USE_HIPBLAS OR USE_THRUST))
  add_library(tvm_rocm_extra_loader_objs OBJECT ${TVM_ROCM_DLL_LOADER_SRC})
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_rocm_extra_loader_objs)
  set(_rocm_extra_dlls ${ROCM_HIP_DLL_NAMES})
  if(USE_HIPBLAS)
    list(APPEND _rocm_extra_dlls ${ROCM_HIPBLAS_DLL_NAMES})
  endif()
  tvm_rocm_windows_delayload(tvm_rocm_extra_loader_objs tvm_runtime_extra ${_rocm_extra_dlls})
endif()
