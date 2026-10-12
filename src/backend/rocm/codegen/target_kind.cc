/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file target_kind.cc
 * \brief ROCm compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <cctype>
#include <string>

namespace tvm {

namespace backend {
namespace rocm {

std::string ExtractStringWithPrefix(const std::string& str, const std::string& prefix) {
  if (str.find(prefix) != 0) return "";
  std::size_t pos = prefix.length();
  while (pos < str.length() && (std::isdigit(str[pos]) || std::isalpha(str[pos]))) {
    ++pos;
  }
  return str.substr(prefix.length(), pos - prefix.length());
}

bool DetectDeviceFlag(Device device, runtime::DeviceAttrKind flag, ffi::Any* val) {
  using runtime::DeviceAPI;
  DeviceAPI* api = DeviceAPI::Get(device, true);
  if (api == nullptr) {
    return false;
  }
  api->GetAttr(device, runtime::kExist, val);
  int exists = val->cast<int>();
  if (!exists) {
    return false;
  }
  DeviceAPI::Get(device)->GetAttr(device, flag, val);
  return true;
}

void CheckOrSetAttr(ffi::Map<ffi::String, ffi::Any>* attrs, const ffi::String& name,
                    const ffi::String& value) {
  auto iter = attrs->find(name);
  if (iter == attrs->end()) {
    attrs->Set(name, value);
  } else {
    auto str = (*iter).second.try_cast<ffi::String>();
    TVM_FFI_CHECK(str && str.value() == value, ValueError)
        << "Expects \"" << name << "\" to be \"" << value << "\", but gets: " << (*iter).second;
  }
}

// GFX major version from the part after "gfx": "1101" -> 11, "90a" -> 9, "11-generic" -> 11.
int GfxMajor(const std::string& arch) {
  if (arch.empty() || !std::isdigit(static_cast<unsigned char>(arch[0]))) return 0;
  size_t n = arch.size() >= 4 ? 2 : (arch.size() == 3 ? 1 : arch.size());
  int major = 0;
  for (size_t i = 0; i < n && std::isdigit(static_cast<unsigned char>(arch[i])); ++i) {
    major = major * 10 + (arch[i] - '0');
  }
  return major;
}

bool HasMattr(const ffi::Array<ffi::String>& mattr, const std::string& feature) {
  for (const ffi::String& attr : mattr) {
    if (std::string(attr) == feature) return true;
  }
  return false;
}

ffi::Map<ffi::String, ffi::Any> UpdateROCmAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  // "amdgcn-amd-amdhsa-hcc" is the pre-ROCm-4 triple; accept it but emit the current one.
  constexpr const char* kTriple = "amdgcn-amd-amdhsa";
  if (target.count("mtriple")) {
    auto triple = target.at("mtriple").try_cast<ffi::String>();
    if (triple && triple.value() == "amdgcn-amd-amdhsa-hcc")
      target.Set("mtriple", ffi::String(kTriple));
  }
  CheckOrSetAttr(&target, "mtriple", kTriple);

  std::string arch = "gfx900";
  bool mcpu_given = target.count("mcpu") != 0;
  if (mcpu_given) {
    ffi::String mcpu = target.at("mcpu").as_or_throw<ffi::String>();
    arch = ExtractStringWithPrefix(mcpu, "gfx");
    TVM_FFI_CHECK(!arch.empty(), ValueError)
        << "ROCm target gets an invalid GFX version: -mcpu=" << mcpu;
  } else {
    if (const auto f_get_rocm_arch = tvm::ffi::Function::GetGlobal("tvm_callback_rocm_get_arch")) {
      arch = (*f_get_rocm_arch)().cast<std::string>();
    }
    target.Set("mcpu", ffi::String(arch));
    arch = ExtractStringWithPrefix(arch, "gfx");
  }

  // Wavefront size: RDNA (gfx10+) compiles as wave32 by default, CDNA/GCN (gfx9 and older)
  // only supports wave64. Derive it from the arch; when the arch was auto-detected, use the
  // local device's value.
  bool rdna = GfxMajor(arch) >= 10;
  int64_t warp_size = rdna ? 32 : 64;
  if (target.count("thread_warp_size")) {
    warp_size = target.at("thread_warp_size").cast<int64_t>();
  } else {
    ffi::Any val;
    if (!mcpu_given && DetectDeviceFlag({kDLROCM, 0}, runtime::kWarpSize, &val)) {
      warp_size = val.cast<int64_t>();
    }
    target.Set("thread_warp_size", warp_size);
  }
  TVM_FFI_CHECK(warp_size == 32 || warp_size == 64, ValueError)
      << "ROCm thread_warp_size must be 32 or 64, got " << warp_size;
  TVM_FFI_CHECK(rdna || warp_size == 64, ValueError)
      << "gfx" << arch << " only supports thread_warp_size=64";
  if (rdna && warp_size == 64) {
    // Make LLVM generate wave64 code to match.
    ffi::Array<ffi::String> mattr;
    if (target.count("mattr")) mattr = target.at("mattr").as_or_throw<ffi::Array<ffi::String>>();
    if (!HasMattr(mattr, "+wavefrontsize64")) {
      mattr.push_back("+wavefrontsize64");
      target.Set("mattr", mattr);
    }
  }
  return target;
}

}  // namespace rocm
}  // namespace backend

namespace codegen {
#ifdef TVM_LLVM_VERSION
void RegisterAMDGPUCodegen();
namespace llvm {
void RegisterROCMIntrinRules();
}  // namespace llvm
#endif
}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace tvm;
  using namespace tvm::backend::rocm;
  namespace refl = tvm::ffi::reflection;

  TargetKindDef("rocm")
      .set_default_device_type(kDLROCM)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("mtriple")
      .def_option<ffi::Array<ffi::String>>("mattr")
      .def_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .def_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .def_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(65536))
      // No default: UpdateROCmAttrs sets it from the device or arch (32 on RDNA, 64 on CDNA).
      .def_option<int64_t>("thread_warp_size")
      .set_default_keys({"rocm", "gpu"})
      .set_target_canonicalizer(UpdateROCmAttrs);
#ifdef TVM_LLVM_VERSION
  tvm::codegen::llvm::RegisterROCMIntrinRules();
  tvm::codegen::RegisterAMDGPUCodegen();
#endif
}
