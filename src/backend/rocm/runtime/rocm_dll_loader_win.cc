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
 * \file rocm_dll_loader_win.cc
 * \brief Windows: load ROCm DLLs from the ROCm install, never from System32.
 *
 * The AMD display driver installs its own amdhip64_*.dll into System32, and the standard
 * DLL search order checks System32 before PATH, so a by-name load picks the driver's copy
 * instead of the ROCm SDK's. Each module that uses ROCm DLLs links them with /DELAYLOAD
 * (see cmake/modules/ROCM.cmake) and includes this file. The hook below loads every
 * delay-loaded DLL found in <ROCm root>\\bin by full path, where the root is ROCM_PATH, then
 * HIP_PATH, then the root found at configure time. Dependencies of those DLLs resolve from
 * the same directory first (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR). If the ROCm copy cannot be
 * loaded, this fails instead of falling back to the driver's copy.
 */
// Only for MSVC-ABI toolchains (MSVC, clang-cl, amdclang++ targeting *-windows-msvc):
// /DELAYLOAD and this const hook declaration are MSVC delayimp features.
#if defined(_WIN32) && defined(_MSC_VER)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h must come first.
#include <delayimp.h>

#include <cstdio>
#include <string>

namespace {

// LoadLibraryExW needs a fully qualified path with backslashes.
std::wstring NormalizeDir(std::wstring path) {
  for (wchar_t& c : path) {
    if (c == L'/') c = L'\\';
  }
  while (!path.empty() && path.back() == L'\\') path.pop_back();
  return path;
}

std::wstring Widen(const char* s) {
  std::wstring out;
  for (; *s; ++s) out.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*s)));
  return out;
}

std::wstring RocmBinDir() {
  for (const wchar_t* var : {L"ROCM_PATH", L"HIP_PATH"}) {
    wchar_t buf[32768];
    DWORD n = GetEnvironmentVariableW(var, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
    // Empty counts as unset.
    if (n > 0 && n < sizeof(buf) / sizeof(buf[0])) {
      std::wstring root = NormalizeDir(std::wstring(buf, n));
      if (!root.empty()) return root + L"\\bin";
    }
  }
#ifdef TVM_ROCM_CONFIGURED_ROOT
  // Narrow string literal set by cmake/modules/ROCM.cmake.
  std::wstring root = NormalizeDir(Widen(TVM_ROCM_CONFIGURED_ROOT));
  if (!root.empty()) return root + L"\\bin";
  return std::wstring();
#else
  return std::wstring();
#endif
}

std::wstring ModulePath(HMODULE module) {
  wchar_t buf[32768];
  DWORD n = GetModuleFileNameW(module, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
  return std::wstring(buf, n);
}

bool SameDir(const std::wstring& file, const std::wstring& dir) {
  size_t slash = file.find_last_of(L"\\/");
  if (slash == std::wstring::npos) return false;
  return CompareStringOrdinal(file.c_str(), static_cast<int>(slash), dir.c_str(),
                              static_cast<int>(dir.size()), TRUE) == CSTR_EQUAL;
}

FARPROC Fail(PDelayLoadInfo info, const wchar_t* what, const std::wstring& detail) {
  std::fwprintf(stderr,
                L"[TVM ROCm] Cannot load %hs: %ls %ls\n"
                L"[TVM ROCm] Refusing to fall back to another copy (e.g. the display driver's "
                L"System32 copy). Set ROCM_PATH or HIP_PATH to the ROCm install root.\n",
                info->szDll, what, detail.c_str());
  std::fflush(stderr);
  ULONG_PTR args[1] = {reinterpret_cast<ULONG_PTR>(info)};
  RaiseException(VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND), 0, 1, args);
  return nullptr;
}

FARPROC WINAPI RocmDelayLoadHook(unsigned notify, PDelayLoadInfo info) {
  if (notify != dliNotePreLoadLibrary) return nullptr;
  std::wstring dir = RocmBinDir();
  if (dir.empty()) return Fail(info, L"ROCm install not found", L"(ROCM_PATH and HIP_PATH unset)");

  std::wstring name = Widen(info->szDll);
  std::wstring full = dir + L"\\" + name;
  if (GetFileAttributesW(full.c_str()) == INVALID_FILE_ATTRIBUTES) {
    // Not a ROCm DLL: keep the default behaviour. The HIP runtime itself must come from ROCm.
    if (name.size() >= 8 &&
        CompareStringOrdinal(name.c_str(), 8, L"amdhip64", 8, TRUE) == CSTR_EQUAL) {
      return Fail(info, L"not found at", full);
    }
    return nullptr;
  }

  // A module with this name may already be loaded (e.g. preloaded by Python). Reuse it only
  // if it is the ROCm copy; loading a second, different copy would split HIP state.
  if (HMODULE loaded = GetModuleHandleW(name.c_str())) {
    std::wstring path = NormalizeDir(ModulePath(loaded));
    if (SameDir(path, dir)) return reinterpret_cast<FARPROC>(loaded);
    return Fail(info, L"a different copy is already loaded:", path);
  }

  HMODULE module = LoadLibraryExW(
      full.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (module == nullptr) {
    return Fail(info, L"LoadLibraryExW failed for",
                full + L" (error " + std::to_wstring(GetLastError()) + L")");
  }
  return reinterpret_cast<FARPROC>(module);
}

}  // namespace

extern "C" const PfnDliHook __pfnDliNotifyHook2 = RocmDelayLoadHook;

#endif  // defined(_WIN32) && defined(_MSC_VER)
