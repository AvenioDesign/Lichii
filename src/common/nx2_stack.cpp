// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_stack.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fmt/format.h>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace Nx2Stack {

std::string CaptureModuleOffsets(unsigned skip, unsigned count) {
    void* frames[32]{};
    count = count > 32 ? 32 : count;
    const USHORT captured = RtlCaptureStackBackTrace(skip + 1, count, frames, nullptr);
    const auto base = reinterpret_cast<uintptr_t>(&__ImageBase);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(
        reinterpret_cast<const char*>(&__ImageBase) + __ImageBase.e_lfanew);
    const uintptr_t end = base + nt->OptionalHeader.SizeOfImage;
    std::string out;
    for (USHORT i = 0; i < captured; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(frames[i]);
        if (address >= base && address < end) {
            out += fmt::format(" +{:#x}", address - base);
        } else {
            out += fmt::format(" ?{:#x}", address);
        }
    }
    return out;
}

std::string CurrentThreadName() {
    PWSTR description = nullptr;
    std::string out;
    if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description) {
        for (const wchar_t* c = description; *c != L'\0'; ++c) {
            out += *c < 0x80 ? static_cast<char>(*c) : '?';
        }
        LocalFree(description);
    }
    return out;
}

} // namespace Nx2Stack

#else

namespace Nx2Stack {
std::string CaptureModuleOffsets(unsigned, unsigned) {
    return {};
}
std::string CurrentThreadName() {
    return {};
}
} // namespace Nx2Stack

#endif
