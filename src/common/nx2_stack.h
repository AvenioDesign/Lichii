// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

namespace Nx2Stack {

/// The calling thread's host return addresses as " +0x<offset>" relative to the executable's
/// image base (symbolize offline against the PDB), " ?0x<address>" for frames in other modules.
/// Windows only; elsewhere returns an empty string.
std::string CaptureModuleOffsets(unsigned skip, unsigned count);

/// The calling thread's name as set by Common::SetCurrentThreadName (Windows only).
std::string CurrentThreadName();

} // namespace Nx2Stack
