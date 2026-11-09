// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Diagnostic build: how the guest threads of the application spend their time. Each guest thread
// already counts its CPU time (KScheduler adds it at every context switch). This adds the wall time
// a thread spends blocked inside IPC, synchronization, lock and sleep SVCs, and logs the busiest
// threads every 10 s. It answers whether the game's main thread is CPU bound (busy close to 100 %)
// or waits on something the emulator could make faster.

#pragma once

#include <chrono>

#include "core/hle/kernel/k_thread.h"

namespace Kernel {

class KProcess;
class KernelCore;

/// Adds the wall time between construction and destruction to the current guest thread.
class Nx2SvcWaitScope {
public:
    Nx2SvcWaitScope(KernelCore& kernel, KThread::Nx2WaitKind kind)
        : m_thread{GetCurrentThreadPointer(kernel)}, m_kind{kind},
          m_start{std::chrono::steady_clock::now()} {}

    ~Nx2SvcWaitScope() {
        if (m_thread != nullptr) {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - m_start)
                                .count();
            m_thread->Nx2AddWait(m_kind, static_cast<u64>(ns));
        }
    }

    Nx2SvcWaitScope(const Nx2SvcWaitScope&) = delete;
    Nx2SvcWaitScope& operator=(const Nx2SvcWaitScope&) = delete;

private:
    KThread* m_thread;
    KThread::Nx2WaitKind m_kind;
    std::chrono::steady_clock::time_point m_start;
};

/// Logs the guest thread profile of @p process when 10 s have passed since the last report.
/// Cheap to call often (one clock read when no report is due).
void Nx2GuestProfileMaybeReport(KernelCore& kernel, KProcess* process);

} // namespace Kernel
