// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/logging.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_light_lock.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/nx2_guest_profile.h"

namespace Kernel {
namespace {

constexpr size_t WaitKinds = static_cast<size_t>(KThread::Nx2WaitKind::Count);
constexpr auto ReportWindow = std::chrono::seconds{10};
constexpr size_t ReportedThreads = 10;

struct Snapshot {
    s64 cpu{};
    std::array<u64, WaitKinds> wait_ns{};
    std::array<u64, WaitKinds> wait_count{};
};

struct Row {
    u64 id{};
    s32 priority{};
    s32 core{};
    u64 mask{};
    double busy{};
    std::array<double, WaitKinds> wait_ms_per_s{};
    std::array<double, WaitKinds> waits_per_s{};
    u64 pc{};
    u64 lr{};
};

struct State {
    std::mutex mutex;
    bool started{};
    std::chrono::steady_clock::time_point last_wall{};
    s64 last_ticks{};
    std::unordered_map<u64, Snapshot> threads;
    std::array<s64, Core::Hardware::NUM_CPU_CORES> idle{};
};

State& GetState() {
    static State state;
    return state;
}

Snapshot Take(const KThread& thread) {
    Snapshot snap;
    snap.cpu = thread.GetCpuTime();
    for (size_t kind = 0; kind < WaitKinds; ++kind) {
        snap.wait_ns[kind] = thread.Nx2WaitNs(static_cast<KThread::Nx2WaitKind>(kind));
        snap.wait_count[kind] = thread.Nx2WaitCount(static_cast<KThread::Nx2WaitKind>(kind));
    }
    return snap;
}

} // Anonymous namespace

void Nx2GuestProfileMaybeReport(KernelCore& kernel, KProcess* process) {
    if (process == nullptr) {
        return;
    }
    State& state = GetState();
    const auto now = std::chrono::steady_clock::now();
    std::unique_lock lock{state.mutex, std::try_to_lock};
    if (!lock.owns_lock() || (state.started && now - state.last_wall < ReportWindow)) {
        return;
    }
    const s64 ticks = static_cast<s64>(kernel.System().CoreTiming().GetClockTicks());
    const bool report = state.started;
    const double tick_span = report ? static_cast<double>(ticks - state.last_ticks) : 0.0;
    const double wall_s =
        report ? std::chrono::duration<double>(now - state.last_wall).count() : 0.0;
    const u64 code_base = GetInteger(process->GetEntryPoint());

    std::vector<Row> rows;
    std::unordered_map<u64, Snapshot> next;
    {
        KScopedLightLock list_lock(process->GetListLock());
        for (KThread& thread : process->GetThreadList()) {
            const u64 id = thread.GetId();
            const Snapshot snap = Take(thread);
            if (report && tick_span > 0.0 && wall_s > 0.0) {
                const auto it = state.threads.find(id);
                const Snapshot prev = it != state.threads.end() ? it->second : Snapshot{};
                Row row;
                row.id = id;
                row.priority = thread.GetPriority();
                row.core = thread.GetActiveCore();
                row.mask = thread.GetAffinityMask().GetAffinityMask();
                row.busy = static_cast<double>(snap.cpu - prev.cpu) / tick_span * 100.0;
                for (size_t kind = 0; kind < WaitKinds; ++kind) {
                    row.wait_ms_per_s[kind] =
                        static_cast<double>(snap.wait_ns[kind] - prev.wait_ns[kind]) / 1e6 / wall_s;
                    row.waits_per_s[kind] =
                        static_cast<double>(snap.wait_count[kind] - prev.wait_count[kind]) / wall_s;
                }
                // Saved context: where the thread was when it last left its core (stale while it
                // runs). As offsets from the process code start, to find the code in the module.
                const auto& context = thread.GetContext();
                row.pc = context.pc >= code_base ? context.pc - code_base : 0;
                row.lr = context.lr >= code_base ? context.lr - code_base : 0;
                rows.push_back(row);
            }
            next.emplace(id, snap);
        }
    }

    std::array<double, Core::Hardware::NUM_CPU_CORES> idle{};
    for (size_t core = 0; core < Core::Hardware::NUM_CPU_CORES; ++core) {
        const KThread* idle_thread = kernel.Scheduler(core).GetIdleThread();
        const s64 value = idle_thread != nullptr ? idle_thread->GetCpuTime() : 0;
        if (report && tick_span > 0.0) {
            idle[core] = static_cast<double>(value - state.idle[core]) / tick_span * 100.0;
        }
        state.idle[core] = value;
    }

    state.threads = std::move(next);
    state.last_wall = now;
    state.last_ticks = ticks;
    state.started = true;
    if (!report || rows.empty()) {
        return;
    }

    std::ranges::sort(rows, [](const Row& a, const Row& b) { return a.busy > b.busy; });
    double total_busy = 0.0;
    for (const Row& row : rows) {
        total_busy += row.busy;
    }
    LOG_INFO(Kernel,
             "Nx2GuestProfile {:.1f} s: {} guest threads, busy sum {:.0f} %, guest cores idle "
             "{:.0f} / {:.0f} / {:.0f} / {:.0f} / {:.0f} / {:.0f} %",
             wall_s, rows.size(), total_busy, idle[0], idle[1], idle[2], idle[3], idle[4],
             idle[5]);
    for (size_t i = 0; i < (std::min)(rows.size(), ReportedThreads); ++i) {
        const Row& row = rows[i];
        LOG_INFO(Kernel,
                 "Nx2GuestProfile   thread {:#x} prio {} core {} mask {:#x}: busy {:.1f} %, "
                 "ipc {:.1f} ms/s ({:.0f}/s), sync {:.1f} ms/s ({:.0f}/s), lock {:.1f} ms/s "
                 "({:.0f}/s), sleep {:.1f} ms/s ({:.0f}/s), pc +{:#x} lr +{:#x}",
                 row.id, row.priority, row.core, row.mask, row.busy, row.wait_ms_per_s[0],
                 row.waits_per_s[0], row.wait_ms_per_s[1], row.waits_per_s[1],
                 row.wait_ms_per_s[2], row.waits_per_s[2], row.wait_ms_per_s[3],
                 row.waits_per_s[3], row.pc, row.lr);
    }
}

} // namespace Kernel
