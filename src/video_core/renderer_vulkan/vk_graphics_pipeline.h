// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <type_traits>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// Graphics pipeline libraries (EDEN_NX2_GPL): non-null `library_cache` builds the pipeline
/// from shared per-stage libraries; `libraries_only` builds only the libraries (warm-up).
struct GraphicsPipelineLibraryParams {
    GraphicsPipelineLibraryCache* library_cache{};
    std::array<u128, Tegra::Engines::Maxwell3D::Regs::MaxShaderStage> spirv_hashes{};
    bool libraries_only{};
};

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    /// Emits the stages' SPIR-V on a worker; with libraries it also returns each stage's SPIR-V
    /// hash (zero for absent stages).
    using EmitStagesFunc = std::function<std::array<vk::ShaderModule, NUM_STAGES>(
        std::array<u128, NUM_STAGES>& spirv_hashes)>;

    /// Graphics pipeline libraries (EDEN_NX2_GPL): non-null `library_cache` builds the pipeline
    /// from per-stage libraries; the stages were emitted with one descriptor set per stage.
    /// `libraries_only` builds the shader libraries and nothing else (warm-up, no draw).
    // Declared outside the class: GCC rejects a default argument `= {}` of a nested struct
    // with member initializers inside its enclosing class.
    using LibraryParams = GraphicsPipelineLibraryParams;

    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, Common::ThreadWorker* worker_thread,
        PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos,
        EmitStagesFunc emit_stages = {}, LibraryParams library_params = {});

    static std::chrono::nanoseconds TakeHoldTime() noexcept;

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    const std::array<u64, 6>& ShaderHashes() const noexcept {
        return key.unique_hashes;
    }

    struct SampledImage {
        u32 stage;
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The sampled textures of the last configured draw, kept while a sampled frame dump runs.
    std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }

    /// Switch 2 task/mesh pipeline: drawn with vkCmdDrawMeshTasksEXT.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return stage_infos[0].is_task || stage_infos[0].is_mesh;
    }
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    bool Configure(bool is_indexed) {
        return configure_func(this, is_indexed);
    }

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it{std::find(transition_keys.begin(), transition_keys.end(), current_key)};
        return it != transition_keys.end() ? transitions[std::distance(transition_keys.begin(), it)]
                                           : nullptr;
    }

    /// The frame the pipeline cache last handed this pipeline out (evicts long unused ones).
    void MarkUsed(u64 frame) noexcept {
        last_use_frame = frame;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use_frame;
    }

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transition_keys.clear();
        transitions.clear();
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return build->IsBuilt();
    }

    [[nodiscard]] bool Failed() const noexcept {
        return build->Failed();
    }

    /// Converted instructions over all stages, a predictor of the driver build time.
    void SetSourceWords(size_t words) noexcept {
        source_words = words;
    }

    [[nodiscard]] size_t SourceWords() const noexcept {
        return source_words;
    }

    /// Some stage writes storage buffers, images or global memory: a skipped draw would lose data.
    [[nodiscard]] bool WritesStorage() const noexcept {
        return writes_storage;
    }

    [[nodiscard]] bool UsesTransformFeedback() const noexcept {
        return key.state.xfb_enabled != 0;
    }

    [[nodiscard]] u64 KeyHash() const noexcept {
        return key.Hash();
    }

    /// Draws skipped while the pipeline was being built (EDEN_NX2_HEAVY_SKIP), and when the
    /// first one was; read and written on the GPU thread only.
    u32 heavy_skipped_draws{};
    /// Why the last unbuilt draw was held instead of skipped (index into the held_draw reasons).
    u8 hold_reason{};
    u32 hold_target_width{};  ///< Render target of that draw (color 0, or depth when colorless)
    u32 hold_target_height{};
    bool hold_target_color{};
    std::chrono::steady_clock::time_point heavy_skip_begin{};

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) { return pl->ConfigureImpl<Spec>(is_indexed); };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    template <typename Spec>
    bool ConfigureImpl(bool is_indexed);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    struct PipelineStates;

    /// The fixed-function state of the pipeline (shared by the monolithic and library builds).
    void FillStates(PipelineStates& states) const;

    vk::Pipeline MakePipeline(VkRenderPass render_pass, bool fast);

    /// Library build (EDEN_NX2_GPL): gets or builds the two shader libraries, then (unless
    /// libraries_only) links them with this pipeline's vertex input and fragment output parts.
    /// False when a library failed; the caller then builds the pipeline the monolithic way.
    bool MakeLibraryPipeline(VkRenderPass render_pass);

    GplLibrary MakeShaderLibrary(GplPart part, const PipelineStates& states,
                                 VkRenderPass render_pass) const;

    std::string ShaderLibraryKey(GplPart part, const PipelineStates& states,
                                 VkRenderPass render_pass) const;

    vk::Pipeline LinkLibraries(VkRenderPass render_pass, bool optimized);

    /// The library route failed: per-stage sets kept, with a layout a monolithic pipeline takes.
    void FallBackToMonolithic(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    bool (*configure_func)(GraphicsPipeline*, bool){};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    std::vector<GraphicsPipeline*> transitions;
    u64 last_use_frame{};
    size_t source_words{};

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    std::vector<SampledImage> sampled_images;
    u32 num_textures{};
    bool fragment_has_color0_output{};
    bool writes_storage{};

    /// Graphics pipeline libraries (EDEN_NX2_GPL): one descriptor set per stage (set index =
    /// stage index), the shader libraries, and this pipeline's own vertex input and fragment
    /// output libraries.
    GraphicsPipelineLibraryCache* const library_cache;
    const bool libraries_only;
    std::array<u128, NUM_STAGES> spirv_hashes{};
    std::array<vk::DescriptorSetLayout, NUM_STAGES> stage_set_layouts;
    std::array<std::string, NUM_STAGES> stage_set_signatures;
    std::array<DescriptorAllocator, NUM_STAGES> stage_allocators;
    std::array<vk::DescriptorUpdateTemplate, NUM_STAGES> stage_templates;
    std::array<size_t, NUM_STAGES> stage_descriptor_offsets{}; ///< Bytes into the update data
    bool has_stage_sets{};
    std::array<std::shared_ptr<const GplLibrary>, 2> shader_libraries;
    vk::Pipeline vertex_input_library;
    vk::Pipeline fragment_output_library;
    vk::DescriptorSetLayout fallback_empty_set_layout;
    bool linked{};

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;
    /// Fast first build (FastFirstBuildEnabled): the optimized pipeline once its build is done, and
    /// the unoptimized one kept alive after the swap for the command buffers that bound it.
    vk::Pipeline optimized_pipeline;
    vk::Pipeline retired_pipeline;
    std::atomic_bool optimized_ready{};
    bool optimized_bound{};
    Common::ThreadWorker* const build_workers;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    const bool worker_only_build;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
    /// Declared before `build`: destroyed after it, so the first build (which starts this one)
    /// has finished when this one is waited for.
    std::optional<PipelineBuild> optimized_build;
    std::optional<PipelineBuild> build;
};

} // namespace Vulkan
