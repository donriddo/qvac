#pragma once

#include <any>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <stable-diffusion.h>

#include <qvac-lib-inference-addon-cpp/ModelInterfaces.hpp>
#include <qvac-lib-inference-addon-cpp/RuntimeStats.hpp>

/**
 * Core stable-diffusion.cpp model wrapper.
 *
 * Wraps the new struct-based stable-diffusion.cpp API:
 *   - sd_ctx_params_t / new_sd_ctx(const sd_ctx_params_t*)
 *   - sd_img_gen_params_t / generate_image()
 *   - sd_vid_gen_params_t / generate_video()
 */
class SdModel
    : public qvac_lib_inference_addon_cpp::model::IModel,
      public qvac_lib_inference_addon_cpp::model::IModelCancel {
public:
  SdModel(const SdModel&)            = delete;
  SdModel& operator=(const SdModel&) = delete;
  SdModel(SdModel&&)                 = delete;
  SdModel& operator=(SdModel&&)      = delete;

  /**
   * @param modelPath   Path to the main weights file (.gguf, .safetensors, .ckpt)
   * @param clipLPath   Optional CLIP-L text encoder  (FLUX.1 / SD3)
   * @param clipGPath   Optional CLIP-G text encoder  (SDXL / SD3)
   * @param t5XxlPath   Optional T5-XXL text encoder  (FLUX.1 / SD3)
   * @param llmPath     Optional LLM text encoder     (FLUX.2 klein → Qwen3 8B)
   * @param vaePath     Optional separate VAE
   * @param configMap   Key/value config options (threads, device, wtype, rng, etc.)
   */
  SdModel(
      std::string modelPath,
      std::string clipLPath,
      std::string clipGPath,
      std::string t5XxlPath,
      std::string llmPath,
      std::string vaePath,
      std::unordered_map<std::string, std::string> configMap);

  ~SdModel() override;

  [[nodiscard]] std::string getName() const final { return "SdModel"; }

  /**
   * Input for a single generation job, passed as std::any through addon-cpp.
   */
  struct GenerationJob {
    std::string paramsJson;

    /** Called each diffusion step with JSON: {"step":N,"total":M,"elapsed_ms":T} */
    std::function<void(const std::string&)> progressCallback;

    /** Called once per output image/frame with PNG-encoded bytes */
    std::function<void(const std::vector<uint8_t>&)> outputCallback;
  };

  std::any process(const std::any& input) final;
  void cancel() const final;
  [[nodiscard]] qvac_lib_inference_addon_cpp::RuntimeStats runtimeStats() const final;

  static void sdLogCallback(sd_log_level_t level, const char* text, void* userData);

private:
  /** Encode sd_image_t raw pixels → PNG bytes using stb_image_write. */
  static std::vector<uint8_t> encodeToPng(const sd_image_t& img);

  /** Decode PNG bytes → sd_image_t (caller must free .data). */
  static sd_image_t decodePng(const std::vector<uint8_t>& pngBytes);

  /** Parse sampler name → sample_method_t enum. */
  static sample_method_t parseSampler(const std::string& name);

  /** Parse scheduler name → scheduler_t enum. */
  static scheduler_t parseScheduler(const std::string& name);

  /** Parse weight type string → sd_type_t enum. */
  static sd_type_t parseWeightType(const std::string& name);

  const std::string modelPath_;
  const std::string clipLPath_;
  const std::string clipGPath_;
  const std::string t5XxlPath_;
  const std::string llmPath_;
  const std::string vaePath_;

  std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> sdCtx_;
  mutable std::atomic<bool> cancelRequested_{ false };
  mutable qvac_lib_inference_addon_cpp::RuntimeStats lastStats_{};
};
