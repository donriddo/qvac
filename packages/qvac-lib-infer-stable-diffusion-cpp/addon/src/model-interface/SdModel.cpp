#include "SdModel.hpp"

#include <chrono>
#include <cstring>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <picojson/picojson.h>
#include <qvac-lib-inference-addon-cpp/Errors.hpp>
#include <qvac-lib-inference-addon-cpp/Logger.hpp>

#include "utils/LoggingMacros.hpp"

using namespace qvac_lib_inference_addon_cpp;
using namespace qvac_errors;

// ---------------------------------------------------------------------------
// Thread-local progress context (stable-diffusion.cpp progress CB is global)
// ---------------------------------------------------------------------------
namespace {

struct ProgressCtx {
  const SdModel::GenerationJob* job = nullptr;
  std::chrono::steady_clock::time_point startTime;
};

thread_local ProgressCtx tl_progressCtx;

void sdProgressCallback(int step, int steps, float /*time*/, void* /*data*/) {
  if (!tl_progressCtx.job || !tl_progressCtx.job->progressCallback) return;

  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - tl_progressCtx.startTime)
          .count();

  std::ostringstream oss;
  oss << R"({"step":)" << step
      << R"(,"total":)" << steps
      << R"(,"elapsed_ms":)" << elapsed << "}";

  tl_progressCtx.job->progressCallback(oss.str());
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SdModel::SdModel(
    std::string modelPath,
    std::string clipLPath,
    std::string clipGPath,
    std::string t5XxlPath,
    std::string llmPath,
    std::string vaePath,
    std::unordered_map<std::string, std::string> configMap)
    : modelPath_(std::move(modelPath)),
      clipLPath_(std::move(clipLPath)),
      clipGPath_(std::move(clipGPath)),
      t5XxlPath_(std::move(t5XxlPath)),
      llmPath_(std::move(llmPath)),
      vaePath_(std::move(vaePath)),
      sdCtx_(nullptr, &free_sd_ctx) {

  qvac_lib_inference_addon_sd::logging::setVerbosityLevel(configMap);

  sd_set_log_callback(SdModel::sdLogCallback, nullptr);

  sd_ctx_params_t params{};
  sd_ctx_params_init(&params);

  params.model_path  = modelPath_.empty()  ? nullptr : modelPath_.c_str();
  params.clip_l_path = clipLPath_.empty()  ? nullptr : clipLPath_.c_str();
  params.clip_g_path = clipGPath_.empty()  ? nullptr : clipGPath_.c_str();
  params.t5xxl_path  = t5XxlPath_.empty()  ? nullptr : t5XxlPath_.c_str();
  params.llm_path    = llmPath_.empty()    ? nullptr : llmPath_.c_str();    // FLUX.2 (Qwen3)
  params.vae_path    = vaePath_.empty()    ? nullptr : vaePath_.c_str();
  params.free_params_immediately = true;

  if (auto it = configMap.find("threads"); it != configMap.end()) {
    try { params.n_threads = std::stoi(it->second); } catch (...) {}
  }
  if (auto it = configMap.find("wtype"); it != configMap.end()) {
    params.wtype = parseWeightType(it->second);
  }
  if (auto it = configMap.find("rng"); it != configMap.end()) {
    params.rng_type = (it->second == "cpu") ? CPU_RNG : CUDA_RNG;
  }
  if (auto it = configMap.find("clip_on_cpu"); it != configMap.end()) {
    params.keep_clip_on_cpu = (it->second == "1" || it->second == "true");
  }
  if (auto it = configMap.find("vae_on_cpu"); it != configMap.end()) {
    params.keep_vae_on_cpu = (it->second == "1" || it->second == "true");
  }
  if (auto it = configMap.find("flash_attn"); it != configMap.end()) {
    params.flash_attn = (it->second == "1" || it->second == "true");
  }

  sd_ctx_t* raw = new_sd_ctx(&params);
  if (!raw) {
    throw StatusError(
        general_error::InternalError,
        "Failed to create stable-diffusion context. "
        "Check model path and format: " + modelPath_);
  }
  sdCtx_.reset(raw);
}

SdModel::~SdModel() = default;

// ---------------------------------------------------------------------------
// IModel::process
// ---------------------------------------------------------------------------

std::any SdModel::process(const std::any& input) {
  const auto& job = std::any_cast<const GenerationJob&>(input);

  cancelRequested_.store(false);

  tl_progressCtx.job       = &job;
  tl_progressCtx.startTime = std::chrono::steady_clock::now();
  sd_set_progress_callback(sdProgressCallback, nullptr);

  // --- Parse params JSON ---
  picojson::value v;
  const std::string parseErr = picojson::parse(v, job.paramsJson);
  if (!parseErr.empty()) {
    throw StatusError(
        general_error::InvalidArgument,
        "Failed to parse generation params JSON: " + parseErr);
  }
  if (!v.is<picojson::object>()) {
    throw StatusError(general_error::InvalidArgument, "Params must be a JSON object");
  }

  const auto& obj = v.get<picojson::object>();

  auto getStr = [&](const std::string& key, const std::string& def = "") -> std::string {
    auto it = obj.find(key);
    return (it != obj.end() && it->second.is<std::string>())
               ? it->second.get<std::string>()
               : def;
  };
  auto getInt = [&](const std::string& key, int def) -> int {
    auto it = obj.find(key);
    return (it != obj.end() && it->second.is<double>())
               ? static_cast<int>(it->second.get<double>())
               : def;
  };
  auto getFloat = [&](const std::string& key, float def) -> float {
    auto it = obj.find(key);
    return (it != obj.end() && it->second.is<double>())
               ? static_cast<float>(it->second.get<double>())
               : def;
  };

  const std::string mode           = getStr("mode", "txt2img");
  const std::string prompt         = getStr("prompt");
  const std::string negativePrompt = getStr("negative_prompt");
  const int width                  = getInt("width", 512);
  const int height                 = getInt("height", 512);
  const int steps                  = getInt("steps", 20);
  const float cfgScale             = getFloat("cfg_scale", 7.0f);
  const int64_t seed               = static_cast<int64_t>(getInt("seed", -1));
  const int batchCount             = getInt("batch_count", 1);
  const float strength             = getFloat("strength", 0.75f);
  const sample_method_t sampler    = parseSampler(getStr("sampler", "euler_a"));
  const scheduler_t scheduler      = parseScheduler(getStr("scheduler", "discrete"));

  auto t0 = std::chrono::steady_clock::now();
  int outputCount = 0;

  if (mode == "txt2img" || mode == "img2img") {
    sd_img_gen_params_t genParams{};
    sd_img_gen_params_init(&genParams);

    genParams.prompt          = prompt.c_str();
    genParams.negative_prompt = negativePrompt.c_str();
    genParams.width           = width;
    genParams.height          = height;
    genParams.seed            = seed;
    genParams.batch_count     = batchCount;
    genParams.strength        = strength;

    genParams.sample_params.sample_method  = sampler;
    genParams.sample_params.scheduler      = scheduler;
    genParams.sample_params.sample_steps   = steps;
    genParams.sample_params.guidance.txt_cfg = cfgScale;

    // img2img: decode PNG from the init_image_bytes array field
    sd_image_t initImg{};
    std::vector<uint8_t> initPng;
    if (mode == "img2img") {
      if (auto it = obj.find("init_image_bytes");
          it != obj.end() && it->second.is<picojson::array>()) {
        const auto& arr = it->second.get<picojson::array>();
        initPng.reserve(arr.size());
        for (const auto& el : arr) {
          initPng.push_back(static_cast<uint8_t>(el.get<double>()));
        }
      }
      if (!initPng.empty()) {
        initImg = decodePng(initPng);
      }
    }
    genParams.init_image = initImg;

    sd_image_t* results = generate_image(sdCtx_.get(), &genParams);

    if (initImg.data) free(initImg.data);

    if (results) {
      for (int i = 0; i < batchCount; ++i) {
        if (results[i].data && !cancelRequested_.load()) {
          auto png = encodeToPng(results[i]);
          if (!png.empty() && job.outputCallback) {
            job.outputCallback(png);
            ++outputCount;
          }
          free(results[i].data);
        }
      }
      free(results);
    }
  } else if (mode == "txt2vid") {
    sd_vid_gen_params_t vidParams{};
    sd_vid_gen_params_init(&vidParams);

    vidParams.prompt          = prompt.c_str();
    vidParams.negative_prompt = negativePrompt.c_str();
    vidParams.width           = width;
    vidParams.height          = height;
    vidParams.seed            = seed;
    vidParams.video_frames    = getInt("frames", 16);

    vidParams.sample_params.sample_method  = sampler;
    vidParams.sample_params.scheduler      = scheduler;
    vidParams.sample_params.sample_steps   = steps;
    vidParams.sample_params.guidance.txt_cfg = cfgScale;

    int numFrames = 0;
    sd_image_t* frames = generate_video(sdCtx_.get(), &vidParams, &numFrames);

    if (frames) {
      for (int i = 0; i < numFrames; ++i) {
        if (frames[i].data && !cancelRequested_.load()) {
          auto png = encodeToPng(frames[i]);
          if (!png.empty() && job.outputCallback) {
            job.outputCallback(png);
            ++outputCount;
          }
          free(frames[i].data);
        }
      }
      free(frames);
    }
  } else {
    throw StatusError(
        general_error::InvalidArgument,
        "Unknown generation mode: " + mode + ". Supported: txt2img, img2img, txt2vid");
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double genMs =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  lastStats_.clear();
  lastStats_.push_back({"generation_time", genMs});
  lastStats_.push_back({"steps",           static_cast<int64_t>(steps)});
  lastStats_.push_back({"width",           static_cast<int64_t>(width)});
  lastStats_.push_back({"height",          static_cast<int64_t>(height)});
  lastStats_.push_back({"output_count",    static_cast<int64_t>(outputCount)});

  tl_progressCtx.job = nullptr;
  sd_set_progress_callback(nullptr, nullptr);

  return lastStats_;
}

// ---------------------------------------------------------------------------
// cancel / runtimeStats
// ---------------------------------------------------------------------------

void SdModel::cancel() const {
  cancelRequested_.store(true);
}

qvac_lib_inference_addon_cpp::RuntimeStats SdModel::runtimeStats() const {
  return lastStats_;
}

// ---------------------------------------------------------------------------
// PNG encode / decode via stb
// ---------------------------------------------------------------------------

std::vector<uint8_t> SdModel::encodeToPng(const sd_image_t& img) {
  std::vector<uint8_t> out;

  auto writeCallback = [](void* ctx, void* data, int size) {
    auto* vec       = static_cast<std::vector<uint8_t>*>(ctx);
    const auto* src = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), src, src + size);
  };

  stbi_write_png_to_func(
      writeCallback, &out,
      static_cast<int>(img.width),
      static_cast<int>(img.height),
      static_cast<int>(img.channel),
      img.data,
      static_cast<int>(img.width * img.channel));

  return out;
}

sd_image_t SdModel::decodePng(const std::vector<uint8_t>& pngBytes) {
  if (pngBytes.empty()) return sd_image_t{};

  int w = 0, h = 0, c = 0;
  uint8_t* data = stbi_load_from_memory(
      pngBytes.data(), static_cast<int>(pngBytes.size()),
      &w, &h, &c, 3);

  if (!data) return sd_image_t{};

  return sd_image_t{
      static_cast<uint32_t>(w),
      static_cast<uint32_t>(h),
      3,
      data};
}

// ---------------------------------------------------------------------------
// Enum parsers
// ---------------------------------------------------------------------------

sample_method_t SdModel::parseSampler(const std::string& name) {
  if (name == "euler_a")     return EULER_A_SAMPLE_METHOD;
  if (name == "euler")       return EULER_SAMPLE_METHOD;
  if (name == "heun")        return HEUN_SAMPLE_METHOD;
  if (name == "dpm2")        return DPM2_SAMPLE_METHOD;
  if (name == "dpm++_2m")    return DPMPP2M_SAMPLE_METHOD;
  if (name == "dpm++_2m_v2") return DPMPP2Mv2_SAMPLE_METHOD;
  if (name == "dpm++_2s_a")  return DPMPP2S_A_SAMPLE_METHOD;
  if (name == "lcm")         return LCM_SAMPLE_METHOD;
  return EULER_A_SAMPLE_METHOD;
}

scheduler_t SdModel::parseScheduler(const std::string& name) {
  if (name == "discrete")    return DISCRETE_SCHEDULER;
  if (name == "karras")      return KARRAS_SCHEDULER;
  if (name == "exponential") return EXPONENTIAL_SCHEDULER;
  if (name == "ays")         return AYS_SCHEDULER;
  if (name == "gits")        return GITS_SCHEDULER;
  if (name == "sgm_uniform") return SGM_UNIFORM_SCHEDULER;
  if (name == "simple")      return SIMPLE_SCHEDULER;
  if (name == "lcm")         return LCM_SCHEDULER;
  return DISCRETE_SCHEDULER;
}

sd_type_t SdModel::parseWeightType(const std::string& name) {
  if (name == "f32")  return SD_TYPE_F32;
  if (name == "f16")  return SD_TYPE_F16;
  if (name == "q4_0") return SD_TYPE_Q4_0;
  if (name == "q4_1") return SD_TYPE_Q4_1;
  if (name == "q5_0") return SD_TYPE_Q5_0;
  if (name == "q5_1") return SD_TYPE_Q5_1;
  if (name == "q8_0") return SD_TYPE_Q8_0;
  return SD_TYPE_COUNT; // auto / default
}

// ---------------------------------------------------------------------------
// Log callback
// ---------------------------------------------------------------------------

void SdModel::sdLogCallback(
    sd_log_level_t level, const char* text, void* /*userData*/) {
  namespace lg = qvac_lib_inference_addon_cpp::logger;

  lg::Priority priority;
  switch (level) {
  case SD_LOG_DEBUG: priority = lg::Priority::DEBUG;   break;
  case SD_LOG_INFO:  priority = lg::Priority::INFO;    break;
  case SD_LOG_WARN:  priority = lg::Priority::WARNING; break;
  case SD_LOG_ERROR: priority = lg::Priority::ERROR;   break;
  default:           priority = lg::Priority::ERROR;   break;
  }

  QLOG_IF(priority, std::string(text));
}
