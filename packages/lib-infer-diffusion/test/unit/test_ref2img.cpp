#include <any>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "handlers/SdCtxHandlers.hpp"
#include "model-interface/SdModel.hpp"
#include "test_common.hpp"

using namespace qvac_lib_inference_addon_sd;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace ref2img_helpers {

inline std::string modelsDir() {
#ifdef PROJECT_ROOT
  return std::string(PROJECT_ROOT) + "/models";
#else
  return "models";
#endif
}

inline std::string tempDir() {
#ifdef PROJECT_ROOT
  return std::string(PROJECT_ROOT) + "/temp";
#else
  return "temp";
#endif
}

inline std::string headshotPath() {
  return tempDir() + "/nik_headshot_832.jpeg";
}

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

void writeFile(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(
      reinterpret_cast<const char*>(data.data()),
      static_cast<std::streamsize>(data.size()));
}

std::pair<int, int> decodeDimensions(const std::vector<uint8_t>& bytes) {
  int w = 0, h = 0, c = 0;
  uint8_t* px = stbi_load_from_memory(
      bytes.data(), static_cast<int>(bytes.size()), &w, &h, &c, 0);
  if (px)
    stbi_image_free(px);
  return {w, h};
}

std::string bytesToJsonArray(const std::vector<uint8_t>& bytes) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i)
      oss << ",";
    oss << static_cast<int>(bytes[i]);
  }
  oss << "]";
  return oss.str();
}

// Build paramsJson for an img2img job (in-context conditioning).
// The image goes through ref_image_bytes → FLUX reference token path.
// cfg_scale=1.0 disables external CFG (distilled model handles guidance internally).
std::string makeRef2ImgParams(
    const std::vector<uint8_t>& refBytes,
    const std::string& prompt,
    const std::string& negPrompt = "blurry, low quality, distorted",
    int steps = 20,
    float guidance = 9.0f,
    int64_t seed = 42,
    int w = 0,
    int h = 0) {
  std::ostringstream oss;
  oss << R"({"mode":"img2img","prompt":")" << prompt << R"(",)"
      << R"("negative_prompt":")" << negPrompt << R"(",)"
      << R"("steps":)" << steps << R"(,)"
      << R"("cfg_scale":1.0,)"
      << R"("guidance":)" << guidance << R"(,)"
      << R"("seed":)" << seed;
  if (w > 0)
    oss << R"(,"width":)" << w;
  if (h > 0)
    oss << R"(,"height":)" << h;
  oss << R"(,"ref_image_bytes":)" << bytesToJsonArray(refBytes) << "}";
  return oss.str();
}

// Build paramsJson for a traditional img2img job (noise-based).
// For comparison with ref2img.
// cfg_scale=1.0 disables external CFG (distilled model handles guidance internally).
std::string makeImg2ImgParams(
    const std::vector<uint8_t>& initBytes,
    const std::string& prompt,
    const std::string& negPrompt = "blurry, low quality, distorted",
    int steps = 20,
    float strength = 1.0f,
    float guidance = 9.0f,
    int64_t seed = 42) {
  std::ostringstream oss;
  oss << R"({"mode":"img2img","prompt":")" << prompt << R"(",)"
      << R"("negative_prompt":")" << negPrompt << R"(",)"
      << R"("steps":)" << steps << R"(,)"
      << R"("strength":)" << strength << R"(,)"
      << R"("cfg_scale":1.0,)"
      << R"("guidance":)" << guidance << R"(,)"
      << R"("seed":)" << seed
      << R"(,"init_image_bytes":)" << bytesToJsonArray(initBytes) << "}";
  return oss.str();
}

} // namespace ref2img_helpers

// ── Fixture ──────────────────────────────────────────────────────────────────

class SdRef2ImgTest : public ::testing::Test {
protected:
  static std::unique_ptr<SdModel> model;

  static void SetUpTestSuite() {
    const auto dir = ref2img_helpers::modelsDir();
    const std::string diffModel = dir + "/flux-2-klein-4b-Q8_0.gguf";
    const std::string llmModel = dir + "/Qwen3-4B-Q4_K_M.gguf";
    const std::string vaeModel = dir + "/flux2-vae.safetensors";

    if (!std::filesystem::exists(diffModel) ||
        !std::filesystem::exists(llmModel) ||
        !std::filesystem::exists(vaeModel)) {
      std::cout << "[SKIP] FLUX2 models not found in: " << dir << "\n"
                << "       Run ./scripts/download-model-i2i.sh first.\n";
      return;
    }

    SdCtxConfig cfg{};
    cfg.diffusionModelPath = diffModel;
    cfg.llmPath = llmModel;
    cfg.vaePath = vaeModel;
    cfg.prediction = FLUX2_FLOW_PRED;
    cfg.nThreads = sd_test_helpers::getTestThreads();
    cfg.device = sd_test_helpers::getTestDevice();

    std::cout << "\n[SdRef2ImgTest] Loading FLUX2-klein...\n"
              << "  diffusion : " << diffModel << "\n"
              << "  llm       : " << llmModel << "\n"
              << "  vae       : " << vaeModel << "\n"
              << "  device    : " << cfg.device << "\n"
              << "  threads   : " << cfg.nThreads << "\n";

    model = std::make_unique<SdModel>(std::move(cfg));
    model->load();
    std::cout << "[SdRef2ImgTest] Model loaded.\n";
  }

  static void TearDownTestSuite() {
    if (model) {
      model->unload();
      model.reset();
    }
  }

  void SetUp() override {
    if (!model)
      GTEST_SKIP() << "FLUX2 models not available — run download-model-i2i.sh";
  }
};

std::unique_ptr<SdModel> SdRef2ImgTest::model = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// TEST: ref2img with the real headshot — the core test for debugging bias.
//
// This bypasses the entire JS layer and calls SdModel::process() directly
// with mode="ref2img", routing the image through ref_image_bytes →
// genParams.ref_images (FLUX in-context conditioning).
//
// The output is saved so you can visually compare with:
//   - Iris output:         temp/nik_headshot_832_transformed_iris.png
//   - JS img2img output:   temp/nik_headshot_832_transformed.jpeg
//   - JS ref2img output:   temp/nik_headshot_832_ref2img.png
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SdRef2ImgTest, Ref2Img_Headshot_InContextConditioning) {
  const auto imgPath = ref2img_helpers::headshotPath();
  if (!std::filesystem::exists(imgPath))
    GTEST_SKIP() << "Headshot not found at: " << imgPath;

  const auto refBytes = ref2img_helpers::readFile(imgPath);
  ASSERT_GT(refBytes.size(), 0u) << "Headshot file is empty";

  const auto [dw, dh] = ref2img_helpers::decodeDimensions(refBytes);
  std::cout << "\n[ref2img] Reference image: " << imgPath << "\n"
            << "[ref2img] Dimensions: " << dw << "x" << dh << "\n"
            << "[ref2img] File size: " << refBytes.size() << " bytes\n"
            << "[ref2img] Mode: ref2img (in-context conditioning)\n"
            << "[ref2img] Steps: 20, Guidance: 9.0, Seed: 42\n"
            << "[ref2img] Prompt: 'a female version of this photo, professional "
               "headshot, corporate lawyer'\n\n";

  std::vector<std::vector<uint8_t>> images;
  std::mutex mu;

  SdModel::GenerationJob job;
  job.paramsJson = ref2img_helpers::makeRef2ImgParams(
      refBytes,
      "a female version of this photo, professional headshot, corporate lawyer",
      "blurry, low quality, distorted",
      /*steps=*/20,
      /*guidance=*/9.0f,
      /*seed=*/42);

  auto t0 = std::chrono::steady_clock::now();

  job.progressCallback = [&t0](const std::string& json) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    std::cout << "\r  [" << (elapsed / 1000.0) << "s] progress: " << json
              << "          " << std::flush;
  };

  job.outputCallback = [&](const std::vector<uint8_t>& png) {
    std::lock_guard<std::mutex> lk(mu);
    images.push_back(png);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    std::cout << "\n[ref2img] Output image: " << png.size() << " bytes"
              << " (took " << (elapsed / 1000.0) << "s)\n";

    const std::string outPath =
        ref2img_helpers::tempDir() + "/cpp-ref2img-headshot-output.png";
    ref2img_helpers::writeFile(outPath, png);
    std::cout << "[ref2img] Saved → " << outPath << "\n";
  };

  std::cout << "[ref2img] Starting generation...\n";
  EXPECT_NO_THROW(model->process(std::any(job)));
  EXPECT_EQ(images.size(), 1u) << "Expected 1 output image";
  if (!images.empty())
    EXPECT_TRUE(sd_test_helpers::isPng(images[0]))
        << "Output must be valid PNG";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST: traditional img2img with same settings — for direct A/B comparison.
//
// Uses mode="img2img" with strength=1.0 so you can compare the traditional
// noise-based approach against ref2img on the same image/prompt/seed.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SdRef2ImgTest, Img2Img_Headshot_TraditionalNoiseBased) {
  const auto imgPath = ref2img_helpers::headshotPath();
  if (!std::filesystem::exists(imgPath))
    GTEST_SKIP() << "Headshot not found at: " << imgPath;

  const auto initBytes = ref2img_helpers::readFile(imgPath);
  ASSERT_GT(initBytes.size(), 0u);

  const auto [dw, dh] = ref2img_helpers::decodeDimensions(initBytes);
  std::cout << "\n[img2img] Init image: " << imgPath << "\n"
            << "[img2img] Dimensions: " << dw << "x" << dh << "\n"
            << "[img2img] Mode: img2img (traditional noise-based)\n"
            << "[img2img] Steps: 20, Strength: 1.0, Guidance: 9.0, Seed: 42\n"
            << "[img2img] Prompt: 'a female version of this photo, professional "
               "headshot, corporate lawyer'\n\n";

  std::vector<std::vector<uint8_t>> images;
  std::mutex mu;

  SdModel::GenerationJob job;
  job.paramsJson = ref2img_helpers::makeImg2ImgParams(
      initBytes,
      "a female version of this photo, professional headshot, corporate lawyer",
      "blurry, low quality, distorted",
      /*steps=*/20,
      /*strength=*/1.0f,
      /*guidance=*/9.0f,
      /*seed=*/42);

  auto t0 = std::chrono::steady_clock::now();

  job.progressCallback = [&t0](const std::string& json) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    std::cout << "\r  [" << (elapsed / 1000.0) << "s] progress: " << json
              << "          " << std::flush;
  };

  job.outputCallback = [&](const std::vector<uint8_t>& png) {
    std::lock_guard<std::mutex> lk(mu);
    images.push_back(png);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    std::cout << "\n[img2img] Output image: " << png.size() << " bytes"
              << " (took " << (elapsed / 1000.0) << "s)\n";

    const std::string outPath =
        ref2img_helpers::tempDir() + "/cpp-img2img-headshot-output.png";
    ref2img_helpers::writeFile(outPath, png);
    std::cout << "[img2img] Saved → " << outPath << "\n";
  };

  std::cout << "[img2img] Starting generation...\n";
  EXPECT_NO_THROW(model->process(std::any(job)));
  EXPECT_EQ(images.size(), 1u) << "Expected 1 output image";
  if (!images.empty())
    EXPECT_TRUE(sd_test_helpers::isPng(images[0]))
        << "Output must be valid PNG";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST: txt2img with same prompt/seed — baseline without any image input.
//
// This shows what the model generates with NO image input at all (pure prompt).
// Useful for isolating whether the bias comes from the model itself or from
// how the reference image is processed.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SdRef2ImgTest, Txt2Img_Baseline_NoImageInput) {
  std::cout << "\n[txt2img] Mode: txt2img (no image input — pure prompt)\n"
            << "[txt2img] Steps: 20, Guidance: 9.0, Seed: 42\n"
            << "[txt2img] Prompt: 'a female version of a white male, "
               "professional headshot, corporate lawyer'\n\n";

  std::vector<std::vector<uint8_t>> images;
  std::mutex mu;

  SdModel::GenerationJob job;
  std::ostringstream oss;
  oss << R"({"mode":"txt2img",)"
      << R"("prompt":"a female professional headshot, corporate lawyer, white woman, caucasian",)"
      << R"("negative_prompt":"blurry, low quality, distorted",)"
      << R"("steps":20,"cfg_scale":1.0,"guidance":9.0,"seed":42,)"
      << R"("width":832,"height":832})";
  job.paramsJson = oss.str();

  auto t0 = std::chrono::steady_clock::now();

  job.progressCallback = [&t0](const std::string& json) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    std::cout << "\r  [" << (elapsed / 1000.0) << "s] progress: " << json
              << "          " << std::flush;
  };

  job.outputCallback = [&](const std::vector<uint8_t>& png) {
    std::lock_guard<std::mutex> lk(mu);
    images.push_back(png);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    std::cout << "\n[txt2img] Output image: " << png.size() << " bytes"
              << " (took " << (elapsed / 1000.0) << "s)\n";

    const std::string outPath =
        ref2img_helpers::tempDir() + "/cpp-txt2img-baseline-output.png";
    ref2img_helpers::writeFile(outPath, png);
    std::cout << "[txt2img] Saved → " << outPath << "\n";
  };

  std::cout << "[txt2img] Starting generation...\n";
  EXPECT_NO_THROW(model->process(std::any(job)));
  EXPECT_EQ(images.size(), 1u) << "Expected 1 output image";
  if (!images.empty())
    EXPECT_TRUE(sd_test_helpers::isPng(images[0]))
        << "Output must be valid PNG";
}
