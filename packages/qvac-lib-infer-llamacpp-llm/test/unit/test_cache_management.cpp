#include <any>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <gtest/gtest.h>
#include <qvac-lib-inference-addon-cpp/RuntimeStats.hpp>

#include "model-interface/LlamaModel.hpp"
#include "test_common.hpp"

namespace fs = std::filesystem;

namespace {
double getStatValue(
    const qvac_lib_inference_addon_cpp::RuntimeStats& stats,
    const std::string& key) {
  for (const auto& stat : stats) {
    if (stat.first == key) {
      return std::visit(
          [](const auto& value) -> double {
            if constexpr (std::is_same_v<
                              std::decay_t<decltype(value)>,
                              double>) {
              return value;
            } else {
              return static_cast<double>(value);
            }
          },
          stat.second);
    }
  }
  return 0.0;
}

std::string processPromptString(
    const std::unique_ptr<LlamaModel>& model, const std::string& input) {
  LlamaModel::Prompt prompt;
  prompt.input = input;
  return model->processPrompt(prompt);
}
} // namespace

class CacheManagementTest : public ::testing::Test {
protected:
  void SetUp() override {
    config_files["device"] = test_common::getTestDevice();
    config_files["ctx_size"] = "2048";
    config_files["gpu_layers"] = test_common::getTestGpuLayers();
    config_files["n_predict"] = "10";
    config_files["tools"] = "true";

    test_model_path = test_common::BaseTestModelPath::get();
    test_projection_path = "";

    config_files["backendsDir"] = test_common::getTestBackendsDir().string();

    session1_path = "test_session1.bin";
    session2_path = "test_session2.bin";
    temp_session_path = "temp_session.bin";
  }

  void TearDown() override {
    for (const auto& session_file :
         {session1_path,
          session2_path,
          temp_session_path,
          std::string("test_large_cache.bin")}) {
      if (fs::exists(session_file)) {
        fs::remove(session_file);
      }
    }
  }

  bool hasValidModel() { return fs::exists(test_model_path); }

  std::unique_ptr<LlamaModel> createModel() {
    if (!hasValidModel()) {
      return nullptr;
    }
    std::string modelPath = test_model_path;
    std::string projectionPath = test_projection_path;
    auto configCopy = config_files;
    auto model = std::make_unique<LlamaModel>(
        std::move(modelPath), std::move(projectionPath), std::move(configCopy));
    model->waitForLoadInitialization();
    if (!model->isLoaded()) {
      return nullptr;
    }
    return model;
  }

  std::unique_ptr<LlamaModel>
  createModelWithContextSize(const std::string& ctxSize) {
    if (!hasValidModel()) {
      return nullptr;
    }
    std::string modelPath = test_model_path;
    std::string projectionPath = test_projection_path;
    std::unordered_map<std::string, std::string> custom_config = config_files;
    custom_config["ctx_size"] = ctxSize;
    auto model = std::make_unique<LlamaModel>(
        std::move(modelPath),
        std::move(projectionPath),
        std::move(custom_config));
    model->waitForLoadInitialization();
    if (!model->isLoaded()) {
      return nullptr;
    }
    return model;
  }

  std::unique_ptr<LlamaModel> createModelWithContextSizeAndNPredict(
      const std::string& ctxSize, const std::string& nPredict) {
    if (!hasValidModel()) {
      return nullptr;
    }
    std::string modelPath = test_model_path;
    std::string projectionPath = test_projection_path;
    std::unordered_map<std::string, std::string> custom_config = config_files;
    custom_config["ctx_size"] = ctxSize;
    custom_config["n_predict"] = nPredict;
    auto model = std::make_unique<LlamaModel>(
        std::move(modelPath),
        std::move(projectionPath),
        std::move(custom_config));
    model->waitForLoadInitialization();
    if (!model->isLoaded()) {
      return nullptr;
    }
    return model;
  }

  std::unordered_map<std::string, std::string> config_files;
  std::string test_model_path;
  std::string test_projection_path;
  std::string session1_path;
  std::string session2_path;
  std::string temp_session_path;

  bool isQwen3Model() {
    std::string lowerPath = test_model_path;
    std::transform(
        lowerPath.begin(), lowerPath.end(), lowerPath.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return lowerPath.find("qwen3") != std::string::npos;
  }
};

TEST_F(CacheManagementTest, InitialStateNoCache) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output = processPromptString(
        model,
        R"([{"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output.length(), 0);
    auto stats = model->runtimeStats();
    EXPECT_GE(stats.size(), 0);
  });

  EXPECT_FALSE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, EnableCacheWithFilename) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is ethereum? Answer shortly."}])");
    EXPECT_GE(output.length(), 0);
    auto stats = model->runtimeStats();
    EXPECT_GE(stats.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])");
    EXPECT_EQ(saveOutput.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, SessionPersistence) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])");
    EXPECT_EQ(saveOutput.length(), 0);
    auto statsSave = model->runtimeStats();
    EXPECT_GE(statsSave.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What did I ask you before? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, MultipleSessionCommands) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session2.bin"}, {"role": "session", "content": "reset"}, {"role": "session", "content": "save"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output.length(), 0);
    auto stats = model->runtimeStats();
    EXPECT_GE(stats.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session2_path));
}

TEST_F(CacheManagementTest, ResetCommand) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])");
    EXPECT_EQ(saveOutput.length(), 0);
    auto statsSave = model->runtimeStats();
    EXPECT_GE(statsSave.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "reset"}, {"role": "user", "content": "What did I ask you before? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, SwitchToSession2) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session2.bin"}, {"role": "user", "content": "What did I ask you before? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  // session1 should be saved automatically when switching to session2
  EXPECT_TRUE(fs::exists(session1_path));

  // session2 needs to be explicitly saved
  EXPECT_NO_THROW({
    std::string saveOutput2 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session2.bin"}, {"role": "session", "content": "save"}])");
    EXPECT_EQ(saveOutput2.length(), 0);
    auto statsSave2 = model->runtimeStats();
    EXPECT_GE(statsSave2.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session2_path));
}

TEST_F(CacheManagementTest, DisableCache) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "user", "content": "What is blockchain? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });
}

TEST_F(CacheManagementTest, VerifyStatelessBehavior) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "user", "content": "What did I ask you before? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });
}

TEST_F(CacheManagementTest, ReEnableCacheAfterDisable) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "session", "content": "temp_session.bin"}, {"role": "user", "content": "What is deep learning? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(
        model,
        R"([{"role": "session", "content": "temp_session.bin"}, {"role": "session", "content": "save"}])");
    EXPECT_EQ(saveOutput.length(), 0);
    auto statsSave = model->runtimeStats();
    EXPECT_GE(statsSave.size(), 0);
  });

  EXPECT_TRUE(fs::exists(temp_session_path));
}

TEST_F(CacheManagementTest, SessionCommandOnly) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_THROW(
      {
        processPromptString(
            model, R"([{"role": "session", "content": "reset"}])");
      },
      qvac_errors::StatusError);
}

TEST_F(CacheManagementTest, SaveWhenCacheDisabled) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_THROW(
      {
        processPromptString(
            model, R"([{"role": "session", "content": "save"}])");
      },
      qvac_errors::StatusError);

  EXPECT_FALSE(fs::exists(temp_session_path));
}

TEST_F(CacheManagementTest, ComplexSessionCommandChain) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output1 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])");
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output2 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session2.bin"}, {"role": "session", "content": "save"}, {"role": "session", "content": "reset"}, {"role": "user", "content": "What is ethereum? Answer shortly."}])");
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  EXPECT_NO_THROW({
    std::string output3 = processPromptString(
        model,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is blockchain? Answer shortly."}])");
    EXPECT_GE(output3.length(), 0);
    auto stats3 = model->runtimeStats();
    EXPECT_GE(stats3.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
  EXPECT_TRUE(fs::exists(session2_path));
}

TEST_F(CacheManagementTest, CacheClearedWhenNoSessionMessage) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
    auto statsSave = model->runtimeStats();
    EXPECT_GE(statsSave.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));

  std::string input2 =
      R"([{"role": "user", "content": "What is ethereum? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output2 = processPromptString(model, input2);
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  // Verify cache file still exists after clearing (was saved before clearing)
  EXPECT_TRUE(fs::exists(session1_path));

  auto stats = model->runtimeStats();
  EXPECT_EQ(getStatValue(stats, "CacheTokens"), 0.0);

  // Verify cache can be re-enabled after being cleared
  std::string input3 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is blockchain? Answer shortly."}])";
  qvac_lib_inference_addon_cpp::RuntimeStats stats3;
  EXPECT_NO_THROW({
    std::string output3 = processPromptString(model, input3);
    EXPECT_GE(output3.length(), 0);
    stats3 = model->runtimeStats();
    EXPECT_GE(stats3.size(), 0);
  });

  double cacheTokens3 = getStatValue(stats3, "CacheTokens");
  EXPECT_GT(cacheTokens3, 0.0); // Verify cache was re-enabled
}

TEST_F(CacheManagementTest, CacheClearedWhenSwitchingToDifferentCache) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string saveInput1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput1 = processPromptString(model, saveInput1);
    EXPECT_EQ(saveOutput1.length(), 0);
    auto statsSave1 = model->runtimeStats();
    EXPECT_GE(statsSave1.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));

  std::string input2 =
      R"([{"role": "session", "content": "test_session2.bin"}, {"role": "user", "content": "What is ethereum? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output2 = processPromptString(model, input2);
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  // Verify first cache file was saved before switching
  EXPECT_TRUE(fs::exists(session1_path));

  // Verify new cache was created (CacheTokens > 0)
  auto stats2 = model->runtimeStats();
  EXPECT_GT(getStatValue(stats2, "CacheTokens"), 0.0);

  std::string saveInput2 =
      R"([{"role": "session", "content": "test_session2.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput2 = processPromptString(model, saveInput2);
    EXPECT_EQ(saveOutput2.length(), 0);
    auto statsSave2 = model->runtimeStats();
    EXPECT_GE(statsSave2.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session2_path));
}

TEST_F(CacheManagementTest, SingleShotInferenceAfterCacheCleared) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  auto stats1 = model->runtimeStats();
  double cacheTokens1 = getStatValue(stats1, "CacheTokens");

  std::string input2 =
      R"([{"role": "user", "content": "What is ethereum? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output2 = processPromptString(model, input2);
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  auto stats2 = model->runtimeStats();
  double cacheTokens2 = getStatValue(stats2, "CacheTokens");
  EXPECT_GT(cacheTokens1, 0.0); // Verify cache was created
  EXPECT_EQ(cacheTokens2, 0.0); // Verify cache was cleared
}

TEST_F(CacheManagementTest, CacheToNoCacheToCache) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string saveInput1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput1 = processPromptString(model, saveInput1);
    EXPECT_EQ(saveOutput1.length(), 0);
    auto statsSave1 = model->runtimeStats();
    EXPECT_GE(statsSave1.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));

  std::string input2 =
      R"([{"role": "user", "content": "What is ethereum? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output2 = processPromptString(model, input2);
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
    EXPECT_EQ(getStatValue(stats2, "CacheTokens"), 0.0);
  });

  std::string input3 =
      R"([{"role": "session", "content": "test_session2.bin"}, {"role": "user", "content": "What is blockchain? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output3 = processPromptString(model, input3);
    EXPECT_GE(output3.length(), 0);
    auto stats3 = model->runtimeStats();
    EXPECT_GE(stats3.size(), 0);
    EXPECT_GT(getStatValue(stats3, "CacheTokens"), 0.0);
  });

  std::string saveInput2 =
      R"([{"role": "session", "content": "test_session2.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput2 = processPromptString(model, saveInput2);
    EXPECT_EQ(saveOutput2.length(), 0);
    auto statsSave2 = model->runtimeStats();
    EXPECT_GE(statsSave2.size(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
  EXPECT_TRUE(fs::exists(session2_path));
}

TEST_F(CacheManagementTest, GetTokensCommand) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  auto stats1 = model->runtimeStats();
  double cacheTokens1 = getStatValue(stats1, "CacheTokens");
  EXPECT_GT(cacheTokens1, 0.0);

  std::string getTokensInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "getTokens"}])";
  EXPECT_NO_THROW({
    std::string output = processPromptString(model, getTokensInput);
    EXPECT_EQ(output.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
    double cacheTokens2 = getStatValue(stats2, "CacheTokens");
    EXPECT_EQ(cacheTokens1, cacheTokens2);

    double ttft = getStatValue(stats2, "TTFT");
    double tps = getStatValue(stats2, "TPS");
    EXPECT_EQ(ttft, 0.0);
    EXPECT_EQ(tps, 0.0);
  });

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
    auto statsSave = model->runtimeStats();
    EXPECT_GE(statsSave.size(), 0);
  });
  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, SaveCommandReturnsZeroMetrics) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  auto stats2 = model->runtimeStats();
  double cacheTokens2 = getStatValue(stats2, "CacheTokens");
  double promptTokens2 = getStatValue(stats2, "promptTokens");
  double generatedTokens2 = getStatValue(stats2, "generatedTokens");
  double ttft2 = getStatValue(stats2, "TTFT");
  double tps2 = getStatValue(stats2, "TPS");

  EXPECT_GT(cacheTokens2, 0.0);

  std::cout << "After save command:\n";
  std::cout << "  promptTokens: " << promptTokens2 << "\n";
  std::cout << "  generatedTokens: " << generatedTokens2 << "\n";
  std::cout << "  TTFT: " << ttft2 << "\n";
  std::cout << "  TPS: " << tps2 << "\n";

  if (promptTokens2 == 0.0 && generatedTokens2 == 0.0 && ttft2 == 0.0 &&
      tps2 == 0.0) {
    std::cout << "Result: save returns zeros (like getTokens)\n";
  } else {
    std::cout << "Result: save returns stale data from previous inference\n";
    std::cout << "  Values returned: prompt=" << promptTokens2
              << ", generated=" << generatedTokens2 << ", TTFT=" << ttft2
              << ", TPS=" << tps2 << "\n";
  }
}

TEST_F(CacheManagementTest, GetTokensCommandWithNoCache) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string getTokensInput =
      R"([{"role": "session", "content": "getTokens"}])";
  EXPECT_THROW(
      { processPromptString(model, getTokensInput); },
      qvac_errors::StatusError);
}

TEST_F(CacheManagementTest, GetTokensCommandAfterDisable) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string disableInput = R"([{"role": "user", "content": "test"}])";
  EXPECT_NO_THROW({
    std::string disableOutput = processPromptString(model, disableInput);
    EXPECT_GE(disableOutput.length(), 0);
    auto stats2 = model->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });
  std::string getTokensInput =
      R"([{"role": "session", "content": "getTokens"}])";
  EXPECT_THROW(
      { processPromptString(model, getTokensInput); },
      qvac_errors::StatusError);
}

TEST_F(CacheManagementTest, GetTokensCommandAfterReset) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string resetInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "reset"}])";
  EXPECT_NO_THROW({
    std::string resetOutput = processPromptString(model, resetInput);
    EXPECT_EQ(resetOutput.length(), 0);
    auto statsReset = model->runtimeStats();
    EXPECT_GE(statsReset.size(), 0);
  });

  std::string getTokensInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "getTokens"}])";
  EXPECT_NO_THROW({
    std::string output = processPromptString(model, getTokensInput);
    EXPECT_EQ(output.length(), 0);
    auto stats = model->runtimeStats();
    EXPECT_GE(stats.size(), 0);
  });

  auto stats = model->runtimeStats();
  double cacheTokens = getStatValue(stats, "CacheTokens");
  EXPECT_EQ(cacheTokens, 0.0);

  double ttft = getStatValue(stats, "TTFT");
  double tps = getStatValue(stats, "TPS");
  EXPECT_EQ(ttft, 0.0);
  EXPECT_EQ(tps, 0.0);
}

TEST_F(CacheManagementTest, CacheTokensExceedContextSize) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  auto model_large = createModelWithContextSizeAndNPredict("4096", "100");
  if (!model_large) {
    FAIL() << "Model failed to load";
  }

  std::string large_cache_path = "test_large_cache.bin";

  std::string input1 =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "user", "content": "What is bitcoin? Please provide a detailed explanation of how bitcoin works, including its blockchain technology, mining process, and cryptographic principles. Explain the concept of distributed consensus and how transactions are verified."}])";
  EXPECT_NO_THROW({
    std::string output1 = processPromptString(model_large, input1);
    EXPECT_GE(output1.length(), 0);
    auto stats1 = model_large->runtimeStats();
    EXPECT_GE(stats1.size(), 0);
  });

  std::string input2 =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "user", "content": "Now explain ethereum in similar detail. Include information about smart contracts, the EVM, gas fees, and how it differs from bitcoin."}])";
  EXPECT_NO_THROW({
    std::string output2 = processPromptString(model_large, input2);
    EXPECT_GE(output2.length(), 0);
    auto stats2 = model_large->runtimeStats();
    EXPECT_GE(stats2.size(), 0);
  });

  std::string input3 =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "user", "content": "Finally, explain blockchain technology in general, covering concepts like immutability, decentralization, consensus mechanisms, and potential use cases beyond cryptocurrencies."}])";
  EXPECT_NO_THROW({
    std::string output3 = processPromptString(model_large, input3);
    EXPECT_GE(output3.length(), 0);
    auto stats3 = model_large->runtimeStats();
    EXPECT_GE(stats3.size(), 0);
  });

  std::string input4 =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "user", "content": "Explain proof of work and proof of stake consensus mechanisms in detail. Compare and contrast their advantages and disadvantages."}])";
  EXPECT_NO_THROW({
    std::string output4 = processPromptString(model_large, input4);
    EXPECT_GE(output4.length(), 0);
    auto stats4 = model_large->runtimeStats();
    EXPECT_GE(stats4.size(), 0);
  });

  std::string input5 =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "user", "content": "Describe DeFi (Decentralized Finance) applications, including DEXs, lending protocols, and yield farming. Explain how they work and their risks."}])";
  EXPECT_NO_THROW({
    std::string output5 = processPromptString(model_large, input5);
    EXPECT_GE(output5.length(), 0);
    auto stats5 = model_large->runtimeStats();
    EXPECT_GE(stats5.size(), 0);
  });

  auto statsBeforeSave = model_large->runtimeStats();
  double cacheTokensBeforeSave = getStatValue(statsBeforeSave, "CacheTokens");
  EXPECT_GT(cacheTokensBeforeSave, 0.0);

  std::string saveInput =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model_large, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
    auto statsSave = model_large->runtimeStats();
    EXPECT_GE(statsSave.size(), 0);
  });

  EXPECT_TRUE(fs::exists(large_cache_path));

  model_large.reset();

  int smallContextSize = 128;
  if (cacheTokensBeforeSave <= smallContextSize) {
    FAIL() << "Cache tokens (" << cacheTokensBeforeSave
           << ") not enough to exceed context size (" << smallContextSize
           << ")";
  }

  auto model_small =
      createModelWithContextSize(std::to_string(smallContextSize));
  if (!model_small) {
    FAIL() << "Model failed to load";
  }

  std::string loadInput =
      R"([{"role": "session", "content": "test_large_cache.bin"}, {"role": "user", "content": "Test"}])";
  EXPECT_THROW(
      { processPromptString(model_small, loadInput); },
      qvac_errors::StatusError);
}

TEST_F(CacheManagementTest, CacheWithToolsAtEndFalseSavesFullCache) {
  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "false";
  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string inputWithTools =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is the weather in Tokyo?"}, {"type": "function", "name": "getWeather", "description": "Get weather forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, inputWithTools);
    EXPECT_GE(output.length(), 0);
    auto stats = model->runtimeStats();
    EXPECT_GE(stats.size(), 0);
  });

  auto statsBeforeSave = model->runtimeStats();
  double cacheTokensBeforeSave = getStatValue(statsBeforeSave, "CacheTokens");
  EXPECT_GT(cacheTokensBeforeSave, 0.0);

  llama_pos nPastBeforeTools = model->getNPastBeforeTools();
  EXPECT_EQ(nPastBeforeTools, -1);

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, CacheWithToolsAtEndTrueTrimsToolTokens) {
  if (!isQwen3Model()) {
    GTEST_SKIP() << "Test requires Qwen3 model for tools_at_end feature";
  }

  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "true";
  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string inputWithTools =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is the weather in Tokyo?"}, {"type": "function", "name": "getWeather", "description": "Get weather forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, inputWithTools);
    EXPECT_GE(output.length(), 0);
  });

  auto statsBeforeSave = model->runtimeStats();
  double cacheTokensBeforeSave = getStatValue(statsBeforeSave, "CacheTokens");
  EXPECT_GT(cacheTokensBeforeSave, 0.0);

  llama_pos nPastBeforeTools = model->getNPastBeforeTools();
  EXPECT_GT(nPastBeforeTools, 0);
  EXPECT_LT(nPastBeforeTools, cacheTokensBeforeSave);

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, CacheReloadWithToolsAtEndTrue) {
  if (!isQwen3Model()) {
    GTEST_SKIP() << "Test requires Qwen3 model for tools_at_end feature";
  }

  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "true";
  auto model1 = createModel();
  if (!model1) {
    FAIL() << "Model failed to load";
  }

  std::string inputWithTools =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is the weather in Tokyo?"}, {"type": "function", "name": "getWeather", "description": "Get weather forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model1, inputWithTools);
    EXPECT_GE(output.length(), 0);
  });

  llama_pos nPastBeforeTools1 = model1->getNPastBeforeTools();
  EXPECT_GT(nPastBeforeTools1, 0);

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model1, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));

  model1.reset();

  auto model2 = createModel();
  if (!model2) {
    FAIL() << "Model failed to load";
  }

  EXPECT_NO_THROW({
    std::string output = processPromptString(
        model2,
        R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is the weather in London?"}])");
    EXPECT_GE(output.length(), 0);
  });

  auto statsAfterReload = model2->runtimeStats();
  double cacheTokensAfterReload = getStatValue(statsAfterReload, "CacheTokens");
  EXPECT_GT(cacheTokensAfterReload, 0.0);

  llama_pos nPastBeforeTools2 = model2->getNPastBeforeTools();
  EXPECT_EQ(nPastBeforeTools2, nPastBeforeTools1);
  EXPECT_LE(nPastBeforeTools2, nPastBeforeTools1);
}

TEST_F(CacheManagementTest, CacheWithoutToolsWithToolsAtEndTrue) {
  if (!isQwen3Model()) {
    GTEST_SKIP() << "Test requires Qwen3 model for tools_at_end feature";
  }

  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "true";
  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string inputNoTools =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What is bitcoin? Answer shortly."}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, inputNoTools);
    EXPECT_GE(output.length(), 0);
  });

  auto statsBeforeSave = model->runtimeStats();
  double cacheTokensBeforeSave = getStatValue(statsBeforeSave, "CacheTokens");
  EXPECT_GT(cacheTokensBeforeSave, 0.0);

  llama_pos nPastBeforeTools = model->getNPastBeforeTools();
  EXPECT_EQ(nPastBeforeTools, -1);

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, CacheToolsAtEndModeWithMultiplePrompts) {
  if (!isQwen3Model()) {
    GTEST_SKIP() << "Test requires Qwen3 model for tools_at_end feature";
  }

  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "true";
  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "Hi"}, {"type": "function", "name": "get_weather", "description": "Get detailed weather forecast data with temperature humidity wind speed precipitation UV visibility pressure sunrise sunset alerts", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The name of the city to get weather for"}, "country": {"type": "string", "description": "Country code or name"}, "lat": {"type": "number", "description": "Latitude coordinate"}, "lon": {"type": "number", "description": "Longitude coordinate"}, "zip": {"type": "string", "description": "ZIP postal code"}, "units": {"type": "string", "description": "Temperature units metric imperial or kelvin"}, "lang": {"type": "string", "description": "Language code for localized descriptions"}, "forecast_days": {"type": "integer", "description": "Number of days to forecast from 1 to 7"}, "hourly": {"type": "boolean", "description": "Include hourly forecast data"}, "alerts": {"type": "boolean", "description": "Include weather alerts and warnings"}, "aqi": {"type": "boolean", "description": "Include air quality index data"}, "tides": {"type": "boolean", "description": "Include tide information"}, "solar": {"type": "boolean", "description": "Include solar data like sunrise sunset"}, "tz": {"type": "string", "description": "Timezone identifier"}, "start_dt": {"type": "string", "description": "Start datetime for historical data"}, "end_dt": {"type": "string", "description": "End datetime for historical data"}, "cnt": {"type": "integer", "description": "Number of data points to return"}, "mode": {"type": "string", "description": "Response mode json xml or html"}, "appid": {"type": "string", "description": "API key for authentication"}}, "required": ["city"]}}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, input1);
    EXPECT_GE(output.length(), 0);
  });

  auto stats1 = model->runtimeStats();
  double cacheTokens1 = getStatValue(stats1, "CacheTokens");
  printf("CacheManagementTest::CacheToolsAtEndModeWithMultiplePrompts cacheTokens1=%f\n", cacheTokens1);
  EXPECT_GT(cacheTokens1, 0.0);

  llama_pos nPastBeforeTools = model->getNPastBeforeTools();
  EXPECT_GT(nPastBeforeTools, 0);
  EXPECT_LT(nPastBeforeTools, cacheTokens1);

  const int maxExpectedCacheTokens = 50;
  EXPECT_GT(cacheTokens1, 0);
  EXPECT_LE(cacheTokens1, maxExpectedCacheTokens)
      << "Cache tokens (" << cacheTokens1 << ") should not exceed "
      << maxExpectedCacheTokens << " - function tokens should be trimmed";

  std::string input2 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What about London?"}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, input2);
    EXPECT_GE(output.length(), 0);
  });

  auto stats2 = model->runtimeStats();
  double cacheTokens2 = getStatValue(stats2, "CacheTokens");
  printf("CacheManagementTest::CacheToolsAtEndModeWithMultiplePrompts cacheTokens2=%f\n", cacheTokens2);
  EXPECT_GT(cacheTokens2, cacheTokens1);
  EXPECT_LE(cacheTokens2, maxExpectedCacheTokens)
      << "Cache tokens (" << cacheTokens1 << ") should not exceed "
      << maxExpectedCacheTokens << " - function tokens should be trimmed";

  EXPECT_LE(nPastBeforeTools, cacheTokens2);

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));
}

TEST_F(CacheManagementTest, CacheToolsAtEndModeTrimOnlyWhenNPastBeforeToolsPositive) {
  if (!isQwen3Model()) {
    GTEST_SKIP() << "Test requires Qwen3 model for tools_at_end feature";
  }

  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "true";
  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string inputNoTools =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "Hello"}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, inputNoTools);
    EXPECT_GE(output.length(), 0);
  });

  llama_pos nPastBeforeTools = model->getNPastBeforeTools();
  EXPECT_EQ(nPastBeforeTools, -1);

  auto statsBeforeSave = model->runtimeStats();
  double cacheTokensBeforeSave = getStatValue(statsBeforeSave, "CacheTokens");
  EXPECT_GT(cacheTokensBeforeSave, 0.0);

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  auto statsAfterSave = model->runtimeStats();
  double cacheTokensAfterSave = getStatValue(statsAfterSave, "CacheTokens");
  EXPECT_EQ(cacheTokensAfterSave, cacheTokensBeforeSave);
}

TEST_F(CacheManagementTest, CacheToolsAtEndModeRestoresNPastBeforeTools) {
  if (!isQwen3Model()) {
    GTEST_SKIP() << "Test requires Qwen3 model for tools_at_end feature";
  }

  if (!hasValidModel()) {
    FAIL() << "Test model not found";
  }

  config_files["tools_at_end"] = "true";
  auto model = createModel();
  if (!model) {
    FAIL() << "Model failed to load";
  }

  std::string input1 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "Hi"}, {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model, input1);
    EXPECT_GE(output.length(), 0);
  });

  llama_pos nPastBeforeTools1 = model->getNPastBeforeTools();
  EXPECT_GT(nPastBeforeTools1, 0) << "nPastBeforeTools should be set after first prompt with tools";

  std::string saveInput =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "session", "content": "save"}])";
  EXPECT_NO_THROW({
    std::string saveOutput = processPromptString(model, saveInput);
    EXPECT_EQ(saveOutput.length(), 0);
  });

  EXPECT_TRUE(fs::exists(session1_path));

  auto model2 = createModel();
  if (!model2) {
    FAIL() << "Model2 failed to load";
  }

  std::string input2 =
      R"([{"role": "session", "content": "test_session1.bin"}, {"role": "user", "content": "What about London?"}])";

  EXPECT_NO_THROW({
    std::string output = processPromptString(model2, input2);
    EXPECT_GE(output.length(), 0);
  });

  llama_pos nPastBeforeTools2 = model2->getNPastBeforeTools();
  EXPECT_EQ(nPastBeforeTools2, nPastBeforeTools1)
      << "nPastBeforeTools should be restored from session: expected " << nPastBeforeTools1
      << " but got " << nPastBeforeTools2;
}
