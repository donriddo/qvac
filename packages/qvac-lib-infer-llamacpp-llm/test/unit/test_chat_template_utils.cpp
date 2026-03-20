#include <filesystem>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>
#include <llama.h>

#include "model-interface/LlamaModel.hpp"
#include "test_common.hpp"
#include "utils/ChatTemplateUtils.hpp"
#include "utils/Qwen3ToolsDynamicTemplate.hpp"
#include "utils/QwenTemplate.hpp"

namespace fs = std::filesystem;
using namespace qvac_lib_inference_addon_llama::utils;

class ChatTemplateUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    config_files["device"] = test_common::getTestDevice();
    config_files["ctx_size"] = "2048";
    config_files["gpu_layers"] = test_common::getTestGpuLayers();
    config_files["n_predict"] = "10";

    test_model_path = test_common::BaseTestModelPath::get();
    test_projection_path = "";

    config_files["backendsDir"] = test_common::getTestBackendsDir().string();
  }

  std::unordered_map<std::string, std::string> config_files;
  std::string test_model_path;
  std::string test_projection_path;

  bool hasValidModel() { return fs::exists(test_model_path); }
};

TEST_F(ChatTemplateUtilsTest, IsQwen3ModelWithNullptr) {
  EXPECT_FALSE(isQwen3Model(nullptr));
}

TEST_F(
    ChatTemplateUtilsTest,
    GetChatTemplateForModelWithManualOverrideToolsAtEndFalse) {
  std::string manual_override = "custom template";
  std::string result = getChatTemplateForModel(nullptr, manual_override, false);
  EXPECT_EQ(result, manual_override);
}

TEST_F(
    ChatTemplateUtilsTest,
    GetChatTemplateForModelWithManualOverrideToolsAtEndTrue) {
  std::string manual_override = "custom template";
  std::string result = getChatTemplateForModel(nullptr, manual_override, true);
  EXPECT_EQ(result, manual_override);
}

TEST_F(
    ChatTemplateUtilsTest,
    GetChatTemplateForModelEmptyOverrideNullptrToolsAtEndFalse) {
  std::string result = getChatTemplateForModel(nullptr, "", false);
  EXPECT_EQ(result, "");
}

TEST_F(
    ChatTemplateUtilsTest,
    GetChatTemplateForModelEmptyOverrideNullptrToolsAtEndTrue) {
  std::string result = getChatTemplateForModel(nullptr, "", true);
  EXPECT_EQ(result, "");
}

TEST_F(ChatTemplateUtilsTest, GetChatTemplateWithNullptrModel) {
  common_params params;
  params.chat_template = "test template";
  params.use_jinja = false;

  std::string result = getChatTemplate(nullptr, params, false);
  EXPECT_EQ(result, params.chat_template);
}

TEST_F(ChatTemplateUtilsTest, GetChatTemplateJinjaDisabled) {
  common_params params;
  params.chat_template = "test template";
  params.use_jinja = false;

  std::string result = getChatTemplate(nullptr, params, false);
  EXPECT_EQ(result, "test template");
}

TEST_F(ChatTemplateUtilsTest, GetChatTemplateJinjaEnabledWithOverride) {
  common_params params;
  params.chat_template = "custom template";
  params.use_jinja = true;

  std::string result = getChatTemplate(nullptr, params, false);
  EXPECT_EQ(result, "custom template");
}

TEST_F(ChatTemplateUtilsTest, GetChatTemplateJinjaEnabledWithoutOverride) {
  common_params params;
  params.chat_template = "";
  params.use_jinja = true;

  std::string result = getChatTemplate(nullptr, params, false);
  EXPECT_EQ(result, "");
}

TEST_F(ChatTemplateUtilsTest, GetChatTemplateParamsNotModified) {
  common_params params;
  params.chat_template = "original template";
  params.use_jinja = false;

  std::string result = getChatTemplate(nullptr, params, false);

  EXPECT_EQ(params.chat_template, "original template");
  EXPECT_FALSE(params.use_jinja);
  EXPECT_EQ(result, "original template");
}

TEST_F(ChatTemplateUtilsTest, GetChatTemplateForModelPreservesWhitespace) {
  std::string overrideWithSpaces = "  template with spaces  ";
  std::string result =
      getChatTemplateForModel(nullptr, overrideWithSpaces, false);
  EXPECT_EQ(result, overrideWithSpaces);
}

TEST_F(
    ChatTemplateUtilsTest, GetChatTemplateForModelPreservesSpecialCharacters) {
  std::string overrideSpecial = "template\nwith\tspecial\rchars";
  std::string result = getChatTemplateForModel(nullptr, overrideSpecial, false);
  EXPECT_EQ(result, overrideSpecial);
}

TEST_F(ChatTemplateUtilsTest, GetFixedQwen3TemplateNotNull) {
  const char* expectedTemplate = getFixedQwen3Template();
  ASSERT_NE(expectedTemplate, nullptr);
  EXPECT_GT(strlen(expectedTemplate), 0u);
}

TEST_F(ChatTemplateUtilsTest, GetToolsDynamicQwen3TemplateNotNull) {
  const char* expectedTemplate = getToolsDynamicQwen3Template();
  ASSERT_NE(expectedTemplate, nullptr);
  EXPECT_GT(strlen(expectedTemplate), 0u);
}

TEST_F(ChatTemplateUtilsTest, TemplatesAreDifferent) {
  const char* fixedTemplate = getFixedQwen3Template();
  const char* dynamicTemplate = getToolsDynamicQwen3Template();
  ASSERT_NE(fixedTemplate, nullptr);
  ASSERT_NE(dynamicTemplate, nullptr);
  EXPECT_STRNE(fixedTemplate, dynamicTemplate);
}

TEST_F(ChatTemplateUtilsTest, ManualOverrideTakesPrecedenceOverToolsAtEnd) {
  common_params params;
  params.chat_template = "my_custom_template";
  params.use_jinja = true;

  std::string result = getChatTemplate(nullptr, params, true);
  EXPECT_EQ(result, "my_custom_template");
}

TEST_F(
    ChatTemplateUtilsTest, ManualOverrideTakesPrecedenceOverToolsAtEndFalse) {
  common_params params;
  params.chat_template = "my_custom_template";
  params.use_jinja = true;

  std::string result = getChatTemplate(nullptr, params, false);
  EXPECT_EQ(result, "my_custom_template");
}

// Tests with actual Qwen3 model loaded
class ChatTemplateUtilsQwen3Test : public ::testing::Test {
protected:
  void SetUp() override {
    config_files["device"] = test_common::getTestDevice();
    config_files["ctx_size"] = "2048";
    config_files["gpu_layers"] = test_common::getTestGpuLayers();
    config_files["n_predict"] = "10";

    // Use Qwen3 model for testing
    test_model_path = test_common::BaseTestModelPath::get(
        "Qwen3-1.7B-Q4_0.gguf", "Llama-3.2-1B-Instruct-Q4_0.gguf");
    test_projection_path = "";

    config_files["backendsDir"] = test_common::getTestBackendsDir().string();
  }

  std::unordered_map<std::string, std::string> config_files;
  std::string test_model_path;
  std::string test_projection_path;

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

  bool hasValidModel() { return fs::exists(test_model_path); }
};

TEST_F(ChatTemplateUtilsQwen3Test, IsQwen3ModelWithQwen3ModelLoaded) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  auto model = createModel();
  ASSERT_NE(model, nullptr) << "Failed to load Qwen3 model";
  ASSERT_TRUE(model->isLoaded()) << "Qwen3 model not loaded successfully";

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  EXPECT_TRUE(isQwen3Model(llamaModel))
      << "Model should be detected as Qwen3 model";
}

TEST_F(ChatTemplateUtilsQwen3Test, GetChatTemplateForModelWithQwen3NoOverride) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  auto model = createModel();
  ASSERT_NE(model, nullptr) << "Failed to load Qwen3 model";
  ASSERT_TRUE(model->isLoaded()) << "Qwen3 model not loaded successfully";

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  // Without override, should return Qwen3 template
  std::string result = getChatTemplateForModel(llamaModel, "", false);
  EXPECT_NE(result, "") << "Should return Qwen3 template when no override provided";
  EXPECT_GT(result.length(), 0u) << "Template should not be empty";
}

TEST_F(ChatTemplateUtilsQwen3Test, GetChatTemplateForModelWithQwen3ToolsAtEnd) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  auto model = createModel();
  ASSERT_NE(model, nullptr) << "Failed to load Qwen3 model";
  ASSERT_TRUE(model->isLoaded()) << "Qwen3 model not loaded successfully";

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  // With toolsAtEnd=true, should return dynamic template
  std::string result = getChatTemplateForModel(llamaModel, "", true);
  EXPECT_NE(result, "") << "Should return Qwen3 tools template when no override provided";
  EXPECT_GT(result.length(), 0u) << "Template should not be empty";
}

TEST_F(
    ChatTemplateUtilsQwen3Test,
    GetChatTemplateForModelWithQwen3ManualOverrideTakesPrecedence) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  auto model = createModel();
  ASSERT_NE(model, nullptr) << "Failed to load Qwen3 model";
  ASSERT_TRUE(model->isLoaded()) << "Qwen3 model not loaded successfully";

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  // Manual override should take precedence
  std::string manualOverride = "custom qwen3 template";
  std::string result = getChatTemplateForModel(llamaModel, manualOverride, false);
  EXPECT_EQ(result, manualOverride)
      << "Manual override should take precedence over Qwen3 template";
}

TEST_F(ChatTemplateUtilsQwen3Test, GetChatTemplateWithQwen3JinjaEnabled) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  auto model = createModel();
  ASSERT_NE(model, nullptr) << "Failed to load Qwen3 model";
  ASSERT_TRUE(model->isLoaded()) << "Qwen3 model not loaded successfully";

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  common_params params;
  params.chat_template = "";
  params.use_jinja = true;

  // With Jinja enabled and no override, should use Qwen3 template
  std::string result = getChatTemplate(llamaModel, params, false);
  EXPECT_NE(result, "") << "Should return Qwen3 template when Jinja is enabled";
  EXPECT_GT(result.length(), 0u) << "Template should not be empty";
}

TEST_F(
    ChatTemplateUtilsQwen3Test,
    GetChatTemplateWithQwen3JinjaEnabledManualOverride) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  auto model = createModel();
  ASSERT_NE(model, nullptr) << "Failed to load Qwen3 model";
  ASSERT_TRUE(model->isLoaded()) << "Qwen3 model not loaded successfully";

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  common_params params;
  params.chat_template = "custom template";
  params.use_jinja = true;

  // With manual override, should use the override
  std::string result = getChatTemplate(llamaModel, params, false);
  EXPECT_EQ(result, "custom template")
      << "Manual override should take precedence";
}

TEST_F(ChatTemplateUtilsQwen3Test, NonQwen3ModelNotDetectedAsQwen3) {
  if (!hasValidModel()) {
    GTEST_SKIP() << "Qwen3 model not found at " << test_model_path;
  }

  // Test with Llama model instead
  std::string llamaModelPath = test_common::BaseTestModelPath::get();
  if (!fs::exists(llamaModelPath)) {
    GTEST_SKIP() << "Llama model not found at " << llamaModelPath;
  }

  std::string modelPath = llamaModelPath;
  std::string projectionPath = "";
  auto configCopy = config_files;
  auto model = std::make_unique<LlamaModel>(
      std::move(modelPath), std::move(projectionPath), std::move(configCopy));
  model->waitForLoadInitialization();

  if (!model->isLoaded()) {
    GTEST_SKIP() << "Llama model failed to load";
  }

  llama_model* llamaModel = model->getModel();
  ASSERT_NE(llamaModel, nullptr) << "Llama model pointer is null";

  EXPECT_FALSE(isQwen3Model(llamaModel))
      << "Llama model should not be detected as Qwen3 model";
}
