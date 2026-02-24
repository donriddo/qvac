#pragma once

#include <memory>
#include <vector>

#include <qvac-lib-inference-addon-cpp/JsInterface.hpp>
#include <qvac-lib-inference-addon-cpp/JsUtils.hpp>
#include <qvac-lib-inference-addon-cpp/ModelInterfaces.hpp>
#include <qvac-lib-inference-addon-cpp/addon/AddonJs.hpp>
#include <qvac-lib-inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <qvac-lib-inference-addon-cpp/handlers/OutputHandler.hpp>
#include <qvac-lib-inference-addon-cpp/queue/OutputCallbackJs.hpp>

#include "model-interface/SdModel.hpp"

namespace qvac_lib_inference_addon_sd {

inline js_value_t* createInstance(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);

  // Extract configuration from JS object at args[1]
  const string modelPath        = args.getMapEntry(1, "path");
  const string clipLPath        = args.getMapEntry(1, "clipLPath");
  const string clipGPath        = args.getMapEntry(1, "clipGPath");
  const string t5XxlPath        = args.getMapEntry(1, "t5XxlPath");
  const string llmPath          = args.getMapEntry(1, "llmPath");   // FLUX.2 [klein] Qwen3
  const string vaePath          = args.getMapEntry(1, "vaePath");
  auto configMap                = args.getSubmap(1, "config");

  auto model = make_unique<SdModel>(
      modelPath, clipLPath, clipGPath, t5XxlPath, llmPath, vaePath, std::move(configMap));

  // Register output handlers for both progress strings and image byte arrays
  out_handl::OutputHandlers<out_handl::JsOutputHandlerInterface> outHandlers;
  outHandlers.add(make_shared<out_handl::JsStringOutputHandler>());
  outHandlers.add(make_shared<out_handl::JsTypedArrayOutputHandler<uint8_t>>());

  unique_ptr<OutputCallBackInterface> callback = make_unique<OutputCallBackJs>(
      env,
      args.get(0, "jsHandle"),
      args.getFunction(2, "outputCallback"),
      std::move(outHandlers));

  auto addon = make_unique<AddonJs>(env, std::move(callback), std::move(model));

  return JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

inline js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));

  auto [type, jsInput] = JsInterface::getInput(args);

  if (type != "text") {
    throw StatusError(
        general_error::InvalidArgument,
        "stable-diffusion runJob expects a single text input with JSON params");
  }

  const string paramsJson =
      js::String(env, jsInput).as<std::string>(env);

  SdModel::GenerationJob job;
  job.paramsJson = paramsJson;

  // Queue step-progress updates as JSON strings (handled by JsStringOutputHandler)
  job.progressCallback = [&instance](const std::string& progressJson) {
    instance.addonCpp->outputQueue->queueResult(std::any(progressJson));
  };

  // Queue final image/frame bytes (handled by JsTypedArrayOutputHandler<uint8_t>)
  job.outputCallback = [&instance](const std::vector<uint8_t>& imageBytes) {
    instance.addonCpp->outputQueue->queueResult(std::any(imageBytes));
  };

  return instance.runJob(std::any(std::move(job)));
}
JSCATCH

} // namespace qvac_lib_inference_addon_sd
