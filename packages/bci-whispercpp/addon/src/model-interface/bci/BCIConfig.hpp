#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include <whisper.h>

namespace qvac_lib_inference_addon_bci {

using JSValueVariant =
    std::variant<std::monostate, int, double, std::string, bool>;

template <typename Params>
using HandlerFunction = std::function<void(Params&, const JSValueVariant&)>;

template <typename Params>
using HandlersMap = std::unordered_map<std::string, HandlerFunction<Params>>;

struct BCIConfig {
  std::map<std::string, JSValueVariant> miscConfig;
  std::map<std::string, JSValueVariant> whisperMainCfg;
  std::map<std::string, JSValueVariant> whisperContextCfg;
  std::map<std::string, JSValueVariant> bciConfig;
};

whisper_full_params toWhisperFullParams(const BCIConfig& bciConfig);
whisper_context_params toWhisperContextParams(const BCIConfig& bciConfig);

std::string convertVariantToString(const JSValueVariant& value);

// Maps of handler functions for setting whisper_full_params fields from JS.
const HandlersMap<whisper_full_params>& getWhisperMainHandlers();
const HandlersMap<whisper_context_params>& getWhisperContextHandlers();

} // namespace qvac_lib_inference_addon_bci
