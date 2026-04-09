#pragma once

#include <cstdint>
#include <string>

#include "qvac-lib-inference-addon-cpp/Errors.hpp"

namespace qvac_lib_inference_addon_bci::errors {
constexpr const char* ADDON_ID = "BCI";

enum BCIErrorCode : std::uint8_t {
  UnableToCreateWhisperContext,
  UnableToTranscribe,
  InvalidNeuralSignal,
  UnsupportedSignalFormat,
  ModelNotLoaded,
  ProcessingFailed,
};

inline std::string toString(BCIErrorCode code) {
  switch (code) {
  case UnableToCreateWhisperContext:
    return "UnableToCreateWhisperContext";
  case UnableToTranscribe:
    return "UnableToTranscribe";
  case InvalidNeuralSignal:
    return "InvalidNeuralSignal";
  case UnsupportedSignalFormat:
    return "UnsupportedSignalFormat";
  case ModelNotLoaded:
    return "ModelNotLoaded";
  case ProcessingFailed:
    return "ProcessingFailed";
  default:
    return "UnknownError";
  }
}
} // namespace qvac_lib_inference_addon_bci::errors

namespace qvac_errors {
namespace bci_error {
enum class Code : std::uint8_t {
  InvalidNeuralSignal,
  UnsupportedSignalFormat,
  ProcessingFailed,
};

inline qvac_errors::StatusError
makeStatus(Code /*code*/, const std::string& message) {
  return qvac_errors::StatusError("BCI", "BCIError", message);
}
} // namespace bci_error
} // namespace qvac_errors
