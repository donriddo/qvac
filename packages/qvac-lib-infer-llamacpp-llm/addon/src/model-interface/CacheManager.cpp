#include "CacheManager.hpp"

#include <filesystem>
#include <system_error>

#include <llama.h>
#include <qvac-lib-inference-addon-cpp/Errors.hpp>

#include "TextLlmContext.hpp"
#include "addon/LlmErrors.hpp"
#include "utils/LoggingMacros.hpp"

using namespace qvac_lib_inference_addon_llama::errors;
using namespace qvac_lib_inference_addon_cpp::logger;
using namespace qvac_lib_inference_addon_llama::logging;

CacheManager::CacheManager(
    LlmContext* llmContext, llama_pos configuredNDiscarded,
    std::function<void(bool)> resetStateCallback)
    : llmContext_(llmContext), configuredNDiscarded_(configuredNDiscarded),
      resetStateCallback_(std::move(resetStateCallback)) {}

bool CacheManager::isFileInitialized(const std::filesystem::path& path) {
  std::error_code errorCode;
  auto size = std::filesystem::file_size(path, errorCode);
  if (errorCode) {
    return false;
  }
  return size != 0;
}

bool CacheManager::handleCache(
    std::vector<common_chat_msg>& chatMsgs,
    std::vector<common_chat_tool>& tools, const std::string& inputPrompt,
    std::function<
        std::pair<std::vector<common_chat_msg>, std::vector<common_chat_tool>>(
            const std::string&)>
        formatPrompt) {

  auto formatted = formatPrompt(inputPrompt);
  chatMsgs = std::move(formatted.first);
  tools = std::move(formatted.second);

  bool hasSessionMessage = !chatMsgs.empty() && chatMsgs[0].role == "session";

  if (!hasSessionMessage) {
    if (hasActiveCache()) {
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "%s: No session message in prompt, clearing existing cache "
              "'%s'\n",
              __func__,
              sessionPath_.c_str()));
      saveCache();
      resetStateCallback_(true);
      sessionPath_.clear();
      cacheDisabled_ = true;
    }
    cacheUsedInLastPrompt_ = false;
    return false;
  }

  bool cacheLoaded = false;
  bool cachePathSetInThisArray = false;

  while (!chatMsgs.empty() && chatMsgs[0].role == "session") {
    std::string sessionCommand = chatMsgs[0].content;
    chatMsgs.erase(chatMsgs.begin());

    if (sessionCommand == "reset") {
      if (!cachePathSetInThisArray) {
        std::string errorMsg = string_format(
            "%s: reset command requires explicit cache file specification in "
            "the same message array\n",
            __func__);
        throw qvac_errors::StatusError(
            ADDON_ID, toString(InvalidInputFormat), errorMsg);
      }
      resetStateCallback_(true);
      cacheUsedInLastPrompt_ = false;
    } else if (sessionCommand == "save") {
      if (!cachePathSetInThisArray) {
        std::string errorMsg = string_format(
            "%s: save command requires explicit cache file specification in "
            "the same message array\n",
            __func__);
        throw qvac_errors::StatusError(
            ADDON_ID, toString(InvalidInputFormat), errorMsg);
      }
      saveCache();
    } else if (sessionCommand == "getTokens") {
      if (!cachePathSetInThisArray) {
        std::string errorMsg = string_format(
            "%s: getTokens command requires explicit cache file specification "
            "in the same message array\n",
            __func__);
        throw qvac_errors::StatusError(
            ADDON_ID, toString(InvalidInputFormat), errorMsg);
      }
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "%s: getTokens command - querying cache tokens for '%s'\n",
              __func__,
              sessionPath_.c_str()));
    } else {
      if (!cacheDisabled_ && !sessionPath_.empty() &&
          sessionCommand == sessionPath_) {
        QLOG_IF(
            Priority::DEBUG,
            string_format(
                "%s: Same session file '%s' - ignoring command, continuing to "
                "inference\n",
                __func__,
                sessionPath_.c_str()));
        cachePathSetInThisArray = true;
        cacheUsedInLastPrompt_ = true;
        continue;
      }

      if (!cacheDisabled_ && !sessionPath_.empty() &&
          sessionCommand != sessionPath_) {
        QLOG_IF(
            Priority::DEBUG,
            string_format(
                "%s: Switching from cache '%s' to '%s', clearing old cache\n",
                __func__,
                sessionPath_.c_str(),
                sessionCommand.c_str()));
        saveCache();
        resetStateCallback_(true);
      }

      if (cacheDisabled_ && sessionPath_.empty()) {
        resetStateCallback_(true);
      }

      sessionPath_ = sessionCommand;
      cachePathSetInThisArray = true;

      if (!sessionPath_.empty()) {
        cacheDisabled_ = false;

        QLOG_IF(
            Priority::DEBUG,
            string_format(
                "%s: Cache enabled with session file '%s'\n",
                __func__,
                sessionPath_.c_str()));

        cacheLoaded = loadCache();
        cacheUsedInLastPrompt_ = true;
      } else {
        std::string errorMsg =
            string_format("%s: session msg content is empty\n", __func__);
        throw qvac_errors::StatusError(
            ADDON_ID, toString(InvalidInputFormat), errorMsg);
      }
    }
  }

  return cacheLoaded;
}

bool CacheManager::loadCache() {
  if (cacheDisabled_ || sessionPath_.empty()) {
    return false;
  }

  auto* ctx = llmContext_->getCtx();
  size_t nTokenCount = 0;

  QLOG_IF(
      Priority::DEBUG,
      string_format(
          "%s: attempting to load saved session from '%s'\n",
          __func__,
          sessionPath_.c_str()));
  if (!isFileInitialized(sessionPath_)) {
    QLOG_IF(
        Priority::DEBUG,
        string_format(
            "%s: session file does not exist or is empty\n", __func__));
    return false;
  }

  // First, get the token count to allocate the buffer
  // We use a larger buffer to accommodate all tokens
  std::vector<llama_token> loadedTokens(llama_n_ctx(ctx));
  size_t maxTokens = loadedTokens.size();

  // Use llama_state_seq_load_file which returns actual token IDs
  // seq_id 0 is the default sequence
  size_t bytesRead = llama_state_seq_load_file(
      ctx,
      sessionPath_.c_str(),
      0,           // dest_seq_id
      loadedTokens.data(),  // tokens_out
      maxTokens,   // n_token_capacity
      &nTokenCount); // n_token_count_out

  if (bytesRead == 0) {
    std::string errorMsg = string_format(
        "%s: failed to load session file '%s'\n",
        __func__,
        sessionPath_.c_str());
    throw qvac_errors::StatusError(
        ADDON_ID, toString(UnableToLoadSessionFile), errorMsg);
  }

  // Resize to actual token count
  loadedTokens.resize(nTokenCount);

  QLOG_IF(Priority::DEBUG, string_format("%s: loaded a session with %zu tokens\n", __func__, nTokenCount));

  if (nTokenCount > 0) {
    if (nTokenCount > static_cast<size_t>(llama_n_ctx(ctx))) {
      std::string errorMsg = string_format(
          "%s: cache file '%s' contains %zu tokens, which exceeds the current "
          "context size of %d tokens\n",
          __func__,
          sessionPath_.c_str(),
          nTokenCount,
          llama_n_ctx(ctx));
      throw qvac_errors::StatusError(
          ADDON_ID, toString(ContextLengthExeeded), errorMsg);
    }

    // Extract metadata from the beginning of the token array
    // Format: [nPast, firstMsgTokens, token0, token1, ...]
    llama_pos savedNPast = loadedTokens[0];
    llama_pos savedFirstMsgTokens = loadedTokens[1];

    if (savedNPast > llama_n_ctx(ctx)) {
      std::string errorMsg = string_format(
          "%s: cache file '%s' has nPast=%lld which exceeds context size of %d\n",
          __func__,
          sessionPath_.c_str(),
          static_cast<long long>(savedNPast),
          llama_n_ctx(ctx));
      throw qvac_errors::StatusError(
          ADDON_ID, toString(ContextLengthExeeded), errorMsg);
    }

    llmContext_->setNPast(savedNPast);
    llmContext_->setFirstMsgTokens(savedFirstMsgTokens);

    if (configuredNDiscarded_ >
        llama_n_ctx(ctx) - llmContext_->getFirstMsgTokens()) {
      llmContext_->setNDiscarded(
          llama_n_ctx(ctx) - llmContext_->getFirstMsgTokens() - 1);
    } else {
      llmContext_->setNDiscarded(configuredNDiscarded_);
    }

    // Restore token tracking buffer in TextLlmContext (skip metadata)
    auto* textCtx = dynamic_cast<TextLlmContext*>(llmContext_);
    if (textCtx && nTokenCount > 2) {
      std::vector<llama_token> actualTokens(loadedTokens.begin() + 2, loadedTokens.end());
      textCtx->setAllTokens(actualTokens);
    }

    // Remove tokens beyond nPast from the KV cache
    auto* mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, -1, savedNPast, -1);
    return true;
  }
  return false;
}

void CacheManager::saveCache() {
  if (cacheDisabled_ || sessionPath_.empty()) {
    std::string errorMsg = string_format(
        "%s: Cannot save cache - caching disabled or no session path set\n",
        __func__);
    throw qvac_errors::StatusError(
        ADDON_ID, toString(InvalidInputFormat), errorMsg);
  }

  auto* ctx = llmContext_->getCtx();
  QLOG_IF(
      Priority::DEBUG,
      string_format(
          "\n%s: saving final output to session file '%s'\n",
          __func__,
          sessionPath_.c_str()));

  // Get all tracked tokens from the context
  // The TextLlmContext maintains a buffer of all processed tokens
  const auto* textCtx = dynamic_cast<const TextLlmContext*>(llmContext_);
  std::vector<llama_token> tokens;
  if (textCtx) {
    tokens = textCtx->getAllTokens();
  } else {
    // Fallback: create minimal token info
    tokens.clear();
  }

  // Prepend metadata: [nPast, firstMsgTokens] at the start of the token array
  // This allows us to restore firstMsgTokens on load
  std::vector<llama_token> tokensWithMetadata;
  tokensWithMetadata.reserve(tokens.size() + 2);
  tokensWithMetadata.push_back(static_cast<llama_token>(llmContext_->getNPast()));
  tokensWithMetadata.push_back(static_cast<llama_token>(llmContext_->getFirstMsgTokens()));
  tokensWithMetadata.insert(tokensWithMetadata.end(), tokens.begin(), tokens.end());

  // Use llama_state_seq_save_file which saves tokens along with KV cache
  llama_state_seq_save_file(ctx, sessionPath_.c_str(), 0, tokensWithMetadata.data(), tokensWithMetadata.size());
}

bool CacheManager::isCacheDisabled() const { return cacheDisabled_; }

bool CacheManager::hasActiveCache() const {
  return !cacheDisabled_ && !sessionPath_.empty();
}
bool CacheManager::wasCacheUsedInLastPrompt() const {
  return cacheUsedInLastPrompt_;
}
