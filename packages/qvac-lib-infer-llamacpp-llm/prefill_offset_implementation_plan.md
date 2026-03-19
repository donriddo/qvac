# Prefill with Offset — Implementation Plan

## Goal

Add an `offset` parameter to the eval methods in the LLM addon's C++ layer. When provided, the KV cache is trimmed to `offset` position before evaluating new tokens. Combined with `prefill=true` (from PR #689), this enables the tools-at-end KV cache optimization entirely in C++, with no JS API changes.

## Context

The tools-at-end optimization (see `Pitch_tools_at_end_of_prompt.md`) requires:
1. Trimming the KV cache tail (remove old tools + response tokens)
2. Prefilling known tokens (previous response + new query + new tools) from a specific cache position
3. Generating from the warm cache

PR #689 adds `prefill=true` which evaluates tokens into the KV cache without generating. But it always appends at `nPast_` — there's no way to say "go back to position X and prefill from there." The `offset` parameter fills that gap.

## How It Works

```
offset = -1 (default): current behavior, eval starts from nPast_
offset >= 0:           trim KV cache to that position, then eval new tokens from there
```

When `evalMessageWithTools` receives `offset >= 0 && offset < nPast_`:
1. Call `removeLastNTokens(nPast_ - offset)` — this trims the KV cache and sets `nPast_ = offset`
2. Proceed with normal token evaluation starting from the new `nPast_`

This is atomic — trim + eval in a single call. No need for the caller to manually call `removeLastNTokens` then `evalMessage` separately.

## End-to-End Flow (tools-at-end optimization)

This is how `processPrompt()` would use offset + prefill internally:

### Turn 1 (cold start)
```
KV cache: empty
Prompt:   <system><user-q-1><tools-1>

1. evalMessageWithTools(msgs, tools1, ..., prefill=false, offset=-1)
   -> tokens evaluated at pos 0..N-1, nPast_ = N
2. generateResponse() -> model generates, nPast_ = N + G
3. Track: toolTokenCount_ = T, responseTokenCount_ = G
   Cache state: [system | user-q-1 | tools-1 | response-1]
                 pos 0              pos N-T    pos N    pos N+G
```

### Turn 2 (tools changed)
```
KV cache: [system | user-q-1 | tools-1 | response-1]
Want:     [system | user-q-1 | response-1 | user-q-2 | tools-2]

offset = N - T  (position just before old tools)

1. Build full message array: [system, user-q-1, assistant-response-1, user-q-2] + tools-2
2. evalMessageWithTools(msgs, tools2, ..., prefill=false, offset=N-T)
   -> removeLastNTokens(nPast_ - offset) trims tools-1 + response-1
   -> KV cache: [system | user-q-1]
   -> tokenizeChat produces full prompt tokens
   -> only NEW tokens (after the cached prefix) are evaluated
   -> nPast_ updated to include response-1 + user-q-2 + tools-2
3. generateResponse() -> model generates turn 2 response
```

Note: The prefill flag is for when you want to eval without generating (preload context). The offset works with both `prefill=true` and `prefill=false`.

---

## Files to Change

All files are under `packages/qvac-lib-infer-llamacpp-llm/addon/src/model-interface/`.

### 1. LlmContext.hpp (virtual interface)

**What:** Add `offset` parameter to both virtual method signatures.

**Current (after PR #689):**
```cpp
virtual bool evalMessage(
    const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
    bool prefill) = 0;

virtual bool evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill) = 0;
```

**New:**
```cpp
virtual bool evalMessage(
    const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
    bool prefill, llama_pos offset = -1) = 0;

virtual bool evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill, llama_pos offset = -1) = 0;
```

**Lines affected:** ~114-128 (the two virtual declarations and their doc comments)

**Doc comment addition for both methods:**
```cpp
* @param offset - KV cache position to trim to before evaluation.
*                 -1 means no trimming (default, eval from current nPast_).
*                 >= 0 trims cache to this position, then evaluates from there.
```

---

### 2. TextLlmContext.hpp (override declarations)

**What:** Update override signatures to match new virtual interface.

**Current (after PR #689):**
```cpp
bool evalMessage(
    const std::vector<common_chat_msg>& chatMsgs,
    bool isCacheLoaded, bool prefill) override;

bool evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill) override;
```

**New:**
```cpp
bool evalMessage(
    const std::vector<common_chat_msg>& chatMsgs,
    bool isCacheLoaded, bool prefill, llama_pos offset = -1) override;

bool evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill, llama_pos offset = -1) override;
```

**Lines affected:** ~30-45

---

### 3. TextLlmContext.cpp (implementation)

**What:** Two changes — update signatures + add trim logic.

#### 3a. evalMessage (forwarding method, ~line 256)

**Current (after PR #689):**
```cpp
bool TextLlmContext::evalMessage(
    const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
    bool prefill) {
  return evalMessageWithTools(chatMsgs, {}, isCacheLoaded, prefill);
}
```

**New:**
```cpp
bool TextLlmContext::evalMessage(
    const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
    bool prefill, llama_pos offset) {
  return evalMessageWithTools(chatMsgs, {}, isCacheLoaded, prefill, offset);
}
```

#### 3b. evalMessageWithTools (main method, ~line 261)

**Current (after PR #689):**
```cpp
bool TextLlmContext::evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill) {
  std::vector<llama_token> inputTokens;
  tokenizeChat(chatMsgs, tools, inputTokens, isCacheLoaded);
  // ... rest of method
```

**New:**
```cpp
bool TextLlmContext::evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill, llama_pos offset) {

  // Trim KV cache to offset position if specified
  if (offset >= 0 && offset < nPast_) {
    removeLastNTokens(nPast_ - offset);
  }

  std::vector<llama_token> inputTokens;
  tokenizeChat(chatMsgs, tools, inputTokens, isCacheLoaded);
  // ... rest of method unchanged
```

**Why this works:** `removeLastNTokens` calls `llama_memory_seq_rm` to remove entries from the KV cache and decrements `nPast_`. The existing eval loop uses `llama_pos count = nPast_` (line ~319) as its starting position, so new tokens are naturally placed right after the offset.

---

### 4. MtmdLlmContext.hpp (override declarations)

**What:** Same signature update as TextLlmContext.hpp.

**Current (after PR #689):**
```cpp
bool evalMessage(
    const std::vector<common_chat_msg>& chatMsgs,
    bool isCacheLoaded, bool prefill) override;

bool evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill) override;
```

**New:**
```cpp
bool evalMessage(
    const std::vector<common_chat_msg>& chatMsgs,
    bool isCacheLoaded, bool prefill, llama_pos offset = -1) override;

bool evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill, llama_pos offset = -1) override;
```

**Lines affected:** ~37-52

---

### 5. MtmdLlmContext.cpp (implementation)

**What:** Same pattern as TextLlmContext.cpp — update signatures + add trim logic.

#### 5a. evalMessage (forwarding method, ~line 210)

**New:**
```cpp
bool MtmdLlmContext::evalMessage(
    const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
    bool prefill, llama_pos offset) {
  return evalMessageWithTools(chatMsgs, {}, isCacheLoaded, prefill, offset);
}
```

#### 5b. evalMessageWithTools (main method, ~line 215)

**New (add trim block at top):**
```cpp
bool MtmdLlmContext::evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill, llama_pos offset) {

  // Trim KV cache to offset position if specified
  if (offset >= 0 && offset < nPast_) {
    removeLastNTokens(nPast_ - offset);
  }

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  tokenizeChat(chatMsgs, tools, chunks, isCacheLoaded);
  // ... rest unchanged
```

---

### 6. LlamaModel.cpp — processPrompt() passes offset

**What:** Thread `offset` from the internal call site into eval methods. For now, always pass `-1` (no trim) to keep existing behavior. The tools-at-end optimization will set a real offset later.

**Current (after PR #689, ~line 262):**
```cpp
bool evalOk =
    resolved.tools.empty()
        ? llmContext_->evalMessage(
              resolved.chatMsgs, resolved.isCacheLoaded, prompt.prefill)
        : llmContext_->evalMessageWithTools(
              resolved.chatMsgs, resolved.tools,
              resolved.isCacheLoaded, prompt.prefill);
```

**No change needed here** — since the default value is `offset = -1` in the virtual declarations, existing call sites don't need to be updated. The offset will only be passed explicitly when the tools-at-end optimization is implemented in `processPrompt()`.

---

## Files NOT Changed

| File | Why |
|------|-----|
| `LlamaModel.hpp` | `Prompt` struct not touched — offset is internal to C++, not from JS |
| `index.js` | JS API unchanged |
| `index.d.ts` | TypeScript types unchanged |
| `CacheManager.hpp/.cpp` | Cache manager deals with disk sessions, not in-flight cache trimming |

---

## Testing

### Unit-level validation for offset

Add to `test/integration/api-behavior.test.js`:

```js
test('prefill with offset trims cache and re-evaluates', { timeout: 600_000 }, async t => {
  const { model } = await setupModel(t)

  // 1. Normal run to populate cache
  const r1 = await model.run(BASE_PROMPT)
  await collectResponse(r1)
  const cacheAfterR1 = toNumber(r1?.stats?.CacheTokens)
  t.ok(cacheAfterR1 > 0, 'cache populated after first run')

  // 2. Prefill with offset=0 should trim entire cache, then re-evaluate
  const r2 = await model.run(BASE_PROMPT, { prefill: true, offset: 0 })
  await collectResponse(r2)
  const cacheAfterR2 = toNumber(r2?.stats?.CacheTokens)
  t.ok(cacheAfterR2 > 0, 'cache repopulated after prefill with offset')

  // 3. Normal run after prefill should still work
  const r3 = await model.run(BASE_PROMPT)
  const output = await collectResponse(r3)
  t.ok(output.length > 0, 'generation works after prefill with offset')
})
```

Note: This test requires exposing `offset` via JS (`RunOptions`). If offset stays internal-only in C++, this test would need to be a C++ unit test instead.

---

## Summary

| Change | Files | Lines of new logic |
|--------|-------|--------------------|
| Add `offset` param to virtual interface | `LlmContext.hpp` | signature only |
| Update TextLlm override signatures | `TextLlmContext.hpp` | signature only |
| Add trim-to-offset logic in TextLlm | `TextLlmContext.cpp` | 3 lines |
| Update MtmdLlm override signatures | `MtmdLlmContext.hpp` | signature only |
| Add trim-to-offset logic in MtmdLlm | `MtmdLlmContext.cpp` | 3 lines |

Total new logic: ~6 lines across 2 files. The rest is signature propagation.
