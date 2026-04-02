import {
  loadModel,
  textToSpeech,
  unloadModel,
  type ModelProgressUpdate,
  TTS_TOKENIZER_EN_CHATTERBOX,
  TTS_SPEECH_ENCODER_EN_CHATTERBOX_FP32,
  TTS_EMBED_TOKENS_EN_CHATTERBOX_FP32,
  TTS_CONDITIONAL_DECODER_EN_CHATTERBOX_FP32,
  TTS_LANGUAGE_MODEL_EN_CHATTERBOX_FP32,
} from "@qvac/sdk";
import {
  createWav,
  playAudio,
  int16ArrayToBuffer,
  createWavHeader,
} from "./utils";

// Chatterbox TTS with LavaSR neural speech enhancement (48kHz output).
// Usage: node chatterbox-enhanced.js <referenceAudioSrc> <enhancerBackbone> <enhancerSpecHead> [denoiserPath]
const [referenceAudioSrc, backboneSrc, specHeadSrc, denoiserSrc] =
  process.argv.slice(2);

if (!referenceAudioSrc || !backboneSrc || !specHeadSrc) {
  console.error(
    "Usage: node chatterbox-enhanced.js <referenceAudioSrc> <enhancerBackbone> <enhancerSpecHead> [denoiserPath]",
  );
  process.exit(1);
}

try {
  const modelId = await loadModel({
    modelSrc: TTS_TOKENIZER_EN_CHATTERBOX.src,
    modelType: "tts",
    modelConfig: {
      ttsEngine: "chatterbox",
      language: "en",
      ttsTokenizerSrc: TTS_TOKENIZER_EN_CHATTERBOX.src,
      ttsSpeechEncoderSrc: TTS_SPEECH_ENCODER_EN_CHATTERBOX_FP32.src,
      ttsEmbedTokensSrc: TTS_EMBED_TOKENS_EN_CHATTERBOX_FP32.src,
      ttsConditionalDecoderSrc: TTS_CONDITIONAL_DECODER_EN_CHATTERBOX_FP32.src,
      ttsLanguageModelSrc: TTS_LANGUAGE_MODEL_EN_CHATTERBOX_FP32.src,
      referenceAudioSrc,
      enhancer: {
        type: "lavasr",
        enhance: true,
        ...(denoiserSrc ? { denoise: true } : {}),
        backboneSrc,
        specHeadSrc,
        ...(denoiserSrc ? { denoiserSrc } : {}),
      },
    },
    onProgress: (progress: ModelProgressUpdate) => {
      console.log(progress);
    },
  });

  console.log(`Model loaded: ${modelId}`);

  console.log("🎵 Synthesizing with LavaSR enhancement...");
  const result = textToSpeech({
    modelId,
    text: "Hello! This audio was synthesized with Chatterbox and enhanced with LavaSR neural bandwidth extension to 48 kilohertz.",
    inputType: "text",
    stream: false,
  });

  const audioBuffer = await result.buffer;
  const sampleRate = await result.sampleRate;
  console.log(`TTS complete. Total samples: ${audioBuffer.length}, sample rate: ${sampleRate}Hz`);

  console.log("💾 Saving audio to file...");
  createWav(audioBuffer, sampleRate ?? 24000, "tts-enhanced-output.wav");
  console.log("✅ Audio saved to tts-enhanced-output.wav");

  console.log("🔊 Playing audio...");
  const audioData = int16ArrayToBuffer(audioBuffer);
  const wavBuffer = Buffer.concat([
    createWavHeader(audioData.length, sampleRate ?? 24000),
    audioData,
  ]);
  playAudio(wavBuffer);
  console.log("✅ Audio playback complete");

  await unloadModel({ modelId });
  console.log("Model unloaded");
  process.exit(0);
} catch (error) {
  console.error("❌ Error:", error);
  process.exit(1);
}
