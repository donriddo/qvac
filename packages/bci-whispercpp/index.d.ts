declare interface BCIWhispercppArgs {
  /** Path to BrainWhisperer .ckpt file */
  checkpoint: string;
  /** Path to rnn_args.yaml */
  rnnArgs: string;
  /** Directory containing model.py, pl_wrapper.py, dataset.py, utils.py */
  modelDir: string;
  /** Path to cleaned_val_data.pkl (required for batch mode) */
  dataPath?: string;
  logger?: {
    debug(...args: unknown[]): void;
    info(...args: unknown[]): void;
    warn(...args: unknown[]): void;
    error(...args: unknown[]): void;
  };
}

declare interface TranscribeOptions {
  /** Expected text for WER computation */
  expected?: string;
  /** Day index for day-specific projection (default: 0) */
  dayIdx?: number;
  /** Timeout in ms (default: 120000) */
  timeout?: number;
}

declare interface TranscriptionResult {
  text: string;
  textClean: string;
  expected?: string;
  expectedClean?: string;
  wer?: number;
}

declare interface BatchTranscriptionResult extends TranscriptionResult {
  index: number;
}

declare interface BatchOptions {
  /** Comma-separated sample indices (default: '0,1,2,3,4') */
  samples?: string;
  /** Timeout in ms (default: 120000) */
  timeout?: number;
}

/**
 * BCI neural signal transcription adapter.
 *
 * Uses the BrainWhisperer Python model with identical beam search
 * parameters to the research notebook, achieving ~8.86% WER.
 * Built on top of @qvac/transcription-whispercpp.
 */
declare class BCIWhispercpp {
  constructor(args: BCIWhispercppArgs);

  /** Transcribe a single .bin neural signal file (exact notebook match). */
  transcribe(signalPath: string, opts?: TranscribeOptions): TranscriptionResult;

  /** Transcribe a batch via DataLoader pipeline (exact notebook match). */
  transcribeBatch(opts?: BatchOptions): BatchTranscriptionResult[];
}

/** Compute Word Error Rate between hypothesis and reference. */
declare function computeWER(hypothesis: string, reference: string): number;

declare namespace BCIWhispercpp {
  export {
    BCIWhispercpp as default,
    BCIWhispercpp,
    BCIWhispercppArgs,
    TranscribeOptions,
    TranscriptionResult,
    BatchTranscriptionResult,
    BatchOptions,
    computeWER,
  };
}

export = BCIWhispercpp;
