export type ModelRuntime = 'mnn' | 'litert-lm';

export interface AttachmentReference {
  id: string;
  kind: 'image' | 'text' | 'markdown' | 'pdf';
  localUri: string;
  displayName: string;
  mimeType: string;
  extractedText?: string;
}

export interface ChatTurn {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string;
  attachments?: AttachmentReference[];
}

export interface GenerationConfig {
  maxNewTokens: number;
  temperature?: number;
  topP?: number;
  threads?: number;
}

export interface GenerationMetrics {
  promptTokens: number;
  generatedTokens: number;
  loadMicros?: number;
  visionMicros?: number;
  prefillMicros: number;
  decodeMicros: number;
  cancelled: boolean;
}

export interface InferenceProvider {
  readonly runtime: ModelRuntime;
  loadModel(modelId: string): Promise<void>;
  unloadModel(): Promise<void>;
  isLoaded(): Promise<boolean>;
  generate(turns: ChatTurn[], config: GenerationConfig, onToken: (text: string) => void): Promise<GenerationMetrics>;
  cancelGeneration(): Promise<void>;
}

