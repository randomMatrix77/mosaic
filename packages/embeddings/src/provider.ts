export interface EmbeddingInputImage {
  localUri: string;
  mimeType: string;
}

export interface EmbeddingVector {
  values: Float32Array;
  modelId: string;
  dimensions: number;
}

/** LiteRT-LM lifecycle is intentionally separate from MNN chat inference. */
export interface EmbeddingProvider {
  loadModel(modelId: string): Promise<void>;
  unloadModel(): Promise<void>;
  embedText(text: string): Promise<EmbeddingVector>;
  embedImage(image: EmbeddingInputImage): Promise<EmbeddingVector>;
  isAvailable(): Promise<boolean>;
}

