export type ModelCapability = 'chat' | 'vision' | 'thinking' | 'embedding-text' | 'embedding-image';

export interface ModelPackageRequirement {
  requiredFiles: string[];
  requiredExtensions: string[];
  configFile: string;
}

export interface ModelDescriptor {
  id: string;
  displayName: string;
  runtime: 'mnn' | 'litert-lm';
  capabilities: ModelCapability[];
  expectedBytes?: number;
  privateStoragePath: string;
  source: { provider: 'huggingface' | 'manual'; repository?: string };
  package: ModelPackageRequirement;
  runtimeRequirements: { abi: 'arm64-v8a'; minSdk: number; singleActiveModel: boolean };
}

export const modelRegistry: readonly ModelDescriptor[] = [
  {
    id: 'qwen35-4b-mnn',
    displayName: 'Qwen3.5 4B (MNN)',
    runtime: 'mnn',
    capabilities: ['chat', 'vision', 'thinking'],
    expectedBytes: 2_845_946_907,
    privateStoragePath: 'models/qwen35-4b-mnn',
    source: { provider: 'huggingface', repository: 'taobao-mnn/Qwen3.5-4B-MNN' },
    package: { configFile: 'config.json', requiredFiles: ['config.json'], requiredExtensions: ['.mnn'] },
    runtimeRequirements: { abi: 'arm64-v8a', minSdk: 26, singleActiveModel: true },
  },
  {
    id: 'embeddinggemma-2-litert-lm',
    displayName: 'EmbeddingGemma 2 (LiteRT-LM)',
    runtime: 'litert-lm',
    capabilities: ['embedding-text', 'embedding-image'],
    privateStoragePath: 'models/embeddinggemma-2-litert-lm',
    source: { provider: 'manual' },
    package: { configFile: 'metadata.json', requiredFiles: ['metadata.json'], requiredExtensions: [] },
    runtimeRequirements: { abi: 'arm64-v8a', minSdk: 26, singleActiveModel: true },
  },
];

export function getModelDescriptor(modelId: string): ModelDescriptor | undefined {
  return modelRegistry.find((model) => model.id === modelId);
}

