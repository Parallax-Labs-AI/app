export interface LocalProgram { name: string; command: string; args: string[]; enabled: boolean }

export interface Settings {
  onboarded: boolean
  webSearch: boolean
  customMcp: LocalProgram[]
  agentCommands: LocalProgram[]
  agentAutoRead: boolean
  agentAutoCreate: boolean
  defaultSystemPrompt: string
  theme: 'dark' | 'light' | 'system'
  reducedMotion: boolean
  chatTextSize: 'small' | 'medium' | 'large'
  accent: 'blue' | 'violet' | 'teal'
  defaultModel: string | null
  autoload: boolean
  modelDirectory: string | null
  acceleration: 'auto' | 'cpu' | 'metal' | 'cuda' | 'vulkan'
  maxTokens: number
  temperature: number
  topP: number
  topK: number
  gpuLayers: number | null
}
export interface Hardware {
  os: string
  arch: string
  cpu: string
  memory: number
  availableMemory: number
  gpu: string | null
  vram: number | null
  metal: boolean
  cuda: boolean
  vulkan: boolean
}
export interface Model {
  id: string
  name: string
  quantization: string | null
  bytes: number
  path: string
  architecture: string
  context: number | null
  imported: boolean
  availableOnDisk: boolean
}
export interface Variant { file: string; quantization: string; bytes: number; sha256: string }
export interface CatalogModel {
  id: string
  name: string
  description: string
  repo: string
  architecture: string | null
  context: number | null
  license: string | null
  status: 'available' | 'coming_soon' | 'incompatible' | 'unavailable'
  reason: string
  variants: Variant[]
}
export interface Progress {
  phase: string
  bytes: number
  total: number
  speed: number | null
  eta: number | null
}
export interface Download { id: string; progress: Progress; error: string | null }
export interface Snapshot {
  hardware: Hardware
  settings: Settings
  runtime: { release: string; suggested: string | null; installed: string[]; previous: string[] }
  runtimeProgress: Progress | null
  server: { phase: 'idle' | 'loading' | 'ready' | 'error'; modelId: string | null; backend: string | null; detail: string | null }
  models: Model[]
  downloads: Download[]
  memoryProfiles: Record<string, MemoryProfile>
  modelPreferences: Record<string, ModelPreferences>
  defaultModelPreferences: ModelPreferences
  activeContext: number | null
  dataDirectory: string
  modelDirectory: string
  importing: Progress | null
}
export interface ModelPreferences {
  contextMode: 'recommended' | 'most' | 'custom'
  customTokens: number
  allowOverBudget: boolean
  reasoning: 'disabled' | 'low' | 'max'
}
export interface MemoryRow {
  kind: string
  name: string
  description: string | null
  total: number | null
  free: number | null
  self: number
  model: number
  context: number
  compute: number
}
export interface Measurement { tokens: number; rows: MemoryRow[] }
export interface MemoryProfile {
  backend: string
  unified: boolean
  systemTotal: number
  systemAvailable: number
  modelLimit: number
  recommended: Measurement | null
  maximum: Measurement | null
  estimatedMaxBeforeStartup: number | null
  minimum: Measurement | null
  custom: Measurement | null
}
export interface Conversation { id: string; title: string; systemPrompt: string; updated: number }
export interface Message { id: number; role: 'user' | 'assistant'; content: string; reasoning: string; search_failed: boolean }
export interface Project { name: string; path: string }
export type ChatMode = 'chat' | 'agent'
export type AgentDecision = 'deny' | 'once' | 'kind'
export interface AgentApproval { id: string; project: string; action: string; path: string; content: string | null; remember: boolean; command: string | null }
export interface Thread { conversation: Conversation; messages: Message[]; project: Project | null }
export type Turn = { type: 'send'; content: string } | { type: 'edit'; message: number; content: string } | { type: 'regenerate' }
export type SearchStatus = 'searching' | 'done' | 'failed'
export type ChatEvent = { type: 'started'; message: number } | { type: 'delta' | 'reasoning'; text: string } | { type: 'search_status'; status: SearchStatus } | { type: 'tool_notice'; text: string } | { type: 'agent_approval'; request: AgentApproval }
export type Page = 'chat' | 'settings' | 'setup'
export type Action = (name: string, args?: Record<string, unknown>) => Promise<boolean>

export const variantId = (model: CatalogModel, variant: Variant) =>
  `${model.id}-${variant.quantization.toLowerCase()}-${variant.sha256.slice(0, 12)}`

export function bytes(amount: number | null) {
  if (amount == null || amount < 0) return 'Unavailable'
  if (amount === 0) return '0 B'
  const [unit, scale] = amount >= 1e9 ? ['GB', 1e9] : amount >= 1e6 ? ['MB', 1e6] : amount >= 1e3 ? ['KB', 1e3] : ['B', 1]
  return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(amount / Number(scale))} ${unit}`
}
export const backendName = (value: string | null) => value === 'metal' ? 'Metal' : value?.toUpperCase() ?? 'Unavailable'
export const errorText = (error: unknown) => typeof error === 'string' ? error : error instanceof Error ? error.message : String(error)

export const tokens = (value: number) => value.toLocaleString()
export const mib = (value: number | null) => value === null ? 'Unavailable' : value >= 1024 ? `${(value / 1024).toFixed(2)} GiB` : `${value} MiB`
