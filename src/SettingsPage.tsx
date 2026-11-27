import { useState, type ReactNode } from 'react'
import { ChevronDown, Cpu, Database, Gauge, MessageSquare, Monitor, RotateCcw, ShieldCheck, Users, Wrench } from 'lucide-react'
import { invoke } from '@tauri-apps/api/core'
import { backendName, bytes, errorText, type Conversation, type ModelPreferences, type Settings, type Snapshot } from './types'
import { About } from './About'
import { AgentTools } from './ToolSettings'
import { Choice } from './Choice'
import { ModelPreferencesForm } from './ModelOptions'
import { ErrorBanner, ProgressView } from './ui'

interface Props {
  snapshot: Snapshot
  updatePanel: ReactNode
  save: (settings: Settings) => Promise<boolean>
  chooseStorage: () => void
  exportHistory: () => void
  eraseHistory: () => void
  clearAgentPermissions: () => void
  models: ReactNode
  saveSquadron: (settings: Settings) => Promise<boolean>
  saveContext: (id: string, preferences: ModelPreferences, reload: boolean) => Promise<boolean>
  measureContext: (id: string, custom: number | null) => Promise<void>
  repair: () => void
  retry: () => void
  busy: boolean
  conversation?: Conversation
  saveConversationPrompt: (prompt: string) => Promise<boolean>
}

function SettingsRow({ label, detail, children }: { label: string; detail?: string; children: ReactNode }) {
  return <div className="setting-row"><div><strong>{label}</strong>{detail && <p>{detail}</p>}</div><div className="setting-control">{children}</div></div>
}

export function SettingsPage({ snapshot, updatePanel, save, chooseStorage, exportHistory, eraseHistory, clearAgentPermissions, models, saveSquadron, saveContext, measureContext, repair, retry, busy, conversation, saveConversationPrompt }: Props) {
  const [settings, setSettings] = useState(snapshot.settings)
  const [accentHex, setAccentHex] = useState(snapshot.settings.customAccent)
  const validAccent = /^#[0-9a-f]{6}$/i.test(accentHex)
  const [searchTest, setSearchTest] = useState('')
  const [checkingSearch, setCheckingSearch] = useState(false)
  const [logs, setLogs] = useState<string[] | null>(null)
  const [logError, setLogError] = useState('')
  const [saving, setSaving] = useState(false)
  const generalSettings = {
    ...settings,
    modelDirectory: snapshot.settings.modelDirectory,
    customMcp: snapshot.settings.customMcp,
    agentCommands: snapshot.settings.agentCommands,
    agentAutoRead: snapshot.settings.agentAutoRead,
    agentAutoCreate: snapshot.settings.agentAutoCreate,
    squadronEnabled: snapshot.settings.squadronEnabled,
    squadronPreferWorkers: snapshot.settings.squadronPreferWorkers,
    squadronMaxSlots: snapshot.settings.squadronMaxSlots,
  }
  const dirty = JSON.stringify(generalSettings) !== JSON.stringify(snapshot.settings)
  const field = <K extends keyof Settings>(key: K, value: Settings[K]) => setSettings(current => ({ ...current, [key]: value }))
  const loadLogs = async () => {
    try { setLogs(await invoke<string[]>('diagnostic_logs')); setLogError('') }
    catch (error) { setLogError(errorText(error)) }
  }
  return <div className="settings page-content">
    <div className="page-heading"><h1>Settings</h1></div>
    {updatePanel}
    <section className="settings-group">
      <h2><ShieldCheck size={19} /> Agent permissions</h2>
      <SettingsRow label="Remembered approvals" detail="Clear remembered workspace permissions and Squadron session approval. Edits, moves, deletions, and custom commands always ask.">
        <button type="button" className="button secondary" disabled={busy} onClick={clearAgentPermissions}>Clear approvals</button>
      </SettingsRow>
    </section>
    <AgentTools programs={snapshot.settings.agentCommands} autoRead={snapshot.settings.agentAutoRead} autoCreate={snapshot.settings.agentAutoCreate} busy={busy}
      savePrograms={agentCommands => save({ ...snapshot.settings, agentCommands })}
      setAutoRead={agentAutoRead => save({ ...snapshot.settings, agentAutoRead })}
      setAutoCreate={agentAutoCreate => save({ ...snapshot.settings, agentAutoCreate })} />
    <SquadronSettings key={`${snapshot.settings.squadronEnabled}-${snapshot.settings.squadronMaxSlots}-${snapshot.settings.squadronPreferWorkers}`} snapshot={snapshot} busy={busy} save={saveSquadron} />
    {models}
    <ContextSettings snapshot={snapshot} busy={busy} save={saveContext} measure={measureContext} />
    <ConversationPrompt key={conversation?.id} conversation={conversation} busy={busy} save={saveConversationPrompt} />
    <form onSubmit={async event => {
      event.preventDefault()
      setSaving(true)
      if (settings.accent !== 'custom' || validAccent) await save(generalSettings)
      setSaving(false)
    }}>
      <section className="settings-group">
        <h2><MessageSquare size={19} /> System prompt for new chats</h2>
        <label className="field"><span>Default instructions</span>
          <textarea rows={3} maxLength={8000} value={settings.defaultSystemPrompt}
            onChange={event => field('defaultSystemPrompt', event.target.value)} placeholder="Optional instructions for every new conversation" />
        </label>
        <p className="settings-note">Saved changes apply when a conversation is created.</p>
      </section>
      <section className="settings-group">
        <h2><Monitor size={19} /> General</h2>
        <SettingsRow label="Appearance"><Choice label="Appearance" value={settings.theme} options={[
          { value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }, { value: 'system', label: 'System' },
        ]} onChange={value => field('theme', value)} /></SettingsRow>
        <SettingsRow label="Accent color"><Choice label="Accent color" value={settings.accent} options={[
          { value: 'blue', label: 'Blue' }, { value: 'violet', label: 'Violet' }, { value: 'teal', label: 'Teal' }, { value: 'custom', label: 'Custom' },
        ]} onChange={value => field('accent', value)} /></SettingsRow>
        {settings.accent === 'custom' && <SettingsRow label="Custom accent">
          <div className="accent-picker">
            <input type="color" aria-label="Pick a custom accent color" value={settings.customAccent} onChange={event => {
              setAccentHex(event.target.value)
              field('customAccent', event.target.value)
            }} />
            <input type="text" aria-label="Custom accent hex color" aria-invalid={!validAccent} maxLength={7} value={accentHex}
              onChange={event => {
                const value = event.target.value
                setAccentHex(value)
                if (/^#[0-9a-f]{6}$/i.test(value)) field('customAccent', value)
              }} />
          </div>
        </SettingsRow>}
        <SettingsRow label="Chat text size"><Choice label="Chat text size" value={settings.chatTextSize} options={[
          { value: 'small', label: 'Small' }, { value: 'medium', label: 'Medium' }, { value: 'large', label: 'Large' },
        ]} onChange={value => field('chatTextSize', value)} /></SettingsRow>
        <SettingsRow label="Reduce motion"><input type="checkbox" className="switch" aria-label="Reduce motion" checked={settings.reducedMotion} onChange={event => field('reducedMotion', event.target.checked)} /></SettingsRow>
        <SettingsRow label="Default model"><Choice label="Default model" value={settings.defaultModel ?? ''} options={[
          { value: '', label: 'None' }, ...snapshot.models.map(model => ({ value: model.id, label: model.name, detail: model.quantization ?? undefined })),
        ]} onChange={value => field('defaultModel', value || null)} /></SettingsRow>
        <SettingsRow label="Load default model on startup" detail="Automatically start the local server when Parallax opens."><input type="checkbox" className="switch" aria-label="Load default model on startup" checked={settings.autoload} onChange={event => field('autoload', event.target.checked)} /></SettingsRow>
      </section>
      <details className="settings-group advanced">
        <summary><Cpu size={19} /> Advanced inference <ChevronDown size={16} /></summary>
        <p className="settings-note">Reload the model to apply acceleration changes.</p>
        <SettingsRow label="Acceleration" detail="Automatic selects a compatible backend and can fall back to CPU.">
          <Choice label="Acceleration" value={settings.acceleration} options={[
            { value: 'auto', label: 'Automatic' }, { value: 'cpu', label: 'CPU' }, { value: 'metal', label: 'Metal' },
            { value: 'cuda', label: 'CUDA' }, { value: 'vulkan', label: 'Vulkan' },
          ]} onChange={value => field('acceleration', value)} />
        </SettingsRow>
        <SettingsRow label="Maximum output tokens" detail="Sets a cap on replies, including reasoning.">
          <input aria-label="Maximum output tokens" type="number" min={16} max={65536} value={settings.maxTokens} onChange={event => field('maxTokens', Number(event.target.value))} />
        </SettingsRow>
        <SettingsRow label="Temperature"><input aria-label="Temperature" type="number" min={0} max={2} step={0.05} value={settings.temperature} onChange={event => field('temperature', Number(event.target.value))} /></SettingsRow>
        <SettingsRow label="Top-p"><input aria-label="Top-p" type="number" min={0.01} max={1} step={0.01} value={settings.topP} onChange={event => field('topP', Number(event.target.value))} /></SettingsRow>
        <SettingsRow label="Top-k"><input aria-label="Top-k" type="number" min={0} max={200} value={settings.topK} onChange={event => field('topK', Number(event.target.value))} /></SettingsRow>
        <SettingsRow label="GPU layers" detail="Leave empty to let llama.cpp decide how many layers to offload.">
          <input aria-label="GPU layers" type="number" min={0} max={256} placeholder="Auto" value={settings.gpuLayers ?? ''} onChange={event => field('gpuLayers', event.target.value === '' ? null : Number(event.target.value))} />
        </SettingsRow>
        <button type="button" className="button ghost reset-button" onClick={() => setSettings(current => ({ ...current, acceleration: 'auto', maxTokens: 1024, temperature: 0.7, topP: 0.95, topK: 40, gpuLayers: null }))}><RotateCcw size={15} /> Reset inference defaults</button>
      </details>
      <div className="settings-save"><button className="button primary" disabled={!dirty || saving || (settings.accent === 'custom' && !validAccent)}>{saving ? 'Saving…' : 'Save changes'}</button></div>
    </form>
    <section className="settings-group">
      <h2><Wrench size={19} /> Runtime</h2>
      <SettingsRow label="Web search"><button type="button" className="button secondary" disabled={busy || checkingSearch} onClick={async () => {
        setCheckingSearch(true)
        setSearchTest('Testing the connection…')
        try { await invoke('test_web_search'); setSearchTest('Web search returned results.') }
        catch (error) { setSearchTest(errorText(error)) }
        finally { setCheckingSearch(false) }
      }}>Test web search</button></SettingsRow>
      {searchTest && <p className="settings-note" role="status">{searchTest}</p>}
      <SettingsRow label="llama.cpp" detail={snapshot.runtime.installed.length ? snapshot.runtime.installed.map(backendName).join(', ') : snapshot.runtime.previous.length ? `Pinned update available from ${snapshot.runtime.previous.join(', ')}` : 'Runtime not installed'}><span className="value-pill">{snapshot.runtime.release}</span></SettingsRow>
      <SettingsRow label="Server status" detail={snapshot.server.detail ?? undefined}><span className={`value-pill ${snapshot.server.phase === 'ready' ? 'good' : ''}`}>{snapshot.server.phase === 'ready' ? `Ready · ${backendName(snapshot.server.backend)}` : snapshot.server.phase === 'idle' ? 'Stopped' : snapshot.server.phase === 'error' ? 'Needs attention' : 'Loading model'}</span></SettingsRow>
      <SettingsRow label="Detected hardware" detail={snapshot.hardware.gpu ?? snapshot.hardware.cpu}><span className="value-pill">{snapshot.hardware.memory ? bytes(snapshot.hardware.memory) : 'Memory unavailable'}</span></SettingsRow>
      {snapshot.runtimeProgress && <ProgressView progress={snapshot.runtimeProgress} />}
      <div className="settings-actions">
        <button type="button" className="button secondary" disabled={busy || snapshot.server.phase === 'loading'} onClick={repair}>{snapshot.runtime.installed.length ? 'Repair runtime' : snapshot.runtime.previous.length ? 'Install runtime update' : 'Install runtime'}</button>
        <button type="button" className="button secondary" disabled={busy || snapshot.server.phase === 'loading' || !snapshot.server.modelId} onClick={retry}>Retry GPU acceleration</button>
        <button type="button" className="button ghost" onClick={() => logs ? setLogs(null) : void loadLogs()}>{logs ? 'Hide diagnostics' : 'View diagnostics'}</button>
      </div>
      {logError && <ErrorBanner text={logError} />}
      {logs && <div className="logs"><pre>{logs.map(line => line.replace(/^\[(\d+)\]/, (_, time: string) => new Date(Number(time) * 1000).toLocaleTimeString())).join('\n') || 'No events in this session.'}</pre>
        <button type="button" className="button ghost" onClick={() => void loadLogs()}>Refresh</button></div>}
    </section>
    <section className="settings-group"><h2>About</h2><About slots={snapshot.squadronSlots} /></section>
    <section className="settings-group">
      <h2><Database size={19} /> Local data</h2>
      <SettingsRow label="Application data folder" detail={snapshot.dataDirectory}><ShieldCheck size={18} className="muted" /></SettingsRow>
      <SettingsRow label="Model storage folder" detail={snapshot.modelDirectory}><button type="button" className="button secondary" onClick={chooseStorage}>Change folder</button></SettingsRow>
      <p className="settings-note">New downloads and imports use the selected folder. Existing models stay in their current locations.</p>
      <p className="settings-note">Chat history, settings, and model files are stored locally. A conversation export contains the text of your chats.</p>
      <div className="settings-actions">
        <button type="button" className="button secondary" onClick={exportHistory}>Export conversations</button>
        <button type="button" className="button ghost danger-text" disabled={busy} onClick={eraseHistory}>Delete conversations</button>
      </div>
    </section>
  </div>
}

function SquadronSettings({ snapshot, busy, save }: {
  snapshot: Snapshot
  busy: boolean
  save: Props['saveSquadron']
}) {
  const [enabled, setEnabled] = useState(snapshot.settings.squadronEnabled)
  const [preferWorkers, setPreferWorkers] = useState(snapshot.settings.squadronPreferWorkers)
  const [limit, setLimit] = useState(String(snapshot.settings.squadronMaxSlots))
  const [saving, setSaving] = useState(false)
  const dirty = enabled !== snapshot.settings.squadronEnabled || Number(limit) !== snapshot.settings.squadronMaxSlots
    || preferWorkers !== snapshot.settings.squadronPreferWorkers
  const ready = snapshot.server.phase === 'ready'
  const slots = snapshot.squadronSlots
  return <form className="settings-group" onSubmit={async event => {
    event.preventDefault()
    setSaving(true)
    try { await save({ ...snapshot.settings, squadronEnabled: enabled, squadronMaxSlots: Number(limit), squadronPreferWorkers: preferWorkers }) }
    finally { setSaving(false) }
  }}>
    <h2><Users size={19} /> Parallax Squadron</h2>
    <SettingsRow label="Enable Squadron" detail="Lets the model request permission to delegate independent tasks.">
      <input type="checkbox" className="switch" aria-label="Enable Squadron" checked={enabled} disabled={saving || busy} onChange={event => setEnabled(event.target.checked)} />
    </SettingsRow>
    <SettingsRow label="Maximum parallel workers" detail="llama.cpp checks how many slots fit on the selected backend when the model loads.">
      <Choice label="Maximum parallel workers" value={limit} disabled={!enabled || saving || busy}
        options={Array.from({ length: 7 }, (_, index) => {
          const value = String(index + 2)
          return { value, label: value === '8' ? 'Auto (up to 8)' : value }
        })} onChange={setLimit} />
    </SettingsRow>
    <SettingsRow label="Prioritize parallel workers" detail="Can use a smaller chat context with Recommended or Max to make room for more workers. Custom keeps its entered context.">
      <input type="checkbox" className="switch" aria-label="Prioritize parallel workers" checked={preferWorkers} disabled={!enabled || saving || busy} onChange={event => setPreferWorkers(event.target.checked)} />
    </SettingsRow>
    <SettingsRow label="Current capacity" detail={ready && slots < 2 && snapshot.settings.squadronEnabled
      ? 'The loaded model has one slot. A smaller context or model quantization may leave room for workers.'
      : undefined}>
      <span className="value-pill">{!snapshot.settings.squadronEnabled ? 'Disabled' : ready ? `${slots} ${slots === 1 ? 'slot' : 'slots'} · ${snapshot.activeContext?.toLocaleString() ?? '—'} tokens` : 'Load a model'}</span>
    </SettingsRow>
    <div className="settings-actions"><button className="button primary" disabled={busy || saving || (!dirty && !ready)}>
      {saving ? 'Saving…' : ready ? dirty ? 'Save and reload model' : 'Reload and check capacity' : 'Save Squadron settings'}
    </button></div>
  </form>
}

function ContextSettings({ snapshot, busy, save, measure }: {
  snapshot: Snapshot
  busy: boolean
  save: Props['saveContext']
  measure: Props['measureContext']
}) {
  const [selected, setSelected] = useState<string | null>(null)
  const models = snapshot.models.filter(model => model.availableOnDisk)
  const model = models.find(model => model.id === selected)
    ?? models.find(model => model.id === snapshot.server.modelId)
    ?? models.find(model => model.id === snapshot.settings.defaultModel)
    ?? models[0]
  const preferences = model && (snapshot.modelPreferences[model.id] ?? snapshot.defaultModelPreferences)
  return <section className="settings-group context-settings">
    <h2><Gauge size={19} /> Context tokens</h2>
    {model ? <>
      <SettingsRow label="Model">
        <Choice label="Model for context settings" value={model.id} disabled={busy} options={models.map(model => ({
          value: model.id, label: model.name, detail: [model.quantization, bytes(model.bytes)].filter(Boolean).join(' · '),
        }))} onChange={setSelected} />
      </SettingsRow>
      <ModelPreferencesForm key={JSON.stringify([model.id, preferences])} model={model} snapshot={snapshot} busy={busy}
        onSave={(preferences, reload) => {
          setSelected(model.id)
          return save(model.id, preferences, reload)
        }} onMeasure={custom => {
          setSelected(model.id)
          return measure(model.id, custom)
        }} />
    </> : <p className="settings-note">Download or import a model to check its context limits.</p>}
  </section>
}

function ConversationPrompt({ conversation, busy, save }: {
  conversation?: Conversation; busy: boolean; save: (prompt: string) => Promise<boolean>
}) {
  const [prompt, setPrompt] = useState(conversation?.systemPrompt ?? '')
  const [saving, setSaving] = useState(false)
  if (!conversation) return null
  return <form className="settings-group" onSubmit={async event => {
    event.preventDefault()
    setSaving(true)
    try { await save(prompt) } finally { setSaving(false) }
  }}>
    <h2><MessageSquare size={19} /> System prompt for this chat</h2>
    <div className="field">
      <textarea aria-label="System prompt for this chat" rows={3} maxLength={8000} value={prompt}
        onChange={event => setPrompt(event.target.value)} placeholder="Optional instructions for this conversation" />
    </div>
    <button className="button secondary" disabled={busy || saving || prompt === conversation.systemPrompt}>
      {saving ? 'Saving…' : 'Save chat prompt'}
    </button>
  </form>
}
