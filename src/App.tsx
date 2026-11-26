import { useCallback, useEffect, useRef, useState } from 'react'
import { Channel, invoke, isTauri } from '@tauri-apps/api/core'
import { open, save } from '@tauri-apps/plugin-dialog'
import { CircleHelp, Ellipsis, FolderOpen, LoaderCircle, Menu, MessageSquare, Plus, Search, Settings2, Terminal, X } from 'lucide-react'
import { Chat, type Reply } from './Chat'
import { ModelSettings } from './ModelSettings'
import { ModelPreferencesForm } from './ModelOptions'
import { SettingsPage } from './SettingsPage'
import { McpPicker } from './ToolSettings'
import { ApprovalDialog, ProjectDialog } from './WorkspaceTools'
import { Setup } from './Setup'
import { Brand, BrandOrbit, ErrorBanner, IconButton, Modal, ProgressView } from './ui'
import { errorText, type AgentApproval, type AgentDecision, type ChatMode, type Project, type Action, type CatalogModel, type ChatEvent, type Conversation, type Model, type ModelPreferences, type Page, type Snapshot, type Thread, type Turn } from './types'

type Dialog = { type: 'chat'; chat: Conversation } | { type: 'delete_chat'; chat: Conversation } | { type: 'delete_all' } | { type: 'delete_model'; model: Model }

export function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [catalog, setCatalog] = useState<CatalogModel[]>([])
  const [mode, setMode] = useState<ChatMode>('chat')
  const [project, setProject] = useState<Project | null>(null)
  const [selectingProject, setSelectingProject] = useState(false)
  const [approval, setApproval] = useState<AgentApproval | null>(null)
  const [usedContext, setUsedContext] = useState<number | null>(null)
  const [page, setPage] = useState<Page>('chat')
  const [search, setSearch] = useState('')
  const [chats, setChats] = useState<Conversation[]>([])
  const [thread, setThread] = useState<Thread | null>(null)
  const [reply, setReply] = useState<Reply | null>(null)
  const [dialog, setDialog] = useState<Dialog | null>(null)
  const [modelOptions, setModelOptions] = useState<Model | null>(null)
  const [savingDialog, setSavingDialog] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [drawer, setDrawer] = useState(false)
  const [sidebarMenu, setSidebarMenu] = useState(false)
  const [starting, setStarting] = useState(true)
  const [operating, setOperating] = useState(false)
  const selected = useRef<string | null>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const generation = useRef(false)
  const queuedStop = useRef(false)
  const activeSearch = useRef('')
  const firstSnapshot = useRef(false)
  const snapshotVersion = useRef(0)
  const historyVersion = useRef(0)

  const refresh = useCallback(async () => {
    const version = ++snapshotVersion.current
    const state = await invoke<Snapshot>('snapshot')
    if (version === snapshotVersion.current) {
      setSnapshot(state)
      if (!firstSnapshot.current) {
        firstSnapshot.current = true
        setPage(state.settings.onboarded ? 'chat' : 'setup')
      }
    }
    return state
  }, [])

  const refreshChats = useCallback(async (query = activeSearch.current) => {
    const version = ++historyVersion.current
    const list = await invoke<Conversation[]>('list_chats', { search: query })
    if (version === historyVersion.current && query === activeSearch.current) setChats(list)
    return list
  }, [])

  const openChat = useCallback(async (id: string) => {
    selected.current = id
    setThread(current => current?.conversation.id === id ? current : null)
    setPage('chat')
    setDrawer(false)
    setSidebarMenu(false)
    if (generation.current) return
    const next = await invoke<Thread>('get_thread', { id })
    if (selected.current === id) {
      setThread(next)
      setProject(next.project)
      setMode(next.project ? 'agent' : 'chat')
      setUsedContext(null)
      localStorage.setItem('parallax.activeChat', id)
    }
  }, [])

  const boot = useCallback(async () => {
    if (!isTauri()) {
      setStarting(false)
      return
    }
    setStarting(true)
    setError('')
    try {
      const [state, models, list] = await Promise.all([refresh(), invoke<CatalogModel[]>('model_catalog'), refreshChats()])
      setCatalog(models)
      if (state.settings.onboarded) {
        const previous = localStorage.getItem('parallax.activeChat')
        const chat = list.find(item => item.id === previous)
        if (chat) await openChat(chat.id)
      }
    } catch (failure) { setError(errorText(failure)) }
    finally { setStarting(false) }
  }, [openChat, refresh, refreshChats])

  useEffect(() => { void boot() }, [boot])
  const connected = snapshot !== null
  useEffect(() => {
    if (!connected) return
    const timer = window.setInterval(() => { void refresh().catch(() => undefined) }, 700)
    return () => clearInterval(timer)
  }, [connected, refresh])
  useEffect(() => {
    if (!connected) return
    activeSearch.current = search
    const timer = window.setTimeout(() => void refreshChats(search).catch(failure => setError(errorText(failure))), 150)
    return () => clearTimeout(timer)
  }, [connected, search, refreshChats])
  useEffect(() => {
    if (!snapshot) return
    const { theme, reducedMotion, accent, chatTextSize } = snapshot.settings
    const system = window.matchMedia('(prefers-color-scheme: light)')
    const update = () => {
      document.documentElement.dataset.theme = theme === 'system' ? system.matches ? 'light' : 'dark' : theme
      document.documentElement.dataset.accent = accent
      document.documentElement.dataset.chatSize = chatTextSize
      document.documentElement.dataset.motion = reducedMotion ? 'reduced' : 'default'
    }
    update()
    system.addEventListener('change', update)
    return () => system.removeEventListener('change', update)
  }, [snapshot?.settings.theme, snapshot?.settings.reducedMotion, snapshot?.settings.accent, snapshot?.settings.chatTextSize])

  useEffect(() => {
    if (page !== 'chat' || snapshot?.server.phase !== 'ready' || reply) return
    let current = true
    if (!thread?.messages.some(message => message.content.trim())) {
      setUsedContext(0)
      return
    }
    void invoke<number>('chat_context', { id: thread.conversation.id })
      .then(used => { if (current) setUsedContext(used) })
      .catch(() => { if (current) setUsedContext(null) })
    return () => { current = false }
  }, [page, thread, reply?.chatId, snapshot?.server.phase, snapshot?.server.modelId, snapshot?.activeContext])

  const navigate = (destination: Page) => {
    setPage(destination)
    setDrawer(false)
    setSidebarMenu(false)
    setError('')
  }

  const newChat = useCallback(() => {
    if (generation.current) { setPage('chat'); setDrawer(false); return }
    selected.current = null
    localStorage.removeItem('parallax.activeChat')
    setThread(null)
    setUsedContext(null)
    setPage('chat')
    setDrawer(false)
    setSidebarMenu(false)
    setError('')
  }, [])
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.defaultPrevented || dialog || modelOptions || selectingProject || approval) return
      if (event.key.toLowerCase() === 'n') { event.preventDefault(); newChat() }
      if (event.key.toLowerCase() === 'k') { event.preventDefault(); setDrawer(true); requestAnimationFrame(() => searchRef.current?.focus()) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [newChat, dialog, modelOptions, selectingProject, approval])

  const action: Action = async (name, args) => {
    setError('')
    try {
      await invoke(name, args)
      await refresh()
      return true
    } catch (failure) {
      setError(errorText(failure))
      void refresh().catch(() => undefined)
      return false
    }
  }

  const finish = async () => {
    if (!snapshot) return
    if (await action('save_settings', { settings: { ...snapshot.settings, onboarded: true } })) navigate('chat')
  }

  const load = async (id: string, retryGpu = false) => {
    setOperating(true)
    try { return await action('load_model', { id, retryGpu }) }
    finally { setOperating(false) }
  }
  const saveModelOptions = async (id: string, preferences: ModelPreferences, reload: boolean, onSaved?: () => void) => {
    setOperating(true)
    try {
      await invoke('save_model_preferences', { id, preferences })
      await refresh()
      onSaved?.()
      return reload ? await load(id) : true
    } finally { setOperating(false) }
  }

  const measureContext = async (id: string, custom: number | null, unload: boolean) => {
    setOperating(true)
    try {
      if (unload && snapshot?.server.phase === 'ready') {
        await invoke('unload_model')
        await refresh()
      }
      await invoke('check_model_memory', { id, custom })
      await refresh()
    } finally { setOperating(false) }
  }
  const prepareRuntime = async (repair = false) => {
    setError('')
    setNotice('')
    setOperating(true)
    try {
      const detail = await invoke<string | null>('setup_runtime', { repair })
      await refresh()
      if (detail) setNotice(detail)
      else if (repair) setNotice('Verified runtime is ready.')
      return true
    } catch (failure) { setError(errorText(failure)); return false }
    finally { setOperating(false) }
  }
  const importModel = async () => {
    try {
      const file = await open({ title: 'Import a local GGUF model', filters: [{ name: 'GGUF model', extensions: ['gguf'] }] })
      if (typeof file === 'string') await action('import_model', { source: file })
    } catch (failure) { setError(errorText(failure)) }
  }
  const chooseStorage = async () => {
    try {
      const directory = await open({ title: 'Choose a model storage folder', directory: true })
      if (typeof directory === 'string') await action('change_storage', { directory })
    } catch (failure) { setError(errorText(failure)) }
  }
  const exportHistory = async () => {
    try {
      const path = await save({ title: 'Export conversations', defaultPath: 'parallax-conversations.json', filters: [{ name: 'JSON', extensions: ['json'] }] })
      if (path && await action('export_chats', { path })) setNotice('Conversations exported.')
    } catch (failure) { setError(errorText(failure)) }
  }

  const saveConversationPrompt = async (prompt: string) => {
    if (!thread) return false
    const { id, title } = thread.conversation
    if (!await action('update_chat', { id, title, systemPrompt: prompt })) return false
    setThread(current => current?.conversation.id === id
      ? { ...current, conversation: { ...current.conversation, systemPrompt: prompt } }
      : current)
    await refreshChats().catch(failure => setError(errorText(failure)))
    return true
  }

  const compose = async (turn: Turn) => {
    if (generation.current || (mode === 'agent' && !project)) return
    generation.current = true
    queuedStop.current = false
    setError('')
    let id = selected.current
    const origin = id
    try {
      if (!id) {
        const created = await invoke<Conversation>('create_chat')
        id = created.id
        if (selected.current === origin) {
          selected.current = id
          localStorage.setItem('parallax.activeChat', id)
          setThread({ conversation: created, messages: [], project: mode === 'agent' ? project : null })
        }
      }
      const chatId = id
      setReply({ chatId, turn, message: null, text: '', reasoning: '', searchStatus: null })
      const events = new Channel<ChatEvent>()
      events.onmessage = (event) => {
        if (event.type === 'tool_notice') { setNotice(event.text); return }
        if (event.type === 'agent_approval') {
          setApproval(event.request)
          return
        }
        if (event.type === 'started' && queuedStop.current) void invoke('stop_chat', { id: chatId }).catch(() => undefined)
        setReply(current => {
          if (!current || current.chatId !== chatId) return current
          if (event.type === 'started') return { ...current, message: event.message }
          if (event.type === 'search_status') return { ...current, searchStatus: event.status }
          if (event.type === 'reasoning') return { ...current, reasoning: current.reasoning + event.text }
          return { ...current, text: current.text + event.text }
        })
      }
      await invoke('chat', { id: chatId, turn, project: mode === 'agent' ? project : null, events })
    } catch (failure) { setError(errorText(failure)) }
    finally {
      try {
        if (id && selected.current === id) {
          const saved = await invoke<Thread>('get_thread', { id })
          if (selected.current === id) setThread(saved)
        }
        await refreshChats()
        await refresh()
      } catch (failure) { setError(errorText(failure)) }
      setReply(null)
      setApproval(null)
      generation.current = false
    }
  }

  const decide = async (decision: AgentDecision) => {
    if (!approval) return
    setApproval(null)
    try { await invoke('agent_decide', { id: approval.id, decision }) }
    catch (failure) { setError(errorText(failure)) }
  }

  const switchMode = (next: ChatMode) => {
    if (generation.current || mode === next) return
    newChat()
    setMode(next)
    if (next === 'agent') setProject(null)
  }

  const stop = () => {
    if (!reply) return
    queuedStop.current = true
    void invoke('stop_chat', { id: reply.chatId }).catch(failure => setError(errorText(failure)))
  }

  if (starting || !snapshot) return <div className="boot-screen"><BrandOrbit /><h1>Parallax App</h1>
    {starting ? <div role="status" className="working"><LoaderCircle size={18} className="spin" /> Starting Parallax…</div>
      : isTauri() ? <><ErrorBanner text={error || 'Unable to open local application data.'} /><button className="button secondary" onClick={() => void boot()}>Retry</button></>
      : <p>Open Parallax App from your desktop to access local models and conversations.</p>}
  </div>

  const busy = reply !== null || operating
  const historyButton = (chat: Conversation) => <div key={chat.id} className={`history-item ${page === 'chat' && chat.id === selected.current ? 'selected' : ''}`}>
    <button className="history-open" disabled={busy && chat.id !== selected.current} title={chat.title} onClick={() => void openChat(chat.id).catch(failure => setError(errorText(failure)))}><MessageSquare size={15} /><span>{chat.title}</span></button>
    <IconButton title={`Options for ${chat.title}`} onClick={() => setDialog({ type: 'chat', chat })} disabled={busy}><Ellipsis size={16} /></IconButton>
  </div>
  const library = { snapshot, catalog, action, load, importModel, confirmDelete: (model: Model) => setDialog({ type: 'delete_model', model }), configureModel: setModelOptions, busy }

  return <div className="app-shell" onKeyDown={event => {
    if (event.key === 'Escape') { setDrawer(false); setSidebarMenu(false) }
  }}>
    {drawer && <button className="drawer-backdrop" aria-label="Close sidebar" onClick={() => setDrawer(false)} />}
    <aside className={`sidebar ${drawer ? 'drawer-open' : ''}`}>
      <div className="sidebar-brand"><Brand /><div><strong>Parallax</strong><span>PARALLAX LABS</span></div></div>
      <div className="side-top">
        <button className="new-chat" disabled={!!reply && page === 'chat'} onClick={newChat}>{reply && page !== 'chat' ? <MessageSquare size={19} /> : <Plus size={19} />} {reply && page !== 'chat' ? 'Back to chat' : 'New chat'} <kbd>{navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'} N</kbd></button>
        <div className="search">
          <Search size={16} /><input ref={searchRef} aria-label="Search conversations" placeholder="Search chats" value={search} maxLength={120} onChange={event => setSearch(event.target.value)} />
          {search && <IconButton title="Clear search" onClick={() => setSearch('')}><X size={14} /></IconButton>}
        </div>
        <nav aria-label="Main navigation">
          {!snapshot.settings.onboarded && page !== 'setup' && <button className="nav-link" onClick={() => navigate('setup')}><CircleHelp size={18} /> Device setup</button>}
        </nav>
      </div>
      <div className="history">
        <h2>{search ? 'Search results' : 'Conversations'}</h2>
        {chats.length ? chats.map(historyButton) : <p>{search ? 'No matching conversations.' : 'Your chats will appear here.'}</p>}
      </div>
      <div className="sidebar-footer">
        <button className={`nav-link ${page === 'settings' ? 'active' : ''}`} aria-current={page === 'settings' ? 'page' : undefined} onClick={() => navigate('settings')}><Settings2 size={18} /> Settings</button>
      </div>
    </aside>

    <main className="workspace">
      {page !== 'setup' && <header className="workspace-header">
        <div className="header-leading">
          <IconButton className="mobile-menu" title="Open sidebar" onClick={() => setDrawer(true)}><Menu size={21} /></IconButton>
          {page === 'settings' && <span className="header-title">Settings</span>}
        </div>
        {page === 'chat' && <div className="mode-switch" role="group" aria-label="Conversation mode">
          <button type="button" aria-pressed={mode === 'chat'} disabled={busy} onClick={() => switchMode('chat')}><MessageSquare size={15} /> Chat</button>
          <button type="button" aria-pressed={mode === 'agent'} disabled={busy} onClick={() => switchMode('agent')}><Terminal size={15} /> Agent</button>
        </div>}
        <div className="header-trailing">
          {snapshot.server.phase === 'loading' && <span className="runtime-status"><LoaderCircle size={14} className="spin" /> Loading model</span>}
          {page === 'chat' && thread && <div className="conversation-menu">
            <IconButton title="Conversation actions" onClick={() => setSidebarMenu(!sidebarMenu)} disabled={busy}><Ellipsis size={21} /></IconButton>
            {sidebarMenu && <div className="dropdown align-right">
              <button onClick={() => { setSidebarMenu(false); setDialog({ type: 'chat', chat: thread.conversation }) }}>Edit conversation</button>
              <button className="danger-text" onClick={() => { setSidebarMenu(false); setDialog({ type: 'delete_chat', chat: thread.conversation }) }}>Delete conversation</button>
            </div>}
          </div>}
        </div>
      </header>}
      {page === 'setup' && <IconButton className="mobile-menu setup-menu" title="Open sidebar" onClick={() => setDrawer(true)}><Menu size={21} /></IconButton>}
      {(error || notice) && <div className="notification">
        {error && <ErrorBanner text={error} onDismiss={() => setError('')} />}
        {notice && <div className="notice" role="status">{notice}<IconButton title="Dismiss notification" onClick={() => setNotice('')}><X size={15} /></IconButton></div>}
      </div>}
      {page === 'chat' && <Chat key={mode} mode={mode} project={project} mcpControl={<McpPicker programs={snapshot.settings.customMcp} busy={busy}
        onChange={customMcp => action('save_settings', { settings: { ...snapshot.settings, customMcp } })} />} usedContext={usedContext} projectControl={mode === 'agent' ? <div className="project-bar">
        <button className="button secondary" disabled={busy || !!thread?.messages.length} onClick={() => setSelectingProject(true)}><FolderOpen size={16} /> {project?.name ?? 'Select project'}</button>
        {project && <span title={project.path}>{project.path}</span>}
      </div> : null} snapshot={snapshot} thread={thread} reply={reply} busy={busy} compose={compose} stop={stop} chooseModels={() => navigate('settings')} selectModel={id => void load(id)} toggleSearch={() => void action('save_settings', { settings: { ...snapshot.settings, webSearch: !snapshot.settings.webSearch } })} toggleThinking={(id, enabled) => void action('set_thinking', { id, enabled })} reload={id => void load(id)} />}
      {page === 'settings' && <div className="page-scroll"><SettingsPage snapshot={snapshot} save={settings => action('save_settings', { settings })} chooseStorage={() => void chooseStorage()} exportHistory={() => void exportHistory()}
        clearAgentPermissions={() => void invoke('clear_agent_permissions').then(() => setNotice('Agent approvals cleared.')).catch(failure => setError(errorText(failure)))}
        eraseHistory={() => setDialog({ type: 'delete_all' })} models={<ModelSettings {...library} />} saveContext={saveModelOptions} measureContext={(id, custom) => measureContext(id, custom, true)} conversation={thread?.conversation} saveConversationPrompt={saveConversationPrompt} repair={() => void prepareRuntime(snapshot.runtime.installed.length > 0)}
        retry={() => snapshot.server.modelId && void load(snapshot.server.modelId, true)} busy={busy} /></div>}
      {page === 'setup' && <Setup {...library} finish={finish} getRuntime={() => prepareRuntime()} load={async id => {
        const ready = await load(id)
        if (ready) await finish()
        return ready
      }} />}
    </main>

    {selectingProject && <ProjectDialog project={project} onSave={next => { setProject(next); setSelectingProject(false) }} onClose={() => setSelectingProject(false)} />}
    {approval && <ApprovalDialog key={approval.id} request={approval} decide={decision => void decide(decision)} />}
    {modelOptions && <ModelPreferencesForm key={modelOptions.id} model={modelOptions} snapshot={snapshot} busy={busy} onClose={() => setModelOptions(null)}
      onMeasure={custom => measureContext(modelOptions.id, custom, false)}
      onSave={(preferences, reload) => saveModelOptions(modelOptions.id, preferences, reload, () => setModelOptions(null))} />}
    {dialog && <Modal title={dialog.type === 'chat' ? 'Edit conversation' : dialog.type === 'delete_model' ? 'Delete model?' : 'Delete conversations?'} onClose={() => !savingDialog && setDialog(null)}>
      {dialog.type === 'chat' ? <form onSubmit={async event => {
        event.preventDefault()
        const fields = new FormData(event.currentTarget)
        setSavingDialog(true)
        if (await action('update_chat', { id: dialog.chat.id, title: fields.get('title') ?? dialog.chat.title, systemPrompt: dialog.chat.systemPrompt })) {
          await refreshChats().catch(failure => setError(errorText(failure)))
          if (selected.current === dialog.chat.id) await openChat(dialog.chat.id).catch(failure => setError(errorText(failure)))
          setDialog(null)
        }
        setSavingDialog(false)
      }}>
        <label className="field"><span>Title</span><input name="title" required maxLength={120} defaultValue={dialog.chat.title} autoFocus /></label>
        {error && <ErrorBanner text={error} />}
        <div className="modal-actions"><button type="button" className="button ghost danger-text" disabled={savingDialog} onClick={() => { setError(''); setDialog({ type: 'delete_chat', chat: dialog.chat }) }}>Delete conversation</button><button className="button primary" disabled={savingDialog}>Save</button></div>
      </form> : <>
        <p>{dialog.type === 'delete_model' ? `Remove ${dialog.model.name} from this device?` : dialog.type === 'delete_chat' ? `Permanently delete “${dialog.chat.title}” and its messages?` : 'Permanently delete all saved conversations from this device?'}</p>
        {error && <ErrorBanner text={error} />}
        <div className="modal-actions">
          <button className="button secondary" disabled={savingDialog} onClick={() => setDialog(null)}>Cancel</button>
          <button className="button danger" disabled={savingDialog} onClick={async () => {
            setSavingDialog(true)
            const ok = dialog.type === 'delete_model'
              ? await action('delete_model', { id: dialog.model.id })
              : await action('delete_chat', { id: dialog.type === 'delete_chat' ? dialog.chat.id : null })
            if (ok) {
              if (dialog.type === 'delete_all' || (dialog.type === 'delete_chat' && selected.current === dialog.chat.id)) {
                selected.current = null
                localStorage.removeItem('parallax.activeChat')
                setThread(null)
              }
              if (dialog.type !== 'delete_model') await refreshChats().catch(failure => setError(errorText(failure)))
              setDialog(null)
            }
            setSavingDialog(false)
          }}>Delete</button>
        </div>
      </>}
    </Modal>}
  </div>
}
