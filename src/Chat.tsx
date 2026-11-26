import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { ArrowRight, ArrowUp, ArrowUpRight, BookOpen, Brain, ChevronRight, Code2, FileText, Globe, LoaderCircle, Pencil, RotateCcw, Square, TriangleAlert } from 'lucide-react'
import { invoke } from '@tauri-apps/api/core'
import Markdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import { ContextWheel } from './WorkspaceTools'
import { Choice } from './Choice'
import { greetingFor } from './greeting'
import { splitSearchSources } from './searchSources'
import type { ChatMode, Message, Project, SearchStatus, Snapshot, Thread, Turn } from './types'
import { Brand, CopyButton, ErrorBanner, IconButton, ProgressView } from './ui'

const examples = [
  { icon: FileText, title: 'Draft an email', prompt: 'Help me draft a short, friendly email to follow up after a meeting.' },
  { icon: Code2, title: 'Explain some code', prompt: 'Explain how a hash map works with a short TypeScript example.' },
  { icon: BookOpen, title: 'Learn something', prompt: 'Explain why the moon has phases in simple terms.' },
]

export interface Reply {
  chatId: string
  turn: Turn
  message: number | null
  text: string
  reasoning: string
  searchStatus: SearchStatus | null
}

function codeText(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(codeText).join('')
  if (node && typeof node === 'object' && 'props' in node) return codeText((node.props as { children?: ReactNode }).children)
  return ''
}

function MarkdownLink({ url, children }: { url?: string; children: ReactNode }) {
  const [error, setError] = useState('')
  if (url?.startsWith("#")) return <a className="markdown-link" href={url}>{children}</a>
  if (!url || !/^https?:\/\//i.test(url)) return <span>{children}</span>
  return <><a className="markdown-link" href={url} onClick={event => {
    event.preventDefault()
    void invoke('open_chat_link', { url }).catch(() => setError('Could not open link.'))
  }}>{children}</a>{error && <span role="alert"> {error}</span>}</>
}

function MessageMarkdown({ content }: { content: string }) {
  const id = useId()
  return <div className="markdown"><Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeHighlight, [rehypeKatex, { trust: false, maxExpand: 200, maxSize: 20 }]]} remarkRehypeOptions={{ clobberPrefix: `message-${id}-` }} skipHtml
    components={{
      img: ({ src, alt }) => <MarkdownLink url={typeof src === "string" ? src : undefined}>{alt || "Open image"}</MarkdownLink>,
      a: ({ children, href }) => <MarkdownLink url={href}>{children}</MarkdownLink>,
      pre: ({ children }) => {
        const code = Array.isArray(children) ? children[0] : children
        const language = code && typeof code === 'object' && 'props' in code
          ? (code.props as { className?: string }).className?.match(/language-([^ ]+)/)?.[1] : undefined
        return <div className="code-block"><div className="code-toolbar"><span>{language ?? 'Code'}</span><CopyButton code text={codeText(children).replace(/\n$/, '')} /></div><pre>{children}</pre></div>
      },
    }}>{content}</Markdown></div>
}

function SourceList({ sources }: { sources: ReturnType<typeof splitSearchSources>['sources'] }) {
  return <details className="chat-sources">
    <summary>
      <Globe size={15} className="sources-globe" aria-hidden="true" />
      <span>Sources</span>
      <span className="source-count" aria-label={`${sources.length} results`}>{sources.length}</span>
      <ChevronRight className="sources-chevron" size={15} aria-hidden="true" />
    </summary>
    <ul>
      {sources.map(({ url, host, title, snippet }) => <li key={url}>
        <MarkdownLink url={url}>
          <span className="source-site" title={url}>{title || host}</span>
          <span className="source-detail" title={url}>{title ? host : url.replace(/^https?:\/\//i, '')}</span>
          {snippet && <span className="source-snippet">{snippet}</span>}
          <ArrowUpRight className="source-arrow" size={14} aria-hidden="true" />
        </MarkdownLink>
      </li>)}
    </ul>
  </details>
}

function MessageBubble({ message, live, searchStatus, edit, regenerate }: {
  message: Message; live?: boolean; searchStatus?: SearchStatus | null; edit?: () => void; regenerate?: () => void
}) {
  const user = message.role === 'user'
  const oldBlock = !user && message.content.trimStart().startsWith('<think>')
    ? message.content.trimStart().slice('<think>'.length) : null
  const end = oldBlock?.indexOf('</think>') ?? -1
  const reasoning = [message.reasoning, oldBlock === null ? '' : end === -1 ? oldBlock : oldBlock.slice(0, end)].filter(Boolean).join('\n\n')
  const content = oldBlock === null ? message.content : end >= 0 ? oldBlock.slice(end + '</think>'.length).trimStart() : ''
  const { body, sources } = user ? { body: content, sources: [] } : splitSearchSources(content)
  return <article className={`message ${user ? 'from-user' : 'from-model'}`} aria-label={user ? 'Your message' : 'Model response'}>
    {!user && <div className="message-mark"><Brand /></div>}
    <div className="message-body">
      {!user && <span className="message-author">Parallax</span>}
      {searchStatus === 'searching' ? <div className="turn-search" role="status"><LoaderCircle size={15} className="spin" aria-hidden="true" /> Searching the web...</div>
        : (searchStatus === 'failed' || message.search_failed) && <div className="turn-search search-failed" role="status"><TriangleAlert size={15} aria-hidden="true" /> Failed to search</div>}
      {user ? <div className="user-content"><MessageMarkdown content={content} /></div> : <>
        {reasoning && <details className="reasoning-block">
          <summary><ChevronRight size={16} /> {live && !content.trim() ? 'Thinking…' : 'Reasoning'}</summary>
          <div className="reasoning-content"><MessageMarkdown content={reasoning} /></div>
        </details>}
        {body.trim() ? <MessageMarkdown content={body} /> : live && !reasoning && !sources.length ?
          <span role="status" className="working"><LoaderCircle size={17} className="spin" /> Generating reply…</span> : null}
        {sources.length > 0 && <SourceList sources={sources} />}
      </>}
      {!live && (content || reasoning) && <div className="message-actions">
        {content && <CopyButton text={content} />}
        {edit && <IconButton title="Edit message" onClick={edit}><Pencil size={15} /></IconButton>}
        {regenerate && <IconButton title="Regenerate reply" onClick={regenerate}><RotateCcw size={15} /></IconButton>}
      </div>}
    </div>
  </article>
}

export function Chat({ mode, project, projectControl, mcpControl, usedContext, snapshot, thread, reply, busy, compose, stop, chooseModels, reload, selectModel, toggleSearch, toggleThinking }: {
  mode: ChatMode
  project: Project | null
  projectControl: ReactNode
  mcpControl: ReactNode
  usedContext: number | null
  snapshot: Snapshot
  thread: Thread | null
  reply: Reply | null
  busy: boolean
  compose: (turn: Turn) => Promise<void>
  stop: () => void
  chooseModels: () => void
  reload: (id: string) => void
  selectModel: (id: string) => void
  toggleSearch: () => void
  toggleThinking: (id: string, enabled: boolean) => void
}) {
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState<number | null>(null)
  const [greeting, setGreeting] = useState(() => greetingFor(new Date(), navigator.language))
  const [revealQuantization, setRevealQuantization] = useState(false)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const sticky = useRef(true)
  const threadId = thread?.conversation.id
  const activeReply = reply && (!threadId || reply.chatId === threadId) ? reply : null
  const ready = snapshot.server.phase === 'ready'
  const loading = snapshot.server.phase === 'loading'
  const canCompose = ready && (mode === 'chat' || !!project)
  const messages = thread?.messages ?? []
  let visible = messages
  if (activeReply) {
    const turn = activeReply.turn
    if (turn.type === 'send') {
      const last = messages.at(-1)
      visible = last?.role === 'user'
        ? [...messages.slice(0, -1), { ...last, content: `${last.content}\n\n${turn.content}` }]
        : [...messages, { id: -1, role: 'user', content: turn.content, reasoning: '', search_failed: false }]
    } else if (turn.type === 'edit') {
      visible = messages.filter(message => message.id <= turn.message).map(message =>
        message.id === turn.message ? { ...message, content: turn.content } : message)
    } else {
      const lastUser = messages.findLastIndex(message => message.role === 'user')
      visible = messages.slice(0, lastUser + 1)
    }
    visible = [...visible, { id: activeReply.message ?? -2, role: 'assistant', content: activeReply.text, reasoning: activeReply.reasoning, search_failed: activeReply.searchStatus === 'failed' }]
  }
  const latest = visible.at(-1)

  useEffect(() => {
    sticky.current = true
    setDraft('')
    setEditing(null)
    if (scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight
  }, [threadId])
  useEffect(() => {
    if (sticky.current && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight
  }, [latest?.id, latest?.content, latest?.reasoning])
  useEffect(() => {
    const field = textarea.current
    if (field) {
      field.style.height = 'auto'
      field.style.height = `${Math.min(field.scrollHeight, 160)}px`
    }
  }, [draft])

  useEffect(() => {
    const update = () => setGreeting(greetingFor(new Date(), navigator.language))
    const timer = window.setInterval(update, 15_000)
    window.addEventListener('focus', update)
    return () => { window.clearInterval(timer); window.removeEventListener('focus', update) }
  }, [])

  useEffect(() => {
    const update = (event: MouseEvent | KeyboardEvent) =>
      setRevealQuantization(snapshot.hardware.os === 'macos' ? event.metaKey : event.ctrlKey)
    const clear = () => setRevealQuantization(false)
    window.addEventListener('keydown', update)
    window.addEventListener('keyup', update)
    window.addEventListener('pointerover', update)
    window.addEventListener('blur', clear)
    return () => {
      window.removeEventListener('keydown', update)
      window.removeEventListener('keyup', update)
      window.removeEventListener('pointerover', update)
      window.removeEventListener('blur', clear)
    }
  }, [snapshot.hardware.os])

  const modelOptions = snapshot.models.filter(model => model.availableOnDisk).map(model => ({
    value: model.id, label: model.name, hoverDetail: model.quantization ?? undefined,
  }))
  const activeModel = modelOptions.some(option => option.value === snapshot.server.modelId) ? snapshot.server.modelId ?? '' : ''

  const thinking = ready && !!activeModel && (snapshot.modelPreferences[activeModel] ?? snapshot.defaultModelPreferences).reasoning !== 'disabled'

  const submit = () => {
    if (!draft.trim() || !canCompose || busy) return
    sticky.current = true
    const content = draft.trim()
    const turn: Turn = editing === null ? { type: 'send', content } : { type: 'edit', message: editing, content }
    setDraft('')
    setEditing(null)
    void compose(turn)
  }

  return <div className="chat">
    <div ref={scroller} className={`chat-scroll ${visible.length ? '' : 'empty-scroll'}`} onScroll={event => {
      const el = event.currentTarget
      sticky.current = el.scrollHeight - el.clientHeight - el.scrollTop < 90
    }}>
      {visible.length === 0 ? <div className="chat-empty">
        <h1>{greeting}</h1>
        <p>{ready ? 'Start a conversation with your local model.' : 'Select a model below to start chatting.'}</p>
        {ready ? <div className="example-grid">{examples.map(({ icon: Icon, title, prompt }) => <button key={title} className="example" onClick={() => {
          setDraft(prompt)
          textarea.current?.focus()
        }}><Icon size={18} /><span>{title}</span><ArrowRight size={15} /></button>)}</div>
          : loading ? <div className="inline-notice"><LoaderCircle size={17} className="spin" /> Loading model…</div>
          : null}
        {snapshot.runtimeProgress && <ProgressView progress={snapshot.runtimeProgress} />}
      </div> : <div className="message-list">
        {visible.map(message => <MessageBubble key={message.id} message={message}
          live={!!activeReply && message.id === latest?.id}
          searchStatus={activeReply && message.id === latest?.id ? activeReply.searchStatus : null}
          edit={message.role === 'user' && !busy && ready ? () => { setEditing(message.id); setDraft(message.content); textarea.current?.focus() } : undefined}
          regenerate={message.role === 'assistant' && message.id === latest?.id && !busy && ready ? () => { sticky.current = true; void compose({ type: 'regenerate' }) } : undefined} />)}
        {latest?.role === 'user' && !busy && ready && <button className="button ghost regenerate" onClick={() => { sticky.current = true; void compose({ type: 'regenerate' }) }}><RotateCcw size={15} /> Regenerate reply</button>}
      </div>}
    </div>
    <div className="composer-region">
      {projectControl}
      {snapshot.server.detail && <div className="chat-status"><ErrorBanner text={snapshot.server.detail} />{snapshot.server.phase === 'error' && snapshot.server.modelId && <button className="button secondary" onClick={() => reload(snapshot.server.modelId!)}>Reload model</button>}</div>}
      <form className="composer" onSubmit={event => { event.preventDefault(); submit() }}>
        {editing !== null && <div className="editing-note">Editing this message replaces replies that follow it.<button className="text-link" type="button" onClick={() => { setEditing(null); setDraft('') }}>Cancel edit</button></div>}
        <div className="composer-input">
          <textarea ref={textarea} rows={1} value={draft} maxLength={60000} onChange={event => setDraft(event.target.value)}
            aria-label="Message" placeholder={mode === 'agent' && !project ? 'Select a project to get started' : ready ? 'Message Parallax…' : loading ? 'Model is loading…' : 'Load a model to send a message'}
            disabled={!canCompose} onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit() }
            }} />
        </div>
        <div className="composer-controls">
          <div className="composer-toggles">
          {mode === 'chat' && <button type="button" className="composer-toggle" aria-pressed={snapshot.settings.webSearch} disabled={busy}
            title="When enabled, the model can send search queries to public providers" onClick={toggleSearch}><Globe size={15} /> Web search</button>}
          <button type="button" className="composer-toggle" aria-pressed={thinking} disabled={busy || !ready || !activeModel}
            title="Thinking applies to this model’s next reply" onClick={() => toggleThinking(activeModel, !thinking)}><Brain size={15} /> Thinking</button>
          {mcpControl}
          </div>
          <Choice label="Chat model" className="composer-model" disabled={busy || loading}
            value={activeModel} revealDetail={revealQuantization}
            options={[...modelOptions, { value: 'settings', label: 'Models in Settings' }]}
            placeholder="Select model" onChange={id => {
              if (id === 'settings') chooseModels()
              else if (id !== activeModel || !ready) selectModel(id)
            }} />

          {activeReply ? <button type="button" className="send-button" aria-label="Stop generation" title="Stop generation" onClick={stop}><Square size={15} fill="currentColor" /></button>
            : <button type="submit" className="send-button" aria-label="Send message" title="Send message" disabled={!canCompose || busy || !draft.trim()}><ArrowUp size={21} /></button>}
        </div>
      </form>
      <div className="composer-footer">
        <span>AI can make mistakes. Check important information.</span>
        {snapshot.activeContext !== null && <ContextWheel used={usedContext ?? (!messages.length ? 0 : null)} total={snapshot.activeContext} />}
      </div>
    </div>
  </div>
}
