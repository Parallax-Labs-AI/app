import { useState } from 'react'
import { open } from '@tauri-apps/plugin-dialog'
import { FolderOpen, MessageSquare } from 'lucide-react'
import { Modal } from './ui'
import { errorText, type AgentApproval, type AgentDecision, type Project, type ProjectConversation } from './types'

export function ProjectList({ entries, filter, selected, busy, open }: {
  entries: ProjectConversation[]
  filter: string
  selected?: string
  busy: boolean
  open: (id: string) => void
}) {
  const query = filter.trim().toLowerCase()
  const groups = new Map<string, ProjectConversation[]>()
  for (const entry of entries) {
    const { project, conversation } = entry
    if (!`${project.name} ${project.path} ${conversation.title}`.toLowerCase().includes(query)) continue
    const group = groups.get(project.path)
    if (group) group.push(entry)
    else groups.set(project.path, [entry])
  }
  if (!groups.size) return <p>{query ? 'No matching projects.' : 'Add a folder to start a project.'}</p>
  return <ul className="project-list">
    {[...groups].map(([path, group]) => {
      const active = group.find(entry => entry.conversation.id === selected)
      const { project, conversation } = active ?? group[0]
      return <li key={path}>
        <button className="project-item" aria-current={active ? 'true' : undefined} disabled={busy} onClick={() => open(conversation.id)}>
          <FolderOpen size={17} aria-hidden="true" />
          <span><strong>{project.name}</strong><small title={path}>{path}</small></span>
        </button>
        {group.length > 1 && <details className="project-conversations">
          <summary>{group.length} conversations</summary>
          {group.map(({ conversation }) => <button key={conversation.id} className="history-open" disabled={busy}
            aria-current={conversation.id === selected ? 'true' : undefined} title={conversation.title} onClick={() => open(conversation.id)}>
            <MessageSquare size={13} aria-hidden="true" /><span>{conversation.title}</span>
          </button>)}
        </details>}
      </li>
    })}
  </ul>
}

export function ProjectDialog({ project, onSave, onClose, saving }: {
  project: Project | null
  onSave: (project: Project) => Promise<void>
  saving: boolean
  onClose: () => void
}) {
  const [name, setName] = useState(project?.name ?? '')
  const [path, setPath] = useState(project?.path ?? '')
  const [error, setError] = useState('')
  return <Modal title="Select project" onClose={onClose}>
    <form className="project-form" onSubmit={async event => {
      event.preventDefault()
      if (!path || !name.trim() || saving) return
      setError('')
      try { await onSave({ name: name.trim(), path }) }
      catch (failure) { setError(errorText(failure)) }
    }}>
      <label className="field"><span>Project name</span><input disabled={saving} autoFocus maxLength={100} required value={name} onChange={event => setName(event.target.value)} /></label>
      <div className="project-folder">
        <button type="button" className="button secondary" disabled={saving} onClick={async () => {
          try {
            const folder = await open({ title: 'Choose project folder', directory: true })
            if (typeof folder !== 'string') return
            setPath(folder)
            setName(current => current.trim() ? current : folder.split(/[\\/]/).filter(Boolean).at(-1)?.slice(0, 100) ?? '')
            setError('')
          } catch (failure) { setError(errorText(failure)) }
        }}><FolderOpen size={17} /> Choose folder</button>
        {path && <span title={path}>{path}</span>}
      </div>
      {error && <p className="danger-text" role="alert">{error}</p>}
      <div className="modal-actions"><button type="button" className="button ghost" onClick={onClose} disabled={saving}>Cancel</button><button className="button primary" disabled={saving || !path || !name.trim()}>{saving ? 'Opening…' : 'Open project'}</button></div>
    </form>
  </Modal>
}

export function ApprovalDialog({ request, decide }: { request: AgentApproval; decide: (decision: AgentDecision) => void }) {
  return <Modal title={request.action} onClose={() => decide('deny')}>
    <div className="approval-details">
      <dl><dt>{request.session ? 'Framework' : request.action === 'Call MCP tool' ? 'Server' : 'Project'}</dt><dd>{request.project}</dd>{request.command && <><dt>Command</dt><dd><code>{request.command}</code></dd></>}<dt>Target</dt><dd><code>{request.path}</code></dd></dl>
      {request.content !== null && <details><summary>{request.session ? 'View assignments' : request.action === 'Call MCP tool' ? 'View arguments shared with the server' : request.action === 'Run configured command' ? 'View executable and arguments' : 'View proposed file contents'}</summary><pre>{request.content}</pre></details>}
      {request.action === 'Delete file' && <p className="warning-text">This permanently deletes the selected file.</p>}
    </div>
    <div className="modal-actions approval-actions">
      <button className="button ghost" onClick={() => decide('deny')}>Deny</button>
      {request.session && <button className="button secondary" onClick={() => decide('session')}>Allow for this session</button>}
      {request.remember && <button className="button secondary" onClick={() => decide('kind')}>Allow this type</button>}
      <button className={`button ${request.remember || request.session ? 'primary' : 'danger'}`} onClick={() => decide('once')}>Allow once</button>
    </div>
  </Modal>
}

export function ContextWheel({ used, total }: { used: number | null; total: number }) {
  const ratio = used === null ? 0 : used / total
  const percentage = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 }).format(ratio)
  const detail = used === null ? 'Context usage unavailable' : `${percentage} used · ${used.toLocaleString()}/${total.toLocaleString()} tokens`
  return <span className="context-wheel" tabIndex={0} aria-label={detail}>
    <span className="context-track" style={{ background: `conic-gradient(var(--context-blue) ${Math.min(100, ratio * 100)}%, var(--context-gray) 0)` }} aria-hidden="true" />
    <span className="context-tooltip" aria-hidden="true">{detail}</span>
  </span>
}
