import { useState } from 'react'
import { open } from '@tauri-apps/plugin-dialog'
import { FolderOpen } from 'lucide-react'
import { Modal } from './ui'
import { errorText, type AgentApproval, type AgentDecision, type Project } from './types'

export function ProjectDialog({ project, onSave, onClose }: {
  project: Project | null
  onSave: (project: Project) => void
  onClose: () => void
}) {
  const [name, setName] = useState(project?.name ?? '')
  const [path, setPath] = useState(project?.path ?? '')
  const [error, setError] = useState('')
  return <Modal title="Select project" onClose={onClose}>
    <form className="project-form" onSubmit={event => {
      event.preventDefault()
      if (path && name.trim()) onSave({ name: name.trim(), path })
    }}>
      <label className="field"><span>Project name</span><input autoFocus maxLength={100} required value={name} onChange={event => setName(event.target.value)} /></label>
      <div className="project-folder">
        <button type="button" className="button secondary" onClick={async () => {
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
      <div className="modal-actions"><button type="button" className="button ghost" onClick={onClose}>Cancel</button><button className="button primary" disabled={!path || !name.trim()}>Use project</button></div>
    </form>
  </Modal>
}

export function ApprovalDialog({ request, decide }: { request: AgentApproval; decide: (decision: AgentDecision) => void }) {
  return <Modal title={request.action} onClose={() => decide('deny')}>
    <div className="approval-details">
      <dl><dt>{request.action === 'Call MCP tool' ? 'Server' : 'Project'}</dt><dd>{request.project}</dd>{request.command && <><dt>Command</dt><dd><code>{request.command}</code></dd></>}<dt>Target</dt><dd><code>{request.path}</code></dd></dl>
      {request.content !== null && <details><summary>{request.action === 'Call MCP tool' ? 'View arguments shared with the server' : request.action === 'Run configured command' ? 'View executable and arguments' : 'View proposed file contents'}</summary><pre>{request.content}</pre></details>}
      {request.action === 'Delete file' && <p className="warning-text">This permanently deletes the selected file.</p>}
    </div>
    <div className="modal-actions approval-actions">
      <button className="button ghost" onClick={() => decide('deny')}>Deny</button>
      {request.remember && <button className="button secondary" onClick={() => decide('kind')}>Allow this type</button>}
      <button className={`button ${request.remember ? 'primary' : 'danger'}`} onClick={() => decide('once')}>Allow once</button>
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
