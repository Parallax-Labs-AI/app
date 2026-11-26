import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Plug, Plus, Terminal, Trash2 } from 'lucide-react'
import { IconButton, Modal } from './ui'
import { errorText, type LocalProgram } from './types'

type ProgramEditorProps = {
  mcp?: boolean
  onSave: (program: LocalProgram) => Promise<boolean>
  onClose: () => void
}

function ProgramEditor({ mcp, onSave, onClose }: ProgramEditorProps) {
  const [name, setName] = useState('')
  const [command, setCommand] = useState('')
  const [args, setArgs] = useState('[]')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  return createPortal(<Modal title={mcp ? 'Add MCP server' : 'Add Agent command'} onClose={() => { if (!saving) onClose() }}>
    <form className="program-form" onSubmit={async event => {
      event.preventDefault()
      event.stopPropagation()
      setError('')
      try {
        const parsed: unknown = JSON.parse(args)
        if (!Array.isArray(parsed) || parsed.some(value => typeof value !== 'string')) throw Error('Enter arguments as a JSON array of strings.')
        setSaving(true)
        if (await onSave({ name: name.trim(), command: command.trim(), args: parsed, enabled: !mcp })) onClose()
        else setError('Could not save the entry. Check its name, executable path, and arguments.')
      } catch (failure) { setError(errorText(failure)) }
      finally { setSaving(false) }
    }}>
      <label className="field"><span>Name</span><input autoFocus value={name} onChange={event => setName(event.target.value)} maxLength={60} required /></label>
      <label className="field"><span>Executable</span><input value={command} onChange={event => setCommand(event.target.value)} placeholder={mcp ? 'Absolute path or uvx' : 'Absolute executable path'} maxLength={1024} required /></label>
      <label className="field"><span>Arguments (JSON array)</span><textarea rows={3} value={args} onChange={event => setArgs(event.target.value)} spellCheck={false} /></label>
      <p className="settings-note">{mcp ? 'Enabling a server lets Parallax start its executable locally. Each tool call asks for approval.' : 'Parallax runs the saved executable and arguments from the project folder after you approve. Local programs run with your account’s access.'}</p>
      {error && <p className="danger-text" role="alert">{error}</p>}
      <div className="modal-actions"><button type="button" className="button ghost" disabled={saving} onClick={onClose}>Cancel</button><button className="button primary" disabled={saving || !name.trim() || !command.trim()}>{saving ? 'Saving…' : 'Add'}</button></div>
    </form>
  </Modal>, document.body)
}

function ProgramList({ programs, busy, onChange }: { programs: LocalProgram[]; busy: boolean; onChange: (items: LocalProgram[]) => Promise<boolean> }) {
  return <ul className="program-list">
    {programs.map(program => <li key={program.name}>
      <span title={program.command}>{program.name}</span>
      <input className="switch" type="checkbox" aria-label={`Enable ${program.name}`} disabled={busy} checked={program.enabled}
        onChange={event => void onChange(programs.map(item => item.name === program.name ? { ...item, enabled: event.target.checked } : item))} />
      <IconButton title={`Remove ${program.name}`} disabled={busy} onClick={() => void onChange(programs.filter(item => item.name !== program.name))}><Trash2 size={14} /></IconButton>
    </li>)}
  </ul>
}

export function McpPicker({ programs, busy, onChange }: { programs: LocalProgram[]; busy: boolean; onChange: (items: LocalProgram[]) => Promise<boolean> }) {
  const [open, setOpen] = useState(false)
  const [adding, setAdding] = useState(false)
  const [saving, setSaving] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open || adding) return
    panel.current?.focus()
    const dismiss = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false) }
    window.addEventListener('pointerdown', dismiss)
    return () => window.removeEventListener('pointerdown', dismiss)
  }, [open, adding])
  const save = async (items: LocalProgram[]) => {
    setSaving(true)
    try { return await onChange(items) }
    finally { setSaving(false) }
  }
  const active = programs.filter(program => program.enabled).length
  return <div className="mcp-picker" ref={root} onKeyDown={event => {
    if (event.key === 'Escape' && open && !adding) { event.stopPropagation(); setOpen(false); button.current?.focus() }
  }}>
    <button type="button" className="composer-toggle" data-active={active > 0} aria-expanded={open} aria-label="Custom MCP servers" disabled={busy}
      ref={button} onClick={() => setOpen(!open)}><Plug size={15} /> MCP <ChevronDown size={12} /></button>
    {open && <div className="mcp-popout" role="group" aria-label="Custom MCP servers" tabIndex={-1} ref={panel}>
      <strong>Custom MCP servers</strong>
      {programs.length ? <ProgramList programs={programs} busy={busy || saving} onChange={save} /> : <p>No custom servers added.</p>}
      <button type="button" className="button ghost" disabled={busy || saving} onClick={() => { setOpen(false); setAdding(true) }}><Plus size={15} /> Add server</button>
    </div>}
    {adding && <ProgramEditor mcp onClose={() => { setAdding(false); setOpen(true) }} onSave={program => save([...programs, program])} />}
  </div>
}

export function AgentTools({ programs, autoRead, autoCreate, busy, savePrograms, setAutoRead, setAutoCreate }: {
  programs: LocalProgram[]
  autoRead: boolean
  autoCreate: boolean
  busy: boolean
  savePrograms: (items: LocalProgram[]) => Promise<boolean>
  setAutoRead: (enabled: boolean) => Promise<boolean>
  setAutoCreate: (enabled: boolean) => Promise<boolean>
}) {
  const [adding, setAdding] = useState(false)
  const [saving, setSaving] = useState(false)
  const save = async (items: LocalProgram[]) => {
    setSaving(true)
    try { return await savePrograms(items) }
    finally { setSaving(false) }
  }
  return <section className="settings-group agent-tools">
    <h2><Terminal size={19} /> Agent tools</h2>
    <label className="setting-row">
      <span><strong>Automatically approve file reads</strong><small>Includes listing folders inside the selected project.</small></span>
      <input type="checkbox" className="switch" checked={autoRead} disabled={busy || saving} onChange={async event => {
        const enabled = event.target.checked
        setSaving(true)
        try { await setAutoRead(enabled) }
        finally { setSaving(false) }
      }} />
    </label>
    <label className="setting-row">
      <span><strong>Automatically approve new files</strong><small>Creating a file this way cannot replace an existing file.</small></span>
      <input type="checkbox" className="switch" checked={autoCreate} disabled={busy || saving} onChange={async event => {
        const enabled = event.target.checked
        setSaving(true)
        try { await setAutoCreate(enabled) }
        finally { setSaving(false) }
      }} />
    </label>
    <h3>Custom commands</h3>
    {programs.length > 0 && <ProgramList programs={programs} busy={busy || saving} onChange={save} />}
    <button type="button" className="button secondary" disabled={busy || saving} onClick={() => setAdding(true)}><Plus size={15} /> Add command</button>
    {adding && <ProgramEditor onClose={() => setAdding(false)} onSave={program => save([...programs, program])} />}
  </section>
}
