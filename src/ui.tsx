import { useEffect, useRef, useState, type ReactNode } from 'react'
import { writeText } from '@tauri-apps/plugin-clipboard-manager'
import { Check, Copy, LoaderCircle, X } from 'lucide-react'
import { bytes, type Progress } from './types'

export function Brand({ className = '' }: { className?: string }) {
  return <img className={`brand-image ${className}`} src="/brand/parallax-logo.png" alt="" draggable={false} />
}

export function BrandOrbit({ small = false }: { small?: boolean }) {
  return <div className={`brand-orbit ${small ? 'compact' : ''}`} aria-hidden="true">
    <span className="orbit-ring" /><span className="orbit-ring inner" /><span className="orbit-star" />
    <div className="orbit-mark"><Brand /></div>
  </div>
}

export function IconButton({ title, children, onClick, disabled, className = '' }: {
  title: string; children: ReactNode; onClick: () => void; disabled?: boolean; className?: string
}) {
  return <button type="button" title={title} aria-label={title} className={`icon-button ${className}`} onClick={onClick} disabled={disabled}>{children}</button>
}

export function CopyButton({ text, code = false }: { text: string; code?: boolean }) {
  const [status, setStatus] = useState<'idle' | 'copied' | 'error'>('idle')
  useEffect(() => {
    if (status === 'idle') return
    const timer = window.setTimeout(() => setStatus('idle'), 1700)
    return () => clearTimeout(timer)
  }, [status])
  const title = status === 'error' ? 'Clipboard unavailable' : status === 'copied' ? 'Copied' : code ? 'Copy code' : 'Copy message'
  return <IconButton title={title} className={code ? 'copy-code' : ''} onClick={() => {
    void writeText(text).then(() => setStatus('copied'), () => setStatus('error'))
  }}>{status === 'copied' ? <Check size={15} /> : status === 'error' ? <X size={15} /> : <Copy size={15} />}</IconButton>
}

const phases: Record<string, string> = {
  connecting: 'Connecting', downloading: 'Downloading', verifying: 'Verifying file', extracting: 'Installing runtime',
  importing: 'Copying model', paused: 'Paused', pausing: 'Pausing', canceling: 'Canceling', error: 'Download interrupted',
}
export function ProgressView({ progress }: { progress: Progress }) {
  const percent = progress.total > 0 ? Math.min(100, 100 * progress.bytes / progress.total) : 0
  const busy = ['connecting', 'verifying', 'extracting', 'importing'].includes(progress.phase)
  const eta = progress.eta == null ? null : progress.eta < 60 ? `${progress.eta}s left` : `${Math.ceil(progress.eta / 60)} min left`
  return <div className="progress-view">
    <div className="progress-caption"><span>{busy && <LoaderCircle className="spin" size={14} />}{phases[progress.phase] ?? progress.phase}</span><span>{progress.total > 0 && `${bytes(progress.bytes)} / ${bytes(progress.total)}`}</span></div>
    <progress aria-label={phases[progress.phase] ?? progress.phase} value={busy ? undefined : percent} max={100} />
    {(progress.speed != null || eta) && <div className="progress-meta">{progress.speed != null && `${bytes(progress.speed)}/s`}{progress.speed != null && eta && ' · '}{eta}</div>}
  </div>
}

export function Modal({ title, children, onClose, wide = false }: { title: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  const element = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const dialog = element.current
    dialog?.showModal()
    return () => dialog?.close()
  }, [])
  return <dialog ref={element} className={`modal ${wide ? "wide-modal" : ""}`} onCancel={event => { event.preventDefault(); onClose() }} aria-label={title}>
    <div className="modal-title"><h2>{title}</h2><IconButton title="Close dialog" onClick={onClose}><X size={19} /></IconButton></div>
    {children}
  </dialog>
}

export function ErrorBanner({ text, onDismiss }: { text: string; onDismiss?: () => void }) {
  return <div role="alert" className="error-banner"><span>{text}</span>{onDismiss && <IconButton title="Dismiss message" onClick={onDismiss}><X size={16} /></IconButton>}</div>
}
