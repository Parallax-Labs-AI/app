import { useCallback, useEffect, useRef, useState } from 'react'
import { Channel, invoke, isTauri } from '@tauri-apps/api/core'
import { save } from '@tauri-apps/plugin-dialog'
import { Download, RefreshCw } from 'lucide-react'
import { bytes, errorText, type Progress } from './types'
import { ErrorBanner, ProgressView } from './ui'

interface UpdateCheck {
  current: string
  latest: string
  status: 'current' | 'available' | 'no_installer'
  installer: { name: string; size: number } | null
}

export function useAppUpdates() {
  const [result, setResult] = useState<UpdateCheck | null>(null)
  const [operation, setOperation] = useState<'checking' | 'downloading' | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState('')
  const working = useRef(false)

  const check = useCallback(async () => {
    if (!isTauri() || working.current) return
    working.current = true
    setOperation('checking')
    setError('')
    setSaved('')
    try { setResult(await invoke<UpdateCheck>('check_app_update')) }
    catch (failure) { setError(errorText(failure)); setResult(null) }
    finally { working.current = false; setOperation(null) }
  }, [])

  useEffect(() => { void check() }, [check])

  const download = async () => {
    if (!result?.installer || working.current) return
    working.current = true
    setOperation('downloading')
    setProgress(null)
    setError('')
    setSaved('')
    try {
      const filename = result.installer.name
      const extension = filename.slice(filename.lastIndexOf('.') + 1)
      const destination = await save({ title: 'Save Parallax installer', defaultPath: filename,
        filters: [{ name: 'Parallax installer', extensions: [extension] }] })
      if (!destination) return
      const events = new Channel<Progress>()
      events.onmessage = setProgress
      await invoke('download_app_update', { tag: result.latest, destination, progress: events })
      setSaved(destination)
    } catch (failure) { setError(errorText(failure)) }
    finally { working.current = false; setOperation(null); setProgress(null) }
  }

  return { result, operation, progress, error, saved, check, download }
}

export function AppUpdates({ updates }: { updates: ReturnType<typeof useAppUpdates> }) {
  const { result, operation, progress, error, saved, check, download } = updates
  const detail = operation === 'checking' ? 'Checking GitHub for the latest release…'
    : result?.status === 'available' ? `${result.latest} is available · ${bytes(result.installer?.size ?? null)}`
    : result?.status === 'no_installer' ? `${result.latest} is available, but has no verified installer for this device.`
    : result?.status === 'current' ? `Parallax ${result.current} is up to date.`
    : 'Check GitHub for the latest Parallax release.'
  return <section className="settings-group update-settings">
    <h2><Download size={19} /> App updates</h2>
    <p className="settings-note" role="status">{detail}</p>
    {result?.installer && <p className="settings-note update-filename">{result.installer.name}</p>}
    {progress && <ProgressView progress={progress} />}
    {error && <ErrorBanner text={error} />}
    {saved && <p className="settings-note update-filename" role="status">Installer saved to {saved}</p>}
    <div className="settings-actions">
      <button type="button" className="button secondary" disabled={!!operation} onClick={() => void check()}><RefreshCw size={15} /> Check for updates</button>
      {result?.installer && <button type="button" className="button primary" disabled={!!operation} onClick={() => void download()}><Download size={15} /> {operation === 'downloading' ? 'Downloading…' : 'Download installer'}</button>}
    </div>
  </section>
}
