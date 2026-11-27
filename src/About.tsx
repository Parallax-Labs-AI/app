import { useEffect, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { Brand } from './ui'

interface Versions { app: string; squadron: string }

export function About({ slots }: { slots: number }) {
  const [versions, setVersions] = useState<Versions | null>(null)
  useEffect(() => {
    let active = true
    void invoke<Versions>('app_versions').then(value => { if (active) setVersions(value) }).catch(() => undefined)
    return () => { active = false }
  }, [])
  return <div className="about-app">
    <div className="about-brand"><Brand /><strong>Parallax App</strong>{versions && <span>v{versions.app}</span>}</div>
    <dl>
      <dt>Parallax Squadron Framework</dt><dd>{versions ? `v${versions.squadron}` : '—'}</dd>
      <dt>Parallel slots</dt><dd>{slots ? slots.toLocaleString() : 'Load a model'}</dd>
    </dl>
  </div>
}
