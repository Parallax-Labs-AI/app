import { useState } from 'react'
import { ArrowDownToLine, ArrowUpRight, Box, ChevronDown, LoaderCircle, Plus, Settings2, Trash2 } from 'lucide-react'
import { openUrl } from '@tauri-apps/plugin-opener'
import { bytes, errorText, variantId, type Action, type CatalogModel, type Model, type Snapshot } from './types'
import { ErrorBanner, IconButton, ProgressView } from './ui'

export interface ModelSettingsProps {
  snapshot: Snapshot
  catalog: CatalogModel[]
  action: Action
  load: (id: string) => Promise<boolean>
  importModel: () => Promise<void>
  confirmDelete: (model: Model) => void
  configureModel: (model: Model) => void
  busy: boolean
  setup?: boolean
}

export function ModelSettings(props: ModelSettingsProps) {
  const { snapshot, catalog, importModel, setup } = props
  const [expanded, setExpanded] = useState<string | null>(null)
  return <section className="settings-group model-settings" id={setup ? undefined : 'model-settings'}>
    {!setup && <h2><Box size={19} /> Models</h2>}
    {catalog.map(model => <div className="model-family" key={model.id}>
      <button type="button" className="model-family-trigger" aria-expanded={expanded === model.id}
        onClick={() => setExpanded(expanded === model.id ? null : model.id)}>
        <strong>{model.name}</strong><ChevronDown size={17} />
      </button>
      {expanded === model.id && <div className="family-variants">
        {model.status === 'available' ? model.variants.map(variant =>
          <VariantRow key={variant.file} id={variantId(model, variant)} label={variant.quantization} size={variant.bytes} {...props} />
        ) : <p>{model.reason || 'A compatible GGUF is currently unavailable.'}</p>}
        <ModelSource repo={model.repo} />
      </div>}
    </div>)}
    {snapshot.models.filter(model => model.imported).map(model => <ModelRow key={model.id} model={model} {...props} />)}
    {snapshot.importing && <ProgressView progress={snapshot.importing} />}
    <button type="button" className="button ghost" onClick={() => void importModel()} disabled={!!snapshot.importing}><Plus size={16} /> Import GGUF</button>
  </section>
}

function ModelSource({ repo }: { repo: string }) {
  const [error, setError] = useState('')
  return <>
    <button type="button" className="text-link" onClick={() => void openUrl(`https://huggingface.co/${repo}`).catch(reason => setError(errorText(reason)))}>Model details <ArrowUpRight size={13} /></button>
    {error && <ErrorBanner text={error} onDismiss={() => setError('')} />}
  </>
}

function VariantRow({ id, label, size, ...props }: ModelSettingsProps & { id: string; label: string; size: number }) {
  const { snapshot, action } = props
  const installed = snapshot.models.find(model => model.id === id && model.availableOnDisk)
  const loaded = snapshot.server.phase === 'ready' && snapshot.server.modelId === id
  const transfer = snapshot.downloads.find(download => download.id === id)
  const active = !!transfer && !['paused', 'error'].includes(transfer.progress.phase)
  const waiting = transfer && ['pausing', 'canceling'].includes(transfer.progress.phase)
  return <div className="variant-row">
    <div className="variant-line">
      <div><strong>{label}</strong><small>{bytes(size)}{loaded ? ' · Loaded' : installed ? ' · Installed' : ''}</small></div>
      {installed ? <InstalledActions model={installed} {...props} /> : active ? <div className="installed-actions">
        <button className="button secondary" disabled={!!waiting} onClick={() => void action('control_download', { id, discard: false })}>Pause</button>
        <button className="button ghost" disabled={!!waiting} onClick={() => void action('control_download', { id, discard: true })}>Cancel</button>
      </div> : <div className="installed-actions">
        <button className="button secondary" onClick={() => void action('start_download', { id })}><ArrowDownToLine size={16} />{transfer?.progress.phase === 'paused' ? 'Resume' : transfer ? 'Retry' : 'Download'}</button>
        {transfer && <button className="button ghost" onClick={() => void action('control_download', { id, discard: true })}>Discard</button>}
      </div>}
    </div>
    {transfer && <ProgressView progress={transfer.progress} />}
    {transfer?.error && <ErrorBanner text={transfer.error} />}
  </div>
}

function ModelRow({ model, ...props }: ModelSettingsProps & { model: Model }) {
  return <div className="imported-row">
    <div className="model-icon"><Box size={20} /></div>
    <div className="imported-name"><strong>{model.name}</strong><p>{model.quantization ?? model.architecture} · {model.availableOnDisk ? bytes(model.bytes) : 'File missing'}</p></div>
    <InstalledActions model={model} {...props} />
  </div>
}

function InstalledActions({ model, snapshot, action, load, confirmDelete, configureModel, busy }: ModelSettingsProps & { model: Model }) {
  const id = model.id
  const active = snapshot.server.modelId === id
  const loading = snapshot.server.phase === 'loading'
  return <div className="installed-actions">
    {active && snapshot.server.phase === 'ready'
      ? <button className="button secondary" disabled={busy} onClick={() => void action('unload_model')}>Unload</button>
      : <button className="button primary" disabled={busy || loading || !model.availableOnDisk} onClick={() => void load(id)}>
        {active && loading && <LoaderCircle size={15} className="spin" />}{active && loading ? 'Checking and loading…' : 'Load model'}
      </button>}
    <IconButton title={`${model.name} options`} disabled={busy || loading || !model.availableOnDisk} onClick={() => configureModel(model)}><Settings2 size={17} /></IconButton>
    <IconButton title={`Delete ${model.name}`} disabled={busy || loading} onClick={() => confirmDelete(model)}><Trash2 size={16} /></IconButton>
  </div>
}
