import { useId, useState } from 'react'
import { Cpu, Gauge, LoaderCircle, RotateCw } from 'lucide-react'
import { Choice } from './Choice'
import { ErrorBanner, Modal } from './ui'
import { backendName, errorText, mib, tokens, type MemoryProfile, type Measurement, type Model, type ModelPreferences, type Snapshot } from './types'

type FormProps = {
  model: Model
  snapshot: Snapshot
  onSave: (preferences: ModelPreferences, reload: boolean) => Promise<boolean>
  onMeasure: (custom: number | null) => Promise<void>
  onClose?: () => void
  busy?: boolean
}

export function ModelPreferencesForm({ model, snapshot, onClose, onSave, onMeasure, busy }: FormProps) {
  const radioGroup = useId()
  const initial = snapshot.modelPreferences[model.id] ?? snapshot.defaultModelPreferences
  const [preferences, setPreferences] = useState(initial)
  const [measuring, setMeasuring] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const profile = snapshot.memoryProfiles[model.id]
  const limit = profile?.modelLimit ?? Math.min(model.context ?? 32_768, 1_048_576)
  const max = profile?.maximum?.tokens
  const custom = preferences.contextMode === 'custom'
  const mounted = snapshot.server.phase === 'ready'
  const customInvalid = custom && (!Number.isInteger(preferences.customTokens) || preferences.customTokens < 512 || preferences.customTokens > limit)
  const blocked = custom && !preferences.allowOverBudget && (max === undefined || preferences.customTokens > max)
  const changedContext = initial.contextMode !== preferences.contextMode || (custom && (initial.customTokens !== preferences.customTokens || initial.allowOverBudget !== preferences.allowOverBudget))
  const reload = changedContext && mounted && snapshot.server.modelId === model.id
  const disabled = !!busy || measuring || saving || snapshot.server.phase === 'loading'
  const measurement = preferences.contextMode === 'most' ? profile?.maximum : custom
    ? profile?.custom?.tokens === preferences.customTokens ? profile.custom
      : profile?.maximum?.tokens === preferences.customTokens ? profile.maximum
        : profile?.recommended?.tokens === preferences.customTokens ? profile.recommended : null
    : profile?.recommended

  async function measure() {
    setError('')
    setMeasuring(true)
    try {
      await onMeasure(custom && !customInvalid ? preferences.customTokens : null)
    } catch (failure) { setError(errorText(failure)) }
    finally { setMeasuring(false) }
  }

  const form = <form className="model-options" onSubmit={async event => {
      event.preventDefault()
      setError('')
      setSaving(true)
      try {
        await onSave({
          ...preferences,
          reasoning: onClose ? preferences.reasoning : initial.reasoning,
          customTokens: custom ? preferences.customTokens : initial.customTokens,
          allowOverBudget: custom ? preferences.allowOverBudget : initial.allowOverBudget,
        }, reload)
      }
      catch (failure) { setError(errorText(failure)) }
      finally { setSaving(false) }
    }}>
      <section className="option-section">
        {onClose && <h3><Gauge size={19} /> Context size</h3>}
        <p>Choose how many tokens the model can use for the conversation and its reply.</p>
        <div className="token-options" role="radiogroup" aria-label="Context size">
          {([
            ['recommended', 'Recommended', profile?.recommended?.tokens, profile ? 'No context fits' : 'Check memory to calculate'],
            ['most', 'Max', max, profile ? 'No context fits' : 'Check memory to calculate'],
            ['custom', 'Custom', undefined, 'Set your own limit'],
          ] as const).map(([mode, title, count, detail]) =>
            <label className="token-option" key={mode}>
              <input type="radio" name={radioGroup} value={mode} checked={preferences.contextMode === mode} disabled={disabled}
                onChange={() => setPreferences(current => ({ ...current, contextMode: mode }))} />
              <span>{title}<small>{count === undefined ? detail : `${tokens(count)} tokens`}</small></span>
            </label>)}
        </div>
        {custom && <div className="custom-tokens">
          <label className="field"><span>Context tokens</span><input type="number" aria-label="Custom context tokens" min={512}
            max={preferences.allowOverBudget ? limit : Math.min(limit, max ?? limit)} step={1} required disabled={disabled} value={Number.isNaN(preferences.customTokens) ? '' : preferences.customTokens}
            onChange={event => setPreferences(current => ({ ...current, customTokens: event.target.valueAsNumber }))} /></label>
          <p>{max !== undefined ? `Measured limit is ${tokens(max)} tokens` : 'Check memory to find this device’s limit.'} · Model limit {tokens(limit)}</p>
          <label className="override-option"><input type="checkbox" disabled={disabled} checked={preferences.allowOverBudget}
            onChange={event => setPreferences(current => ({ ...current, allowOverBudget: event.target.checked }))} /> Allow above measured limit</label>
          {preferences.allowOverBudget && <p className="warning-text">A larger context can slow this device or prevent the model from loading.</p>}
        </div>}
      </section>
      <section className="option-section readout-section">
        <div className="readout-header"><h3><Cpu size={19} /> Memory estimate</h3>
          <button type="button" className="button secondary" disabled={disabled || (!!onClose && mounted)} onClick={() => void measure()}>
            {measuring ? <LoaderCircle size={15} className="spin" /> : <RotateCw size={15} />}{measuring ? 'Checking…' : mounted && !onClose ? 'Unload and check memory' : 'Check memory'}
          </button></div>
        {mounted && snapshot.server.modelId === model.id && snapshot.activeContext !== null && <p>
          Active context {tokens(snapshot.activeContext)} tokens · {backendName(snapshot.server.backend)}
        </p>}
        {mounted && snapshot.server.modelId === model.id && snapshot.server.backend === 'cpu' && snapshot.server.detail && <p>{snapshot.server.detail}</p>}
        {profile ? <MemoryReadout profile={profile} measurement={measurement ?? null} /> :
          <p>{mounted && onClose ? 'Unload the active model to check the available memory.' : 'llama.cpp measures the selected model against available GPU and system memory. Parallax checks again each time it loads.'}</p>}
      </section>
      {onClose && <section className="option-section">
        <h3>Reasoning</h3>
        <Choice label="Reasoning" value={preferences.reasoning} options={[
          { value: 'disabled', label: 'Fully disable reasoning', detail: 'Answer directly' },
          { value: 'low', label: 'Reason less', detail: 'Spend up to 256 tokens reasoning' },
          { value: 'max', label: 'Max reasoning', detail: 'Use the reply’s full output budget' },
        ]} onChange={reasoning => setPreferences(current => ({ ...current, reasoning }))} />
        <p>{model.imported ? 'The amount of control depends on the model’s chat template.' : 'Applies to the next reply.'}</p>
      </section>}
      {error && <ErrorBanner text={error} />}
      <div className="modal-actions">
        {onClose && <button type="button" className="button secondary" disabled={disabled} onClick={onClose}>Cancel</button>}
        <button className="button primary" disabled={disabled || customInvalid || blocked || (!onClose && !changedContext)}>{saving ? 'Saving…' : reload ? 'Save and reload' : onClose ? 'Save options' : 'Save context'}</button>
      </div>
    </form>
  return onClose ? <Modal wide title={`${model.name} options`} onClose={() => { if (!measuring && !saving) onClose() }}>{form}</Modal> : form
}

function MemoryReadout({ profile, measurement }: { profile: MemoryProfile; measurement: Measurement | null }) {
  const sample = measurement ?? profile.maximum ?? profile.minimum
  const gpu = sample?.rows.filter(row => row.kind === 'device' && row.self > 0) ?? []
  const systemRows = sample?.rows.filter(row => row.kind !== 'device' || profile.unified)
  const systemUsage = systemRows?.reduce((sum, row) => sum + row.self, 0)
  return <div className="memory-readout">
    <p><strong>{backendName(profile.backend)}</strong>{profile.unified ? ' · Shared system memory' : ''}{sample && ` · ${tokens(sample.tokens)} tokens measured`}</p>
    {gpu.map(row => <div className="memory-row" key={row.name}>
      <span>{row.description ?? row.name}{profile.unified ? ' GPU budget' : ' VRAM'}</span>
      <strong>{mib(row.self)} expected · {mib(row.free)} free{row.total !== null ? ` of ${mib(row.total)}` : ''}</strong>
      <small>Weights {mib(row.model)} · Context {mib(row.context)} · Compute {mib(row.compute)}</small>
      {row.free !== null && row.free > 0 && <progress aria-label={`${row.name} estimated memory use`} max={row.free} value={Math.min(row.self, row.free)} />}
    </div>)}
    {systemUsage !== undefined && <div className="memory-row">
      <span>{profile.unified ? 'Combined shared memory' : 'System memory'}</span>
      <strong>{mib(systemUsage)} expected · {mib(profile.systemAvailable)} available</strong>
      {systemRows && <small>Weights {mib(systemRows.reduce((sum, row) => sum + row.model, 0))} · Context {mib(systemRows.reduce((sum, row) => sum + row.context, 0))} · Compute {mib(systemRows.reduce((sum, row) => sum + row.compute, 0))}</small>}
    </div>}
    {profile.estimatedMaxBeforeStartup != null && profile.maximum && <p>
      Startup verified {tokens(profile.maximum.tokens)} tokens after the {tokens(profile.estimatedMaxBeforeStartup)}-token estimate failed.
    </p>}
    {!profile.maximum && <p>No measured context fits this backend’s available memory.</p>}
    {profile.maximum && !measurement && <p>Choose Check memory to measure the entered context. Parallax checks it again before loading.</p>}
  </div>
}
