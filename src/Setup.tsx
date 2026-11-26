import { useState } from 'react'
import { ArrowLeft, ArrowRight, Check, Cpu, Download, HardDrive, LoaderCircle, ShieldCheck } from 'lucide-react'
import { BrandOrbit, ErrorBanner, ProgressView } from './ui'
import { ModelSettings, type ModelSettingsProps } from './ModelSettings'
import { backendName, bytes } from './types'

export function Setup({ finish, getRuntime, ...library }: ModelSettingsProps & {
  finish: () => Promise<void>
  getRuntime: () => Promise<boolean>
}) {
  const { snapshot } = library
  const [step, setStep] = useState(snapshot.runtime.installed.length ? 2 : 0)
  const [installing, setInstalling] = useState(false)
  const loading = snapshot.server.phase === 'loading'
  return <div className="setup-page">
    <div className="setup-header"><span>Device setup</span><button className="button ghost" onClick={() => void finish()}>Set up later</button></div>
    <div className={`setup-stage ${step === 2 ? 'wide' : ''}`}>
      <ol className="step-list" aria-label="Setup steps">
        {['Welcome', 'Runtime', 'Model'].map((label, index) => <li key={label} aria-current={step === index ? 'step' : undefined} className={step === index ? 'current' : step > index ? 'passed' : ''}><span>{step > index ? <Check size={12} /> : index + 1}</span>{label}</li>)}
      </ol>
      {step === 0 && <div className="setup-welcome">
        <BrandOrbit />
        <h1>AI that runs on<br />your hardware.</h1>
        <p>Download a Parallax model to chat, write, and code on your device. After setup, inference works offline.</p>
        <button className="button primary hero-action" onClick={() => setStep(1)}>Get started <ArrowRight size={18} /></button>
        <div className="privacy-line"><ShieldCheck size={15} /> Conversations stay on this device</div>
      </div>}
      {step === 1 && <section className="setup-body">
        <div className="model-icon large"><Cpu size={27} /></div>
        <h1>Get your device ready</h1>
        <p>Parallax will install a verified llama.cpp runtime for this device. You’ll need an internet connection for the download.</p>
        <dl className="hardware-list">
          <div><dt>Platform</dt><dd>{{ macos: 'macOS', windows: 'Windows', linux: 'Linux' }[snapshot.hardware.os] ?? snapshot.hardware.os} · {snapshot.hardware.arch === 'aarch64' ? 'ARM64' : snapshot.hardware.arch === 'x86_64' ? 'x64' : snapshot.hardware.arch}</dd></div>
          <div><dt>Processor</dt><dd>{snapshot.hardware.cpu}</dd></div>
          <div><dt>System memory</dt><dd>{snapshot.hardware.memory ? bytes(snapshot.hardware.memory) : 'Unavailable'}</dd></div>
          <div><dt>Graphics</dt><dd>{snapshot.hardware.gpu ?? 'No compatible GPU detected'}</dd></div>
          <div><dt>Selected runtime</dt><dd>{backendName(snapshot.runtime.suggested)}</dd></div>
        </dl>
        {snapshot.runtimeProgress && <ProgressView progress={snapshot.runtimeProgress} />}
        <div className="setup-actions">
          <button className="button ghost" disabled={installing} onClick={() => setStep(0)}><ArrowLeft size={17} /> Back</button>
          <button className="button primary" disabled={installing} onClick={async () => {
            setInstalling(true)
            if (await getRuntime()) setStep(2)
            setInstalling(false)
          }}>{installing ? <LoaderCircle size={16} className="spin" /> : <Download size={16} />}{installing ? 'Setting up runtime' : 'Set up runtime'}</button>
        </div>
      </section>}
      {step === 2 && <section className="setup-body choice">
        <div className="model-icon large"><HardDrive size={25} /></div>
        <h1>Choose a model</h1>
        <p>Download a model and select Load model to start chatting.</p>
        {loading && <div className="inline-notice"><LoaderCircle className="spin" size={17} /> Loading the model into memory…</div>}
        {snapshot.runtimeProgress && <ProgressView progress={snapshot.runtimeProgress} />}
        {snapshot.server.detail && <ErrorBanner text={snapshot.server.detail} />}
        <ModelSettings {...library} setup />
        <div className="setup-actions">
          <button className="button ghost" disabled={loading} onClick={() => setStep(1)}><ArrowLeft size={17} /> Back</button>
          {snapshot.server.phase === 'ready' && <button className="button primary" onClick={() => void finish()}>Open chat <ArrowRight size={17} /></button>}
        </div>
      </section>}
    </div>
  </div>
}
