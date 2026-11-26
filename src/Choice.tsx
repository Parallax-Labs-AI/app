import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { Check, ChevronDown } from 'lucide-react'

export interface ChoiceOption<T extends string> { value: T; label: string; detail?: string; hoverDetail?: string }

export function Choice<T extends string>({ label, value, options, onChange, disabled, className = '', revealDetail = false, placeholder = 'Choose' }: {
  label: string; value: T; options: ChoiceOption<T>[]; onChange: (value: T) => void
  disabled?: boolean; className?: string; revealDetail?: boolean; placeholder?: string
}) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const [open, setOpen] = useState(false)
  const [index, setIndex] = useState(0)
  useEffect(() => { if (disabled) setOpen(false) }, [disabled])

  const selected = options.find(option => option.value === value)
  const selectedIndex = Math.max(0, options.findIndex(option => option.value === value))

  useEffect(() => {
    if (!open || disabled) return
    list.current?.focus()
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open, disabled])
  useEffect(() => {
    if (open) list.current?.children[index]?.scrollIntoView({ block: 'nearest' })
  }, [open, index])

  const select = (option: ChoiceOption<T>) => {
    onChange(option.value)
    setOpen(false)
    button.current?.focus()
  }
  const navigate = (event: KeyboardEvent<HTMLUListElement>) => {
    if (event.key === 'Tab') { setOpen(false); button.current?.focus(); return }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter', ' ', 'Escape'].includes(event.key)) return
    event.preventDefault()
    if (event.key === 'Escape') {
      event.stopPropagation()
      setOpen(false)
      button.current?.focus()
    } else if (event.key === 'Enter' || event.key === ' ') {
      if (options[index]) select(options[index])
    } else {
      setIndex(current => {
        if (event.key === 'Home') return 0
        if (event.key === 'End') return options.length - 1
        return Math.max(0, Math.min(options.length - 1, current + (event.key === 'ArrowDown' ? 1 : -1)))
      })
    }
  }

  return <div className={`choice ${className}`} data-reveal-detail={revealDetail} ref={root}>
    <button ref={button} type="button" disabled={disabled} className="choice-trigger" aria-label={`${label}, ${selected?.label ?? placeholder}`}
      aria-haspopup="listbox" aria-expanded={open} aria-controls={open ? id : undefined}
      onClick={() => { setIndex(selectedIndex); setOpen(!open) }}
      onKeyDown={event => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
          event.preventDefault()
          setIndex(selectedIndex)
          setOpen(true)
        }
      }}>
      <span>{selected?.label ?? placeholder}{selected?.detail && <small>{selected.detail}</small>}</span><ChevronDown size={16} />{selected?.hoverDetail && <small className="choice-hover-detail" aria-hidden="true">{selected.hoverDetail}</small>}
    </button>
    {open && <ul ref={list} id={id} className="choice-list" tabIndex={-1} role="listbox" aria-label={label} aria-activedescendant={`${id}-${index}`} onKeyDown={navigate}>
      {options.map((option, position) =>
        <li id={`${id}-${position}`} key={option.value} role="option" aria-label={option.hoverDetail ? `${option.label}, ${option.hoverDetail}` : undefined} aria-selected={option.value === value}
          className={position === index ? 'highlighted' : ''} onPointerMove={() => setIndex(position)} onClick={() => select(option)}>
          <span>{option.label}{option.detail && <small>{option.detail}</small>}</span>{option.hoverDetail && <small className="choice-hover-detail" aria-hidden="true">{option.hoverDetail}</small>}{option.value === value && <Check size={16} />}
        </li>)}
    </ul>}
  </div>
}
