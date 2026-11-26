interface SearchSource {
  url: string
  host: string
  title?: string
  snippet?: string
}

const searchHeading = '**Search results**'
const fallbackIntro = 'I couldn’t produce an answer with a citation to the retrieved sources. Here are the results from the last successful search.'

function webSource(value: string): SearchSource | undefined {
  if (!/^https?:\/\/[^\s<>]+$/i.test(value)) return
  try {
    const url = new URL(value)
    if (url.username || url.password) return
    return { url: url.href, host: url.hostname.replace(/^www\./i, '') }
  } catch {
    return
  }
}

export function splitSearchSources(markdown: string): { body: string; sources: SearchSource[] } {
  if (!markdown.includes(searchHeading) && !markdown.includes(fallbackIntro)) {
    return { body: markdown, sources: [] }
  }

  const lines = markdown.trimEnd().split(/\r?\n/)
  let fence: string | undefined
  let start = -1
  let fallback = false
  let fallbackOffset = 0
  for (const [index, line] of lines.entries()) {
    const match = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line)
    if (match) {
      const [, marker, rest] = match
      if (!fence) fence = marker
      else if (marker[0] === fence[0] && marker.length >= fence.length && !rest.trim()) fence = undefined
      continue
    }
    if (fence) continue
    if (line === searchHeading && (index === 0 || !lines[index - 1].trim())) {
      start = index
      fallback = false
    } else if (line.endsWith(fallbackIntro)) {
      start = index
      fallback = true
      fallbackOffset = line.length - fallbackIntro.length
    }
  }

  const sources = new Map<string, SearchSource>()
  if (start !== -1) {
    const entries = lines.slice(start + 1).filter(line => line.trim())
    for (let index = 0; index < entries.length;) {
      const line = entries[index++]
      const link = fallback ? entries[index++]?.match(/^ {2}(https?:\/\/\S+)$/i)?.[1]
        : line.match(/^- <(https?:\/\/[^<>]+)>$/i)?.[1]
      const source = link && webSource(link)
      if (!source || (fallback && !line.startsWith('- '))) return { body: markdown, sources: [] }
      if (fallback) {
        source.title = line.slice(2)
        if (entries[index]?.startsWith('  ')) source.snippet = entries[index++].trim()
      }
      if (!sources.has(source.url)) sources.set(source.url, source)
    }
  }
  return sources.size
    ? { body: (fallback ? [...lines.slice(0, start), lines[start].slice(0, fallbackOffset), '', fallbackIntro] : lines.slice(0, start)).join('\n').trimEnd(), sources: [...sources.values()] }
    : { body: markdown, sources: [] }
}
