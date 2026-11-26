import { copyFile, mkdir, readdir } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

async function findCrtDir() {
  // 1. Native Windows Visual Studio Developer Shell
  if (process.env.VCToolsRedistDir) {
    const root = join(process.env.VCToolsRedistDir, 'x64')
    const entries = await readdir(root, { withFileTypes: true })
    const crt = entries.find(e => e.isDirectory() && /^Microsoft\.VC\d+\.CRT$/.test(e.name))
    if (crt) return join(root, crt.name)
  }

  // 2. macOS / Linux cargo-xwin cache fallback
  const xwinCache = process.env.XWIN_CACHE_DIR || join(homedir(), 'Library/Caches/cargo-xwin')
  const allFiles = await readdir(xwinCache, { recursive: true })
  const match = allFiles.find(p => p.includes('x64') && p.endsWith('vcruntime140.dll'))
  if (match) {
    return join(xwinCache, match, '..')
  }

  throw new Error('Could not locate x64 Visual C++ Redistributable DLLs in VCToolsRedistDir or cargo-xwin cache.')
}

const source = await findCrtDir()
const destination = fileURLToPath(new URL('../src-tauri/resources/runtime-crt/', import.meta.url))
const dependencies = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll']

await mkdir(destination, { recursive: true })
for (const file of dependencies) {
  await copyFile(join(source, file), join(destination, file))
}
