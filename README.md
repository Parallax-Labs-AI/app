# Parallax App

Parallax App is a desktop workspace for running Parallax Labs models on your own device. It installs a verified llama.cpp runtime, downloads the model you choose, and streams replies into a local conversation. Once the runtime and model are installed, chat works offline.

## Using the app

1. Open Parallax App and select **Get started**.
2. Review the detected hardware and choose **Set up runtime**.
3. Choose a model size. After downloading it, open its options to check memory, choose a context size, or set reasoning.
4. Download the model, then choose **Load model**. Parallax opens chat after the runtime confirms that the model is ready.
5. To use a file you already have, select **Import GGUF**. Choose a compatible single-file chat GGUF with an embedded template.

Open **Settings → Models** and expand **Parallax 2 Mini** or **Parallax 2** to see quantizations and manage downloads. Use the picker beside Send to switch installed models. Hold Command on macOS or Ctrl on Windows and Linux while hovering over the picker to see quantization. Settings contains the default system prompt and the open conversation’s prompt. **Settings** includes appearance, a startup model, inference controls, runtime repair, diagnostics, and conversation export. Press **⌘N** on macOS or **Ctrl+N** on Windows and Linux for a new chat. Press **⌘K** or **Ctrl+K** to search. **Shift+Enter** inserts a line break in the composer.

## Development

Install Git, Node.js 22 LTS or later, stable Rust (the lockfile was resolved with Rust 1.100), and Tauri's platform prerequisites. These are development requirements. End users need the platform's standard webview and compatible graphics drivers for hardware acceleration. [Tauri prerequisites](https://tauri.app/start/prerequisites/)

```sh
npm ci
npm run tauri -- dev
npm run build
cargo test --locked --manifest-path src-tauri/Cargo.toml
```

The Rust tests that simulate HTTP transfers and local readiness need permission to bind loopback. The development Vite server runs on port 1420. Tauri supplies the native IPC bridge.

## Building and packaging

Run `npm ci` on the build host first. Bundles appear beneath `src-tauri/target/<target>/release/bundle/` when a target is specified, or `src-tauri/target/release/bundle/` for a native build.

| Target | Build host and prerequisites | Command |
| --- | --- | --- |
| macOS Apple Silicon | macOS 13.3 or newer with Xcode command-line tools | `npm run tauri -- build --target aarch64-apple-darwin --bundles app,dmg` |
| macOS Intel | Intel Mac with Xcode command-line tools, or a Mac with the Rust Intel target installed | `npm run tauri -- build --target x86_64-apple-darwin --bundles app,dmg` |
| macOS universal | Mac with both Rust Darwin targets installed | `npm run tauri -- build --target universal-apple-darwin --bundles app,dmg` |
| Windows 10/11 x64 | Windows x64 Visual Studio developer shell, C++ build tools and redistributable component, Windows SDK, and WebView2 | `npm run tauri -- build --target x86_64-pc-windows-msvc --bundles nsis` |
| Linux x64 | Ubuntu x64 with WebKitGTK 4.1, GTK, libxdo, librsvg, and the other Tauri build dependencies | `npm run tauri -- build --target x86_64-unknown-linux-gnu --bundles deb,appimage` |

The Windows build hook copies the redistributable Visual C++ x64 libraries from the build host into the installer for use by llama-server. Review the Microsoft redistribution terms for the build toolchain you use. [Microsoft deployment documentation](https://learn.microsoft.com/en-us/cpp/windows/deploying-native-desktop-applications-visual-cpp?view=msvc-170)

Build and test Linux packages on the oldest distribution you intend to ship. The provided llama.cpp Linux binaries come from upstream Ubuntu builds and need compatible system libraries. Validate the AppImage on clean target distributions with FUSE where applicable. Windows installer testing should include an offline machine with WebView2 preinstalled and a clean machine that needs the WebView2 bootstrapper. [Tauri distribution guide](https://tauri.app/distribute/)

## Runtime and model catalog

The runtime is pinned to **llama.cpp b12675**. [Upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b12675) · [Release attestations](https://github.com/ggml-org/llama.cpp/attestations/71542828)

| Platform | Runtime assets in the manifest | Selection |
| --- | --- | --- |
| macOS arm64 and x64 | `llama-b12675-bin-macos-arm64.tar.gz`, `llama-b12675-bin-macos-x64.tar.gz` | Metal when detected. CPU mode uses the same archive with accelerator offload disabled. |
| Windows x64 | `llama-b12675-bin-win-cpu-x64.zip`, `llama-b12675-bin-win-vulkan-x64.zip`, `llama-b12675-bin-win-cuda-12.4-x64.zip` and the matching CUDA DLL ZIP | CUDA if the driver probe qualifies, then Vulkan if CUDA is unavailable, or CPU. A failed accelerated load retries on CPU. |
| Linux x64 | `llama-b12675-bin-ubuntu-x64.tar.gz`, `llama-b12675-bin-ubuntu-vulkan-x64.tar.gz`, `llama-b12675-bin-ubuntu-cuda-12.8-x64.tar.gz` and the matching CUDA library tarball | CUDA if the driver probe qualifies, then a Vulkan 1.2 capable physical GPU or CPU. A failed accelerated load retries on CPU. |

The runtime manifest records the exact filenames, byte sizes and GitHub release SHA-256 digests. Installations from 0.1.0 pick up the bundled release’s `llama-fit-params` tool the next time Parallax sets up or loads the runtime. Parallax resumes interrupted runtime archives where the host honors HTTP range requests, verifies the full archive, and stages the server and supporting libraries before publishing the installation. The runtime supervisor watches a pipe held by the desktop app. If the app exits or the pipe closes, the supervisor terminates and reaps the server.

To prepare a runtime update, select a specific upstream release, check its CLI flags and server API, and update every supported entry in the manifest from the [GitHub Releases API](https://docs.github.com/en/rest/releases/releases#get-a-release-by-tag-name). Run the runtime, extraction, readiness, and streaming tests on each supported platform before shipping an app update. Settings offers **Install runtime update** when a newer app build pins a different release and an older runtime is present. Runtime repair reinstalls the current pinned release. Older versions remain in the app data directory until removed by maintenance.

The bundled catalog pins official Hugging Face commits and includes ten single-file GGUF variants across Parallax 2 Mini and Parallax 2. Both are public Apache 2.0 Qwen 3.5 repositories. The app uses the GGUF's embedded chat template and limits the context to the lower of the user's selected size and the model's declared limit. Parallax models default to direct replies. In each installed model’s options, choose **Fully disable reasoning**, **Reason less**, or **Max reasoning**. The lower mode budgets up to 256 reasoning tokens, within the global output limit. Reasoning streams into a collapsible block and stays separate from the answer in local history. [Parallax 2 Mini](https://huggingface.co/parallaxlabs/parallax-2-mini) · [Parallax 2](https://huggingface.co/parallaxlabs/parallax-2)

To update `src-tauri/data/models.json`:

1. Look up the official repository with `GET /api/models/parallaxlabs/<id>`. Check its access status, commit SHA, license and GGUF architecture.
2. List files with `GET /api/models/parallaxlabs/<id>/tree/<commit>?recursive=true`. For each full GGUF, record the exact filename, its quantization, byte size and the Git LFS `lfs.oid` SHA-256. Compare the tree file size and LFS size.
3. Add the immutable commit, app-facing name and description, known context length, architecture, license and verified variants. Run catalog validation and load at least one representative file with the pinned runtime. Expand the Rust compatibility check only after verifying a new architecture.
4. Use `coming_soon`, `incompatible` or `unavailable` with an explanatory `reason` when a compatible public file cannot be offered. For restricted repositories, describe the access requirement in the reason. Only public official files are downloadable by this app.

## Memory and context sizing

Parallax measures installed GGUF files with the pinned release’s `llama-fit-params --fit-print on --log-jsonl`. It uses llama.cpp’s estimates for model weights, context state, and compute buffers with the same GPU layer placement as the server. Device memory comes from llama.cpp’s backend report. Parallax also checks available system memory and accounts for shared allocations on Apple Silicon. [Pinned profiler implementation](https://github.com/ggml-org/llama.cpp/blob/b12675/tools/fit-params/fit-params.cpp) · [Pinned memory reporting code](https://github.com/ggml-org/llama.cpp/blob/b12675/common/fit.cpp)

Open **Settings → Context tokens** and select an installed model.

- **Recommended** searches up to 32,768 tokens with more memory reserved for the rest of the system. When available memory is tight, it checks a smaller context up to 4,096 tokens with the standard reserve.
- **Max** searches across the permitted model context in 256-token steps and keeps a smaller reserve. If the estimate fails at startup, Parallax measures and tries smaller contexts on the same backend before falling back to CPU. Settings shows the resulting startup-verified limit. Check memory again to reset that limit and get a fresh estimate.
- **Custom** tests the requested size when loading. **Allow above measured limit** bypasses the memory guard while preserving the model’s hard context limit.

The context includes the prompt, conversation history, reasoning, and reply. Maximum output tokens is a separate setting. Model-specific options apply per installed quantization, and changing context reloads an active model. **Check memory** runs the profiler against the selected GGUF. If a model is loaded, **Unload and check memory** frees that model’s allocations before measuring.

## Project structure

| Path | Responsibility |
| --- | --- |
| `src/` | React views, onboarding, chat, models, settings, and typed Tauri IPC |
| `src-tauri/src/app.rs` | Application service, model switching, transfer control, diagnostics |
| `src-tauri/src/hardware.rs`, `runtime.rs`, `server.rs` | Hardware probes, runtime installation, and authenticated server supervision |
| `src-tauri/src/catalog.rs`, `models.rs`, `transfer.rs` | Catalog validation, GGUF import, range downloads, file integrity |
| `src-tauri/src/db.rs`, `settings.rs`, `inference.rs` | SQLite migration and persistence, preference validation, streaming chat |
| `src-tauri/data/` | Pinned model catalog and runtime manifest |

## Local data and limitations

The database, default model folder, and runtime are under the app data folder shown in Settings. Changing the model folder applies to new downloads and imports. In-progress downloads resume in their original folder. A JSON export contains conversation text. Diagnostics retain recent app events for the current session and exclude raw server logs, prompts, and authentication secrets.

Only Rust connects to the loopback inference server. Requests authenticate with a secret generated per server launch. Prompt text stays within this local connection. The app uses HTTPS for its runtime and model downloads and can open official model links in the system browser. Model output renders as Markdown with HTML skipped and remote image loads disabled. Model responses are text and the app supplies no execution tools.

CUDA, Vulkan, and Intel Mac deployment still require native verification. Memory estimates reflect the runtime and available memory at the time of each check. System library and driver compatibility vary. The app accepts single-file little-endian GGUF v2/v3 chat models from the import compatibility list. Multi-shard, multimodal, and restricted repositories need additional support.

Chat supports CommonMark and GitHub Flavored Markdown, including highlighted code, tables, task lists, and links. External links and images open through a user click. Chat skips embedded HTML and does not load remote images automatically.

**Web search** starts enabled and can be turned off in the composer. While it’s enabled, Parallax prepares search in the background at launch and retries preparation when needed for a search. Parallax installs a checksum-verified standalone uv release in its own data folder and uses its `uvx` to prepare managed Python, install the pinned `free-search-mcp` package, and verify its MCP connection over stdio. System Python and PATH stay independent of the app’s environment. The model can request up to three successive public searches per reply. A search sends a short model-generated query to public providers. Parallax requests strict SafeSearch and keeps web search files in its own app data folder. [uv releases](https://github.com/astral-sh/uv/releases/tag/0.13.10) · [free-search-mcp 0.18.2](https://pypi.org/project/free-search-mcp/0.18.2/)

The **Thinking** toggle beside Web search disables thinking for the selected model’s next reply. Turning it back on selects lower reasoning. Each turn gives the model the computer’s current local date alongside the configured system prompt, including when Web search is off.

Hover over or focus the blue and gray ring below the composer to see the token count of saved user messages and replies. Parallax counts message text with the loaded model’s tokenizer. A new conversation starts at zero. System prompts, template overhead, and tool definitions still take context during inference. Settings offers accent colors and chat text sizes.

Choose **Agent** in the header, then **Select project** to name a project and choose its folder. Agent conversations keep their project assignment. The agent can list folders, read and write UTF-8 files, delete individual files, and use the built-in `pwd`, `ls`, `cat`, and `wc` commands within the project. Approve actions in the app dialog. Remembered read and command approvals last for that project conversation until you quit or clear them in **Settings → Agent permissions**. Web search is available in Chat mode.

**Settings → Agent tools** lets you auto-approve project reads and creation of new files. Creating or moving a file to an occupied path leaves that file intact. Edits, moves, and deletions require one-time approval. Custom Agent commands use a fixed executable and argument list from Settings, run from the selected project folder, and require one-time approval.

The **MCP** button beside the composer controls manages custom local stdio servers. Use an absolute executable path or app-managed `uvx` and enter arguments as a JSON array of strings. Entries start disabled. Enabling a server allows Parallax to launch it to discover tools. Each tool call asks for one-time approval and shows its arguments. Use servers and commands you trust, since their executables run with your account’s access. [MCP stdio transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)

The charcoal and periwinkle palette, official logo, and DM Sans typography follow the Parallax Labs website. DM Sans is bundled under the SIL Open Font License. [Parallax Labs](https://labs.parallaxstudios.xyz)

## Release checklist

- Build and install each target bundle on a clean system.
- Test first-run setup, a real official GGUF download, an interrupted download, a local import, streaming, persistence after restart, and closing while inference is active.
- Validate Metal, supported CUDA and Vulkan driver combinations, and CPU fallback on native hardware.
- Sign Windows installers and binaries and verify Authenticode.
- Sign macOS bundles with Developer ID, notarize them, staple the ticket, and verify with Gatekeeper.
- Verify Linux package dependencies and publish installer checksums.
- Review bundled licenses and publish the tested OS, architecture, and backend matrix.

See [Verification](docs/verification.md) for results from this workspace.
