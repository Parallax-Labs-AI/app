# Verification

## Automated checks

`npm run build` passed TypeScript checking and bundled the frontend. All 13 Rust tests passed on Apple Silicon. The suite covers runtime selection and extraction, catalog validation, resumable downloads and cancellation, readiness authentication, SQLite persistence and migration from 0.1.0, measured memory budgets and custom guards, and streamed UTF-8 reasoning and split `<think>` tags.

The Apple Silicon smoke test installed the pinned runtime into a clean app directory, resumed and verified the official Parallax 2 Mini Q3_K_M download, measured Metal memory and context limits, loaded the recommended context, and blocked an oversized custom context. It streamed responses with reasoning disabled and lowered, verified that **Max reasoning** could be stopped while streaming, and reopened SQLite with answers and reasoning intact. It also tested shutdown, CPU recovery after a simulated CUDA startup failure, and importing and removing a local GGUF.

To repeat the smoke test on Apple Silicon, build the app and download the official Q3_K_M fixture. This test uses a temporary directory and needs outbound HTTPS and local loopback.

```sh
npm run tauri -- build --bundles app
export PARALLAX_TEST_APP="$PWD/src-tauri/target/release/parallax-app"
export PARALLAX_TEST_GGUF="/absolute/path/to/parallax-2-mini.Q3_K_M.gguf"
cargo test --locked --manifest-path src-tauri/Cargo.toml -- --include-ignored --nocapture
```

The smoke test reads the expected fixture hash from the catalog before seeding the partial download. [Pinned Mini repository](https://huggingface.co/parallaxlabs/parallax-2-mini/tree/6067374875c236c5998d2cafdce705711d7cd3e5)

## Real memory reports

llama.cpp b12675 reported the following Metal allocations on an Apple M1. These are examples captured on this computer.

| Variant | Context size | Metal context state | Total Metal allocation |
| --- | ---: | ---: | ---: |
| Parallax 2 Mini Q3_K_M | 4,096 | 178 MiB | 2,834 MiB |
| Parallax 2 Mini Q3_K_M | 32,768 | 1,074 MiB | 3,758 MiB |
| Parallax 2 Q2_K | 4,096 | 178 MiB | 4,020 MiB |
| Parallax 2 Q2_K | 32,768 | 1,074 MiB | 4,944 MiB |

Mini was tested with the complete verified GGUF. Parallax 2 was profiled using the official GGUF’s metadata and llama.cpp’s allocation estimator. No Parallax 2 inference was run. [Pinned Parallax 2 repository](https://huggingface.co/parallaxlabs/parallax-2/tree/a42e0c2211757f114b9925f8abb6360ee63c8135) · [Pinned profiler implementation](https://github.com/ggml-org/llama.cpp/blob/b12675/tools/fit-params/fit-params.cpp)

## Platform coverage

| Platform | Result |
| --- | --- |
| Apple Silicon macOS | Release build and real backend smoke test with Metal and CPU. |
| Intel macOS | Release cross-build. The pinned Intel runtime previously reported its version under Rosetta. Native Intel inference remains unverified. |
| Windows x64 and Linux x64 | Runtime manifest and CPU archives checked against upstream. Native builds, installations, and inference remain unverified. |

Physical NVIDIA CUDA and Vulkan paths remain unverified. Manual verification of the updated UI is still pending.

The 0.2.0 universal DMG passed integrity and SHA-256 checks. The installed bundle contains arm64 and x86_64 slices, and both started in supervisor mode. The Intel slice ran under Rosetta. The app is ad hoc signed. Developer ID signing and notarization are pending.

## 0.3.0 checks

The frontend production build passed. All 14 standard Rust tests passed on Apple Silicon. A separate live smoke test downloaded the pinned uv release, verified its checksum, started managed Python and free-search-mcp through uvx, initialized MCP over stdio, and retrieved public links. The official Mini model separately streamed a tool call and generated a follow-up answer from a supplied test result. The updated interface still needs manual desktop testing.

## 0.4.0 checks

The frontend production build and all 15 standard Rust tests passed. The new SQLite test verified that a default prompt applies to new chats and that a conversation can override it. Eight calendar checks covered regional Thanksgiving dates, locale differences, weekday and weekend greetings, a month boundary, Christmas, and late night. Manual desktop UI testing remains pending.

## 0.5.0 checks

The frontend production build and all 18 Rust tests passed on Apple Silicon, including the live smoke tests. Starting with a clean app directory, the smoke test prepared uvx and the pinned MCP server, verified reasoning disabled and enabled through the Thinking command, and ran a live search from a Parallax 2 Mini chat before streaming its answer. A SQLite regression test verified that a search failure remains visible after reopening an empty assistant reply.

The universal DMG passed integrity, bundle signature, and SHA-256 checks. Both packaged executable slices started in supervisor mode, with the Intel slice running under Rosetta. The app is ad hoc signed. Manual desktop UI testing remains pending.

## 0.5.1 checks

Regression tests replayed the text-form search call seen in chat with tags split across stream events, missing inner closing tags, and a trailing quote. They also checked complete calls, incomplete calls, and Markdown code examples. The Apple Silicon smoke test replayed the raw call from a local inference fixture, ran the real MCP search, verified that source links reached the follow-up request, and checked that the stored response contained only the final answer. The same suite separately verified a live tool call from Parallax 2 Mini.

The universal DMG passed integrity, bundle signature, architecture startup, and SHA-256 checks. The Intel slice ran under Rosetta.

## 0.5.2 checks

The frontend production build passed. The date regression test verified that the current date is added to a single system message and that an existing custom prompt is preserved. On Apple Silicon, Parallax 2 Mini answered with the computer’s current local date with Web Search disabled.

All 21 Rust tests passed on Apple Silicon. The follow-up regression replayed successive raw tool calls through the app and verified three live MCP searches, unique call IDs and matching tool results in the inference history, and source links in the final reply when the model attempted a fourth search.

The 0.5.2 universal DMG passed integrity, bundle signature, architecture startup, and SHA-256 checks. The Intel slice ran under Rosetta.

## 0.5.2 Agent update

The frontend production build passed. On Apple Silicon, the real Mini model returned generation usage, and the model tokenizer counted the saved conversation. The model completed a live MCP search and requested approval before reading and quoting a project file. The permission tests checked confinement to the selected folder, remembered read approvals, revocation, and one-time approval for writes and deletions. The new universal DMG passed integrity, signature, SHA-256, and startup checks for both slices. Desktop UI automation was unavailable.

## 0.6.0 checks

The frontend production build and 30 standard Rust tests passed on Apple Silicon. The Squadron fixture verified overlapping requests on one inference endpoint, ordered reports, and one-time and session approvals. A live Parallax 2 Mini test profiled two slots and completed two simultaneous requests in one llama.cpp process. The managed uvx search path also passed a live MCP test on macOS.

The universal macOS disk image passed integrity, signature, SHA-256, and startup checks for both slices. The Windows x64 code cross-checked and the release executable linked successfully. The Windows update archive passed integrity and SHA-256 checks. Native Windows startup and Web Search remain unverified.

## 0.6.0 news and accent update

Parallax 2 Mini answered the reported news prompt with a structured Squadron batch containing eight assignments and eight matching search queries. A separate live MCP test returned article links using the updated news filters. Rust tests cover query gating, batch shape, permissions, and concurrent worker requests. The Settings render check verified the native color swatch and hex field for the custom accent option. Windows was cross-built.

## 0.6.0 Squadron news article update

All 35 standard Rust tests passed. The article fixtures cover the reported topic, hub, category, and trending URLs, undated results, stale entries, and publisher RSS article dates. A live lookup through the final news routing returned article links. The full Squadron desktop flow and native Windows execution were not exercised for this update.
