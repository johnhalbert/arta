# Cactus Rust bindings in Arta

Arta vendors `cactus-rs` and `cactus-sys` 0.3.0 from [upstream commit `50a0cbec`](https://github.com/hxphsts/cactus-rs/commit/50a0cbec29d7b2724a3c4ae90829c614313ff7a8). The upstream MIT and Apache-2.0 licenses are included.

Arta enables the `dynamic-link` feature. It resolves the shared C API at runtime, including on Windows MSVC. The shared engine and base Needle weights use the same pins as `cactus-compute` 4.2.7. The engine is extracted from the checksum-verified upstream wheel using Rust. Upstream static linking remains available for other callers.

The Needle builder exposes the configured thread count and response buffer size. Its optional text grounding, local date handling, and custom confidence-head detection follow the corresponding Apache-2.0 Cactus Compute SDK behavior. Process isolation and tool execution live in Arta's Cactus provider.
