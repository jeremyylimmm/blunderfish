
# blunderfish

Work In Progress — a small, experimental chess engine written in Rust that uses bitboards for board representation and move generation.

## Overview

`blunderfish` is an in-development chess engine. The implementation uses 64-bit bitboards to represent piece sets and to implement fast move generation and position operations. The project is experimental and evolving; internal APIs, on-disk formats, and module boundaries will change.

## Current Status

- Work In Progress (WIP): core board representation and move generation are the primary focus.
- Expect incomplete features, rough edges, and breaking changes.

## Goals (non-exhaustive)

- Implement efficient bitboard-based move generation and utilities.
- Add perft tests and correctness checks for move generation.
- Implement search (alpha-beta / iterative deepening) and a simple evaluation function.

## Building & Running

Requires the Rust toolchain (rustc + cargo).

Build:

```bash
cargo build
```

## Notes

- This repository is experimental and incomplete; use at your own risk.
- Tests and examples may be missing or incomplete; adding perft tests is a recommended first contribution.