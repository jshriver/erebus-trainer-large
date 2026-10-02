# erebus-trainer-large

NNUE trainer for the Erebus chess engine's large network variant, built on the [bullet](https://github.com/jw1912/bullet) training framework.

## Network Architecture

**Inputs per perspective: 86,896**

| Feature set | Size | Description |
|---|---|---|
| HalfKAv2_hm | 24,576 | 768 features × 32 horizontally-mirrored king-position buckets |
| Full_Threats | varies | Piece-threat features (separate STM/NTM encodings) |
| PP_3Wide | 4,560 | Pawn-pair features within a 3-file band |

**Network topology:**

```
[86,896 sparse inputs] × 2 perspectives
        ↓ sparse linear + pairwise CReLU
   [512] STM   [512] NTM
        ↓ concat
      [1024]
        ↓ L1 dense (32) → SqrCReLU + CReLU → [64]
        ↓ L2 dense (32) → SqrCReLU + CReLU → [64]
        ↓ skip concat → [128]
        ↓ output linear → [1]  (8 material-count output buckets)
```

A shared 768-wide factoriser weight is broadcast across all king-position buckets in the feature transformer and merged into the saved weights at checkpoint time.

## Requirements

- Rust 1.87+
- CUDA toolkit (for GPU training)

## Building

```bash
# GPU build (required for training)
cargo build --release --features cuda

# Syntax check without GPU
cargo check --no-default-features
```

## Running

```bash
./target/release/erebus-trainer-large <PATH>...
```

Each `PATH` is a `.binpack` file or a directory. Directories are scanned for `*.binpack` files sorted by name. Multiple paths are accepted.

```bash
# single file
./target/release/erebus-trainer-large /data/jan2024.binpack

# whole directory
./target/release/erebus-trainer-large /data/binpacks/

# mix of files and directories
./target/release/erebus-trainer-large /data/extra.binpack /data/binpacks/
```

## Configuration

All hyper-parameters are compile-time constants in the `CONFIG` block at the top of [`src/main.rs`](src/main.rs). Recompile after any changes.

| Constant | Default | Description |
|---|---|---|
| `L0_SIZE` | 512 | Accumulator width per perspective after pairwise activation |
| `L1_SIZE` | 32 | First dense hidden layer width |
| `L2_SIZE` | 32 | Second dense hidden layer width |
| `OUTPUT_BUCKETS` | 8 | Material-count output buckets |
| `QA` | 255 | Feature transformer quantisation scale (saved as i16) |
| `QB` | 64 | Dense layer quantisation scale (saved as i8) |
| `TOTAL_PASSES` | 1.0 | Full passes over the corpus |
| `BATCH_SIZE` | 16,384 | Positions per batch |
| `BATCHES_PER_SUPERBATCH` | 6,104 | Batches per superbatch (~100M positions) |
| `SAVE_RATE` | 10 | Checkpoint every N superbatches |
| `LR_START` / `LR_FINAL` | 1e-3 / 2.5e-6 | Cosine LR schedule bounds |
| `WDL_START` / `WDL_END` | 0.2 / 0.6 | Linear WDL blend across the full plan |
| `DATA_THREADS` | 8 | Parallel data loading threads |

## Position counts

`POSITION_COUNTS` in `src/main.rs` maps each binpack filename (basename only) to its raw position count. This drives the global training plan and per-session budgets. Alternatively, place a `<file>.binpack.count` sidecar file (containing just the integer) next to each binpack.

To cap a session manually:

```bash
EREBUS_LARGE_END_SB=200 ./target/release/erebus-trainer-large /data/binpacks/
```

## Resume

Training resumes automatically from the latest checkpoint found in `checkpoints/`. A `.session` file tracks the active session window and is cleaned up on normal completion.

## Checkpoints

Saved to `checkpoints/erebus-large-<N>/`:

| File | Contents |
|---|---|
| `quantised.bin` | Quantised network ready for the engine |
| `raw.bin` | Full-precision weights |
| `optimiser_state/` | Adam optimiser state for seamless resume |
