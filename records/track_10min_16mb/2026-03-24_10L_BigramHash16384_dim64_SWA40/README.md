# 10L Int5-MLP + BigramHash(16384, dim=64) + SWA(0.4/50) — val_bpb 1.14595

**Track**: `track_10min_16mb` | **Hardware**: 8×H200 | **Date**: 2026-03-24

Built on top of thwu1's submission ([PR #180](https://github.com/openai/parameter-golf/pull/180)) with a single change to the bigram hash embedding configuration.

## Result

| Seed | val_bpb (roundtrip) |
|------|---------------------|
| 1337 | **1.14594585** |

*Single seed — additional seeds pending.*

## Change from PR #180 (thwu1)

Only 2 lines changed from `2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`:

```diff
- bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))
- bigram_dim        = int(os.environ.get("BIGRAM_DIM", 128))
+ bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 16384))
+ bigram_dim        = int(os.environ.get("BIGRAM_DIM", 64))
```

Everything else — architecture, optimizer, SWA, quantization, compression — is identical to PR #180.

## Why this change?

thwu1 uses `vocab=10240, dim=128` → **1,310,720 bigram embedding params**.
This submission uses `vocab=16384, dim=64` → **1,048,576 bigram embedding params** (actually ~20% fewer params).

The hypothesis: at the same (or smaller) parameter budget, a wider vocabulary captures more distinct bigram hash collisions than a higher embedding dimension. Proxy sweeps across multiple vocab/dim configs at equal budget consistently showed wider vocab winning:

| Config | L40s proxy rt_bpb |
|--------|------------------|
| `vocab=3072, dim=128, offsets=1,4` | 1.22082 |
| `vocab=16384, dim=64, offset=1` | 1.22286 |
| `vocab=10240, dim=128, offset=1` (thwu1) | 1.22395 |
| `vocab=8192, dim=64, offsets=1,2` | 1.22409 |
| `vocab=10240, dim=64, offsets=1,2` | 1.22544 |

## Config (unchanged from PR #180 except above)

| Hyperparameter | Value |
|---|---|
| Layers | 10 |
| Model dim | 512 |
| MLP mult | 3× |
| Quantization | Int5 (MLP) / Int6 (attn) |
| BigramHash vocab | **16384** |
| BigramHash dim | **64** |
| SWA start_frac | 0.40 |
| SWA every | 50 steps |
| Muon WD | 0.04 |
| Adam WD | 0.01 |
| Warmdown | 3000 steps |
| Compression | zstd-22 |
| Eval stride | 64 |
| Steps (10-min cap) | 7306 on 8×H200 |
| Artifact size | 15,923,771 bytes (15.19 MiB) |
