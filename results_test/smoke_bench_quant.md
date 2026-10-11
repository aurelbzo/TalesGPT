| variant | memory (MB) | file (MB) | val loss | perplexity | decode tok/s (B=1) | prefill (ms) |
|---|---|---|---|---|---|---|
| fp32 | 1.9 | 1.9 | 0.349 | 1.418 | 2276.6 | 0.7 |
| int8 (hand-written, weight-only) | 0.7 | 0.7 | 0.349 | 1.418 | 1343.0 | 1.0 |
| int8 (torch dynamic quant) | n/a | 0.7 | 0.3486 | 1.417 | 1287.9 | 1.3 |
