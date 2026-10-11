| variant | memory (MB) | file (MB) | val loss | perplexity | decode tok/s (B=1) | prefill (ms) |
|---|---|---|---|---|---|---|
| fp32 | 119.7 | 119.6 | 1.4095 | 4.094 | 145.3 | 13.0 |
| int8 (hand-written, weight-only) | 42.8 | 42.7 | 1.4095 | 4.094 | 79.1 | 20.1 |
| int8 (torch dynamic quant) | n/a | 47.0 | 1.4432 | 4.234 | 197.6 | 11.9 |
