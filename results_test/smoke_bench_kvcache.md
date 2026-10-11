| device | prompt | new | KV cache | batch | TTFT (ms) | decode tok/s (total over batch) | speedup vs naive | KV mem (MB) |
|---|---|---|---|---|---|---|---|---|
| cpu | 16 | 48 | off | 1 | 0.6 | 1674.1 | - | 0.0 |
| cpu | 16 | 48 | on | 1 | 0.7 | 2374.7 | 1.4x | 0.3 |
| cpu | 16 | 48 | on | 8 | 1.7 | 13142.4 | - | 2.1 |
| cpu | 64 | 48 | off | 1 | 1.0 | 1047.0 | - | 0.0 |
| cpu | 64 | 48 | on | 1 | 0.8 | 2400.5 | 2.3x | 0.3 |
| cpu | 64 | 48 | on | 8 | 4.3 | 12206.4 | - | 2.1 |
