#!/usr/bin/env python
"""Closed-loop load test against /generate/stream. No extra tooling needed.

  python scripts/load_test.py --url http://localhost:8000 --concurrency 1,2,4,8 --duration 30 --label docker-int8

Each of the C virtual users sends requests back-to-back for --duration seconds.
Reports: requests/s, aggregate tokens/s, p50/p95/p99 end-to-end latency, p50/p95 time-to-first-token, errors.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Once upon a time", "One day, a little girl named Lily", "Tom and his dog went to the park",
    "The sun was shining and the birds", "Ben found a big red ball", "Mia loved to paint",
]


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def worker(client, url, deadline, max_new, out, wid):
    i = wid
    while time.perf_counter() < deadline:
        body = {"prompt": PROMPTS[i % len(PROMPTS)], "max_new_tokens": max_new, "temperature": 0.8}
        i += 1
        t0 = time.perf_counter()
        ttft, n, ok = None, 0, True
        try:
            async with client.stream("POST", url + "/generate/stream", json=body) as r:
                if r.status_code != 200:
                    ok = False
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if "token" in ev:
                        n += 1
                        if ttft is None:
                            ttft = time.perf_counter() - t0
        except Exception:
            ok = False
        out.append(dict(ok=ok, latency=time.perf_counter() - t0, ttft=ttft, tokens=n))


async def run_level(url, c, duration, max_new):
    out = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=httpx.Limits(max_connections=c + 4)) as client:
        deadline = time.perf_counter() + duration
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, url, deadline, max_new, out, w) for w in range(c)])
        wall = time.perf_counter() - t0
    good = [r for r in out if r["ok"]]
    lat = [r["latency"] for r in good]
    ttft = [r["ttft"] for r in good if r["ttft"] is not None]
    return dict(concurrency=c, requests=len(good), errors=len(out) - len(good),
                rps=round(len(good) / wall, 2), tok_s=round(sum(r["tokens"] for r in good) / wall, 1),
                p50_s=round(pct(lat, 50), 2), p95_s=round(pct(lat, 95), 2), p99_s=round(pct(lat, 99), 2),
                ttft_p50_ms=round(pct(ttft, 50) * 1e3), ttft_p95_ms=round(pct(ttft, 95) * 1e3))


async def main_async(a):
    rows = []
    for c in map(int, a.concurrency.split(",")):
        print(f"concurrency={c} ...", flush=True)
        row = await run_level(a.url, c, a.duration, a.max_new_tokens)
        rows.append(row)
        print(row, flush=True)
    md = ["| concurrency | requests | errors | req/s | tokens/s (total) | p50 (s) | p95 (s) | p99 (s) | TTFT p50 (ms) | TTFT p95 (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['concurrency']} | {r['requests']} | {r['errors']} | {r['rps']} | {r['tok_s']} | "
                  f"{r['p50_s']} | {r['p95_s']} | {r['p99_s']} | {r['ttft_p50_ms']} | {r['ttft_p95_ms']} |")
    out = Path(f"results/loadtest_{a.label}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--label", default="local")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
