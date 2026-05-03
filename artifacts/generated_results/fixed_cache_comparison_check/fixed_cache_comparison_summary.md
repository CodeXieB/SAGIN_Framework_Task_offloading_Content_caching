# Fixed Cache Comparison

Tail window: last 50 episodes.

| Method | Episodes | Reward | Completed | Hits | Dropped | Violations | Safety Cost | r_perf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint Cache | 800 | -16.37 | 518.28 | 485.12 | 1554.36 | 43.86 | 2.79 | 1.30 |
| Fixed Cache | 800 | -16.65 | 404.06 | 394.92 | 1383.30 | 34.64 | 2.48 | 1.01 |

## Tail-Mean Delta

| Metric | Joint Cache - Fixed Cache |
|---|---:|
| Reward | 0.28 |
| Completed | 114.22 |
| Hits | 90.20 |
| Dropped | 171.06 |
| Violations | 9.22 |
| Safety Cost | 0.30 |
| r_perf | 0.29 |
