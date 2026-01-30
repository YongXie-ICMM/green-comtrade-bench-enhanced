# Purple Comtrade Baseline v2

Baseline Purple agent for [Green Comtrade Bench v2](https://github.com/yonghongzhang-io/green-comtrade-bench-v2).

This is a minimal, deterministic Purple agent implementation that validates the evaluation contract without requiring an LLM.

## Features

- **No LLM**: Pure HTTP client implementation
- **Deterministic**: Stable sorting, fixed retry schedule (no random jitter)
- **Contract-compliant**: Outputs match EVALUATION_CONTRACT.md schema
- **Handles all fault modes**: Pagination, duplicates, 429, 500, page drift, totals trap

## Output Files

For each task, the agent creates files in `_purple_output/{task_id}/`:

| File | Description |
|------|-------------|
| `data.jsonl` | Deduplicated, sorted records (one JSON object per line) |
| `metadata.json` | Task metadata with query, row_count, schema, totals_handling |
| `run.log` | Execution log with retry evidence for fault tasks |

## Tasks (T1-T7)

| Task ID | Name | What It Tests |
|---------|------|---------------|
| `T1_single_page` | Single Page | Basic API calls and response parsing |
| `T2_multi_page` | Multi-Page | Pagination correctness |
| `T3_duplicates` | Duplicates | De-duplication under `dedup_key` |
| `T4_rate_limit_429` | Rate Limit 429 | Retry/backoff on HTTP 429 |
| `T5_server_error_500` | Server Error 500 | Retry on HTTP 500 |
| `T6_page_drift` | Page Drift | Canonical sort + convergence |
| `T7_totals_trap` | Totals Trap | Drop totals rows + report handling |

## Local Usage

### Run Single Task

```bash
python3 run.py --task-id T1_single_page --mock-url http://localhost:8000
```

### Run with Custom Output Directory

```bash
python3 run.py --task-id T6_page_drift --output-dir /tmp/purple_out/T6 --mock-url http://localhost:8000
```

## Docker Usage

### Build Image

```bash
docker build -t purple-comtrade-baseline:latest .
```

### Run Container

```bash
# Run single task against mock service
docker run --rm \
  --network host \
  purple-comtrade-baseline:latest \
  --task-id T1_single_page \
  --mock-url http://localhost:8000

# Run with output volume
docker run --rm \
  -v $(pwd)/_purple_output:/workspace/purple_output \
  --network host \
  purple-comtrade-baseline:latest \
  --task-id T7_totals_trap \
  --mock-url http://localhost:8000
```

### Run All Tasks

```bash
for task in T1_single_page T2_multi_page T3_duplicates T4_rate_limit_429 T5_server_error_500 T6_page_drift T7_totals_trap; do
  docker run --rm \
    -v $(pwd)/_purple_output:/workspace/purple_output \
    --network host \
    purple-comtrade-baseline:latest \
    --task-id $task \
    --mock-url http://localhost:8000
done
```

## Implementation Notes

- **Configure step**: Calls `POST /configure` with task definition
- **Fetch**: Uses `GET /records` with pagination (page or offset mode)
- **Retry logic**: Exponential backoff (1s, 2s, 4s) for HTTP 429 and 500
- **Totals handling**: For T7, drops rows where `isTotal=true AND partner=WLD AND hs=TOTAL`
- **Deduplication**: By `dedup_key` fields (year, reporter, partner, flow, hs, record_id)
- **Sorting**: Stable sort by dedup_key for deterministic output

## Dependencies

- Python 3.11+
- `requests` library

## Related

- [Green Comtrade Bench v2](https://github.com/yonghongzhang-io/green-comtrade-bench-v2) - The evaluation benchmark
- [AgentBeats Leaderboard v2](https://github.com/yonghongzhang-io/agentbeats-leaderboard-v2) - Submission tracking

## License

MIT
