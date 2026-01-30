"""
Enhanced Fixture Generator - Generates FULL datasets matching task constraints

This replaces the minimal fixture generator with one that creates realistic
datasets matching the expected row counts for each task.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("PYTHONPATH", str(ROOT_DIR))

from src.tasks import get_tasks

# Generate fixtures directly to mock_service/fixtures for the mock service to use
ROOT = Path("mock_service/fixtures")
ROOT.mkdir(parents=True, exist_ok=True)

SCHEMA = [
    "year",
    "reporter",
    "partner",
    "flow",
    "hs",
    "tradeValue",
    "netWeight",
    "qty",
    "record_id",
]


def generate_realistic_row(task_id: str, query: dict, index: int, is_totals: bool = False) -> dict:
    """Generate a realistic data row with varying values."""
    if is_totals:
        # Generate a totals row for T7
        return {
            "year": query.get("year"),
            "reporter": query.get("reporter"),
            "partner": "WLD",  # World totals
            "flow": query.get("flow"),
            "hs": "TOTAL",  # Totals marker
            "tradeValue": random.randint(100000, 1000000),
            "netWeight": random.randint(50000, 500000),
            "qty": random.randint(5000, 50000),
            "record_id": f"{task_id}_{index:04d}",
            "isTotal": True,  # Totals marker
        }
    else:
        # Generate a normal data row
        return {
            "year": query.get("year"),
            "reporter": query.get("reporter"),
            "partner": query.get("partner"),
            "flow": query.get("flow"),
            "hs": query.get("hs"),
            "tradeValue": random.randint(100, 10000),
            "netWeight": random.randint(50, 5000),
            "qty": random.randint(1, 100),
            "record_id": f"{task_id}_{index:04d}",
        }


def _log_for_mode(mode: str) -> str:
    base = "INFO start task\nINFO fetched data\nINFO done\n"
    if mode == "rate_limit":
        return base + "WARN HTTP 429 received, retry backoff\n"
    if mode == "server_error":
        return base + "WARN HTTP 500 received, retry\n"
    if mode == "duplicates":
        return base + "INFO dedup strategy applied\n"
    if mode == "pagination":
        return base + "INFO fetched page 1/3\n"
    if mode == "page_drift":
        return base + "INFO canonical sort and dedup\n"
    if mode == "totals_trap":
        return base + "INFO dropped totals rows\n"
    return base


def main() -> None:
    print("=" * 60)
    print("Enhanced Fixture Generator - Generating FULL datasets")
    print("=" * 60)
    
    for task in get_tasks():
        out_dir = ROOT / task.task_id
        out_dir.mkdir(parents=True, exist_ok=True)

        q = task.query
        total_rows = task.constraints.get("total_rows", 100)
        
        print(f"\n📦 Generating {task.task_id}: {total_rows} rows")
        
        # Special handling for T7_totals_trap
        if task.task_id == "T7_totals_trap":
            # For T7, we need to generate data rows + totals rows
            # The constraint says 750 total rows, but we need to add totals rows
            # Let's generate 750 data rows + some totals rows (which will be filtered)
            data_rows = []
            totals_rows = []
            
            # Generate normal data rows (750)
            for i in range(total_rows):
                data_rows.append(generate_realistic_row(task.task_id, q, i + 1, is_totals=False))
            
            # Add totals rows (these should be filtered out by Purple Agent)
            # Let's add ~100 totals rows interspersed
            num_totals = total_rows // 10  # 10% totals rows
            for i in range(num_totals):
                totals_rows.append(generate_realistic_row(task.task_id, q, total_rows + i + 1, is_totals=True))
            
            # Combine and shuffle to make it realistic
            all_rows = data_rows + totals_rows
            random.shuffle(all_rows)
            
            # Write all rows (including totals - Purple Agent should filter them)
            (out_dir / "data.jsonl").write_text("\n".join(json.dumps(row) for row in data_rows) + "\n")
            
            # Metadata reflects FILTERED data (after totals removal)
            meta = {
                "task_id": task.task_id,
                "query": task.query,
                "row_count": len(data_rows),
                "schema": SCHEMA + ["isTotal"],  # Include isTotal for T7
                "dedup_key": ["year", "reporter", "partner", "flow", "hs", "record_id"],
                "sorted_by": ["year", "reporter", "partner", "flow", "hs", "record_id"],
                "pagination_stats": {
                    "paging_mode": task.constraints.get("paging_mode", "page"),
                    "page_size": task.constraints.get("page_size", 500),
                    "pages_fetched": (len(data_rows) // task.constraints.get("page_size", 500)) + 1,
                    "stop_reason": "complete",
                },
                "request_stats": {
                    "requests_total": (len(data_rows) // task.constraints.get("page_size", 500)) + 1,
                    "retries_total": 0,
                    "http_429": 0,
                    "http_500": 0,
                },
                "retry_policy": {"max_retries": 3, "backoff": "exponential", "base_seconds": 1},
                "totals_handling": {
                    "enabled": True,
                    "rows_dropped": len(totals_rows),
                    "rule": "drop rows where isTotal=true AND partner=WLD AND hs=TOTAL",
                },
                "output_hashes": {"data_sha256": "optional", "metadata_sha256": "optional"},
                "created_at": "2026-01-30T00:00:00Z",
                "tool_versions": {"purple": "enhanced-fixture-generator-v1", "python": "3.x"},
                "notes": f"Enhanced fixture with {len(data_rows)} data rows (filtered from {len(all_rows)} total)",
            }
            (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")
            
            log_content = _log_for_mode(task.fault_injection.get("mode", "none"))
            log_content += f"INFO Dropped {len(totals_rows)} totals rows\n"
            (out_dir / "run.log").write_text(log_content)
            
            print(f"  ✓ Generated {len(data_rows)} data rows + {len(totals_rows)} totals rows (to be filtered)")
            
        else:
            # Default handling for all other tasks
            data_rows = []
            
            # Generate the expected number of rows
            for i in range(total_rows):
                data_rows.append(generate_realistic_row(task.task_id, q, i + 1))
            
            # For T3 (duplicates), add some duplicate rows
            if task.task_id == "T3_duplicates":
                # Add ~8% duplicates as per task definition
                num_dupes = int(total_rows * 0.08)
                for i in range(num_dupes):
                    # Duplicate a random existing row
                    dupe_row = data_rows[random.randint(0, len(data_rows) - 1)].copy()
                    data_rows.append(dupe_row)
                print(f"  ✓ Added {num_dupes} duplicate rows for deduplication testing")
            
            # Write data
            (out_dir / "data.jsonl").write_text("\n".join(json.dumps(row) for row in data_rows) + "\n")
            
            # Calculate pages
            page_size = task.constraints.get("page_size", 500)
            pages_fetched = (len(data_rows) // page_size) + (1 if len(data_rows) % page_size else 0)
            
            meta = {
                "task_id": task.task_id,
                "query": task.query,
                "row_count": len(data_rows),
                "schema": SCHEMA,
                "dedup_key": ["year", "reporter", "partner", "flow", "hs", "record_id"],
                "sorted_by": ["year", "reporter", "partner", "flow", "hs", "record_id"],
                "pagination_stats": {
                    "paging_mode": task.constraints.get("paging_mode", "page"),
                    "page_size": page_size,
                    "pages_fetched": pages_fetched,
                    "stop_reason": "complete",
                },
                "request_stats": {
                    "requests_total": pages_fetched,
                    "retries_total": 0,
                    "http_429": 0,
                    "http_500": 0,
                },
                "retry_policy": {"max_retries": 3, "backoff": "exponential", "base_seconds": 1},
                "totals_handling": {
                    "enabled": False,
                    "rows_dropped": 0,
                    "rule": "drop rows where isTotal=true AND partner=WLD AND hs=TOTAL",
                },
                "output_hashes": {"data_sha256": "optional", "metadata_sha256": "optional"},
                "created_at": "2026-01-30T00:00:00Z",
                "tool_versions": {"purple": "enhanced-fixture-generator-v1", "python": "3.x"},
                "notes": f"Enhanced fixture with {len(data_rows)} rows",
            }
            (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")
            (out_dir / "run.log").write_text(_log_for_mode(task.fault_injection.get("mode", "none")))
            
            print(f"  ✓ Generated {len(data_rows)} rows")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced fixture generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
