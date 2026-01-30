"""
Enhanced Mock Service Fixture Generator

Generates realistic JSONL fixture files for the mock service.
These files are read by the mock service to serve data to Purple agents.
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

# Output to mock_service/fixtures where the mock service reads from
FIXTURES_DIR = Path("mock_service/fixtures")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


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
        row = {
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
        return row


def main() -> None:
    print("=" * 70)
    print("Enhanced Mock Service Fixture Generator")
    print("=" * 70)
    
    for task in get_tasks():
        total_rows = task.constraints.get("total_rows", 100)
        fixture_path = FIXTURES_DIR / f"{task.task_id}.jsonl"
        
        print(f"\n📦 {task.task_id}: Generating {total_rows} rows")
        
        rows = []
        
        # Special handling for T7_totals_trap
        if task.task_id == "T7_totals_trap":
            # Generate normal data rows
            for i in range(total_rows):
                rows.append(generate_realistic_row(task.task_id, task.query, i + 1, is_totals=False))
            
            # Add totals rows (these should be filtered out by Purple Agent)
            # Add ~10% totals rows interspersed
            num_totals = total_rows // 10
            for i in range(num_totals):
                rows.append(generate_realistic_row(task.task_id, task.query, total_rows + i + 1, is_totals=True))
            
            # Shuffle to make it realistic
            random.shuffle(rows)
            
            print(f"  ✓ Generated {total_rows} data rows + {num_totals} totals rows")
            
        else:
            # Generate the expected number of rows
            for i in range(total_rows):
                rows.append(generate_realistic_row(task.task_id, task.query, i + 1))
            
            # For T3 (duplicates), add some duplicate rows
            if task.task_id == "T3_duplicates":
                # Add ~8% duplicates as per task definition
                num_dupes = int(total_rows * 0.08)
                for i in range(num_dupes):
                    # Duplicate a random existing row
                    dupe_row = rows[random.randint(0, len(rows) - 1)].copy()
                    rows.append(dupe_row)
                print(f"  ✓ Generated {total_rows} rows + {num_dupes} duplicates")
            else:
                print(f"  ✓ Generated {len(rows)} rows")
        
        # Write JSONL file
        fixture_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        print(f"  📝 Written to: {fixture_path}")
    
    print("\n" + "=" * 70)
    print("✅ Mock service fixture generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Restart the mock service to load new fixtures:")
    print("   cd green-comtrade-bench-v3-main && docker compose restart mock-comtrade")
    print("2. Run Purple agents to test with full datasets")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
