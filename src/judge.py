from __future__ import annotations

import json
import hashlib
import errno
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

REQUIRED_FILES = ["data.jsonl", "metadata.json", "run.log"]

# macOS Docker bind-mounts can intermittently raise EDEADLK (Errno 35)
_RETRY_ERRNOS = {getattr(errno, "EDEADLK", 35), 35}

# ============ GOVERNANCE CONSTANTS ============
# These thresholds enforce scoring fairness and anti-gaming rules.
# See README.md "Scoring Philosophy / Governance" for rationale.

# Threshold: correctness must reach 70% for full efficiency/observability credit
CORRECTNESS_GATE_THRESHOLD = 0.70  # 70% of max 30 points = 21 points
CORRECTNESS_GATE_PENALTY = 0.50    # efficiency/observability capped at 50% if below threshold

# Completeness must be 100% to earn any efficiency points
COMPLETENESS_GATE_FULL = 1.0  # Must be fully complete

# Time penalty: only penalize if execution exceeds this threshold (seconds)
# Avoids penalizing normal variance; only flags truly slow runs
EXECUTION_TIME_PENALTY_THRESHOLD = 45.0  # seconds
EXECUTION_TIME_PENALTY_POINTS = 3.0      # max points lost for slow execution

# Per-task efficiency baselines (expected request counts for fair comparison)
# Prevents pagination-heavy tasks from being unfairly penalized
TASK_EFFICIENCY_BASELINES: Dict[str, int] = {
    "T1_single_page": 1,       # Single page, 1 request expected
    "T2_multi_page": 5,        # Multi-page, ~5 requests expected
    "T3_duplicates": 3,        # Duplicates task, ~3 requests
    "T4_rate_limit_429": 4,    # Rate limit with retries, ~4 requests
    "T5_server_error_500": 4,  # Server error with retries, ~4 requests
    "T6_page_drift": 3,        # Page drift, ~3 requests
    "T7_totals_trap": 8,       # Totals trap, more pages, ~8 requests
}

# Observability: required traceable fields (must be present in log or metadata)
REQUIRED_OBSERVABILITY_FIELDS = [
    "task_id",          # Which task was executed
    "page",             # Pagination tracking (page/offset/cursor)
    "request",          # Request tracking
    "complete",         # Stop reason / completion indicator
]


def _with_retries(
    func,
    *,
    attempts: int = 10,
    base_sleep: float = 0.05,
    max_sleep: float = 0.5,
    max_elapsed: float = 5.0,
):
    last_exc: Exception | None = None
    start = time.monotonic()
    for i in range(attempts):
        try:
            return func()
        except OSError as e:
            last_exc = e
            if getattr(e, "errno", None) in _RETRY_ERRNOS:
                if time.monotonic() - start >= max_elapsed:
                    break
                time.sleep(min(base_sleep * (2**i), max_sleep))
                continue
            raise
    if last_exc is not None:
        if isinstance(last_exc, OSError) and getattr(last_exc, "errno", None) in _RETRY_ERRNOS:
            raise TimeoutError("I/O retry deadline exceeded")
        raise last_exc


def _read_text_retry(p: Path, *, encoding: str = "utf-8", errors: str = "strict") -> str:
    return _with_retries(lambda: p.read_text(encoding=encoding, errors=errors))


@dataclass
class ScoreResult:
    total: float
    breakdown: Dict[str, float]
    errors: List[str]
    details: Dict[str, Any]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()

    def _read_all():
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

    _with_retries(_read_all)
    return h.hexdigest()


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(_read_text_retry(p, encoding="utf-8", errors="strict"))


def _count_jsonl_rows(p: Path) -> int:
    def _count():
        n = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    return _with_retries(_count)


def _dedup_check_jsonl(p: Path, key_fields: List[str]) -> Tuple[int, int]:
    def _check():
        seen = set()
        total = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                obj = json.loads(line)
                k = tuple(obj.get(kf) for kf in key_fields)
                seen.add(k)
        return total, len(seen)

    return _with_retries(_check)


# ============ ENHANCED OBSERVABILITY HELPERS ============

def _check_structured_logging(log_text: str) -> float:
    """
    Check structured logging with multi-level granularity.
    Returns score 0.0 - 1.0 based on logging quality:
    - 1.0: Full structured (all required fields in key=value format)
    - 0.6: Partial structured (some fields in key=value format)
    - 0.25: Basic (keywords present but not structured)
    - 0.0: None
    """
    lines = [l.strip() for l in log_text.split('\n') if l.strip()]
    if not lines:
        return 0.0
    
    # Count lines with different levels of structure
    full_structured = 0  # Has all: task_id=X, page=X, request=X
    partial_structured = 0  # Has some: task_id=X, page=X
    basic_keywords = 0  # Has keywords but not key=value
    
    for line in lines:
        # Level 3: Full structured (V1 style)
        # Example: [task_id=T1] [page=1] [request=2] [complete=true]
        has_task_id = bool(re.search(r'task_id=\S+', line))
        has_page = bool(re.search(r'page=\d+', line))
        has_request = bool(re.search(r'request=\d+', line))
        
        if has_task_id and has_page and has_request:
            full_structured += 1
        # Level 2: Partial structured (V2 style)
        # Example: [task_id=T1] [page=1] INFO: Message
        elif has_task_id and has_page:
            partial_structured += 1
        # Level 1: Basic keywords (V3 style)
        # Example: INFO: Task T1 complete
        elif 'task' in line.lower() or 'page' in line.lower():
            basic_keywords += 1
    
    total_lines = len(lines)
    
    # Calculate weighted score
    if full_structured > total_lines * 0.8:
        # V1: 80%+ lines are fully structured
        return 1.0
    elif partial_structured > total_lines * 0.5:
        # V2: 50%+ lines are partially structured
        return 0.6
    elif basic_keywords > total_lines * 0.3:
        # V3: 30%+ lines have basic keywords
        return 0.25
    else:
        return 0.0



def _check_traceable_fields_strict(log_text: str, meta: dict) -> Tuple[float, List[str]]:
    """
    Strictly check for required traceable fields with format validation.
    Returns (score 0.0-1.0, list of errors)
    """
    errors = []
    fields_found = 0
    total_fields = 5
    
    # Define strict patterns for each field
    patterns = {
        'task_id': r'task_id=\S+',
        'page': r'page=\d+',
        'request': r'request=\d+',
        'complete': r'complete=(true|false)',
        'timestamp': r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}'
    }
    
    for field, pattern in patterns.items():
        if re.search(pattern, log_text, re.IGNORECASE):
            fields_found += 1
        else:
            errors.append(f"Missing or incorrectly formatted field: {field}")
    
    return fields_found / total_fields, errors


def _check_log_levels(log_text: str) -> float:
    """Check if proper log levels are used. Returns 0.0-1.0"""
    has_info = bool(re.search(r'\bINFO\b', log_text, re.IGNORECASE))
    has_warn = bool(re.search(r'\bWARN\b', log_text, re.IGNORECASE))
    has_error = bool(re.search(r'\bERROR\b', log_text, re.IGNORECASE))
    
    levels_count = sum([has_info, has_warn, has_error])
    return min(1.0, levels_count / 3.0)


def _check_error_tracking(log_text: str) -> float:
    """Check if errors have proper tracking info. Returns 0.0-1.0"""
    # Look for stack traces, error details, or line numbers
    has_traceback = bool(re.search(r'traceback|stack trace', log_text, re.IGNORECASE))
    has_error_detail = bool(re.search(r'error:.*line \d+|exception:', log_text, re.IGNORECASE))
    has_error_code = bool(re.search(r'error_code|err_code', log_text, re.IGNORECASE))
    
    tracking_features = sum([has_traceback, has_error_detail, has_error_code])
    return min(1.0, tracking_features / 2.0)  # Need at least 2 features for full score


def _check_performance_metrics(log_text: str, meta: dict) -> float:
    """Check if performance metrics are logged. Returns 0.0-1.0"""
    # Check log for timing information
    has_timing_in_log = bool(re.search(r'(time|duration|elapsed).*\d+', log_text, re.IGNORECASE))
    
    # Check metadata for performance metrics
    has_exec_time = 'execution_time_seconds' in meta or 'execution_time' in meta
    has_request_count = 'request_count' in meta
    has_request_stats = 'request_stats' in meta
    
    metrics_count = sum([has_timing_in_log, has_exec_time, has_request_count, has_request_stats])
    return min(1.0, metrics_count / 3.0)


# ============ ENHANCED EFFICIENCY HELPERS ============

def _score_request_efficiency(meta: dict, task_id: str) -> Tuple[float, List[str]]:
    """
    Score request efficiency relative to baseline.
    Returns (score 0.0-8.0, list of errors)
    """
    errors = []
    request_count = meta.get('request_count', 0)
    baseline = TASK_EFFICIENCY_BASELINES.get(task_id, 5)
    
    if request_count == 0:
        return 8.0, []  # Benefit of doubt if not measurable
    
    if request_count <= baseline:
        return 8.0, []
    elif request_count <= baseline * 1.5:
        errors.append(f"Request count {request_count} slightly above baseline {baseline}")
        return 6.0, errors
    elif request_count <= baseline * 2:
        errors.append(f"Request count {request_count} moderately above baseline {baseline}")
        return 4.0, errors
    else:
        errors.append(f"Request count {request_count} significantly above baseline {baseline}")
        return 2.0, errors


def _score_retry_efficiency(meta: dict) -> float:
    """
    Score retry efficiency based on retry count vs errors.
    Returns 0.0-4.0
    """
    retry_count = meta.get('request_stats', {}).get('retries_total', 0)
    http_429 = meta.get('request_stats', {}).get('http_429', 0)
    http_500 = meta.get('request_stats', {}).get('http_500', 0)
    
    expected_retries = http_429 + http_500
    
    if expected_retries == 0 and retry_count == 0:
        return 4.0  # No errors, no retries - perfect
    
    if retry_count == expected_retries:
        return 4.0  # Perfect: one retry per error
    elif retry_count <= expected_retries * 2:
        return 3.0  # Good: some extra retries
    elif retry_count > 0:
        return 1.0  # Has retries but excessive
    
    return 2.0  # Default


def _check_concurrency_support(meta: dict) -> float:
    """Check if agent supports concurrency. Returns 0.0-1.0"""
    concurrency_fields = ['max_workers', 'concurrent_requests', 'parallelism', 'thread_count']
    has_concurrency = any(field in meta for field in concurrency_fields)
    return 1.0 if has_concurrency else 0.0


def _check_caching_strategy(meta: dict) -> float:
    """Check if agent implements caching. Returns 0.0-1.0"""
    cache_fields = ['cache_hits', 'cache_enabled', 'cached_responses', 'cache_size']
    has_cache = any(field in meta for field in cache_fields)
    return 1.0 if has_cache else 0.0


def _score_execution_time(meta: dict) -> float:
    """Score execution time. Returns 0.0-3.0"""
    exec_time = meta.get('execution_time_seconds', 0)
    
    if exec_time == 0:
        return 3.0  # Benefit of doubt
    
    if exec_time <= 10:
        return 3.0
    elif exec_time <= 30:
        return 2.0
    elif exec_time <= 45:
        return 1.0
    else:
        return 0.0


# ============ ENHANCED ROBUSTNESS HELPERS ============

def _verify_error_detection(log_text: str, mode: str) -> float:
    """Verify that errors are properly detected. Returns 0.0-5.0"""
    if mode == 'rate_limit':
        has_429 = bool(re.search(r'HTTP 429|429 received', log_text, re.IGNORECASE))
        return 5.0 if has_429 else 0.0
    elif mode == 'server_error':
        has_500 = bool(re.search(r'HTTP 500|500 received', log_text, re.IGNORECASE))
        return 5.0 if has_500 else 0.0
    return 0.0


def _verify_retry_strategy(log_text: str, meta: dict) -> Tuple[float, List[str]]:
    """
    Verify retry strategy implementation.
    Returns (score 0.0-5.0, list of errors)
    """
    errors = []
    score = 0.0
    
    # Check for retry evidence
    has_retry = bool(re.search(r'\bretry\b', log_text, re.IGNORECASE))
    if not has_retry:
        errors.append("No retry evidence found in logs")
        return 0.0, errors
    
    score += 2.0  # Has retry
    
    # Check for backoff strategy
    has_backoff = bool(re.search(r'exponential|backoff', log_text, re.IGNORECASE))
    if has_backoff:
        score += 2.0
    
    # Check for retry timing evidence
    retry_times = re.findall(r'retry after (\d+)s', log_text, re.IGNORECASE)
    if retry_times:
        times = [int(t) for t in retry_times]
        # Check if exponential (each time >= previous)
        is_increasing = all(times[i] <= times[i+1] for i in range(len(times)-1))
        if is_increasing:
            score += 1.0
    
    return min(5.0, score), errors


def _verify_recovery(log_text: str, meta: dict) -> float:
    """Verify that agent recovers from errors. Returns 0.0-5.0"""
    # Look for success after retry
    has_success_after_retry = bool(re.search(
        r'(retry.*success|retry.*complete|429.*retry.*\d+.*success|500.*retry.*\d+.*success)',
        log_text,
        re.IGNORECASE
    ))
    
    if has_success_after_retry:
        return 5.0
    
    # Check if final status is success
    has_complete = bool(re.search(r'complete|success|done', log_text, re.IGNORECASE))
    if has_complete:
        return 3.0
    
    return 0.0


def _check_degradation_strategy(log_text: str, meta: dict) -> float:
    """Check for degradation/circuit breaker strategy. Returns 0.0-1.0"""
    degradation_keywords = [
        'circuit breaker',
        'max retries',
        'retry limit',
        'giving up',
        'fallback',
        'degraded mode'
    ]
    
    for keyword in degradation_keywords:
        if keyword in log_text.lower():
            return 1.0
    
    # Check metadata for retry limits
    retry_policy = meta.get('retry_policy', {})
    if 'max_retries' in retry_policy:
        return 1.0
    
    return 0.0


def _check_error_reporting(meta: dict, mode: str) -> float:
    """Check if errors are properly reported in metadata. Returns 0.0-1.0"""
    request_stats = meta.get('request_stats', {})
    
    if mode == 'rate_limit':
        has_429_count = 'http_429' in request_stats
        return 1.0 if has_429_count else 0.0
    elif mode == 'server_error':
        has_500_count = 'http_500' in request_stats
        return 1.0 if has_500_count else 0.0
    
    return 0.0



def score_output(output_dir: Path, task_expected: Dict[str, Any]) -> ScoreResult:
    """
    ENHANCED comprehensive scoring with 6 dimensions (100-point scale):
    
    - correctness (25 points): Data accuracy with gradient scoring
    - completeness (13 points): Required files and fields
    - robustness (17 points): Error handling capability with behavior verification
    - efficiency (17 points): Request efficiency, retry strategy, concurrency, caching
    - data_quality (13 points): Content validation, type checking
    - observability (15 points): Multi-level structured logging, traceable fields, metrics
    
    Total: 100 points (normalized from 120)
    
    GOVERNANCE RULES (anti-gaming, fairness):
    - If correctness < 70%, efficiency and observability are capped at 50%
    - If completeness < 100%, efficiency is 0 (no gaming incomplete outputs)
    - Efficiency uses task-specific baselines for cross-task fairness
    - Observability uses multi-level detection: full/partial/basic structured logging
    
    ENHANCEMENTS:
    - Observability: 3-level detection (V1=full, V2=partial, V3=basic)
    - Efficiency: Evaluates retry efficiency, concurrency, and caching strategies
    - Robustness: Verifies actual retry behavior and recovery, not just keywords
    """
    errors: List[str] = []
    breakdown: Dict[str, float] = {
        "correctness": 0.0, 
        "completeness": 0.0,
        "robustness": 0.0, 
        "efficiency": 0.0,
        "data_quality": 0.0,
        "observability": 0.0
    }
    details: Dict[str, Any] = {}

    if not output_dir.exists():
        return ScoreResult(0.0, breakdown, [f"Missing output dir: {output_dir}"], details)

    # Check required files
    missing_files = []
    for fn in REQUIRED_FILES:
        if not (output_dir / fn).exists():
            missing_files.append(fn)
            errors.append(f"Missing required file: {fn}")

    if missing_files:
        # Partial credit for partial files
        files_present = len(REQUIRED_FILES) - len(missing_files)
        breakdown["completeness"] = (files_present / len(REQUIRED_FILES)) * 8.7
        return ScoreResult(breakdown["completeness"], breakdown, errors, details)

    data_path = output_dir / "data.jsonl"
    meta_path = output_dir / "metadata.json"
    log_path = output_dir / "run.log"

    try:
        meta = _load_json(meta_path)
    except json.JSONDecodeError as e:
        return ScoreResult(0.0, breakdown, [f"metadata.json is not valid JSON: {e}"], details)
    except OSError as e:
        return ScoreResult(0.0, breakdown, [f"metadata.json could not be read: {e}"], details)

    # ============ COMPLETENESS (15 points) ============
    needed_meta_fields = ["task_id", "query", "row_count", "schema", "dedup_key"]
    fields_present = sum(1 for f in needed_meta_fields if f in meta)
    breakdown["completeness"] = (fields_present / len(needed_meta_fields)) * 13.0
    
    for f in needed_meta_fields:
        if f not in meta:
            errors.append(f"metadata.json missing field: {f}")

    row_count_actual = _count_jsonl_rows(data_path)
    details["row_count_actual"] = row_count_actual
    details["row_count_declared"] = meta.get("row_count")

    # ============ CORRECTNESS (30 points) - Gradient scoring ============
    correctness = 0.0
    
    # Row count accuracy (12 points) - gradient based on accuracy
    expected_rows = task_expected.get("constraints", {}).get("total_rows", 0)
    declared_rows = meta.get("row_count", 0)
    
    if expected_rows > 0:
        # Calculate accuracy percentage
        row_accuracy = 1.0 - abs(row_count_actual - expected_rows) / max(expected_rows, 1)
        row_accuracy = max(0.0, min(1.0, row_accuracy))
        correctness += row_accuracy * 12.0
        details["row_accuracy_pct"] = round(row_accuracy * 100, 1)
        
        if row_count_actual != expected_rows:
            errors.append(f"Row count: got {row_count_actual}, expected {expected_rows} (accuracy: {row_accuracy*100:.1f}%)")
    elif meta.get("row_count") == row_count_actual:
        correctness += 12.0
    else:
        errors.append(f"Row count mismatch: declared={meta.get('row_count')} actual={row_count_actual}")

    # Schema validation (4 points)
    schema = meta.get("schema") or []
    if isinstance(schema, list):
        expected_schema_len = 9  # Full schema has 9 columns
        schema_completeness = min(len(schema) / expected_schema_len, 1.0)
        correctness += schema_completeness * 4.0
        details["schema_completeness_pct"] = round(schema_completeness * 100, 1)
        if len(schema) < 5:
            errors.append(f"Schema incomplete: {len(schema)} columns, expected >= 5")
    else:
        errors.append("Schema must be a list")

    # Query matching (4 points)
    expected_query = task_expected.get("query", {})
    got_query = meta.get("query", {})
    query_fields = ["reporter", "partner", "flow", "hs", "year"]
    query_matches = sum(1 for k in query_fields 
                       if expected_query.get(k) is None or got_query.get(k) == expected_query.get(k))
    correctness += (query_matches / len(query_fields)) * 4.0
    if query_matches < len(query_fields):
        errors.append(f"Query mismatch: {query_matches}/{len(query_fields)} fields correct")

    # Deduplication check (6 points) - gradient based on duplicate rate
    dedup_key = meta.get("dedup_key") or []
    if isinstance(dedup_key, list) and len(dedup_key) >= 3:
        total_rows, unique_rows = _dedup_check_jsonl(data_path, dedup_key)
        details["dedup_total_rows"] = total_rows
        details["dedup_unique_rows"] = unique_rows
        
        if total_rows > 0:
            dedup_quality = unique_rows / total_rows
            correctness += dedup_quality * 6.0
            details["dedup_quality_pct"] = round(dedup_quality * 100, 1)
            if unique_rows < total_rows:
                dup_count = total_rows - unique_rows
                errors.append(f"Found {dup_count} duplicates ({(1-dedup_quality)*100:.1f}% duplicate rate)")
        else:
            correctness += 6.0
    else:
        errors.append("dedup_key invalid; expect list with >= 3 fields.")

    # Declared vs actual consistency (4 points)
    if meta.get("row_count") == row_count_actual:
        correctness += 4.0
    else:
        errors.append(f"Declared row_count ({meta.get('row_count')}) != actual ({row_count_actual})")

    # Totals handling check for T7_totals_trap
    if task_expected.get("fault_injection", {}).get("mode") == "totals_trap":
        totals_handling = meta.get("totals_handling", {})
        if not totals_handling.get("enabled"):
            errors.append("T7_totals_trap requires totals_handling.enabled=true")
            correctness -= 4.0
        elif totals_handling.get("rows_dropped", 0) < 1:
            errors.append("T7_totals_trap: no rows dropped")
            correctness -= 2.0

    breakdown["correctness"] = max(0.0, min(25.0, correctness * 25.0 / 30.0))

    # ============ ROBUSTNESS (20 points - ENHANCED) ============
    log_text = _read_text_retry(log_path, encoding="utf-8", errors="ignore")
    log_text_lower = log_text.lower()
    robustness = 0.0
    mode = task_expected.get("fault_injection", {}).get("mode")

    if mode in ["rate_limit", "server_error"]:
        # 1. Error detection (5 points)
        robustness += _verify_error_detection(log_text, mode)
        
        # 2. Retry strategy (5 points)
        retry_score, retry_errors = _verify_retry_strategy(log_text, meta)
        robustness += retry_score
        errors.extend(retry_errors)
        
        # 3. Recovery capability (5 points)
        robustness += _verify_recovery(log_text, meta)
        
        # 4. Degradation strategy (3 points)
        robustness += _check_degradation_strategy(log_text, meta) * 3.0
        
        # 5. Error reporting (2 points)
        robustness += _check_error_reporting(meta, mode) * 2.0
    else:
        # For non-fault tasks, give default good score
        robustness = 15.0

    breakdown["robustness"] = min(17.0, robustness * 17.0 / 20.0)

    # ============ EFFICIENCY (20 points - ENHANCED) ============
    efficiency = 0.0
    task_id = meta.get("task_id", "")
    
    # 1. Request efficiency (8 points)
    request_score, req_errors = _score_request_efficiency(meta, task_id)
    efficiency += request_score
    errors.extend(req_errors)
    details["request_efficiency_score"] = request_score
    
    # 2. Retry efficiency (4 points)
    retry_score = _score_retry_efficiency(meta)
    efficiency += retry_score
    details["retry_efficiency_score"] = retry_score
    
    # 3. Concurrency support (3 points)
    concurrency_score = _check_concurrency_support(meta) * 3.0
    efficiency += concurrency_score
    details["concurrency_score"] = concurrency_score
    
    # 4. Caching strategy (2 points)
    cache_score = _check_caching_strategy(meta) * 2.0
    efficiency += cache_score
    details["cache_score"] = cache_score
    
    # 5. Execution time (3 points)
    time_score = _score_execution_time(meta)
    efficiency += time_score
    details["execution_time_score"] = time_score
    details["execution_time_seconds"] = meta.get("execution_time_seconds", 0)

    breakdown["efficiency"] = min(17.0, efficiency * 17.0 / 20.0)

    # ============ DATA QUALITY (15 points) - NEW ============
    data_quality = 0.0
    
    # Content validation (5 points) - check data integrity
    try:
        data_valid = _validate_data_content(data_path, meta.get("schema", []))
        data_quality += data_valid * 5.0
        details["data_integrity_pct"] = round(data_valid * 100, 1)
    except Exception as e:
        errors.append(f"Data validation error: {e}")
    
    # Type consistency (5 points) - check if values have consistent types
    try:
        type_score = _check_type_consistency(data_path)
        data_quality += type_score * 5.0
        details["type_consistency_pct"] = round(type_score * 100, 1)
    except Exception as e:
        errors.append(f"Type check error: {e}")
    
    # Value range validation (5 points) - check if numeric values are reasonable
    try:
        range_score = _check_value_ranges(data_path, task_expected)
        data_quality += range_score * 5.0
        details["value_range_pct"] = round(range_score * 100, 1)
    except Exception as e:
        errors.append(f"Value range check error: {e}")
    
    breakdown["data_quality"] = min(13.0, data_quality * 13.0 / 15.0)

    # ============ OBSERVABILITY (20 points - ENHANCED) ============
    observability = 0.0
    
    # 1. Structured logging (8 points)
    structured_score = _check_structured_logging(log_text) * 8.0
    observability += structured_score
    details["structured_logging_score"] = structured_score
    
    # 2. Required traceable fields with format validation (6 points)
    field_score, field_errors = _check_traceable_fields_strict(log_text, meta)
    observability += field_score * 6.0
    errors.extend(field_errors)
    details["traceable_fields_score"] = field_score * 6.0
    
    # 3. Log levels (2 points)
    log_level_score = _check_log_levels(log_text) * 2.0
    observability += log_level_score
    details["log_levels_score"] = log_level_score
    
    # 4. Error tracking (2 points)
    error_tracking_score = _check_error_tracking(log_text) * 2.0
    observability += error_tracking_score
    details["error_tracking_score"] = error_tracking_score
    
    # 5. Performance metrics (2 points)
    perf_metrics_score = _check_performance_metrics(log_text, meta) * 2.0
    observability += perf_metrics_score
    details["performance_metrics_score"] = perf_metrics_score

    breakdown["observability"] = min(15.0, observability * 15.0 / 20.0)

    details["data_sha256"] = _sha256_file(data_path)
    details["metadata_sha256"] = _sha256_file(meta_path)

    # ============ GOVERNANCE GATES ============
    # Apply threshold-based caps to prevent gaming
    
    governance_applied = []
    
    # Gate 1: Completeness gate - if not 100% complete, efficiency = 0
    # Rationale: Can't claim efficiency credit for incomplete work
    completeness_ratio = breakdown["completeness"] / 15.0
    if completeness_ratio < COMPLETENESS_GATE_FULL:
        original_efficiency = breakdown["efficiency"]
        breakdown["efficiency"] = 0.0
        governance_applied.append(f"completeness_gate: efficiency {original_efficiency:.1f} → 0 (completeness {completeness_ratio*100:.0f}% < 100%)")
    
    # Gate 2: Correctness gate - if < 70%, cap efficiency and observability at 50%
    # Rationale: Can't claim quality signals if core task is largely wrong
    correctness_ratio = breakdown["correctness"] / 30.0
    if correctness_ratio < CORRECTNESS_GATE_THRESHOLD:
        # Apply 50% cap to efficiency (if not already zeroed by completeness gate)
        if breakdown["efficiency"] > 0:
            original_efficiency = breakdown["efficiency"]
            capped_efficiency = original_efficiency * CORRECTNESS_GATE_PENALTY
            breakdown["efficiency"] = capped_efficiency
            governance_applied.append(f"correctness_gate: efficiency {original_efficiency:.1f} → {capped_efficiency:.1f} (correctness {correctness_ratio*100:.0f}% < 70%)")
        
        # Apply 50% cap to observability
        original_observability = breakdown["observability"]
        capped_observability = original_observability * CORRECTNESS_GATE_PENALTY
        breakdown["observability"] = capped_observability
        governance_applied.append(f"correctness_gate: observability {original_observability:.1f} → {capped_observability:.1f} (correctness {correctness_ratio*100:.0f}% < 70%)")
    
    if governance_applied:
        details["governance_rules_applied"] = governance_applied
    
    total = sum(breakdown.values())
    return ScoreResult(total=round(total, 1), breakdown=breakdown, errors=errors, details=details)


def _validate_data_content(data_path: Path, schema: List[str]) -> float:
    """Validate data content quality. Returns score 0.0 - 1.0"""
    def _validate():
        valid_rows = 0
        total_rows = 0
        
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    obj = json.loads(line)
                    # Check if row has expected fields
                    if isinstance(obj, dict):
                        # At least 50% of schema fields should be present
                        if schema:
                            fields_present = sum(1 for s in schema if s in obj)
                            if fields_present >= len(schema) * 0.5:
                                valid_rows += 1
                        else:
                            valid_rows += 1
                except json.JSONDecodeError:
                    pass
        
        return valid_rows / total_rows if total_rows > 0 else 0.0
    
    return _with_retries(_validate)


def _check_type_consistency(data_path: Path) -> float:
    """Check if fields have consistent types across rows. Returns score 0.0 - 1.0"""
    def _check():
        field_types: Dict[str, set] = {}
        total_rows = 0
        
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k not in field_types:
                                field_types[k] = set()
                            # Track type (allow None as compatible with any type)
                            if v is not None:
                                field_types[k].add(type(v).__name__)
                except json.JSONDecodeError:
                    pass
        
        if not field_types:
            return 0.0
        
        # Score based on type consistency (1 type per field is best)
        consistent_fields = sum(1 for types in field_types.values() if len(types) <= 1)
        return consistent_fields / len(field_types)
    
    return _with_retries(_check)


def _check_value_ranges(data_path: Path, task_expected: Dict[str, Any]) -> float:
    """Check if numeric values are within reasonable ranges. Returns score 0.0 - 1.0"""
    def _check():
        total_checks = 0
        valid_checks = 0
        
        # Expected year range
        expected_year = task_expected.get("query", {}).get("year")
        
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    
                    # Check year field if present
                    if "year" in obj:
                        total_checks += 1
                        year_val = obj["year"]
                        if isinstance(year_val, (int, float)):
                            # Year should be reasonable (1900-2100)
                            if 1900 <= year_val <= 2100:
                                valid_checks += 1
                                # Bonus if matches expected year
                                if expected_year and year_val == expected_year:
                                    valid_checks += 0.5
                                    total_checks += 0.5
                    
                    # Check trade value if present (should be non-negative)
                    for val_field in ["value", "trade_value", "tradeValue", "primaryValue"]:
                        if val_field in obj:
                            total_checks += 1
                            val = obj[val_field]
                            if isinstance(val, (int, float)) and val >= 0:
                                valid_checks += 1
                    
                    # Check quantity if present (should be non-negative)
                    for qty_field in ["qty", "quantity", "netWgt"]:
                        if qty_field in obj:
                            total_checks += 1
                            qty = obj[qty_field]
                            if qty is None or (isinstance(qty, (int, float)) and qty >= 0):
                                valid_checks += 1
                                
                except json.JSONDecodeError:
                    pass
        
        return valid_checks / total_checks if total_checks > 0 else 1.0
    
    return _with_retries(_check)
