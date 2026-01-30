# Purple Agents - Three Performance Variants

This directory contains three Purple Agent implementations at different performance levels, demonstrating the Green Agent scoring system's ability to differentiate code quality.

## 📁 Directory Structure

```
purple_agents/
├── v1_high_performance/     # V1 - High Performance (75.3%)
├── v2_medium_performance/   # V2 - Medium Performance (70.7%)
└── v3_baseline/             # V3 - Baseline (66.4%)
```

## 📊 Performance Comparison

| Version | Score | Average | Observability | Key Features |
|:---|:---:|:---:|:---:|:---|
| **V1** | 527.4/700 | **75.3%** | 15/15 | Full structured logging |
| **V2** | 495.0/700 | **70.7%** | ~9/15 | Partial structured logging |
| **V3** | 464.5/700 | **66.4%** | ~3.75/15 | Basic logging |

## 🔍 Version Details

### V1 - High Performance (v1_high_performance/)

**Score: 527.4/700 (75.3%)**

**Features:**
- ✅ Full structured logging: `[task_id=X] [page=Y] [request=Z] [complete=true]`
- ✅ Detailed performance metrics tracking
- ✅ Complete error statistics (http_429, http_500)
- ✅ Precise retry efficiency recording

**Log Example:**
```
INFO: [task_id=T1_single_page] [page=1] [request=2] [complete=true] Fetched 800 rows
```

**Recommended for:**
- 🏢 Production environments
- 📊 Detailed performance tracking needs
- 🔍 Complex debugging scenarios

### V2 - Medium Performance (v2_medium_performance/)

**Score: 495.0/700 (70.7%)**

**Features:**
- ✅ Partial structured logging: `[task_id=X] [page=Y] INFO: Message`
- ✅ Basic traceability
- ✅ Standard retry logic
- ⚠️ Missing request field and complete marker

**Log Example:**
```
[task_id=T1_single_page] [page=1] INFO: Fetching page 1
```

**Recommended for:**
- 🧪 Testing environments
- 📋 Medium complexity projects
- 👥 Small team collaboration

### V3 - Baseline (v3_baseline/)

**Score: 464.5/700 (66.4%)**

**Features:**
- ⚠️ Basic logging: `INFO: Message`
- ⚠️ Minimal implementation
- ⚠️ No structured fields
- ✅ Core functionality complete

**Log Example:**
```
INFO: Fetching page 1
INFO: Task T1_single_page complete
```

**Recommended for:**
- 📚 Learning reference
- 🔬 Minimal dependencies
- 🎓 Educational purposes

## 🚀 Usage

### Run V1 (High Performance)

```bash
cd purple_agents/v1_high_performance
python3 run.py --task-id T1_single_page --mock-url http://localhost:8000 --output-dir ./output
```

### Run V2 (Medium Performance)

```bash
cd purple_agents/v2_medium_performance
python3 run.py --task-id T1_single_page --mock-url http://localhost:8000 --output-dir ./output
```

### Run V3 (Baseline)

```bash
cd purple_agents/v3_baseline
python3 run.py --task-id T1_single_page --mock-url http://localhost:8000 --output-dir ./output
```

## 📈 Score Breakdown Analysis

### Observability Score Comparison

| Version | Structure Level | Score | Gap |
|:---|:---|:---:|:---|
| V1 | Full structured | 15/15 | Baseline |
| V2 | Partial structured | ~9/15 | -6 pts |
| V3 | Basic logging | ~3.75/15 | -11.25 pts |

### Why V2 Scores 30.5 Points Higher Than V3?

**V2 Advantages:**
1. Partial structured logging (+4-5 pts/task)
2. Traceable task_id and page fields (+1-2 pts/task)
3. Better debugging experience

**V3 Disadvantages:**
1. Missing structured fields (-4-5 pts/task)
2. Difficult to parse automatically (-1-2 pts/task)
3. Poor traceability

## 🎯 Selection Guide

| Scenario | Recommended | Reason |
|:---|:---:|:---|
| Production | V1 | Best observability and debugging |
| Testing | V2 | Balanced performance and complexity |
| Learning | V3 | Simplest implementation |
| High performance needs | V1 | Detailed performance metrics |
| Quick prototyping | V3 | Minimal dependencies |

## 📝 Technical Details

### Common Features

All three versions implement:
- ✅ Correct pagination logic
- ✅ Exponential backoff retry
- ✅ Deduplication and sorting
- ✅ Totals row filtering (T7 task)
- ✅ Complete output files (data.jsonl, metadata.json, run.log)

### Key Differences

| Feature | V1 | V2 | V3 |
|:---|:---:|:---:|:---:|
| Full structured logging | ✅ | ❌ | ❌ |
| Partial structured logging | ✅ | ✅ | ❌ |
| Request field | ✅ | ❌ | ❌ |
| Complete marker | ✅ | ❌ | ❌ |
| Detailed error stats | ✅ | ⚠️ | ❌ |

## 🔗 Related Resources

- [Green Agent Scoring System](../src/judge.py)
- [Scoring Documentation](../README.md)
- [Test Fixtures](../mock_service/fixtures/)

---

**Note:** All three versions have identical core functionality. The main differences are in logging structure and observability. Choose based on your specific requirements and environment.
