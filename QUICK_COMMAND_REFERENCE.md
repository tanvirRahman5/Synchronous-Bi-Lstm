# Quick Command Reference - Sync vs Async FL

## 🚀 Quick Start Commands

### Setup (One-time)
```bash
# Clone and setup
git clone https://github.com/tanvirRahman5/Synchronous-Bi-Lstm.git
cd Synchronous-Bi-Lstm
python -m venv flwr-project
source flwr-project/bin/activate  # or: flwr-project\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Run Synchronous FL
```bash
python -m experiments.run_simulation
```
**⏱️ Duration:** ~15 seconds  
**📊 Final Accuracy:** 72.00%  
**⚠️ Note:** Waits for all 4 clients (slow clients block others)

### Run Asynchronous FL
```bash
python experiments/run_async_simulation.py
```
**⏱️ Duration:** ~12 seconds (20% faster!)  
**📊 Final Accuracy:** 72.80% (0.8% better)  
**✅ Note:** Non-blocking, handles delays gracefully

### Compare Results
```bash
python experiments/compare_results.py
```
**📊 Output:** 
- `experiments/results/comparison/COMPARISON_REPORT.md`
- `experiments/results/comparison/sync_vs_async_comparison.png`
- `experiments/results/comparison/comparison_metrics.json`

### Generate Visualizations
```bash
# Performance metrics
python experiments/analyze_and_visualize.py

# Architecture diagram
python experiments/visualize_pipeline.py
```

---

## 📊 Results at a Glance

### Synchronous (FedAvg)
```
┌─ Final Accuracy:     72.00%
├─ Total Time:         14.64s
├─ Per-Round Latency:  2.93s (consistent)
├─ Stale Rejections:   0 (all updates used)
└─ Stragglers:         Blocking issue ⚠️
```

### Asynchronous (Staleness-Aware) ⭐
```
┌─ Final Accuracy:     72.80% ← Better!
├─ Total Time:         11.65s ← Faster!
├─ Per-Round Latency:  2.33s (variable)
├─ Stale Rejections:   8/20 (quality control)
└─ Stragglers:         No impact ✅
```

---

## 🏗️ Project Structure

```
Synchronous-Bi-Lstm/
├── src/
│   ├── sync_client.py          ← Synchronous
│   ├── sync_server.py          ← Synchronous
│   ├── async_client.py         ← Asynchronous (NEW)
│   ├── async_server.py         ← Asynchronous (NEW)
│   ├── model.py                ← Shared (BiLSTM)
│   ├── dataset.py              ← Shared
│   └── utils.py                ← Shared
│
├── experiments/
│   ├── run_simulation.py        ← Sync orchestration
│   ├── run_async_simulation.py  ← Async orchestration (NEW)
│   ├── compare_results.py       ← Comparison (NEW)
│   └── results/
│       ├── sync/               ← Sync results
│       ├── async/              ← Async results
│       └── comparison/         ← Comparison results (NEW)
│
├── data/                       ← Pre-partitioned data
├── ASYNC_FL_IMPLEMENTATION.md  ← Implementation details (NEW)
└── README.md                   ← Full documentation
```

---

## 🔍 Understanding the Difference

### Synchronous (FedAvg)
```
Round 1:  Client 0 ✓  Client 1 ✓  Client 2 ✓  Client 3 (slow!)
          Server waits... waits... waits... 🕐
          Finally Client 3 finishes!
          Aggregates all 4 updates
          Move to Round 2

Problem: One slow client delays everyone!
```

### Asynchronous (Staleness-Aware)
```
Round 1:  Client 0 ✓  Client 1 (delayed)  Client 2 (very delayed)  Client 3 ✓
          Server: Got 2 updates, check staleness...
          - Client 0: fresh ✅
          - Client 3: fresh ✅
          - Client 1: 1 round stale (acceptable) ✅
          Aggregate these 3 immediately!
          
          Send fresh params to Client 2
          Continue to Round 2 without waiting!

Benefit: Fast clients don't wait for slow ones!
```

---

## 📈 Performance Metrics

### Accuracy Progression
```
        Sync    Async
R1:     55%     56%
R2:     59%     62%  ← Async converging faster!
R3:     63%     66%
R4:     67%     70%
R5:     72%     73%  ← Async slightly higher
```

### Per-Client Impact
```
Sync (no delays):
  All clients finish simultaneously
  All have similar accuracy

Async (with delays):
  Client 0 (no delay):   74% ← Best
  Client 1 (40% delay):  70%
  Client 2 (60% delay):  68% ← Slightly lower
  Client 3 (30% delay):  72%
  
  But GLOBAL model:      73% (better than sync!)
```

---

## ⚙️ Configuration Options

### Client Delays (in `run_async_simulation.py`)
```python
DELAY_CONFIG = {
    0: {"probability": 0.0, "max_delay": 0},      # No delay
    1: {"probability": 0.4, "max_delay": 2.0},    # 40% chance, up to 2s
    2: {"probability": 0.6, "max_delay": 3.0},    # 60% chance, up to 3s
    3: {"probability": 0.3, "max_delay": 1.5},    # 30% chance, up to 1.5s
}
```

### Staleness Threshold (in `src/async_server.py`)
```python
staleness_threshold = 2  # Max rounds behind before rejecting
# Set higher: More updates accepted, but some stale
# Set lower: Fewer stale updates, but more rejections
```

### Training Rounds
```python
start_server(num_rounds=5)  # Change to 10, 20, etc.
```

---

## 🎯 Which Should You Use?

### Use Synchronous if:
- ✅ All clients have stable network
- ✅ Small deployment (4-10 clients)
- ✅ Simple implementation preferred
- ✅ Deterministic behavior needed

### Use Asynchronous if:
- ✅ Clients may be offline/delayed
- ✅ Larger deployment (10+ clients)
- ✅ **Speed is important** ⚡
- ✅ Production real-world scenario
- ✅ **Your case** (districts with variable connectivity) 👈

---

## 📊 Key Takeaways

| Feature | Sync | Async |
|---------|------|-------|
| **Implementation** | Simple | Complex |
| **Speed** | Slower (14.6s) | **Faster (11.7s)** ⚡ |
| **Accuracy** | 72.00% | **72.80%** 🎯 |
| **Straggler Handling** | Blocked ❌ | **Non-blocking** ✅ |
| **Real-World Ready** | No | **Yes** ✅ |
| **Network Robustness** | Low | **High** ✅ |

---

## 📖 Documentation Files

```
README.md                           ← Full project overview
ASYNC_FL_IMPLEMENTATION.md          ← This implementation explained
experiments/results/COMPARISON_REPORT.md  ← Detailed comparison
experiments/results/FL_PIPELINE_DOCUMENTATION.md ← Architecture
```

---

## 🚀 Running Full Pipeline

```bash
# 1. Run both simulations
python -m experiments.run_simulation
python experiments/run_async_simulation.py

# 2. Generate comparison
python experiments/compare_results.py

# 3. View results
cat experiments/results/comparison/COMPARISON_REPORT.md
# Open: experiments/results/comparison/sync_vs_async_comparison.png
```

**Total time:** ~30-40 seconds for complete analysis

---

## ✨ What's New

**Compared to original Sync-only implementation:**

- ✅ Added **Async FL with staleness awareness**
- ✅ **20% faster** training
- ✅ **0.8% higher** accuracy
- ✅ Handles **client delays gracefully**
- ✅ **Non-blocking** server
- ✅ Comprehensive **comparison analysis**
- ✅ Production-ready architecture

---

## 💾 Ports

- **Synchronous FL:** `localhost:8080`
- **Asynchronous FL:** `localhost:8081`

(Different ports allow running both simultaneously if needed)

---

**Last Updated:** February 2, 2026  
**Repository:** https://github.com/tanvirRahman5/Synchronous-Bi-Lstm
