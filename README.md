# Federated Learning with Synchronous BiLSTM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flower](https://img.shields.io/badge/Flower-1.25.0-green.svg)](https://flower.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A complete implementation of **Federated Learning** using **Bidirectional LSTM** (BiLSTM) for crop classification on a distributed, non-IID dataset. This project demonstrates privacy-preserving machine learning with synchronous parameter aggregation using the Flower framework.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements **dual federated learning approaches** for crop classification:

### Synchronous FL (FedAvg)
- **Framework**: Flower (Federated Learning framework)
- **Strategy**: Synchronous FedAvg
- **Key Trait**: All clients synchronized (waits for stragglers)
- **Final Accuracy**: 72.00%
- **Training Time**: 14.64 seconds (5 rounds)
- **Per-Round Latency**: ~2.93 seconds (consistent)

### Asynchronous FL (Staleness-Aware)
- **Framework**: Flower with custom async strategy
- **Strategy**: Continuous aggregation with staleness checking
- **Key Trait**: Non-blocking (rejects overly stale updates)
- **Final Accuracy**: 72.80%
- **Training Time**: 11.65 seconds (5 rounds) - **20.4% faster**
- **Per-Round Latency**: ~2.33 seconds (variable)
- **Staleness Handling**: Automatic sync for delayed clients

### Shared Components
- **Model**: Bidirectional LSTM (BiLSTM) with 32 hidden units
- **Clients**: 4 district-based clients with non-IID data distribution
- **Communication**: gRPC for efficient parameter transmission
- **Comparison**: Full side-by-side analysis included

---

## ✨ Features

- ✅ **Privacy-Preserving**: Only model parameters shared, not raw data
- ✅ **Non-IID Data**: District-based heterogeneous data partitioning
- ✅ **Synchronous Training**: FedAvg with coordinated client updates
- ✅ **Comprehensive Metrics**: Accuracy, latency, and client-wise analysis
- ✅ **Rich Visualizations**: Performance dashboards and architecture diagrams
- ✅ **Production-Ready**: Complete documentation and reproducible results

---

## 🏗️ Architecture

### Model Architecture

```
Input [batch, 13 timesteps, 1 feature]
    ↓
Bidirectional LSTM (hidden=32)
    ↓
Dense Layer (64 → 16 classes)
    ↓
Output [batch, 16] (Crop predictions)
```

### Federated Learning Workflow

```
Server (FedAvg Strategy)
    ↓ (Broadcast parameters)
Client 0, 1, 2, 3 (Parallel training)
    ↓ (Upload local updates)
Server (Aggregate parameters)
    ↓ (Evaluate & repeat)
Next Round...
```

**Complete architecture diagrams available in**: [`experiments/results/`](experiments/results/)

---

## 📊 Dataset

**Source**: Crop Fertilizer Dataset  
**Samples**: ~1000+  
**Features**: 13 (6 numeric + categorical encoded)  
**Target**: 16 crop classes  
**Split Strategy**: District-based Non-IID  

### Data Partitioning

- **Client 0**: Districts A+B (~500 samples)
- **Client 1**: District C (~500 samples)
- **Client 2**: Districts D+E (~500 samples)
- **Client 3**: Districts F+G (~500 samples)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/tanvirRahman5/Synchronous-Bi-Lstm.git
cd Synchronous-Bi-Lstm
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv flwr-project

# Activate virtual environment
# On Linux/Mac:
source flwr-project/bin/activate

# On Windows:
flwr-project\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `flwr` - Federated learning framework
- `torch` - PyTorch deep learning
- `torchvision` - PyTorch vision utilities
- `numpy` - Numerical computing
- `matplotlib` - Visualization

---

## 📖 Usage

### 1. Data Preprocessing (Optional - Already Done)

The data is already preprocessed and partitioned. To re-preprocess:

```bash
python data/preprocess_data.py
```

### 2. Run Synchronous FL Simulation

**Synchronous FedAvg** - All clients wait for each other

```bash
python -m experiments.run_simulation
```

**Characteristics:**
- ✅ Simple, deterministic
- ✅ All updates used (no rejection)
- ❌ Blocked by slow clients (stragglers)
- **Results**: 72.00% accuracy in 14.64s

### 3. Run Asynchronous FL Simulation

**Asynchronous with Staleness-Aware Aggregation** - Non-blocking, continuous aggregation

```bash
python experiments/run_async_simulation.py
```

**Characteristics:**
- ✅ Faster (20.4% speedup)
- ✅ Handles delays gracefully
- ✅ Automatic sync for stale clients
- ⚠️ Some updates rejected (if too stale)
- **Results**: 72.80% accuracy in 11.65s

**Simulated Delays:**
```
Client 0: Always on-time (0% delay)
Client 1: 40% chance of 2s delay
Client 2: 60% chance of 3s delay  ← frequently delayed
Client 3: 30% chance of 1.5s delay
```

### 4. Compare Synchronous vs Asynchronous

```bash
python experiments/compare_results.py
```

**Generates:**
- 📊 Detailed comparison report
- 📈 Side-by-side visualization (6-panel dashboard)
- 📋 Metrics JSON for integration

**Quick Comparison:**

| Metric | Sync | Async | Winner |
|--------|------|-------|--------|
| **Final Accuracy** | 72.00% | 72.80% | Async 🎯 |
| **Total Time** | 14.64s | 11.65s | Async ⚡ (20% faster) |
| **Avg Latency/Round** | 2.93s | 2.33s | Async ⏱️ |
| **Stale Rejections** | 0 | 8 | Sync (no rejections) |
| **Handles Delays** | No ❌ | Yes ✅ | Async |
| **Stragglers Impact** | HIGH | NONE | Async |

### 5. Generate Performance Metrics & Visualizations

```bash
python experiments/analyze_and_visualize.py
```

**Outputs:**
- `fl_metrics_visualization.png` - 4-panel performance dashboard
- `performance_summary.json` - Machine-readable metrics
- `PERFORMANCE_REPORT.md` - Detailed analysis

### 6. Generate Architecture Diagrams

```bash
python experiments/visualize_pipeline.py
```

**Output:**
- `fl_pipeline_architecture.png` - Complete pipeline visualization

---

## 📁 Project Structure

```
Synchronous-Bi-Lstm/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── configs/
│   └── fl_config.yaml            # Configuration file
│
├── data/
│   ├── README.md                 # Data documentation
│   ├── preprocess_data.py        # Preprocessing script
│   ├── raw/
│   │   └── crop_fertilizer.csv   # Original dataset
│   ├── processed/
│   │   └── cleaned.csv           # Cleaned dataset
│   └── partitions/
│       ├── client_0.npz, client_1.npz, client_2.npz, client_3.npz
│
├── src/
│   ├── model.py                  # BiLSTM model (shared)
│   ├── dataset.py                # Data loading (shared)
│   ├── utils.py                  # Utilities (shared)
│   │
│   ├── sync_client.py            # Synchronous FL client
│   ├── sync_server.py            # Synchronous FL server (FedAvg)
│   │
│   ├── async_client.py           # Asynchronous FL client (NEW)
│   └── async_server.py           # Async server with staleness awareness (NEW)
│
├── experiments/
│   ├── run_simulation.py         # Synchronous FL orchestration
│   ├── run_async_simulation.py   # Asynchronous FL orchestration (NEW)
│   ├── compare_results.py        # Sync vs Async comparison (NEW)
│   ├── analyze_and_visualize.py  # Metrics & visualization
│   ├── visualize_pipeline.py     # Architecture diagrams
│   │
│   └── results/
│       ├── sync/                 # Sync FL results
│       ├── async/                # Async FL results (NEW)
│       │
│       ├── comparison/           # Comparison results (NEW)
│       │   ├── COMPARISON_REPORT.md
│       │   ├── sync_vs_async_comparison.png
│       │   └── comparison_metrics.json
│       │
│       ├── fl_metrics_visualization.png
│       ├── fl_pipeline_architecture.png
│       ├── PERFORMANCE_REPORT.md
│       ├── QUICK_REFERENCE_PIPELINE.md
│       ├── FL_PIPELINE_DOCUMENTATION.md
│       └── performance_summary.json
│
├── QUICK_ACCESS_GUIDE.md         # Quick reference guide
└── RESULTS_SUMMARY.md            # Results overview
```

**New Async FL Components:**
- `src/async_client.py` - Client with staleness tracking & delay simulation
- `src/async_server.py` - Continuous aggregation with staleness-aware gradient rejection
- `experiments/run_async_simulation.py` - Async orchestration with client delays
- `experiments/compare_results.py` - Comprehensive sync vs async comparison
- `experiments/results/comparison/` - All comparison outputs

---

## 📈 Results

### Synchronous FL (FedAvg) Results

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 72.00% |
| **Initial Accuracy** | 55.25% |
| **Improvement** | +16.75% |
| **Total Time** | 14.64 seconds |
| **Per-Round Time** | ~2.93 seconds (consistent) |
| **Number of Rounds** | 5 |
| **Number of Clients** | 4 |

### Asynchronous FL (Staleness-Aware) Results

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 72.80% |
| **Initial Accuracy** | 55.80% |
| **Improvement** | +17.00% |
| **Total Time** | 11.65 seconds |
| **Per-Round Time** | ~2.33 seconds (variable) |
| **Number of Rounds** | 5 |
| **Number of Clients** | 4 |
| **Stale Updates Rejected** | 8 out of 20 |
| **Speedup vs Sync** | **1.26x faster** |

### Key Findings

**Synchronous FL:**
- ✅ Predictable, consistent behavior
- ✅ All client updates included
- ❌ Vulnerable to stragglers
- ❌ Blocked waiting for slow clients

**Asynchronous FL:**
- ✅ **20.4% faster** convergence
- ✅ **0.80% higher** final accuracy
- ✅ Handles client delays gracefully
- ✅ Automatic sync for stale clients
- ⚠️ Some updates rejected (if too stale)

### Per-Client Performance

**Synchronous:**
```
Client 0: 72.00% (+17%) | Client 1: 73.00% (+15%) ⭐
Client 2: 71.00% (+19%) | Client 3: 72.00% (+16%)
```

**Asynchronous (with delays):**
```
Client 0 (no delay):    74.00% (+18%) ⭐
Client 1 (40% delayed): 70.00% (+16%)
Client 2 (60% delayed): 68.00% (+18%)  ← frequently delayed
Client 3 (30% delayed): 72.00% (+16%)
```

---

## 📚 Documentation

### Core Documentation

1. **[FL_PIPELINE_DOCUMENTATION.md](experiments/results/FL_PIPELINE_DOCUMENTATION.md)**
   - Complete architecture breakdown
   - Data preprocessing pipeline
   - Model details and training process
   - Parameter communication flow
   - Synchronous FL explanation

2. **[PERFORMANCE_REPORT.md](experiments/results/PERFORMANCE_REPORT.md)**
   - Detailed accuracy metrics (Sync)
   - Latency analysis
   - Client-wise performance
   - Key insights and findings

3. **[QUICK_REFERENCE_PIPELINE.md](experiments/results/QUICK_REFERENCE_PIPELINE.md)**
   - ASCII architecture diagrams
   - Quick facts and statistics
   - Data flow visualization
   - Timing breakdown

### Comparison Documentation (NEW)

4. **[COMPARISON_REPORT.md](experiments/results/comparison/COMPARISON_REPORT.md)** ⭐
   - **Detailed sync vs async comparison**
   - Accuracy progression comparison
   - Convergence speed analysis
   - Robustness & staleness handling
   - Per-client performance analysis
   - Quantitative summary table
   - Recommendations for each approach

5. **[sync_vs_async_comparison.png](experiments/results/comparison/sync_vs_async_comparison.png)** ⭐
   - 6-panel visualization dashboard
   - Accuracy progression (both approaches)
   - Per-round latency comparison
   - Client accuracy distribution
   - Performance metrics summary

6. **[comparison_metrics.json](experiments/results/comparison/comparison_metrics.json)**
   - Machine-readable comparison metrics
   - Integration-ready format

---

## 🔧 Configuration

Modify federated learning parameters in [`experiments/run_simulation.py`](experiments/run_simulation.py):

```python
# Server configuration
start_server(
    num_rounds=5,        # Number of training rounds
    num_clients=4,       # Expected number of clients
    input_size=1,        # LSTM input feature size
    num_classes=16       # Number of crop classes
)
```

Or adjust model hyperparameters in [`src/model.py`](src/model.py):

```python
model = BiLSTMModel(
    input_size=1,        # Features per timestep
    hidden_size=32,      # LSTM hidden units
    num_layers=1,        # LSTM layers
    num_classes=16       # Output classes
)
```

---

## 🐛 Troubleshooting

### Issue: Port already in use

```bash
# Kill process using port 8080
lsof -ti:8080 | xargs kill -9
```

### Issue: Module not found

```bash
# Ensure virtual environment is activated
source flwr-project/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Data files not found

```bash
# Re-run preprocessing
python data/preprocess_data.py
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👥 Authors

**Tanvir Rahman**
- GitHub: [@tanvirRahman5](https://github.com/tanvirRahman5)

---

## 🙏 Acknowledgments

- **Flower Framework** - For providing an excellent federated learning platform
- **PyTorch** - For the deep learning framework
- **Crop Fertilizer Dataset** - For the agricultural data

---

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
