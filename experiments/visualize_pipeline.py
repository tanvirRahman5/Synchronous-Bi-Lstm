# experiments/visualize_pipeline.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram():
    """Create comprehensive FL pipeline architecture diagram"""
    fig = plt.figure(figsize=(16, 20))
    
    # Define colors
    color_data = '#E8F4F8'
    color_process = '#B3E5FC'
    color_model = '#81D4FA'
    color_server = '#4FC3F7'
    color_client = '#29B6F6'
    color_result = '#039BE5'
    
    # Title
    fig.suptitle('Federated Learning Pipeline - Complete Architecture', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis('off')
    
    y_pos = 27
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: DATA PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "1. DATA PREPROCESSING", 0.5, y_pos, color_data)
    y_pos -= 1.5
    
    # Raw data
    box(ax, 1, y_pos, 3, 0.8, "Raw Dataset\ncrop_fertilizer.csv\n(1000+ samples)", color_data)
    arrow_down(ax, 2.5, y_pos - 0.4)
    y_pos -= 1.5
    
    # Data cleaning
    box(ax, 1, y_pos, 3, 1, "Data Cleaning\n✓ Drop unused columns\n✓ Remove duplicates", color_process)
    arrow_down(ax, 2.5, y_pos - 0.5)
    y_pos -= 2
    
    # Feature separation
    box(ax, 0.5, y_pos, 2.5, 1, "Separate Features\nInput: 13 features\nTarget: Crop (16 classes)", color_process)
    arrow_down(ax, 1.75, y_pos - 0.5)
    y_pos -= 2
    
    # Encoding
    box(ax, 0.5, y_pos, 2.5, 1.2, "Label & Categorical\nEncoding\n✓ LabelEncoder (Crop)\n✓ OneHot (Soil_color)", color_process)
    arrow_down(ax, 1.75, y_pos - 0.6)
    y_pos -= 2.2
    
    # Normalization
    box(ax, 0.5, y_pos, 2.5, 1, "Normalization\n(StandardScaler)\n6 numeric features", color_process)
    arrow_down(ax, 1.75, y_pos - 0.5)
    y_pos -= 2
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: DATA PARTITIONING
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "2. NON-IID DATA PARTITIONING (District-based)", 0.5, y_pos, color_data)
    y_pos -= 1.5
    
    box(ax, 0.5, y_pos, 2.5, 1.2, "District-based Split\n✓ 4 unique districts\n✓ Non-IID distribution\n✓ Heterogeneous data", color_process)
    
    # Partition visualization
    arrow_right(ax, 3, y_pos + 0.6)
    
    # Client partitions
    partition_y = y_pos
    for i, label in enumerate(['Client 0', 'Client 1', 'Client 2', 'Client 3']):
        x_pos = 4.5 + (i % 2) * 2.5
        y_p = partition_y - (i // 2) * 1.3
        box(ax, x_pos, y_p, 2, 0.8, f"{label}\nClient_{i}.npz\n[samples, 13, 1]", color_client)
    
    y_pos -= 2.5
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: MODEL ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "3. BI-LSTM MODEL ARCHITECTURE", 0.5, y_pos, color_model)
    y_pos -= 1.5
    
    # Model layers
    ax.text(0.5, y_pos, "Input Layer", fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor=color_model, alpha=0.7))
    ax.text(1.5, y_pos, "[batch_size, 13 timesteps, 1 feature]", fontsize=9, style='italic')
    y_pos -= 0.8
    
    arrow_down(ax, 1.25, y_pos)
    y_pos -= 0.6
    
    ax.text(0.5, y_pos, "BiLSTM Layer", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color_model, alpha=0.7))
    ax.text(1.5, y_pos, "input_size=1, hidden=32, bidirectional=True", fontsize=9, style='italic')
    ax.text(1.5, y_pos - 0.3, "Output: [batch, 13, 64]", fontsize=9, style='italic')
    y_pos -= 1
    
    arrow_down(ax, 1.25, y_pos)
    y_pos -= 0.6
    
    ax.text(0.5, y_pos, "Dense Layer", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color_model, alpha=0.7))
    ax.text(1.5, y_pos, "Linear(64, 16) - 16 crop classes", fontsize=9, style='italic')
    y_pos -= 0.8
    
    arrow_down(ax, 1.25, y_pos)
    y_pos -= 0.6
    
    ax.text(0.5, y_pos, "Output Layer", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color_model, alpha=0.7))
    ax.text(1.5, y_pos, "[batch_size, 16] - logits", fontsize=9, style='italic')
    
    # Training parameters on the right
    ax.text(4.5, y_pos + 2.5, "Training Config", fontsize=11, fontweight='bold')
    train_params = [
        "Optimizer: Adam(lr=0.001)",
        "Loss: CrossEntropyLoss",
        "Batch Size: 32",
        "Local Epochs: 1 per round"
    ]
    for i, param in enumerate(train_params):
        ax.text(4.5, y_pos + 2 - i*0.35, f"• {param}", fontsize=9)
    
    y_pos -= 2.5
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: FEDERATED LEARNING STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "4. FEDERATED LEARNING STRUCTURE", 0.5, y_pos, color_server)
    y_pos -= 1.5
    
    # Server box
    box(ax, 2.5, y_pos, 5, 0.9, "FL SERVER (localhost:8080)\nFedAvg Strategy - Synchronous Aggregation", color_server)
    server_y = y_pos
    y_pos -= 1.8
    
    # Communication arrows
    for i in range(4):
        client_x = 1 + i * 2
        # Down arrow from server
        ax.annotate('', xy=(client_x, y_pos), xytext=(client_x, server_y - 0.9),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        # Label
        ax.text(client_x - 0.3, server_y - 0.5, "Params", fontsize=8, style='italic')
    
    y_pos -= 0.5
    
    # Clients
    for i in range(4):
        client_x = 0.5 + i * 2
        box(ax, client_x, y_pos, 1.8, 1.2, 
            f"CLIENT {i}\n• Load data\n• Train locally\n• Send updates", 
            color_client)
    
    y_pos -= 2.2
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: ROUND-BY-ROUND PROCESS
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "5. FEDERATED LEARNING ROUNDS (5 Rounds Total)", 0.5, y_pos, color_result)
    y_pos -= 1.5
    
    # Round details
    rounds_info = [
        ("Round 1", "55.25%", "Initial parameters → All clients"),
        ("Round 2", "58.75%", "Aggregated weights from Round 1"),
        ("Round 3", "62.50%", "Aggregated weights from Round 2"),
        ("Round 4", "66.50%", "Aggregated weights from Round 3"),
        ("Round 5", "72.00%", "Final aggregated weights"),
    ]
    
    for i, (round_name, accuracy, description) in enumerate(rounds_info):
        x_pos = 0.5 + (i % 5) * 1.8
        y_p = y_pos if i < 5 else y_pos - 1
        
        color = color_result if accuracy == "72.00%" else color_process
        box(ax, x_pos, y_p, 1.6, 0.9, f"{round_name}\n{accuracy}", color)
        ax.text(x_pos + 0.1, y_p - 0.5, description, fontsize=7, style='italic')
    
    y_pos -= 2
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: PARAMETER COMMUNICATION
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "6. PARAMETER COMMUNICATION FLOW", 0.5, y_pos, color_server)
    y_pos -= 1.5
    
    # Step 1: Broadcast
    ax.text(0.5, y_pos, "Step 1: Broadcast (Server → Clients)", fontsize=10, fontweight='bold')
    y_pos -= 0.5
    box(ax, 0.7, y_pos, 4, 0.8, 
        "Server sends initial/aggregated model parameters to all clients\n" +
        f"Parameters per layer: ~8,000 floating point values",
        color_process)
    y_pos -= 1.2
    
    # Step 2: Local training
    ax.text(0.5, y_pos, "Step 2: Local Training (Clients)", fontsize=10, fontweight='bold')
    y_pos -= 0.5
    box(ax, 0.7, y_pos, 4, 1, 
        "Each client:\n" +
        "  • Loads local data from .npz file\n" +
        "  • Trains model for 1 epoch\n" +
        "  • Computes gradients and updates weights",
        color_process)
    y_pos -= 1.3
    
    # Step 3: Upload
    ax.text(0.5, y_pos, "Step 3: Upload (Clients → Server)", fontsize=10, fontweight='bold')
    y_pos -= 0.5
    box(ax, 0.7, y_pos, 4, 0.8, 
        "Clients send updated model weights (NOT gradients)\n" +
        "Communication: ~100KB per client per round",
        color_process)
    y_pos -= 1.2
    
    # Step 4: Aggregation
    ax.text(0.5, y_pos, "Step 4: Aggregation & Evaluation (Server)", fontsize=10, fontweight='bold')
    y_pos -= 0.5
    box(ax, 0.7, y_pos, 4, 1, 
        "Server:\n" +
        "  • Averages all received model weights (FedAvg)\n" +
        "  • Evaluates on client data\n" +
        "  • Prepares for next round",
        color_server)
    y_pos -= 1.3
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: TIMING & LATENCY
    # ═══════════════════════════════════════════════════════════════
    section_title(ax, "7. TIMING & LATENCY BREAKDOWN", 0.5, y_pos, color_result)
    y_pos -= 1.5
    
    timing_data = [
        ("Total Simulation", "14.64 seconds", "5 complete rounds"),
        ("Per Round", "2.93 seconds", "avg across all rounds"),
        ("Client Training", "~2.5 seconds", "per client per round"),
        ("Evaluation", "~200-300ms", "per round"),
        ("Communication", "~100-200ms", "param transfer")
    ]
    
    for i, (metric, value, note) in enumerate(timing_data):
        y_p = y_pos - i * 0.5
        ax.text(0.7, y_p, f"• {metric}:", fontsize=9, fontweight='bold')
        ax.text(2.2, y_p, value, fontsize=9, style='italic')
        ax.text(4, y_p, f"({note})", fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig('experiments/results/fl_pipeline_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ Pipeline architecture diagram saved to experiments/results/fl_pipeline_architecture.png")
    plt.close()

def section_title(ax, text, x, y, color):
    """Add section title with background"""
    ax.text(x, y, text, fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, edgecolor='black', linewidth=2))

def box(ax, x, y, width, height, text, color):
    """Draw a box with text"""
    rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color, alpha=0.7, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', fontsize=9, fontweight='bold', wrap=True)

def arrow_down(ax, x, y, length=0.4):
    """Draw downward arrow"""
    ax.annotate('', xy=(x, y - length), xytext=(x, y),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

def arrow_right(ax, x, y, length=0.8):
    """Draw rightward arrow"""
    ax.annotate('', xy=(x + length, y), xytext=(x, y),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

if __name__ == "__main__":
    create_pipeline_diagram()
