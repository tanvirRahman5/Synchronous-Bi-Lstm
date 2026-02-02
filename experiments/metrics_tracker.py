# experiments/metrics_tracker.py
import json
from pathlib import Path

class MetricsTracker:
    def __init__(self, output_file="experiments/fl_metrics.json"):
        self.output_file = output_file
        self.metrics = {
            'rounds': [],
            'global_loss': [],
            'global_accuracy': [],
            'client_accuracies': {0: [], 1: [], 2: [], 3: []},
            'round_times': [],
            'total_time': 0
        }
    
    def add_round_metrics(self, round_num, loss, accuracy, client_accs, round_time):
        """Add metrics for a completed round"""
        self.metrics['rounds'].append(round_num)
        self.metrics['global_loss'].append(loss)
        self.metrics['global_accuracy'].append(accuracy)
        self.metrics['round_times'].append(round_time)
        
        for client_id, acc in client_accs.items():
            self.metrics['client_accuracies'][client_id].append(acc)
    
    def set_total_time(self, total_time):
        """Set total simulation time"""
        self.metrics['total_time'] = total_time
    
    def save(self):
        """Save metrics to JSON file"""
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {self.output_file}")
    
    def get_summary(self):
        """Return summary statistics"""
        summary = {
            'avg_accuracy': sum(self.metrics['global_accuracy']) / len(self.metrics['global_accuracy']) if self.metrics['global_accuracy'] else 0,
            'final_accuracy': self.metrics['global_accuracy'][-1] if self.metrics['global_accuracy'] else 0,
            'best_accuracy': max(self.metrics['global_accuracy']) if self.metrics['global_accuracy'] else 0,
            'avg_latency_per_round': sum(self.metrics['round_times']) / len(self.metrics['round_times']) if self.metrics['round_times'] else 0,
            'total_time': self.metrics['total_time']
        }
        return summary
