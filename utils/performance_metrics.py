"""
Performance Metrics System for Ultra-High Performance Optimizations
Real-time performance tracking and benchmarking
"""

import time
import sys
import gc
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

class PerformanceMetrics:
    """Ultra-high performance metrics tracking system"""
    
    def __init__(self):
        self.metrics = {}
        self.benchmarks = {}
        self.start_times = {}
        self.memory_usage = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str, item_count: int = 1) -> float:
        """End timing and calculate performance metrics"""
        if operation not in self.start_times:
            return 0.0
            
        elapsed_time = time.time() - self.start_times[operation]
        
        # Calculate performance metrics
        items_per_second = item_count / elapsed_time if elapsed_time > 0 else 0
        
        # Store metrics
        self.metrics[operation] = {
            'elapsed_time': elapsed_time,
            'item_count': item_count,
            'items_per_second': items_per_second,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up
        del self.start_times[operation]
        
        return elapsed_time
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        summary = {
            'total_operations': len(self.metrics),
            'operations': {}
        }
        
        for operation, metrics in self.metrics.items():
            summary['operations'][operation] = {
                'speed': f"{metrics['items_per_second']:.1f} items/sec",
                'time': f"{metrics['elapsed_time']:.3f}s",
                'items': metrics['item_count']
            }
            
        return summary
        
    def get_benchmark_comparison(self) -> Dict:
        """Compare current performance with benchmarks"""
        comparison = {}
        
        # Define benchmark targets
        benchmark_targets = {
            'dxf_processing': 500,  # entities/sec
            'ilot_placement': 100,  # îlots/sec
            'corridor_generation': 50,  # corridors/sec
            'visualization': 1000   # elements/sec
        }
        
        for operation, metrics in self.metrics.items():
            speed = metrics['items_per_second']
            target = benchmark_targets.get(operation, 100)
            
            performance_ratio = speed / target
            
            if performance_ratio >= 1.0:
                status = "✅ EXCEEDS TARGET"
            elif performance_ratio >= 0.8:
                status = "⚡ MEETS TARGET"
            elif performance_ratio >= 0.5:
                status = "⚠️ BELOW TARGET"
            else:
                status = "❌ NEEDS OPTIMIZATION"
                
            comparison[operation] = {
                'current_speed': speed,
                'target_speed': target,
                'performance_ratio': performance_ratio,
                'status': status
            }
            
        return comparison
        
    def log_performance_metrics(self):
        """Log performance metrics to console"""
        logging.info("=== ULTRA-HIGH PERFORMANCE METRICS ===")
        
        for operation, metrics in self.metrics.items():
            logging.info(f"{operation}: {metrics['items_per_second']:.1f} items/sec "
                        f"({metrics['item_count']} items in {metrics['elapsed_time']:.3f}s)")
                        
        logging.info("=" * 40)
        
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        # Get memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / (1024 * 1024),  # MB
                'vms': memory_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss': 0, 'vms': 0, 'percent': 0}
            
    def force_garbage_collection(self):
        """Force garbage collection for memory optimization"""
        gc.collect()
        
    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics.clear()
        self.benchmarks.clear()
        self.start_times.clear()
        self.memory_usage.clear()
        
# Global performance tracker instance
performance_tracker = PerformanceMetrics()

def track_performance(operation: str):
    """Decorator to track performance of functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_tracker.start_timer(operation)
            result = func(*args, **kwargs)
            
            # Try to get item count from result
            item_count = 1
            if isinstance(result, list):
                item_count = len(result)
            elif isinstance(result, dict):
                item_count = len(result.get('entities', result.get('items', [result])))
                
            performance_tracker.end_timer(operation, item_count)
            return result
        return wrapper
    return decorator