"""
Render Memory Optimizer - Prevents memory issues on Render deployment
Handles file size limits and memory management for 512MB constraint
"""

import os
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RenderMemoryOptimizer:
    """Memory optimizer for Render deployment"""
    
    def __init__(self):
        self.max_file_size = 3 * 1024 * 1024  # 3MB in bytes
        self.warning_file_size = 1 * 1024 * 1024  # 1MB warning
        self.max_entities = 500
        self.max_ilots = 30
        
    def check_file_size(self, file_size: int) -> bool:
        """Check if file size is within limits"""
        return file_size <= self.max_file_size
    
    def create_memory_warning(self, file_size: int) -> Optional[str]:
        """Create warning message for file size"""
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            return f"File too large: {size_mb:.1f}MB. Maximum allowed: 3MB. Large files cause memory issues on Render."
        elif file_size > self.warning_file_size:
            size_mb = file_size / (1024 * 1024)
            return f"Large file detected: {size_mb:.1f}MB. Consider reducing file size for better performance."
        return None
    
    def optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear any cached data
            if hasattr(self, '_cached_data'):
                self._cached_data = {}
                
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
    
    def get_file_size_mb(self, file_size: int) -> float:
        """Convert file size to MB"""
        return file_size / (1024 * 1024)
    
    def is_memory_safe(self, file_size: int) -> bool:
        """Check if file processing is memory safe"""
        return file_size <= self.max_file_size
    
    def reduce_file_complexity(self, entities: list) -> list:
        """Reduce file complexity for memory efficiency"""
        if len(entities) > self.max_entities:
            return entities[:self.max_entities]
        return entities
    
    def limit_ilots(self, ilots: list) -> list:
        """Limit number of ilots for memory efficiency"""
        if len(ilots) > self.max_ilots:
            return ilots[:self.max_ilots]
        return ilots

# Global instance
render_optimizer = RenderMemoryOptimizer()