"""
Memory Manager - Optimize memory usage for Render deployment
Prevents memory limit crashes on 512MB instances
"""

import gc
import os
import sys
import logging
from typing import Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manage memory usage for Render deployment"""
    
    def __init__(self):
        self.memory_limit = 400 * 1024 * 1024  # 400MB limit (safe margin)
        self.max_file_size = 10 * 1024 * 1024  # 10MB max file
        self.max_entities = 1000  # Limit entities processed
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage"""
        try:
            # Get memory info
            process = os.getpid()
            memory_info = self._get_memory_info()
            
            return {
                'current_mb': memory_info / (1024 * 1024),
                'limit_mb': self.memory_limit / (1024 * 1024),
                'usage_percent': (memory_info / self.memory_limit) * 100,
                'safe': memory_info < self.memory_limit
            }
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return {
                'current_mb': 0,
                'limit_mb': 400,
                'usage_percent': 0,
                'safe': True
            }
    
    def _get_memory_info(self) -> int:
        """Get memory usage in bytes"""
        try:
            # Try psutil first
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback to sys.getsizeof estimation
            return sys.getsizeof(st.session_state) * 100
    
    def optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Clear unused session state
            self._clear_unused_session_data()
            
            # Force garbage collection
            gc.collect()
            
            # Clear matplotlib cache if available
            try:
                import matplotlib
                matplotlib.pyplot.close('all')
            except ImportError:
                pass
            
            logger.info("Memory optimized")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
    
    def _clear_unused_session_data(self):
        """Clear unused session state data"""
        try:
            # Keep only essential session data
            essential_keys = [
                'analysis_results', 'placed_ilots', 'placement_metrics',
                'ilot_distribution', 'user_authenticated', 'current_page'
            ]
            
            # Clear non-essential keys
            keys_to_remove = []
            for key in st.session_state:
                if key not in essential_keys:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del st.session_state[key]
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")
    
    def is_file_too_large(self, file_size: int) -> bool:
        """Check if file is too large for memory"""
        return file_size > self.max_file_size
    
    def limit_entities(self, entities: list) -> list:
        """Limit entities to prevent memory issues"""
        try:
            if len(entities) > self.max_entities:
                logger.warning(f"Limiting entities from {len(entities)} to {self.max_entities}")
                return entities[:self.max_entities]
            return entities
        except Exception as e:
            logger.error(f"Entity limiting failed: {str(e)}")
            return entities[:100]  # Safe fallback
    
    def create_memory_warning(self) -> Optional[str]:
        """Create memory warning message"""
        try:
            memory_info = self.check_memory_usage()
            
            if memory_info['usage_percent'] > 80:
                return f"High memory usage: {memory_info['current_mb']:.1f}MB / {memory_info['limit_mb']:.1f}MB"
            elif memory_info['usage_percent'] > 60:
                return f"Memory usage: {memory_info['current_mb']:.1f}MB / {memory_info['limit_mb']:.1f}MB"
            
            return None
            
        except Exception as e:
            logger.error(f"Memory warning creation failed: {str(e)}")
            return None
    
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        try:
            # Clear all non-essential session state
            essential_keys = ['user_authenticated', 'current_page']
            
            keys_to_remove = []
            for key in st.session_state:
                if key not in essential_keys:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    del st.session_state[key]
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
            
            logger.warning("Emergency memory cleanup performed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {str(e)}")

# Global instance
memory_manager = MemoryManager()