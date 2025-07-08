"""
Render Memory Optimizer - Optimize for 512MB memory limit
Prevents memory crashes on Render deployment
"""

import gc
import streamlit as st
import logging
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

class RenderMemoryOptimizer:
    """Memory optimization for Render deployment"""
    
    def __init__(self):
        self.max_memory_mb = 400  # Safe limit for 512MB instance
        self.max_file_size = 3 * 1024 * 1024  # 3MB max file
        self.max_entities = 500  # Limit entities
        self.max_ilots = 30  # Limit îlots for memory
        
    def optimize_on_startup(self):
        """Optimize memory on app startup"""
        try:
            # Clear any existing session state
            st.session_state.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Set streamlit config for memory efficiency
            st.set_page_config(
                page_title="CAD Analyzer Pro",
                page_icon="🏨",
                layout="wide",
                initial_sidebar_state="collapsed"  # Save memory
            )
            
            logger.info("Memory optimized on startup")
            
        except Exception as e:
            logger.error(f"Startup optimization failed: {str(e)}")
    
    def check_file_size(self, file_size: int) -> bool:
        """Check if file size is acceptable"""
        return file_size <= self.max_file_size
    
    def limit_entities(self, entities: List[Dict]) -> List[Dict]:
        """Limit entities to prevent memory issues"""
        try:
            if len(entities) > self.max_entities:
                logger.warning(f"Limiting entities from {len(entities)} to {self.max_entities}")
                return entities[:self.max_entities]
            return entities
        except Exception as e:
            logger.error(f"Entity limiting failed: {str(e)}")
            return entities[:100]  # Safe fallback
    
    def optimize_analysis_data(self, analysis_data: Dict) -> Dict:
        """Optimize analysis data for memory"""
        try:
            optimized = {
                'bounds': analysis_data.get('bounds', {}),
                'entities': self.limit_entities(analysis_data.get('entities', [])),
                'zones': analysis_data.get('zones', [])[:50]  # Limit zones
            }
            
            # Force garbage collection
            gc.collect()
            
            return optimized
            
        except Exception as e:
            logger.error(f"Analysis data optimization failed: {str(e)}")
            return {
                'bounds': {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80},
                'entities': [],
                'zones': []
            }
    
    def create_memory_efficient_ilots(self, bounds: Dict, target_count: int = None) -> List[Dict]:
        """Create memory-efficient îlots"""
        try:
            if target_count is None:
                target_count = self.max_ilots
            
            # Limit îlots for memory efficiency
            actual_count = min(target_count, self.max_ilots)
            
            ilots = []
            
            # Size distribution (client requirement)
            size_distribution = {
                'size_0_1': 0.10,   # 10%
                'size_1_3': 0.25,   # 25%
                'size_3_5': 0.30,   # 30%
                'size_5_10': 0.35   # 35%
            }
            
            # Generate îlots efficiently
            for i, (category, percentage) in enumerate(size_distribution.items()):
                category_count = max(1, int(actual_count * percentage))
                
                for j in range(category_count):
                    # Simple grid placement for memory efficiency
                    x = bounds.get('min_x', 0) + 10 + (j % 5) * 12
                    y = bounds.get('min_y', 0) + 10 + (j // 5) * 12
                    
                    # Size based on category
                    if category == 'size_0_1':
                        size = 2.5
                    elif category == 'size_1_3':
                        size = 4.0
                    elif category == 'size_3_5':
                        size = 6.0
                    else:  # size_5_10
                        size = 8.0
                    
                    ilot = {
                        'id': f'{category}_{j}',
                        'x': x,
                        'y': y,
                        'width': size,
                        'height': size,
                        'size_category': category,
                        'area': size * size
                    }
                    
                    ilots.append(ilot)
            
            logger.info(f"Created {len(ilots)} memory-efficient îlots")
            return ilots
            
        except Exception as e:
            logger.error(f"Memory-efficient îlot creation failed: {str(e)}")
            return []
    
    def cleanup_session_state(self):
        """Clean up session state to free memory"""
        try:
            # Keep only essential keys
            essential_keys = [
                'analysis_results', 'placed_ilots', 'placement_metrics',
                'ilot_distribution', 'user_authenticated', 'current_page'
            ]
            
            # Remove non-essential keys
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
            
            logger.info("Session state cleaned for memory")
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")
    
    def create_memory_warning(self, file_size: int) -> Optional[str]:
        """Create memory warning for file size"""
        try:
            if file_size > self.max_file_size:
                return f"File too large: {file_size} bytes. Maximum: {self.max_file_size} bytes (3MB)"
            elif file_size > 1 * 1024 * 1024:  # 1MB warning
                return f"Large file: {file_size} bytes. May use significant memory."
            
            return None
            
        except Exception as e:
            logger.error(f"Memory warning creation failed: {str(e)}")
            return None
    
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup"""
        try:
            # Clear all session state except authentication
            essential_keys = ['user_authenticated']
            
            keys_to_remove = []
            for key in st.session_state:
                if key not in essential_keys:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    del st.session_state[key]
                except:
                    pass
            
            # Force aggressive garbage collection
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            logger.warning("Emergency memory cleanup performed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {str(e)}")

# Global optimizer instance
render_optimizer = RenderMemoryOptimizer()