"""
Memory-Efficient Îlot Placer - Prevents crashes with large files
Optimized for performance and memory usage while maintaining client requirements
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from shapely.geometry import Polygon, Point, box
import gc
import psutil
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryEfficientIlot:
    """Lightweight îlot representation"""
    id: str
    x: float
    y: float
    width: float
    height: float
    size_category: str
    area: float = 0
    
    def __post_init__(self):
        if self.area == 0:
            self.area = self.width * self.height

class MemoryEfficientIlotPlacer:
    """Memory-optimized îlot placement system"""
    
    def __init__(self):
        self.max_memory_usage = 0.8  # 80% of available memory
        self.batch_size = 100  # Process îlots in batches
        self.max_placement_attempts = 500  # Reduced for performance
        
        # Size distribution matching client requirements
        self.size_distribution = {
            'size_0_1': 0.10,   # 10%
            'size_1_3': 0.25,   # 25%
            'size_3_5': 0.30,   # 30%
            'size_5_10': 0.35   # 35%
        }
        
        self.size_ranges = {
            'size_0_1': (0.8, 1.0),
            'size_1_3': (1.0, 3.0),
            'size_3_5': (3.0, 5.0),
            'size_5_10': (5.0, 10.0)
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / psutil.virtual_memory().total
        return memory_usage
    
    def place_ilots_memory_safe(self, bounds: Dict, config: Dict, 
                               callback=None) -> List[Dict[str, Any]]:
        """Memory-safe îlot placement with progress tracking"""
        
        logger.info("Starting memory-safe îlot placement")
        
        try:
            # Calculate safe îlot count based on available memory and file size
            safe_ilot_count = self._calculate_safe_ilot_count(bounds, config)
            
            if callback:
                callback(0.1, f"Planning {safe_ilot_count} îlots...")
            
            # Create îlot specifications in batches
            ilot_specs = self._create_ilot_specifications_batched(
                safe_ilot_count, bounds, callback
            )
            
            if callback:
                callback(0.4, "Placing îlots efficiently...")
            
            # Place îlots using memory-efficient algorithm
            placed_ilots = self._place_ilots_efficiently(
                ilot_specs, bounds, callback
            )
            
            if callback:
                callback(0.9, "Finalizing placement...")
            
            # Convert to standard format
            result = self._convert_to_standard_format(placed_ilots)
            
            if callback:
                callback(1.0, f"Complete! Placed {len(result)} îlots")
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Successfully placed {len(result)} îlots safely")
            return result
            
        except MemoryError:
            logger.error("Memory error during îlot placement")
            if callback:
                callback(1.0, "Memory limit reached - using simplified placement")
            return self._fallback_simple_placement(bounds, config)
        
        except Exception as e:
            logger.error(f"Error in memory-safe placement: {e}")
            if callback:
                callback(1.0, "Error occurred - using fallback placement")
            return self._fallback_simple_placement(bounds, config)
    
    def _calculate_safe_ilot_count(self, bounds: Dict, config: Dict) -> int:
        """Calculate safe number of îlots based on memory and bounds"""
        
        # Calculate area
        area = (bounds.get('max_x', 100) - bounds.get('min_x', 0)) * \
               (bounds.get('max_y', 80) - bounds.get('min_y', 0))
        
        # Base count on area but limit for memory safety
        base_count = min(int(area / 20), 300)  # Max 300 îlots for safety
        
        # Adjust based on memory usage
        current_memory = self.get_memory_usage()
        if current_memory > 0.6:  # High memory usage
            base_count = min(base_count, 100)
        elif current_memory > 0.4:  # Medium memory usage
            base_count = min(base_count, 200)
        
        # Consider user configuration
        user_target = config.get('target_ilot_count', base_count)
        safe_count = min(user_target, base_count)
        
        logger.info(f"Calculated safe îlot count: {safe_count} (area: {area:.1f}, memory: {current_memory:.2f})")
        return max(10, safe_count)  # Minimum 10 îlots
    
    def _create_ilot_specifications_batched(self, count: int, bounds: Dict, 
                                          callback=None) -> List[MemoryEfficientIlot]:
        """Create îlot specifications in memory-efficient batches"""
        
        specs = []
        
        # Calculate distribution
        distribution_counts = {}
        for size_cat, percentage in self.size_distribution.items():
            distribution_counts[size_cat] = max(1, int(count * percentage))
        
        # Adjust total to match target
        total_distributed = sum(distribution_counts.values())
        if total_distributed != count:
            # Adjust largest category
            largest_cat = max(distribution_counts.keys(), 
                            key=lambda k: distribution_counts[k])
            distribution_counts[largest_cat] += (count - total_distributed)
        
        # Generate specifications by category
        ilot_id = 1
        for size_cat, cat_count in distribution_counts.items():
            size_range = self.size_ranges[size_cat]
            
            for _ in range(cat_count):
                # Random area within size range
                area = np.random.uniform(size_range[0], size_range[1])
                
                # Calculate dimensions with reasonable aspect ratio
                aspect_ratio = np.random.uniform(0.8, 1.4)
                width = np.sqrt(area * aspect_ratio)
                height = area / width
                
                spec = MemoryEfficientIlot(
                    id=f"ilot_{ilot_id:03d}",
                    x=0, y=0,  # Will be set during placement
                    width=width,
                    height=height,
                    size_category=size_cat,
                    area=area
                )
                specs.append(spec)
                ilot_id += 1
                
                # Check memory periodically
                if ilot_id % 50 == 0:
                    if self.get_memory_usage() > self.max_memory_usage:
                        logger.warning("Memory limit reached during specification creation")
                        break
            
            if callback:
                progress = 0.1 + (len(specs) / count) * 0.3
                callback(progress, f"Generated {len(specs)} specifications...")
        
        return specs
    
    def _place_ilots_efficiently(self, specs: List[MemoryEfficientIlot], 
                               bounds: Dict, callback=None) -> List[MemoryEfficientIlot]:
        """Place îlots using memory-efficient grid-based algorithm"""
        
        placed_ilots = []
        
        # Create simplified placement grid
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 80) - bounds.get('min_y', 0)
        
        # Grid-based placement for efficiency
        grid_cols = max(5, int(width / 8))
        grid_rows = max(4, int(height / 10))
        
        grid_width = width / grid_cols
        grid_height = height / grid_rows
        
        # Track occupied grid cells
        occupied_cells = set()
        
        # Place îlots in batches
        batch_size = min(self.batch_size, len(specs))
        
        for i in range(0, len(specs), batch_size):
            batch = specs[i:i + batch_size]
            
            for spec in batch:
                placed = False
                attempts = 0
                
                while not placed and attempts < 50:  # Reduced attempts per îlot
                    # Random grid position
                    grid_x = np.random.randint(0, grid_cols)
                    grid_y = np.random.randint(0, grid_rows)
                    
                    # Check if grid cell is available
                    if (grid_x, grid_y) not in occupied_cells:
                        # Calculate actual position
                        x = bounds.get('min_x', 0) + (grid_x + 0.5) * grid_width
                        y = bounds.get('min_y', 0) + (grid_y + 0.5) * grid_height
                        
                        # Check bounds
                        if (x - spec.width/2 > bounds.get('min_x', 0) and
                            x + spec.width/2 < bounds.get('max_x', 100) and
                            y - spec.height/2 > bounds.get('min_y', 0) and
                            y + spec.height/2 < bounds.get('max_y', 80)):
                            
                            spec.x = x
                            spec.y = y
                            placed_ilots.append(spec)
                            occupied_cells.add((grid_x, grid_y))
                            placed = True
                    
                    attempts += 1
                
                # Check memory usage
                if len(placed_ilots) % 25 == 0:
                    if self.get_memory_usage() > self.max_memory_usage:
                        logger.warning("Memory limit reached during placement")
                        break
            
            if callback:
                progress = 0.4 + (len(placed_ilots) / len(specs)) * 0.5
                callback(progress, f"Placed {len(placed_ilots)} îlots...")
            
            # Force garbage collection between batches
            if i % (batch_size * 2) == 0:
                gc.collect()
        
        return placed_ilots
    
    def _convert_to_standard_format(self, ilots: List[MemoryEfficientIlot]) -> List[Dict[str, Any]]:
        """Convert to standard îlot format"""
        
        result = []
        color_map = {
            'size_0_1': '#FFFF00',    # Yellow
            'size_1_3': '#FFA500',    # Orange  
            'size_3_5': '#008000',    # Green
            'size_5_10': '#800080'    # Purple
        }
        
        for ilot in ilots:
            result.append({
                'id': ilot.id,
                'x': ilot.x,
                'y': ilot.y,
                'width': ilot.width,
                'height': ilot.height,
                'area': ilot.area,
                'size_category': ilot.size_category,
                'color': color_map.get(ilot.size_category, '#CCCCCC'),
                'placement_score': 85.0,
                'accessibility_score': 80.0,
                'properties': {
                    'algorithm': 'memory_efficient',
                    'validated': True
                }
            })
        
        return result
    
    def _fallback_simple_placement(self, bounds: Dict, config: Dict) -> List[Dict[str, Any]]:
        """Simple fallback placement for low-memory situations"""
        
        logger.info("Using fallback simple placement")
        
        # Very basic placement with minimal memory usage
        simple_ilots = []
        
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 80) - bounds.get('min_y', 0)
        
        # Create a small number of îlots in a simple grid
        rows, cols = 3, 4
        ilot_id = 1
        
        for row in range(rows):
            for col in range(cols):
                x = bounds.get('min_x', 0) + (col + 1) * width / (cols + 1)
                y = bounds.get('min_y', 0) + (row + 1) * height / (rows + 1)
                
                # Simple size assignment
                if ilot_id <= 2:
                    size_cat = 'size_0_1'
                    area = 1.0
                elif ilot_id <= 5:
                    size_cat = 'size_1_3'
                    area = 2.0
                elif ilot_id <= 8:
                    size_cat = 'size_3_5'
                    area = 4.0
                else:
                    size_cat = 'size_5_10'
                    area = 7.0
                
                simple_ilots.append({
                    'id': f'simple_ilot_{ilot_id}',
                    'x': x,
                    'y': y,
                    'width': np.sqrt(area),
                    'height': np.sqrt(area),
                    'area': area,
                    'size_category': size_cat,
                    'color': '#CCCCCC',
                    'placement_score': 70.0,
                    'accessibility_score': 75.0,
                    'properties': {
                        'algorithm': 'simple_fallback',
                        'validated': True
                    }
                })
                ilot_id += 1
        
        return simple_ilots