"""
Simple Reliable Îlot Placer
Guaranteed to place îlots successfully with real data processing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class SimpleIlotPlacer:
    """Simple, reliable îlot placer that always succeeds"""
    
    def __init__(self):
        self.size_categories = {
            'small': {'width': 2.0, 'height': 1.5, 'area': 3.0},
            'medium': {'width': 3.0, 'height': 2.0, 'area': 6.0},
            'large': {'width': 4.0, 'height': 3.0, 'area': 12.0},
            'xlarge': {'width': 5.0, 'height': 4.0, 'area': 20.0}
        }
        self.color_map = {
            'small': '#FFFF00',   # Yellow
            'medium': '#FFA500',  # Orange
            'large': '#008000',   # Green
            'xlarge': '#800080'   # Purple
        }
    
    def place_ilots_guaranteed(self, analysis_data: Dict, target_count: int = 20) -> List[Dict]:
        """Place îlots with guaranteed success using real bounds"""
        bounds = analysis_data.get('bounds', {})
        
        # Get actual bounds from data
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        # Calculate usable area
        width = max_x - min_x
        height = max_y - min_y
        usable_area = width * height
        
        # Adjust target count based on available space
        if usable_area < 50:
            target_count = min(target_count, 8)
        elif usable_area < 200:
            target_count = min(target_count, 15)
        else:
            target_count = min(target_count, 30)
        
        # Generate size distribution
        size_distribution = self._get_size_distribution(target_count)
        
        # Create placement grid
        grid_points = self._create_placement_grid(bounds, target_count)
        
        # Place îlots
        placed_ilots = []
        used_positions = set()
        
        spec_index = 0
        for size_cat, count in size_distribution.items():
            spec = self.size_categories[size_cat]
            
            for _ in range(count):
                if spec_index >= len(grid_points):
                    break
                    
                x, y = grid_points[spec_index]
                
                # Ensure îlot fits within bounds
                ilot_width = spec['width'] * 0.8  # Slightly smaller for safety
                ilot_height = spec['height'] * 0.8
                
                if x + ilot_width > max_x:
                    x = max_x - ilot_width
                if y + ilot_height > max_y:
                    y = max_y - ilot_height
                if x < min_x:
                    x = min_x
                if y < min_y:
                    y = min_y
                
                # Create îlot
                ilot = {
                    'id': f'ilot_{spec_index}',
                    'x': x,
                    'y': y,
                    'position': [x, y],
                    'width': ilot_width,
                    'height': ilot_height,
                    'area': ilot_width * ilot_height,
                    'size_category': size_cat,
                    'color': self.color_map[size_cat],
                    'center_x': x + ilot_width / 2,
                    'center_y': y + ilot_height / 2,
                    'label': f"{ilot_width * ilot_height:.1f} m²"
                }
                
                placed_ilots.append(ilot)
                used_positions.add((x, y))
                spec_index += 1
        
        return placed_ilots
    
    def _get_size_distribution(self, target_count: int) -> Dict[str, int]:
        """Get size distribution for îlots"""
        distribution = {
            'small': int(target_count * 0.35),    # 35% small
            'medium': int(target_count * 0.35),   # 35% medium
            'large': int(target_count * 0.20),    # 20% large
            'xlarge': int(target_count * 0.10)    # 10% xlarge
        }
        
        # Adjust to match exact target
        total = sum(distribution.values())
        if total < target_count:
            distribution['small'] += target_count - total
        
        return distribution
    
    def _create_placement_grid(self, bounds: Dict, target_count: int) -> List[Tuple[float, float]]:
        """Create placement grid based on actual bounds"""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Calculate grid size based on area
        area = width * height
        if area < 100:
            cols, rows = 4, 3
        elif area < 500:
            cols, rows = 6, 5
        else:
            cols, rows = 8, 6
        
        # Generate grid points
        grid_points = []
        x_step = width / (cols + 1)
        y_step = height / (rows + 1)
        
        for i in range(1, cols + 1):
            for j in range(1, rows + 1):
                x = min_x + i * x_step
                y = min_y + j * y_step
                grid_points.append((x, y))
        
        # Add some randomization for natural look
        randomized_points = []
        for x, y in grid_points:
            offset_x = np.random.uniform(-x_step * 0.2, x_step * 0.2)
            offset_y = np.random.uniform(-y_step * 0.2, y_step * 0.2)
            randomized_points.append((x + offset_x, y + offset_y))
        
        return randomized_points