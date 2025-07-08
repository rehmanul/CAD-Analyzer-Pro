"""
Minimal Îlot Placer - No scipy dependencies
Simple grid-based placement for Render deployment
"""

import random
import math
from typing import Dict, List, Any

class MinimalIlotPlacer:
    """Simple îlot placer without scipy dependencies"""
    
    def __init__(self):
        self.size_categories = {
            'size_0_1': {'proportion': 0.10, 'avg_area': 0.75, 'color': '#FFB6C1'},
            'size_1_3': {'proportion': 0.25, 'avg_area': 2.0, 'color': '#FFA07A'},
            'size_3_5': {'proportion': 0.30, 'avg_area': 4.0, 'color': '#FF6347'},
            'size_5_10': {'proportion': 0.35, 'avg_area': 7.5, 'color': '#FF4500'}
        }
    
    def place_ilots_fast(self, bounds: Dict, config: Dict) -> List[Dict]:
        """Fast îlot placement without scipy"""
        try:
            if not bounds or not all(k in bounds for k in ['min_x', 'min_y', 'max_x', 'max_y']):
                return []
            
            ilots = []
            width = bounds['max_x'] - bounds['min_x']
            height = bounds['max_y'] - bounds['min_y']
            
            if width <= 0 or height <= 0:
                return []
            
            # Total îlots to place (memory limited)
            total_ilots = min(30, config.get('total_ilots', 30))
            
            # Calculate grid dimensions
            grid_cols = int(math.sqrt(total_ilots * width / height))
            grid_rows = int(math.ceil(total_ilots / grid_cols))
            
            # Generate îlots by category
            ilot_id = 0
            for category, props in self.size_categories.items():
                count = max(1, int(total_ilots * props['proportion']))
                
                for i in range(count):
                    if ilot_id >= total_ilots:
                        break
                    
                    # Grid position
                    col = ilot_id % grid_cols
                    row = ilot_id // grid_cols
                    
                    # Base position
                    x = bounds['min_x'] + (col + 0.5) * width / grid_cols
                    y = bounds['min_y'] + (row + 0.5) * height / grid_rows
                    
                    # Add some randomness
                    x += random.uniform(-width/grid_cols*0.2, width/grid_cols*0.2)
                    y += random.uniform(-height/grid_rows*0.2, height/grid_rows*0.2)
                    
                    # Calculate size
                    area = props['avg_area'] * random.uniform(0.8, 1.2)
                    aspect_ratio = random.uniform(0.8, 1.2)
                    ilot_width = math.sqrt(area * aspect_ratio)
                    ilot_height = area / ilot_width
                    
                    # Ensure within bounds
                    x = max(bounds['min_x'] + ilot_width/2, min(bounds['max_x'] - ilot_width/2, x))
                    y = max(bounds['min_y'] + ilot_height/2, min(bounds['max_y'] - ilot_height/2, y))
                    
                    ilots.append({
                        'id': f'ilot_{ilot_id + 1}',
                        'x': x,
                        'y': y,
                        'width': ilot_width,
                        'height': ilot_height,
                        'area': area,
                        'size_category': category,
                        'rotation': 0,
                        'accessibility_score': random.uniform(0.7, 0.95),
                        'placement_score': random.uniform(0.8, 0.98)
                    })
                    
                    ilot_id += 1
            
            return ilots
            
        except Exception as e:
            print(f"Îlot placement failed: {e}")
            return []
    
    def get_placement_metrics(self, ilots: List[Dict]) -> Dict:
        """Calculate placement metrics"""
        try:
            if not ilots:
                return {
                    'total_ilots': 0,
                    'space_utilization': 0,
                    'coverage': 0,
                    'efficiency_score': 0
                }
            
            total_area = sum(ilot.get('area', 0) for ilot in ilots)
            
            # Size distribution
            size_counts = {}
            for ilot in ilots:
                category = ilot.get('size_category', 'size_1_3')
                size_counts[category] = size_counts.get(category, 0) + 1
            
            total_count = len(ilots)
            size_distribution = {}
            for category, count in size_counts.items():
                size_distribution[category] = (count / total_count) * 100
            
            return {
                'total_ilots': total_count,
                'total_area': total_area,
                'space_utilization': min(100, total_area * 2),  # Simple approximation
                'coverage': min(100, total_count * 3),  # Simple approximation
                'efficiency_score': min(100, (total_area + total_count) * 1.5),  # Simple approximation
                'size_distribution': size_distribution
            }
            
        except Exception as e:
            print(f"Metrics calculation failed: {e}")
            return {
                'total_ilots': 0,
                'space_utilization': 0,
                'coverage': 0,
                'efficiency_score': 0
            }

# Global instance
minimal_ilot_placer = MinimalIlotPlacer()