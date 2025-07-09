"""
Smart Îlot Placer
Places rectangular îlots intelligently in available floor space
Creates green rectangular îlots matching the reference image
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import random

class SmartIlotPlacer:
    """Intelligent îlot placement for clean floor plans"""
    
    def __init__(self):
        self.ilot_colors = {
            'small': '#10B981',    # Green for all îlots
            'medium': '#10B981',
            'large': '#10B981',
            'xlarge': '#10B981'
        }
        
    def place_ilots_smart(self, analysis_data: Dict, config: Dict) -> List[Dict]:
        """Place îlots intelligently in available floor space"""
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        walls = analysis_data.get('walls', [])
        restricted_areas = analysis_data.get('restricted_areas', [])
        
        # Find available rooms/spaces for îlot placement
        available_spaces = self._find_available_spaces(bounds, walls, restricted_areas)
        
        # Generate îlots based on size distribution
        target_ilots = self._calculate_target_ilots(available_spaces, config)
        
        # Place îlots in available spaces
        placed_ilots = self._place_ilots_in_spaces(available_spaces, target_ilots, config)
        
        return placed_ilots
    
    def _find_available_spaces(self, bounds: Dict, walls: List[Dict], restricted_areas: List[Dict]) -> List[Dict]:
        """Find available rectangular spaces for îlot placement"""
        # Define grid of potential spaces
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Create grid-based spaces
        grid_size = 10  # 10x10 meter grid
        spaces = []
        
        for x in range(int(min_x), int(max_x), grid_size):
            for y in range(int(min_y), int(max_y), grid_size):
                space = {
                    'x': x,
                    'y': y,
                    'width': min(grid_size, max_x - x),
                    'height': min(grid_size, max_y - y),
                    'area': min(grid_size, max_x - x) * min(grid_size, max_y - y)
                }
                
                # Check if space is available (not in restricted areas)
                if self._is_space_available(space, restricted_areas):
                    spaces.append(space)
        
        return spaces
    
    def _is_space_available(self, space: Dict, restricted_areas: List[Dict]) -> bool:
        """Check if space is available for îlot placement"""
        space_x1 = space['x']
        space_y1 = space['y']
        space_x2 = space['x'] + space['width']
        space_y2 = space['y'] + space['height']
        
        for area in restricted_areas:
            bounds = area.get('bounds', {})
            if bounds:
                area_x1 = bounds.get('min_x', 0)
                area_y1 = bounds.get('min_y', 0)
                area_x2 = bounds.get('max_x', 0)
                area_y2 = bounds.get('max_y', 0)
                
                # Check for overlap
                if not (space_x2 <= area_x1 or space_x1 >= area_x2 or 
                       space_y2 <= area_y1 or space_y1 >= area_y2):
                    return False
        
        return True
    
    def _calculate_target_ilots(self, available_spaces: List[Dict], config: Dict) -> Dict:
        """Calculate number of îlots needed per size category"""
        total_area = sum(space['area'] for space in available_spaces)
        utilization = config.get('utilization_target', 0.7)
        target_ilot_area = total_area * utilization
        
        # Size distributions from config
        size_0_1_pct = config.get('size_0_1_percent', 10) / 100
        size_1_3_pct = config.get('size_1_3_percent', 25) / 100
        size_3_5_pct = config.get('size_3_5_percent', 30) / 100
        size_5_10_pct = config.get('size_5_10_percent', 35) / 100
        
        # Average sizes per category
        avg_sizes = {
            'small': 0.5,    # 0-1 m²
            'medium': 2.0,   # 1-3 m²
            'large': 4.0,    # 3-5 m²
            'xlarge': 7.5    # 5-10 m²
        }
        
        # Calculate number of îlots per category
        target_counts = {
            'small': int((target_ilot_area * size_0_1_pct) / avg_sizes['small']),
            'medium': int((target_ilot_area * size_1_3_pct) / avg_sizes['medium']),
            'large': int((target_ilot_area * size_3_5_pct) / avg_sizes['large']),
            'xlarge': int((target_ilot_area * size_5_10_pct) / avg_sizes['xlarge'])
        }
        
        return target_counts
    
    def _place_ilots_in_spaces(self, available_spaces: List[Dict], target_ilots: Dict, config: Dict) -> List[Dict]:
        """Place îlots in available spaces"""
        placed_ilots = []
        used_spaces = []
        
        min_spacing = config.get('min_spacing', 1.0)
        
        # Size ranges for each category
        size_ranges = {
            'small': (0.8, 1.2),    # 0.8-1.2 m
            'medium': (1.2, 2.0),   # 1.2-2.0 m
            'large': (2.0, 2.5),    # 2.0-2.5 m
            'xlarge': (2.5, 3.5)    # 2.5-3.5 m
        }
        
        # Place îlots for each size category
        for category, count in target_ilots.items():
            size_range = size_ranges[category]
            
            for _ in range(count):
                # Find available space
                space = self._find_best_space(available_spaces, used_spaces, size_range, min_spacing)
                if space:
                    # Create îlot
                    width = random.uniform(size_range[0], size_range[1])
                    height = random.uniform(size_range[0], size_range[1])
                    
                    # Position within space with some randomness
                    max_x_offset = max(0, space['width'] - width - min_spacing)
                    max_y_offset = max(0, space['height'] - height - min_spacing)
                    
                    x_offset = random.uniform(0, max_x_offset) if max_x_offset > 0 else 0
                    y_offset = random.uniform(0, max_y_offset) if max_y_offset > 0 else 0
                    
                    ilot = {
                        'x': space['x'] + x_offset,
                        'y': space['y'] + y_offset,
                        'width': width,
                        'height': height,
                        'size_category': category,
                        'color': self.ilot_colors[category],
                        'bounds': {
                            'min_x': space['x'] + x_offset,
                            'max_x': space['x'] + x_offset + width,
                            'min_y': space['y'] + y_offset,
                            'max_y': space['y'] + y_offset + height
                        }
                    }
                    
                    placed_ilots.append(ilot)
                    used_spaces.append({
                        'x': space['x'],
                        'y': space['y'],
                        'width': space['width'],
                        'height': space['height']
                    })
        
        return placed_ilots
    
    def _find_best_space(self, available_spaces: List[Dict], used_spaces: List[Dict], 
                        size_range: Tuple[float, float], min_spacing: float) -> Dict:
        """Find best available space for îlot"""
        min_size = size_range[0] + min_spacing
        
        # Filter spaces that can fit the îlot
        suitable_spaces = [
            space for space in available_spaces 
            if (space['width'] >= min_size and space['height'] >= min_size and
                not self._space_overlaps_used(space, used_spaces))
        ]
        
        if not suitable_spaces:
            return None
        
        # Return random suitable space
        return random.choice(suitable_spaces)
    
    def _space_overlaps_used(self, space: Dict, used_spaces: List[Dict]) -> bool:
        """Check if space overlaps with already used spaces"""
        for used in used_spaces:
            if not (space['x'] + space['width'] <= used['x'] or 
                   space['x'] >= used['x'] + used['width'] or
                   space['y'] + space['height'] <= used['y'] or 
                   space['y'] >= used['y'] + used['height']):
                return True
        return False
    
    def calculate_placement_stats(self, placed_ilots: List[Dict]) -> Dict:
        """Calculate placement statistics"""
        if not placed_ilots:
            return {
                'total_ilots': 0,
                'total_area': 0,
                'average_size': 0,
                'size_distribution': {}
            }
        
        total_area = sum(ilot['width'] * ilot['height'] for ilot in placed_ilots)
        size_distribution = {}
        
        for ilot in placed_ilots:
            category = ilot.get('size_category', 'unknown')
            if category not in size_distribution:
                size_distribution[category] = 0
            size_distribution[category] += 1
        
        return {
            'total_ilots': len(placed_ilots),
            'total_area': total_area,
            'average_size': total_area / len(placed_ilots),
            'size_distribution': size_distribution
        }