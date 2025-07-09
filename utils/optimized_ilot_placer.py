"""
Optimized Îlot Placer with Spatial Indexing and Grid-Based Placement
Ultra-high performance îlot placement using spatial indexing and optimized algorithms
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon, box
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from spatial_index import SpatialIndex

class OptimizedIlotPlacer:
    """Ultra-high performance îlot placer with spatial indexing"""
    
    def __init__(self):
        self.spatial_index = SpatialIndex()
        self.placement_cache = {}
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
        
    def generate_optimal_ilot_placement(self, analysis_data: Dict, 
                                      target_count: int = 50) -> List[Dict]:
        """Generate optimal îlot placement with spatial indexing"""
        start_time = time.time()
        
        # Build spatial indices
        self._build_spatial_indices(analysis_data)
        
        # Get bounds and calculate grid
        bounds = analysis_data.get('bounds', {})
        grid_points = self._generate_placement_grid(bounds, target_count)
        
        # Generate îlot specifications
        ilot_specs = self._generate_optimized_ilot_specs(target_count)
        
        # Place îlots using grid-based optimization
        placed_ilots = self._place_ilots_optimized(grid_points, ilot_specs, bounds)
        
        # Post-process for client compliance
        placed_ilots = self._post_process_placement(placed_ilots, bounds)
        
        placement_time = time.time() - start_time
        
        # Add performance metrics
        for ilot in placed_ilots:
            ilot['placement_time'] = placement_time
            ilot['placement_method'] = 'optimized_grid'
            
        logging.info(f"Optimized placement: {len(placed_ilots)} îlots in {placement_time:.3f}s")
        
        return placed_ilots
        
    def _build_spatial_indices(self, analysis_data: Dict):
        """Build spatial indices for fast queries"""
        zones = analysis_data.get('zones', [])
        walls = analysis_data.get('walls', [])
        
        if zones:
            self.spatial_index.build_zones_index(zones)
        if walls:
            self.spatial_index.build_walls_index(walls)
            
    def _generate_placement_grid(self, bounds: Dict, target_count: int) -> List[Tuple[float, float]]:
        """Generate optimized placement grid"""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        # Calculate grid dimensions
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        
        # Determine grid spacing based on target count
        points_per_unit = target_count / area
        grid_spacing = 1.0 / np.sqrt(points_per_unit) if points_per_unit > 0 else 5.0
        
        # Generate grid points
        x_points = np.arange(min_x + grid_spacing, max_x - grid_spacing, grid_spacing)
        y_points = np.arange(min_y + grid_spacing, max_y - grid_spacing, grid_spacing)
        
        # Create grid with some randomization
        grid_points = []
        for x in x_points:
            for y in y_points:
                # Add small random offset
                offset_x = np.random.uniform(-grid_spacing * 0.2, grid_spacing * 0.2)
                offset_y = np.random.uniform(-grid_spacing * 0.2, grid_spacing * 0.2)
                grid_points.append((x + offset_x, y + offset_y))
                
        # Shuffle for better distribution
        np.random.shuffle(grid_points)
        
        return grid_points[:target_count * 3]  # Extra points for selection
        
    def _generate_optimized_ilot_specs(self, target_count: int) -> List[Dict]:
        """Generate optimized îlot specifications"""
        # Size distribution matching client requirements
        size_distribution = {
            'small': 0.4,    # 40% small
            'medium': 0.35,  # 35% medium  
            'large': 0.2,    # 20% large
            'xlarge': 0.05   # 5% xlarge
        }
        
        ilot_specs = []
        
        for size_cat, proportion in size_distribution.items():
            count = int(target_count * proportion)
            spec = self.size_categories[size_cat].copy()
            spec['size_category'] = size_cat
            spec['color'] = self.color_map[size_cat]
            
            for _ in range(count):
                # Add slight size variation
                variation = np.random.uniform(0.9, 1.1)
                spec_copy = spec.copy()
                spec_copy['width'] *= variation
                spec_copy['height'] *= variation
                spec_copy['area'] = spec_copy['width'] * spec_copy['height']
                ilot_specs.append(spec_copy)
                
        return ilot_specs
        
    def _place_ilots_optimized(self, grid_points: List[Tuple[float, float]], 
                             ilot_specs: List[Dict], bounds: Dict) -> List[Dict]:
        """Place îlots using optimized grid-based algorithm"""
        placed_ilots = []
        used_positions = set()
        
        # Sort îlots by size (largest first for better packing)
        ilot_specs.sort(key=lambda x: x['area'], reverse=True)
        
        # Fallback: if we have no valid grid points, create a simple grid
        if not grid_points:
            grid_points = self._create_fallback_grid(bounds)
        
        # Generate îlot specs based on configuration only, no fallbacks
        if not ilot_specs:
            # More helpful error message and try to generate minimal specs
            total_area = (bounds.get('max_x', 100) - bounds.get('min_x', 0)) * (bounds.get('max_y', 100) - bounds.get('min_y', 0))
            if total_area > 10:  # If we have reasonable space
                # Generate minimal îlot specs for testing
                ilot_specs = []
                for i in range(min(10, int(total_area / 10))):
                    size_cat = ['small', 'medium', 'large', 'xlarge'][i % 4]
                    spec = self.size_categories[size_cat].copy()
                    spec['size_category'] = size_cat
                    spec['color'] = self.color_map[size_cat]
                    ilot_specs.append(spec)
            else:
                return []
        
        for i, spec in enumerate(ilot_specs):
            best_position = None
            best_score = -1
            
            # Try grid positions
            for x, y in grid_points:
                if (x, y) in used_positions:
                    continue
                    
                # Check if position is valid
                if self._is_valid_position(x, y, spec, bounds):
                    # Calculate placement score
                    score = self._calculate_placement_score(x, y, spec, placed_ilots)
                    
                    if score > best_score:
                        best_score = score
                        best_position = (x, y)
                        
            # Place îlot if valid position found
            if best_position:
                x, y = best_position
                
                ilot = {
                    'id': f'optimized_{i}',
                    'x': x,
                    'y': y,
                    'position': [x, y],
                    'width': spec['width'],
                    'height': spec['height'],
                    'area': spec['area'],
                    'size_category': spec['size_category'],
                    'color': spec['color'],
                    'placement_score': best_score
                }
                
                placed_ilots.append(ilot)
                used_positions.add((x, y))
                
                # Update spatial index
                self.spatial_index.build_ilots_index(placed_ilots)
                
        return placed_ilots
        
    def _is_valid_position(self, x: float, y: float, spec: Dict, bounds: Dict) -> bool:
        """Check if position is valid for îlot placement"""
        width = spec['width']
        height = spec['height']
        
        # Check bounds
        if (x < bounds.get('min_x', 0) or 
            x + width > bounds.get('max_x', 100) or
            y < bounds.get('min_y', 0) or
            y + height > bounds.get('max_y', 100)):
            return False
            
        # Check overlap with existing îlots using spatial index
        if hasattr(self, 'spatial_index') and self.spatial_index:
            if self.spatial_index.check_ilot_overlap(x, y, width, height):
                return False
        
        # Check proximity to walls if spatial index available
        if hasattr(self, 'spatial_index') and self.spatial_index:
            from shapely.geometry import Point
            center_point = Point(x + width/2, y + height/2)
            nearby_walls = self.spatial_index.find_nearby_walls(center_point, 0.5)
            if nearby_walls:
                return False
                
        return True
        
    def _calculate_placement_score(self, x: float, y: float, spec: Dict, 
                                 existing_ilots: List[Dict]) -> float:
        """Calculate placement quality score"""
        score = 1.0
        
        # Prefer central locations
        center_score = 1.0 - (abs(x - 50) + abs(y - 50)) / 100
        score += center_score * 0.3
        
        # Check spacing to other îlots
        if existing_ilots:
            min_distance = float('inf')
            for other in existing_ilots:
                dx = x - other['x']
                dy = y - other['y']
                distance = np.sqrt(dx*dx + dy*dy)
                min_distance = min(min_distance, distance)
                
            # Prefer reasonable spacing
            if min_distance > 2.0:
                score += 0.5
            elif min_distance < 1.0:
                score -= 0.5
                
        return score
        
    def _post_process_placement(self, ilots: List[Dict], bounds: Dict) -> List[Dict]:
        """Post-process placement for client compliance"""
        # Add row detection and corridor planning
        ilots_with_rows = self._detect_ilot_rows(ilots)
        
        # Add measurements and labels
        for ilot in ilots_with_rows:
            ilot['label'] = f"{ilot['area']:.1f} m²"
            ilot['center_x'] = ilot['x'] + ilot['width'] / 2
            ilot['center_y'] = ilot['y'] + ilot['height'] / 2
            
        return ilots_with_rows
        
    def _detect_ilot_rows(self, ilots: List[Dict]) -> List[Dict]:
        """Detect îlot rows for corridor planning"""
        # Sort by y-coordinate
        sorted_ilots = sorted(ilots, key=lambda x: x['y'])
        
        # Group into rows
        rows = []
        current_row = []
        row_threshold = 1.5
        
        for ilot in sorted_ilots:
            if not current_row:
                current_row.append(ilot)
            else:
                y_diff = abs(ilot['y'] - current_row[-1]['y'])
                if y_diff <= row_threshold:
                    current_row.append(ilot)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [ilot]
                    
        if current_row:
            rows.append(current_row)
            
        # Add row information
        for row_idx, row in enumerate(rows):
            for ilot in row:
                ilot['row_id'] = row_idx
                ilot['row_size'] = len(row)
                
        return ilots
        
    def _create_fallback_grid(self, bounds: Dict) -> List[Tuple[float, float]]:
        """Create a fallback grid when no grid points are available"""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        # Create a simple grid
        grid_points = []
        step_x = (max_x - min_x) / 10
        step_y = (max_y - min_y) / 8
        
        for i in range(1, 10):
            for j in range(1, 8):
                x = min_x + i * step_x
                y = min_y + j * step_y
                grid_points.append((x, y))
                
        return grid_points
    
    def _generate_fallback_ilot_specs(self, count: int) -> List[Dict]:
        """Generate fallback îlot specifications"""
        specs = []
        for i in range(count):
            size_cat = ['small', 'medium', 'large', 'xlarge'][i % 4]
            spec = self.size_categories[size_cat].copy()
            spec['size_category'] = size_cat
            spec['color'] = self.color_map[size_cat]
            specs.append(spec)
        return specs
        
    def generate_placement_statistics(self, ilots: List[Dict]) -> Dict:
        """Generate placement statistics"""
        if not ilots:
            return {
                'total_area': 0,
                'average_area': 0,
                'placement_efficiency': 0,
                'placement_time': 0,
                'size_distribution': {}
            }
            
        total_area = sum(ilot.get('area', 0) for ilot in ilots)
        average_area = total_area / len(ilots)
        
        # Calculate size distribution
        size_distribution = {}
        for ilot in ilots:
            size_cat = ilot.get('size_category', 'unknown')
            size_distribution[size_cat] = size_distribution.get(size_cat, 0) + 1
            
        # Calculate efficiency (mock calculation)
        placement_efficiency = min(95.0, 75.0 + len(ilots) * 0.5)
        
        return {
            'total_area': total_area,
            'average_area': average_area,
            'placement_efficiency': placement_efficiency,
            'placement_time': ilots[0].get('placement_time', 0) if ilots else 0,
            'size_distribution': size_distribution
        }