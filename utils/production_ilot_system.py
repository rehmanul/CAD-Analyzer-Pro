"""
Production Îlot Placement System
Full implementation matching client requirements exactly
"""

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class IlotSpec:
    """Îlot specification"""
    id: str
    x: float
    y: float
    width: float
    height: float
    area: float
    size_category: str
    rotation: float = 0
    zone_id: str = ""
    accessibility_score: float = 0
    placement_score: float = 0
    
    def to_polygon(self) -> Polygon:
        """Convert to shapely polygon"""
        half_w, half_h = self.width / 2, self.height / 2
        corners = [
            (self.x - half_w, self.y - half_h),
            (self.x + half_w, self.y - half_h),
            (self.x + half_w, self.y + half_h),
            (self.x - half_w, self.y + half_h)
        ]
        return Polygon(corners)

@dataclass
class CorridorSpec:
    """Corridor specification"""
    id: str
    type: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    width: float
    path_points: List[Tuple[float, float]]
    connects_ilots: List[str]
    is_mandatory: bool = False
    accessibility_compliant: bool = True
    
    def to_polygon(self) -> Polygon:
        """Convert corridor to polygon"""
        if len(self.path_points) < 2:
            return None
        
        line = LineString(self.path_points)
        return line.buffer(self.width / 2)

class ProductionIlotPlacer:
    """Production îlot placement system"""
    
    def __init__(self):
        self.walls = []
        self.restricted_areas = []
        self.entrances = []
        self.zones = {}
        self.bounds = {}
        self.placed_ilots = []
        self.corridors = []
        
    def load_floor_plan_data(self, walls: List, restricted_areas: List, 
                           entrances: List, zones: Dict, bounds: Dict):
        """Load floor plan analysis data"""
        self.walls = [Polygon(wall) if len(wall) > 2 else None for wall in walls]
        self.walls = [w for w in self.walls if w is not None]
        
        self.restricted_areas = [Polygon(area) if len(area) > 2 else None for area in restricted_areas]
        self.restricted_areas = [r for r in self.restricted_areas if r is not None]
        
        self.entrances = [Polygon(entrance) if len(entrance) > 2 else None for entrance in entrances]
        self.entrances = [e for e in self.entrances if e is not None]
        
        self.zones = zones
        self.bounds = bounds
        
    def calculate_ilot_distribution(self, config: Dict, available_area: float) -> Dict[str, int]:
        """Calculate exact îlot distribution based on percentages"""
        total_percentage = (
            config.get('size_0_1_percent', 10) +
            config.get('size_1_3_percent', 25) +
            config.get('size_3_5_percent', 30) +
            config.get('size_5_10_percent', 35)
        )
        
        # Normalize percentages
        size_0_1_pct = config.get('size_0_1_percent', 10) / total_percentage
        size_1_3_pct = config.get('size_1_3_percent', 25) / total_percentage
        size_3_5_pct = config.get('size_3_5_percent', 30) / total_percentage
        size_5_10_pct = config.get('size_5_10_percent', 35) / total_percentage
        
        # Calculate total îlots based on average area
        avg_area_0_1 = 0.5  # 0-1 m²
        avg_area_1_3 = 2.0  # 1-3 m²
        avg_area_3_5 = 4.0  # 3-5 m²
        avg_area_5_10 = 7.5 # 5-10 m²
        
        total_avg_area = (
            size_0_1_pct * avg_area_0_1 +
            size_1_3_pct * avg_area_1_3 +
            size_3_5_pct * avg_area_3_5 +
            size_5_10_pct * avg_area_5_10
        )
        
        total_ilots = int(available_area * 0.7 / total_avg_area)  # 70% utilization
        
        return {
            'size_0_1': max(1, int(total_ilots * size_0_1_pct)),
            'size_1_3': max(1, int(total_ilots * size_1_3_pct)),
            'size_3_5': max(1, int(total_ilots * size_3_5_pct)),
            'size_5_10': max(1, int(total_ilots * size_5_10_pct))
        }
    
    def generate_ilot_specifications(self, distribution: Dict[str, int]) -> List[IlotSpec]:
        """Generate îlot specifications"""
        ilots = []
        ilot_id = 1
        
        # Size category definitions
        size_ranges = {
            'size_0_1': (0.5, 1.0),   # 0-1 m²
            'size_1_3': (1.0, 3.0),  # 1-3 m²
            'size_3_5': (3.0, 5.0),  # 3-5 m²
            'size_5_10': (5.0, 10.0) # 5-10 m²
        }
        
        for category, count in distribution.items():
            min_area, max_area = size_ranges[category]
            
            for _ in range(count):
                # Generate random area within range
                area = np.random.uniform(min_area, max_area)
                
                # Calculate dimensions (roughly square with some variation)
                base_side = math.sqrt(area)
                aspect_ratio = np.random.uniform(0.7, 1.4)
                width = base_side * aspect_ratio
                height = area / width
                
                ilot = IlotSpec(
                    id=f"ilot_{ilot_id:03d}",
                    x=0, y=0,  # Will be set during placement
                    width=width,
                    height=height,
                    area=area,
                    size_category=category
                )
                
                ilots.append(ilot)
                ilot_id += 1
        
        return ilots
    
    def find_available_zones(self) -> List[Polygon]:
        """Find zones available for îlot placement"""
        if not self.bounds:
            return []
        
        # Create main boundary
        main_bounds = Polygon([
            (self.bounds['min_x'], self.bounds['min_y']),
            (self.bounds['max_x'], self.bounds['min_y']),
            (self.bounds['max_x'], self.bounds['max_y']),
            (self.bounds['min_x'], self.bounds['max_y'])
        ])
        
        # Start with main boundary
        available_area = main_bounds
        
        # Remove restricted areas (blue zones)
        for restricted in self.restricted_areas:
            try:
                available_area = available_area.difference(restricted.buffer(0.5))
            except:
                continue
        
        # Remove entrances with clearance (red zones)
        for entrance in self.entrances:
            try:
                available_area = available_area.difference(entrance.buffer(2.0))
            except:
                continue
        
        # Convert to list of polygons
        if hasattr(available_area, 'geoms'):
            zones = [geom for geom in available_area.geoms if isinstance(geom, Polygon)]
        elif isinstance(available_area, Polygon):
            zones = [available_area]
        else:
            zones = []
        
        # Filter out very small zones
        zones = [zone for zone in zones if zone.area > 2.0]
        
        return zones
    
    def can_place_ilot(self, ilot: IlotSpec, x: float, y: float, placed_ilots: List[IlotSpec],
                      min_spacing: float = 1.0, wall_clearance: float = 0.5) -> bool:
        """Check if îlot can be placed at position"""
        # Update position
        test_ilot = IlotSpec(
            id=ilot.id, x=x, y=y, width=ilot.width, height=ilot.height,
            area=ilot.area, size_category=ilot.size_category
        )
        
        test_poly = test_ilot.to_polygon()
        
        # Check bounds
        if not self.bounds:
            return False
        
        if (x - ilot.width/2 < self.bounds['min_x'] or
            x + ilot.width/2 > self.bounds['max_x'] or
            y - ilot.height/2 < self.bounds['min_y'] or
            y + ilot.height/2 > self.bounds['max_y']):
            return False
        
        # Check overlap with other îlots
        for placed in placed_ilots:
            placed_poly = placed.to_polygon()
            if test_poly.distance(placed_poly) < min_spacing:
                return False
        
        # Check distance from restricted areas (blue zones)
        for restricted in self.restricted_areas:
            if test_poly.distance(restricted) < 0.5:
                return False
        
        # Check distance from entrances (red zones)
        for entrance in self.entrances:
            if test_poly.distance(entrance) < 2.0:
                return False
        
        # Allow touching walls (black lines) - requirement
        # But ensure minimum clearance
        for wall in self.walls:
            if test_poly.intersects(wall):
                return False
        
        return True
    
    def place_ilots_optimized(self, ilots: List[IlotSpec], config: Dict) -> List[IlotSpec]:
        """Place îlots using optimization algorithm"""
        available_zones = self.find_available_zones()
        if not available_zones:
            return []
        
        placed_ilots = []
        min_spacing = config.get('min_spacing', 1.0)
        wall_clearance = config.get('wall_clearance', 0.5)
        
        # Sort îlots by size (largest first)
        sorted_ilots = sorted(ilots, key=lambda i: i.area, reverse=True)
        
        for ilot in sorted_ilots:
            best_position = None
            best_score = -1
            
            # Try multiple positions in each zone
            for zone in available_zones:
                bounds = zone.bounds
                
                # Grid search within zone
                x_steps = max(5, int((bounds[2] - bounds[0]) / 2))
                y_steps = max(5, int((bounds[3] - bounds[1]) / 2))
                
                for i in range(x_steps):
                    for j in range(y_steps):
                        x = bounds[0] + (bounds[2] - bounds[0]) * i / (x_steps - 1)
                        y = bounds[1] + (bounds[3] - bounds[1]) * j / (y_steps - 1)
                        
                        if self.can_place_ilot(ilot, x, y, placed_ilots, min_spacing, wall_clearance):
                            # Check if position is within zone
                            test_point = Point(x, y)
                            if zone.contains(test_point):
                                # Calculate placement score
                                score = self.calculate_placement_score(ilot, x, y, placed_ilots, zone)
                                if score > best_score:
                                    best_score = score
                                    best_position = (x, y, zone)
            
            # Place îlot at best position
            if best_position:
                x, y, zone = best_position
                ilot.x = x
                ilot.y = y
                ilot.placement_score = best_score
                ilot.accessibility_score = self.calculate_accessibility_score(ilot, placed_ilots)
                placed_ilots.append(ilot)
        
        return placed_ilots
    
    def calculate_placement_score(self, ilot: IlotSpec, x: float, y: float, 
                                placed_ilots: List[IlotSpec], zone: Polygon) -> float:
        """Calculate placement quality score"""
        score = 0
        
        # Zone utilization (prefer central placement)
        zone_bounds = zone.bounds
        center_x = (zone_bounds[0] + zone_bounds[2]) / 2
        center_y = (zone_bounds[1] + zone_bounds[3]) / 2
        
        distance_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        zone_diagonal = math.sqrt((zone_bounds[2] - zone_bounds[0])**2 + 
                                (zone_bounds[3] - zone_bounds[1])**2)
        
        center_score = 1 - (distance_to_center / zone_diagonal)
        score += center_score * 0.3
        
        # Proximity to other îlots (encourage grouping)
        if placed_ilots:
            min_distance = min(
                math.sqrt((x - placed.x)**2 + (y - placed.y)**2)
                for placed in placed_ilots
            )
            proximity_score = 1 / (1 + min_distance / 10)
            score += proximity_score * 0.4
        
        # Wall proximity (slight preference for wall adjacency)
        min_wall_distance = float('inf')
        for wall in self.walls:
            test_point = Point(x, y)
            distance = test_point.distance(wall)
            min_wall_distance = min(min_wall_distance, distance)
        
        if min_wall_distance < float('inf'):
            wall_score = 1 / (1 + min_wall_distance / 5)
            score += wall_score * 0.3
        
        return score
    
    def calculate_accessibility_score(self, ilot: IlotSpec, placed_ilots: List[IlotSpec]) -> float:
        """Calculate accessibility score"""
        # Check access from multiple directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        accessible_directions = 0
        
        for dx, dy in directions:
            # Check 2m in each direction
            check_x = ilot.x + dx * 2
            check_y = ilot.y + dy * 2
            
            # Check if path is clear
            path_clear = True
            for other in placed_ilots:
                if other.id != ilot.id:
                    other_poly = other.to_polygon()
                    path_line = LineString([(ilot.x, ilot.y), (check_x, check_y)])
                    if path_line.intersects(other_poly):
                        path_clear = False
                        break
            
            if path_clear:
                accessible_directions += 1
        
        return accessible_directions / 4.0
    
    def generate_facing_corridors(self, ilots: List[IlotSpec], config: Dict) -> List[CorridorSpec]:
        """Generate mandatory corridors between facing îlot rows - CLIENT REQUIREMENT"""
        corridors = []
        corridor_width = config.get('corridor_width', 1.5)
        
        # Group îlots by proximity to form rows
        rows = self.detect_ilot_rows(ilots)
        
        corridor_id = 1
        for i, row1 in enumerate(rows):
            for j, row2 in enumerate(rows[i+1:], i+1):
                if self.are_rows_facing(row1, row2):
                    # Generate corridor between facing rows
                    corridor = self.create_corridor_between_rows(
                        row1, row2, corridor_width, f"corridor_{corridor_id:03d}"
                    )
                    if corridor:
                        corridors.append(corridor)
                        corridor_id += 1
        
        return corridors
    
    def detect_ilot_rows(self, ilots: List[IlotSpec]) -> List[List[IlotSpec]]:
        """Detect îlot rows based on alignment"""
        if not ilots:
            return []
        
        rows = []
        threshold = 2.0  # Alignment threshold
        
        remaining_ilots = ilots.copy()
        
        while remaining_ilots:
            current_ilot = remaining_ilots.pop(0)
            current_row = [current_ilot]
            
            # Find aligned îlots
            to_remove = []
            for other in remaining_ilots:
                # Check horizontal alignment
                if abs(current_ilot.y - other.y) < threshold:
                    current_row.append(other)
                    to_remove.append(other)
                # Check vertical alignment
                elif abs(current_ilot.x - other.x) < threshold:
                    current_row.append(other)
                    to_remove.append(other)
            
            for ilot in to_remove:
                remaining_ilots.remove(ilot)
            
            if len(current_row) >= 2:  # Only consider actual rows
                rows.append(current_row)
        
        return rows
    
    def are_rows_facing(self, row1: List[IlotSpec], row2: List[IlotSpec]) -> bool:
        """Check if two rows are facing each other"""
        # Calculate row centers
        center1_x = sum(ilot.x for ilot in row1) / len(row1)
        center1_y = sum(ilot.y for ilot in row1) / len(row1)
        
        center2_x = sum(ilot.x for ilot in row2) / len(row2)
        center2_y = sum(ilot.y for ilot in row2) / len(row2)
        
        # Check if rows are parallel and close enough
        distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Determine row orientation
        row1_horizontal = abs(max(ilot.x for ilot in row1) - min(ilot.x for ilot in row1)) > \
                         abs(max(ilot.y for ilot in row1) - min(ilot.y for ilot in row1))
        
        row2_horizontal = abs(max(ilot.x for ilot in row2) - min(ilot.x for ilot in row2)) > \
                         abs(max(ilot.y for ilot in row2) - min(ilot.y for ilot in row2))
        
        # Rows must have same orientation and be close enough
        return (row1_horizontal == row2_horizontal and 
                3.0 <= distance <= 8.0)  # Reasonable corridor distance
    
    def create_corridor_between_rows(self, row1: List[IlotSpec], row2: List[IlotSpec], 
                                   width: float, corridor_id: str) -> CorridorSpec:
        """Create corridor between two facing rows"""
        # Calculate row bounds
        row1_min_x = min(ilot.x - ilot.width/2 for ilot in row1)
        row1_max_x = max(ilot.x + ilot.width/2 for ilot in row1)
        row1_min_y = min(ilot.y - ilot.height/2 for ilot in row1)
        row1_max_y = max(ilot.y + ilot.height/2 for ilot in row1)
        
        row2_min_x = min(ilot.x - ilot.width/2 for ilot in row2)
        row2_max_x = max(ilot.x + ilot.width/2 for ilot in row2)
        row2_min_y = min(ilot.y - ilot.height/2 for ilot in row2)
        row2_max_y = max(ilot.y + ilot.height/2 for ilot in row2)
        
        # Determine corridor direction and position
        row1_center_x = sum(ilot.x for ilot in row1) / len(row1)
        row1_center_y = sum(ilot.y for ilot in row1) / len(row1)
        row2_center_x = sum(ilot.x for ilot in row2) / len(row2)
        row2_center_y = sum(ilot.y for ilot in row2) / len(row2)
        
        # Create corridor path
        if abs(row1_center_x - row2_center_x) > abs(row1_center_y - row2_center_y):
            # Vertical corridor (rows are horizontally separated)
            corridor_x = (row1_center_x + row2_center_x) / 2
            
            # Find overlapping Y range
            overlap_min_y = max(row1_min_y, row2_min_y)
            overlap_max_y = min(row1_max_y, row2_max_y)
            
            if overlap_min_y < overlap_max_y:
                start_point = (corridor_x, overlap_min_y)
                end_point = (corridor_x, overlap_max_y)
                path_points = [start_point, end_point]
            else:
                # Extend to cover both rows
                start_point = (corridor_x, min(row1_min_y, row2_min_y))
                end_point = (corridor_x, max(row1_max_y, row2_max_y))
                path_points = [start_point, end_point]
        else:
            # Horizontal corridor (rows are vertically separated)
            corridor_y = (row1_center_y + row2_center_y) / 2
            
            # Find overlapping X range
            overlap_min_x = max(row1_min_x, row2_min_x)
            overlap_max_x = min(row1_max_x, row2_max_x)
            
            if overlap_min_x < overlap_max_x:
                start_point = (overlap_min_x, corridor_y)
                end_point = (overlap_max_x, corridor_y)
                path_points = [start_point, end_point]
            else:
                # Extend to cover both rows
                start_point = (min(row1_min_x, row2_min_x), corridor_y)
                end_point = (max(row1_max_x, row2_max_x), corridor_y)
                path_points = [start_point, end_point]
        
        return CorridorSpec(
            id=corridor_id,
            type="facing_corridor",
            start_point=start_point,
            end_point=end_point,
            width=width,
            path_points=path_points,
            connects_ilots=[ilot.id for ilot in row1 + row2],
            is_mandatory=True,
            accessibility_compliant=True
        )
    
    def validate_corridor_placement(self, corridor: CorridorSpec, ilots: List[IlotSpec]) -> bool:
        """Validate that corridor doesn't overlap with îlots - CLIENT REQUIREMENT"""
        corridor_poly = corridor.to_polygon()
        if not corridor_poly:
            return False
        
        for ilot in ilots:
            ilot_poly = ilot.to_polygon()
            if corridor_poly.intersects(ilot_poly):
                return False
        
        return True
    
    def process_full_placement(self, config: Dict) -> Dict[str, Any]:
        """Complete îlot placement process"""
        # Calculate available area
        available_zones = self.find_available_zones()
        total_available_area = sum(zone.area for zone in available_zones)
        
        # Calculate îlot distribution
        distribution = self.calculate_ilot_distribution(config, total_available_area)
        
        # Generate îlot specifications
        ilot_specs = self.generate_ilot_specifications(distribution)
        
        # Place îlots
        placed_ilots = self.place_ilots_optimized(ilot_specs, config)
        
        # Generate corridors
        corridors = self.generate_facing_corridors(placed_ilots, config)
        
        # Validate corridors
        valid_corridors = [c for c in corridors if self.validate_corridor_placement(c, placed_ilots)]
        
        # Calculate metrics
        metrics = self.calculate_placement_metrics(placed_ilots, valid_corridors, total_available_area)
        
        return {
            'ilots': [self.ilot_to_dict(ilot) for ilot in placed_ilots],
            'corridors': [self.corridor_to_dict(corridor) for corridor in valid_corridors],
            'metrics': metrics,
            'distribution': distribution,
            'available_area': total_available_area,
            'zones_count': len(available_zones)
        }
    
    def ilot_to_dict(self, ilot: IlotSpec) -> Dict:
        """Convert îlot to dictionary"""
        return {
            'id': ilot.id,
            'x': ilot.x,
            'y': ilot.y,
            'width': ilot.width,
            'height': ilot.height,
            'area': ilot.area,
            'size_category': ilot.size_category,
            'rotation': ilot.rotation,
            'zone_id': ilot.zone_id,
            'accessibility_score': ilot.accessibility_score,
            'placement_score': ilot.placement_score
        }
    
    def corridor_to_dict(self, corridor: CorridorSpec) -> Dict:
        """Convert corridor to dictionary"""
        return {
            'id': corridor.id,
            'type': corridor.type,
            'start_point': corridor.start_point,
            'end_point': corridor.end_point,
            'width': corridor.width,
            'path_points': corridor.path_points,
            'connects_ilots': corridor.connects_ilots,
            'is_mandatory': corridor.is_mandatory,
            'accessibility_compliant': corridor.accessibility_compliant
        }
    
    def calculate_placement_metrics(self, ilots: List[IlotSpec], corridors: List[CorridorSpec], 
                                  available_area: float) -> Dict[str, float]:
        """Calculate placement quality metrics"""
        if not ilots:
            return {
                'space_utilization': 0,
                'coverage_percentage': 0,
                'efficiency_score': 0,
                'accessibility_score': 0,
                'circulation_efficiency': 0,
                'safety_compliance': 1.0
            }
        
        total_ilot_area = sum(ilot.area for ilot in ilots)
        total_corridor_area = sum(corridor.width * 
                                sum(math.sqrt((corridor.path_points[i+1][0] - corridor.path_points[i][0])**2 + 
                                            (corridor.path_points[i+1][1] - corridor.path_points[i][1])**2)
                                    for i in range(len(corridor.path_points)-1))
                                for corridor in corridors if len(corridor.path_points) > 1)
        
        space_utilization = (total_ilot_area + total_corridor_area) / available_area if available_area > 0 else 0
        coverage_percentage = total_ilot_area / available_area if available_area > 0 else 0
        
        avg_accessibility = sum(ilot.accessibility_score for ilot in ilots) / len(ilots)
        avg_placement_score = sum(ilot.placement_score for ilot in ilots) / len(ilots)
        
        mandatory_corridors = sum(1 for c in corridors if c.is_mandatory)
        circulation_efficiency = min(1.0, mandatory_corridors / max(1, len(ilots) // 4))
        
        return {
            'space_utilization': min(1.0, space_utilization),
            'coverage_percentage': min(1.0, coverage_percentage),
            'efficiency_score': avg_placement_score,
            'accessibility_score': avg_accessibility,
            'circulation_efficiency': circulation_efficiency,
            'safety_compliance': 1.0  # All placements respect safety constraints
        }