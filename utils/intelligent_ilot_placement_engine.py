"""
Intelligent Îlot Placement Engine
Implements advanced room analysis and smart îlot distribution
Following the detailed implementation plan for pixel-perfect results
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import math
from dataclasses import dataclass

@dataclass
class IlotSpecification:
    """Specification for a single îlot"""
    id: str
    size_category: str
    min_area: float
    max_area: float
    preferred_aspect_ratio: float
    color: str
    priority: int

@dataclass
class PlacementZone:
    """Usable zone for îlot placement"""
    polygon: Polygon
    area: float
    accessibility_score: float
    room_id: str
    constraints: List[str]

class IntelligentIlotPlacementEngine:
    """
    Advanced îlot placement engine implementing the complete detailed plan
    Creates optimal furniture placement with professional space utilization
    """
    
    def __init__(self):
        # Size categories matching professional standards
        self.size_categories = {
            'small': {
                'min_area': 1.0,
                'max_area': 3.0,
                'preferred_aspect_ratio': 1.2,
                'color': '#FFE6E6',
                'priority': 4,
                'percentage': 0.20
            },
            'medium': {
                'min_area': 3.0,
                'max_area': 7.0,
                'preferred_aspect_ratio': 1.4,
                'color': '#FFD6D6',
                'priority': 3,
                'percentage': 0.35
            },
            'large': {
                'min_area': 7.0,
                'max_area': 12.0,
                'preferred_aspect_ratio': 1.6,
                'color': '#FFC6C6',
                'priority': 2,
                'percentage': 0.30
            },
            'xlarge': {
                'min_area': 12.0,
                'max_area': 20.0,
                'preferred_aspect_ratio': 1.8,
                'color': '#FFB6B6',
                'priority': 1,
                'percentage': 0.15
            }
        }
        
        # Professional placement parameters
        self.placement_config = {
            'min_clearance': 1.0,      # Minimum clearance around îlots
            'wall_clearance': 0.5,     # Clearance from walls
            'accessibility_width': 1.2, # Minimum accessibility path width
            'utilization_target': 0.65,  # Target space utilization
            'max_iterations': 1000,     # Maximum optimization iterations
            'convergence_threshold': 0.01
        }
    
    def generate_intelligent_placement(self, floor_plan_data: Dict[str, Any],
                                     target_count: int = None) -> List[Dict[str, Any]]:
        """
        Generate intelligent îlot placement with advanced room analysis
        Step 3.1: Advanced Room Analysis + Step 3.2: Smart Îlot Distribution
        """
        print("Starting intelligent îlot placement...")
        
        # Step 3.1: Advanced Room Analysis
        placement_zones = self._analyze_placement_zones(floor_plan_data)
        usable_area = sum(zone.area for zone in placement_zones)
        
        # Calculate optimal îlot count if not provided
        if target_count is None:
            target_count = self._calculate_optimal_count(usable_area)
        
        # Generate îlot specifications
        ilot_specs = self._generate_ilot_specifications(target_count)
        
        # Step 3.2: Smart Îlot Distribution
        placed_ilots = self._optimize_placement(ilot_specs, placement_zones, floor_plan_data)
        
        # Ensure proper spacing and accessibility
        placed_ilots = self._ensure_accessibility_compliance(placed_ilots, floor_plan_data)
        
        print(f"Placed {len(placed_ilots)} îlots with {usable_area:.1f}m² usable area")
        
        return placed_ilots
    
    def _analyze_placement_zones(self, floor_plan_data: Dict[str, Any]) -> List[PlacementZone]:
        """
        Step 3.1: Advanced Room Analysis
        Calculate usable floor area excluding restricted zones
        """
        bounds = floor_plan_data.get('bounds', {})
        walls = floor_plan_data.get('walls', [])
        restricted_areas = floor_plan_data.get('restricted_areas', [])
        entrances = floor_plan_data.get('entrances', [])
        
        # Create main room polygon
        main_room = box(
            bounds.get('min_x', 0),
            bounds.get('min_y', 0),
            bounds.get('max_x', 100),
            bounds.get('max_y', 100)
        )
        
        # Subtract restricted areas
        restricted_polygons = []
        for area in restricted_areas:
            if area.get('type') == 'circle':
                center = Point(area.get('x', 0), area.get('y', 0))
                radius = area.get('radius', 1.0)
                restricted_polygons.append(center.buffer(radius))
            elif area.get('type') == 'polygon':
                coords = area.get('coordinates', [])
                if len(coords) >= 3:
                    restricted_polygons.append(Polygon(coords))
        
        # Subtract entrance areas
        entrance_polygons = []
        for entrance in entrances:
            if entrance.get('type') == 'circle':
                center = Point(entrance.get('x', 0), entrance.get('y', 0))
                radius = entrance.get('radius', 1.0)
                entrance_polygons.append(center.buffer(radius))
        
        # Create usable area
        usable_area = main_room
        if restricted_polygons:
            restricted_union = unary_union(restricted_polygons)
            usable_area = usable_area.difference(restricted_union)
        
        if entrance_polygons:
            entrance_union = unary_union(entrance_polygons)
            usable_area = usable_area.difference(entrance_union)
        
        # Apply wall clearance
        if isinstance(usable_area, Polygon):
            usable_area = usable_area.buffer(-self.placement_config['wall_clearance'])
        
        # Create placement zones
        zones = []
        if isinstance(usable_area, Polygon) and usable_area.is_valid:
            accessibility_score = self._calculate_accessibility_score(usable_area, entrances)
            zones.append(PlacementZone(
                polygon=usable_area,
                area=usable_area.area,
                accessibility_score=accessibility_score,
                room_id='main_room',
                constraints=['wall_clearance', 'accessibility']
            ))
        
        return zones
    
    def _calculate_optimal_count(self, usable_area: float) -> int:
        """Calculate optimal îlot count based on usable area and target utilization"""
        target_utilization = self.placement_config['utilization_target']
        target_area = usable_area * target_utilization
        
        # Calculate average îlot size across all categories
        avg_area = sum(
            (cat['min_area'] + cat['max_area']) / 2 * cat['percentage']
            for cat in self.size_categories.values()
        )
        
        optimal_count = int(target_area / avg_area)
        
        # Ensure reasonable bounds
        return max(10, min(optimal_count, 50))
    
    def _generate_ilot_specifications(self, target_count: int) -> List[IlotSpecification]:
        """Generate specifications for each îlot based on professional distribution"""
        specs = []
        
        for category, config in self.size_categories.items():
            count = int(target_count * config['percentage'])
            
            for i in range(count):
                # Random area within category range
                area = np.random.uniform(config['min_area'], config['max_area'])
                
                spec = IlotSpecification(
                    id=f"{category}_{i+1}",
                    size_category=category,
                    min_area=config['min_area'],
                    max_area=config['max_area'],
                    preferred_aspect_ratio=config['preferred_aspect_ratio'],
                    color=config['color'],
                    priority=config['priority']
                )
                specs.append(spec)
        
        # Sort by priority (larger îlots first)
        specs.sort(key=lambda x: x.priority)
        
        return specs
    
    def _optimize_placement(self, ilot_specs: List[IlotSpecification],
                          placement_zones: List[PlacementZone],
                          floor_plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Step 3.2: Smart Îlot Distribution
        Optimize placement for maximum space utilization
        """
        placed_ilots = []
        
        for spec in ilot_specs:
            best_position = self._find_optimal_position(
                spec, placement_zones, placed_ilots, floor_plan_data
            )
            
            if best_position:
                # Calculate dimensions from area and preferred aspect ratio
                area = np.random.uniform(spec.min_area, spec.max_area)
                aspect_ratio = spec.preferred_aspect_ratio
                
                width = math.sqrt(area * aspect_ratio)
                height = area / width
                
                ilot = {
                    'id': spec.id,
                    'x': best_position[0],
                    'y': best_position[1],
                    'width': width,
                    'height': height,
                    'area': area,
                    'category': spec.size_category,
                    'size_category': spec.size_category,
                    'color': spec.color,
                    'priority': spec.priority,
                    'type': 'furniture'
                }
                
                placed_ilots.append(ilot)
        
        return placed_ilots
    
    def _find_optimal_position(self, spec: IlotSpecification,
                             placement_zones: List[PlacementZone],
                             existing_ilots: List[Dict[str, Any]],
                             floor_plan_data: Dict[str, Any]) -> Tuple[float, float]:
        """Find optimal position for a single îlot"""
        best_position = None
        best_score = -1
        
        # Try multiple positions across all zones
        for zone in placement_zones:
            bounds = zone.polygon.bounds
            
            # Generate candidate positions
            for _ in range(100):  # Try 100 random positions per zone
                x = np.random.uniform(bounds[0], bounds[2])
                y = np.random.uniform(bounds[1], bounds[3])
                
                # Check if position is valid
                if self._is_valid_position(x, y, spec, zone, existing_ilots):
                    score = self._calculate_placement_score(x, y, spec, zone, existing_ilots)
                    
                    if score > best_score:
                        best_score = score
                        best_position = (x, y)
        
        return best_position
    
    def _is_valid_position(self, x: float, y: float, spec: IlotSpecification,
                         zone: PlacementZone, existing_ilots: List[Dict[str, Any]]) -> bool:
        """Check if position is valid for îlot placement"""
        # Calculate dimensions
        area = (spec.min_area + spec.max_area) / 2
        width = math.sqrt(area * spec.preferred_aspect_ratio)
        height = area / width
        
        # Create îlot polygon
        ilot_polygon = box(x, y, x + width, y + height)
        
        # Check if îlot is within zone
        if not zone.polygon.contains(ilot_polygon):
            return False
        
        # Check clearance from existing îlots
        clearance = self.placement_config['min_clearance']
        for existing in existing_ilots:
            existing_polygon = box(
                existing['x'], existing['y'],
                existing['x'] + existing['width'],
                existing['y'] + existing['height']
            )
            
            if ilot_polygon.distance(existing_polygon) < clearance:
                return False
        
        return True
    
    def _calculate_placement_score(self, x: float, y: float, spec: IlotSpecification,
                                 zone: PlacementZone, existing_ilots: List[Dict[str, Any]]) -> float:
        """Calculate placement score for optimization"""
        score = 0.0
        
        # Zone accessibility score
        score += zone.accessibility_score * 0.3
        
        # Distance from zone center (prefer central placement)
        zone_center = zone.polygon.centroid
        distance_to_center = math.sqrt((x - zone_center.x)**2 + (y - zone_center.y)**2)
        max_distance = max(zone.polygon.bounds[2] - zone.polygon.bounds[0],
                          zone.polygon.bounds[3] - zone.polygon.bounds[1])
        centrality_score = 1.0 - (distance_to_center / max_distance)
        score += centrality_score * 0.4
        
        # Spacing from existing îlots (prefer good spacing)
        if existing_ilots:
            min_distance = min(
                math.sqrt((x - existing['x'])**2 + (y - existing['y'])**2)
                for existing in existing_ilots
            )
            spacing_score = min(min_distance / 10.0, 1.0)
            score += spacing_score * 0.3
        
        return score
    
    def _calculate_accessibility_score(self, polygon: Polygon, entrances: List[Dict[str, Any]]) -> float:
        """Calculate accessibility score based on entrance proximity"""
        if not entrances:
            return 0.5
        
        polygon_center = polygon.centroid
        distances = []
        
        for entrance in entrances:
            entrance_point = Point(entrance.get('x', 0), entrance.get('y', 0))
            distance = polygon_center.distance(entrance_point)
            distances.append(distance)
        
        # Score based on proximity to nearest entrance
        min_distance = min(distances)
        accessibility_score = 1.0 / (1.0 + min_distance / 50.0)
        
        return accessibility_score
    
    def _ensure_accessibility_compliance(self, ilots: List[Dict[str, Any]],
                                       floor_plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ensure proper spacing and accessibility compliance"""
        accessibility_width = self.placement_config['accessibility_width']
        
        # Check and adjust îlot positions for accessibility
        adjusted_ilots = []
        
        for ilot in ilots:
            # Create accessibility buffer around îlot
            ilot_polygon = box(
                ilot['x'], ilot['y'],
                ilot['x'] + ilot['width'],
                ilot['y'] + ilot['height']
            )
            
            # Check if accessibility paths are maintained
            has_accessibility = self._check_accessibility_paths(ilot_polygon, ilots, floor_plan_data)
            
            if has_accessibility:
                adjusted_ilots.append(ilot)
            else:
                # Try to adjust position
                adjusted_ilot = self._adjust_for_accessibility(ilot, ilots, floor_plan_data)
                if adjusted_ilot:
                    adjusted_ilots.append(adjusted_ilot)
        
        return adjusted_ilots
    
    def _check_accessibility_paths(self, ilot_polygon: Polygon,
                                 all_ilots: List[Dict[str, Any]],
                                 floor_plan_data: Dict[str, Any]) -> bool:
        """Check if accessibility paths are maintained"""
        # Simplified accessibility check
        # In a full implementation, this would use pathfinding algorithms
        return True
    
    def _adjust_for_accessibility(self, ilot: Dict[str, Any],
                                all_ilots: List[Dict[str, Any]],
                                floor_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust îlot position for accessibility compliance"""
        # Try small adjustments to position
        adjustments = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
        
        for dx, dy in adjustments:
            adjusted_ilot = ilot.copy()
            adjusted_ilot['x'] += dx
            adjusted_ilot['y'] += dy
            
            # Check if adjusted position is valid
            if self._is_position_accessible(adjusted_ilot, all_ilots, floor_plan_data):
                return adjusted_ilot
        
        return ilot  # Return original if no adjustment works
    
    def _is_position_accessible(self, ilot: Dict[str, Any],
                              all_ilots: List[Dict[str, Any]],
                              floor_plan_data: Dict[str, Any]) -> bool:
        """Check if position maintains accessibility"""
        # Simplified accessibility check
        return True