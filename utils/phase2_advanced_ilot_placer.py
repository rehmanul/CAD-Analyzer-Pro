"""
Phase 2: Advanced Îlot Placement Engine
High-performance algorithms for optimal îlot placement with size distribution,
spatial optimization, and intelligent collision detection
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import networkx as nx
from dataclasses import dataclass
from enum import Enum
import time
import random

class IlotSize(Enum):
    SMALL = "0-1m²"      # 0-1 m²
    MEDIUM = "1-3m²"     # 1-3 m²
    LARGE = "3-5m²"      # 3-5 m²
    EXTRA_LARGE = "5-10m²"  # 5-10 m²

class PlacementStrategy(Enum):
    GRID_BASED = "grid_based"
    PHYSICS_BASED = "physics_based"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HYBRID = "hybrid"

@dataclass
class IlotConfiguration:
    """Configuration for îlot placement"""
    size_distribution: Dict[IlotSize, float]  # Percentage for each size
    min_spacing: float = 1.0  # Minimum spacing between îlots (meters)
    wall_clearance: float = 0.5  # Minimum clearance from walls (meters)
    utilization_target: float = 0.7  # Target space utilization (0.0-1.0)
    placement_strategy: PlacementStrategy = PlacementStrategy.HYBRID
    max_iterations: int = 1000
    convergence_threshold: float = 0.01

@dataclass 
class PlacedIlot:
    """Represents a placed îlot with all properties"""
    id: str
    geometry: Polygon
    size_category: IlotSize
    area: float
    center: Point
    room_id: Optional[str] = None
    placement_score: float = 0.0
    constraints_satisfied: bool = True

class AdvancedIlotPlacer:
    """
    Advanced îlot placement engine with multiple optimization strategies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Size category definitions (in m²)
        self.size_ranges = {
            IlotSize.SMALL: (0.5, 1.0),
            IlotSize.MEDIUM: (1.0, 3.0),
            IlotSize.LARGE: (3.0, 5.0),
            IlotSize.EXTRA_LARGE: (5.0, 10.0)
        }
        
        # Color coding for visualization
        self.size_colors = {
            IlotSize.SMALL: "#FDE047",      # Yellow
            IlotSize.MEDIUM: "#FB923C",     # Orange  
            IlotSize.LARGE: "#22C55E",      # Green
            IlotSize.EXTRA_LARGE: "#A855F7" # Purple
        }
        
        # Performance tracking
        self.placement_stats = {}

    def place_ilots_advanced(self, floor_plan_data: Dict[str, Any], 
                           config: IlotConfiguration) -> Dict[str, Any]:
        """
        Main advanced îlot placement method with multiple strategies
        """
        start_time = time.time()
        
        try:
            # Extract spatial data from floor plan
            spatial_data = self._extract_spatial_data(floor_plan_data)
            
            if not spatial_data['placeable_areas']:
                return self._create_empty_result("No placeable areas found")
            
            # Calculate total îlot counts based on available space and distribution
            total_ilots = self._calculate_optimal_ilot_count(spatial_data, config)
            
            # Generate îlot specifications
            ilot_specs = self._generate_ilot_specifications(total_ilots, config)
            
            # Choose placement strategy based on configuration and data complexity
            strategy = self._select_optimal_strategy(spatial_data, config, len(ilot_specs))
            
            # Place îlots using selected strategy
            placed_ilots = self._execute_placement_strategy(
                spatial_data, ilot_specs, config, strategy
            )
            
            # Optimize placement through post-processing
            optimized_ilots = self._optimize_placement(placed_ilots, spatial_data, config)
            
            # Calculate placement metrics
            placement_metrics = self._calculate_placement_metrics(
                optimized_ilots, spatial_data, config
            )
            
            processing_time = time.time() - start_time
            
            # Generate comprehensive result
            result = self._generate_placement_result(
                optimized_ilots, placement_metrics, processing_time, strategy, config
            )
            
            self.logger.info(f"Advanced îlot placement completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced îlot placement: {str(e)}")
            return self._create_error_result(str(e))

    def _extract_spatial_data(self, floor_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and process spatial data for placement"""
        spatial_data = {
            'walls': [],
            'rooms': [],
            'restricted_areas': [],
            'entrances': [],
            'placeable_areas': [],
            'bounds': None,
            'total_area': 0.0
        }
        
        try:
            # Extract walls
            walls = floor_plan_data.get('walls', [])
            for wall in walls:
                coords = wall.get('coordinates', [])
                if len(coords) >= 2:
                    wall_geom = LineString(coords)
                    spatial_data['walls'].append(wall_geom)
            
            # Extract rooms
            rooms = floor_plan_data.get('rooms', [])
            for room in rooms:
                coords = room.get('coordinates', [])
                if len(coords) >= 3:
                    room_geom = Polygon(coords)
                    if room_geom.is_valid and room_geom.area > 1.0:  # Minimum 1m² room
                        spatial_data['rooms'].append(room_geom)
            
            # Extract restricted areas
            restricted = floor_plan_data.get('restricted_areas', [])
            for area in restricted:
                coords = area.get('coordinates', [])
                if len(coords) >= 3:
                    restricted_geom = Polygon(coords)
                    if restricted_geom.is_valid:
                        spatial_data['restricted_areas'].append(restricted_geom)
            
            # Extract entrances  
            entrances = floor_plan_data.get('entrances', [])
            for entrance in entrances:
                coords = entrance.get('coordinates', [])
                if len(coords) >= 1:
                    if len(coords) == 1:
                        entrance_geom = Point(coords[0]).buffer(1.0)  # 1m buffer for entrance
                    else:
                        entrance_geom = Polygon(coords)
                    if entrance_geom.is_valid:
                        spatial_data['entrances'].append(entrance_geom)
            
            # Calculate bounds
            bounds = floor_plan_data.get('floor_plan_bounds', {})
            if bounds:
                spatial_data['bounds'] = bounds
                spatial_data['total_area'] = bounds.get('width', 0) * bounds.get('height', 0)
            
            # Generate placeable areas
            spatial_data['placeable_areas'] = self._generate_placeable_areas(spatial_data)
            
            return spatial_data
            
        except Exception as e:
            self.logger.error(f"Error extracting spatial data: {str(e)}")
            return spatial_data

    def _generate_placeable_areas(self, spatial_data: Dict[str, Any]) -> List[Polygon]:
        """Generate areas where îlots can be placed"""
        placeable_areas = []
        
        try:
            # Start with room areas or overall bounds
            if spatial_data['rooms']:
                base_areas = spatial_data['rooms']
            elif spatial_data['bounds']:
                # Create a single large area from bounds
                bounds = spatial_data['bounds']
                base_area = box(
                    bounds.get('min_x', 0), bounds.get('min_y', 0),
                    bounds.get('max_x', 1000), bounds.get('max_y', 1000)
                )
                base_areas = [base_area]
            else:
                return placeable_areas
            
            # Remove restricted areas and entrance zones
            exclusion_zones = []
            exclusion_zones.extend(spatial_data['restricted_areas'])
            exclusion_zones.extend(spatial_data['entrances'])
            
            # Add wall buffers as exclusion zones
            for wall in spatial_data['walls']:
                wall_buffer = wall.buffer(500)  # 0.5m clearance from walls (mm)
                exclusion_zones.append(wall_buffer)
            
            # Subtract exclusion zones from base areas
            for base_area in base_areas:
                remaining_area = base_area
                
                for exclusion in exclusion_zones:
                    try:
                        remaining_area = remaining_area.difference(exclusion)
                    except Exception:
                        continue
                
                # Split large areas into smaller placeable zones if needed
                if hasattr(remaining_area, 'geoms'):
                    # MultiPolygon
                    for geom in remaining_area.geoms:
                        if isinstance(geom, Polygon) and geom.area > 0.5:  # Minimum 0.5m²
                            placeable_areas.append(geom)
                elif isinstance(remaining_area, Polygon) and remaining_area.area > 0.5:
                    placeable_areas.append(remaining_area)
            
            return placeable_areas
            
        except Exception as e:
            self.logger.error(f"Error generating placeable areas: {str(e)}")
            return []

    def _calculate_optimal_ilot_count(self, spatial_data: Dict[str, Any], 
                                    config: IlotConfiguration) -> int:
        """Calculate optimal number of îlots based on available space"""
        
        # Calculate total placeable area
        total_placeable_area = sum(area.area for area in spatial_data['placeable_areas'])
        
        if total_placeable_area == 0:
            return 0
        
        # Convert area from mm² to m² (assuming coordinates are in mm)
        total_area_m2 = total_placeable_area / 1000000
        
        # Calculate average îlot size based on distribution
        avg_ilot_size = 0
        for size_cat, percentage in config.size_distribution.items():
            min_size, max_size = self.size_ranges[size_cat]
            avg_size = (min_size + max_size) / 2
            avg_ilot_size += avg_size * (percentage / 100)
        
        # Account for spacing between îlots (approximately 20% overhead)
        spacing_factor = 1.2
        
        # Calculate optimal count considering utilization target
        optimal_count = int((total_area_m2 * config.utilization_target) / (avg_ilot_size * spacing_factor))
        
        # Ensure minimum and maximum bounds
        optimal_count = max(1, min(optimal_count, 200))  # Between 1 and 200 îlots
        
        self.logger.info(f"Calculated optimal îlot count: {optimal_count} for {total_area_m2:.1f}m²")
        return optimal_count

    def _generate_ilot_specifications(self, total_count: int, 
                                    config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Generate specifications for each îlot to be placed"""
        specs = []
        
        # Calculate count for each size category
        for size_cat, percentage in config.size_distribution.items():
            count = int(total_count * percentage / 100)
            min_size, max_size = self.size_ranges[size_cat]
            
            for i in range(count):
                # Random size within category range
                size = np.random.uniform(min_size, max_size)
                
                # Generate square îlot geometry (can be extended for other shapes)
                side_length = np.sqrt(size) * 1000  # Convert to mm
                
                spec = {
                    'id': f"{size_cat.value}_{i+1}",
                    'size_category': size_cat,
                    'area': size,
                    'dimensions': {'width': side_length, 'height': side_length},
                    'color': self.size_colors[size_cat],
                    'priority': self._get_placement_priority(size_cat)
                }
                specs.append(spec)
        
        # Sort by priority (larger îlots placed first)
        specs.sort(key=lambda x: x['priority'], reverse=True)
        
        return specs

    def _get_placement_priority(self, size_category: IlotSize) -> int:
        """Get placement priority for size category"""
        priority_map = {
            IlotSize.EXTRA_LARGE: 4,
            IlotSize.LARGE: 3,
            IlotSize.MEDIUM: 2,
            IlotSize.SMALL: 1
        }
        return priority_map.get(size_category, 1)

    def _select_optimal_strategy(self, spatial_data: Dict[str, Any], 
                               config: IlotConfiguration, ilot_count: int) -> PlacementStrategy:
        """Select optimal placement strategy based on problem characteristics"""
        
        # Analyze problem complexity
        area_count = len(spatial_data['placeable_areas'])
        total_area = sum(area.area for area in spatial_data['placeable_areas'])
        complexity_score = area_count * ilot_count / max(total_area, 1)
        
        # Choose strategy based on complexity and configuration
        if config.placement_strategy != PlacementStrategy.HYBRID:
            return config.placement_strategy
        
        # Automatic strategy selection
        if complexity_score < 0.001:  # Simple case
            return PlacementStrategy.GRID_BASED
        elif complexity_score < 0.01:  # Medium complexity
            return PlacementStrategy.PHYSICS_BASED
        else:  # High complexity
            return PlacementStrategy.GENETIC_ALGORITHM

    def _execute_placement_strategy(self, spatial_data: Dict[str, Any], 
                                  ilot_specs: List[Dict[str, Any]], 
                                  config: IlotConfiguration, 
                                  strategy: PlacementStrategy) -> List[PlacedIlot]:
        """Execute the selected placement strategy"""
        
        if strategy == PlacementStrategy.GRID_BASED:
            return self._place_ilots_grid_based(spatial_data, ilot_specs, config)
        elif strategy == PlacementStrategy.PHYSICS_BASED:
            return self._place_ilots_physics_based(spatial_data, ilot_specs, config)
        elif strategy == PlacementStrategy.GENETIC_ALGORITHM:
            return self._place_ilots_genetic_algorithm(spatial_data, ilot_specs, config)
        else:
            # Hybrid approach: try multiple strategies and pick best result
            return self._place_ilots_hybrid(spatial_data, ilot_specs, config)

    def _place_ilots_grid_based(self, spatial_data: Dict[str, Any], 
                              ilot_specs: List[Dict[str, Any]], 
                              config: IlotConfiguration) -> List[PlacedIlot]:
        """Grid-based placement strategy for simple, regular layouts"""
        placed_ilots = []
        
        try:
            for area in spatial_data['placeable_areas']:
                area_bounds = area.bounds
                width = area_bounds[2] - area_bounds[0]
                height = area_bounds[3] - area_bounds[1]
                
                # Calculate grid spacing
                min_spacing_mm = config.min_spacing * 1000
                
                # Place îlots in grid pattern
                for spec in ilot_specs:
                    if len(placed_ilots) >= len(ilot_specs):
                        break
                    
                    ilot_size = spec['dimensions']['width']
                    
                    # Try to place îlot in grid positions
                    for x in np.arange(area_bounds[0] + ilot_size/2, 
                                     area_bounds[2] - ilot_size/2, 
                                     ilot_size + min_spacing_mm):
                        for y in np.arange(area_bounds[1] + ilot_size/2,
                                         area_bounds[3] - ilot_size/2,
                                         ilot_size + min_spacing_mm):
                            
                            # Create îlot geometry
                            ilot_center = Point(x, y)
                            ilot_geom = self._create_ilot_geometry(ilot_center, spec)
                            
                            # Check if placement is valid
                            if self._is_valid_placement(ilot_geom, area, placed_ilots, config):
                                placed_ilot = PlacedIlot(
                                    id=spec['id'],
                                    geometry=ilot_geom,
                                    size_category=spec['size_category'],
                                    area=spec['area'],
                                    center=ilot_center,
                                    placement_score=1.0
                                )
                                placed_ilots.append(placed_ilot)
                                break
                        else:
                            continue
                        break
            
            return placed_ilots
            
        except Exception as e:
            self.logger.error(f"Error in grid-based placement: {str(e)}")
            return placed_ilots

    def _place_ilots_physics_based(self, spatial_data: Dict[str, Any], 
                                 ilot_specs: List[Dict[str, Any]], 
                                 config: IlotConfiguration) -> List[PlacedIlot]:
        """Physics-based placement using force simulation"""
        placed_ilots = []
        
        try:
            # Initial random placement
            for spec in ilot_specs:
                attempts = 0
                max_attempts = 100
                
                while attempts < max_attempts:
                    # Random placement in available areas
                    area = random.choice(spatial_data['placeable_areas'])
                    bounds = area.bounds
                    
                    x = np.random.uniform(bounds[0], bounds[2])
                    y = np.random.uniform(bounds[1], bounds[3])
                    center = Point(x, y)
                    
                    ilot_geom = self._create_ilot_geometry(center, spec)
                    
                    if self._is_valid_placement(ilot_geom, area, placed_ilots, config):
                        placed_ilot = PlacedIlot(
                            id=spec['id'],
                            geometry=ilot_geom,
                            size_category=spec['size_category'],
                            area=spec['area'],
                            center=center
                        )
                        placed_ilots.append(placed_ilot)
                        break
                    
                    attempts += 1
            
            # Physics simulation to optimize positions
            placed_ilots = self._simulate_physics_optimization(placed_ilots, spatial_data, config)
            
            return placed_ilots
            
        except Exception as e:
            self.logger.error(f"Error in physics-based placement: {str(e)}")
            return placed_ilots

    def _place_ilots_genetic_algorithm(self, spatial_data: Dict[str, Any], 
                                     ilot_specs: List[Dict[str, Any]], 
                                     config: IlotConfiguration) -> List[PlacedIlot]:
        """Genetic algorithm for complex optimization scenarios"""
        # Simplified genetic algorithm implementation
        # In production, this would be a full GA with population, crossover, mutation
        
        best_placement = []
        best_score = 0
        
        try:
            # Multiple random attempts with scoring
            for attempt in range(config.max_iterations // 10):  # Reduced iterations for performance
                candidate_placement = []
                
                # Generate candidate solution
                for spec in ilot_specs:
                    placed = False
                    for _ in range(50):  # Attempts per îlot
                        area = random.choice(spatial_data['placeable_areas'])
                        bounds = area.bounds
                        
                        x = np.random.uniform(bounds[0], bounds[2])
                        y = np.random.uniform(bounds[1], bounds[3])
                        center = Point(x, y)
                        
                        ilot_geom = self._create_ilot_geometry(center, spec)
                        
                        if self._is_valid_placement(ilot_geom, area, candidate_placement, config):
                            placed_ilot = PlacedIlot(
                                id=spec['id'],
                                geometry=ilot_geom,
                                size_category=spec['size_category'],
                                area=spec['area'],
                                center=center
                            )
                            candidate_placement.append(placed_ilot)
                            placed = True
                            break
                    
                    if not placed:
                        break
                
                # Score this candidate
                score = self._score_placement(candidate_placement, spatial_data, config)
                
                if score > best_score:
                    best_score = score
                    best_placement = candidate_placement
            
            return best_placement
            
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm placement: {str(e)}")
            return best_placement

    def _place_ilots_hybrid(self, spatial_data: Dict[str, Any], 
                          ilot_specs: List[Dict[str, Any]], 
                          config: IlotConfiguration) -> List[PlacedIlot]:
        """Hybrid strategy combining multiple approaches"""
        
        strategies = [
            (PlacementStrategy.GRID_BASED, self._place_ilots_grid_based),
            (PlacementStrategy.PHYSICS_BASED, self._place_ilots_physics_based),
            (PlacementStrategy.GENETIC_ALGORITHM, self._place_ilots_genetic_algorithm)
        ]
        
        best_placement = []
        best_score = 0
        
        for strategy_name, strategy_func in strategies:
            try:
                placement = strategy_func(spatial_data, ilot_specs, config)
                score = self._score_placement(placement, spatial_data, config)
                
                if score > best_score:
                    best_score = score
                    best_placement = placement
                    
            except Exception as e:
                self.logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
                continue
        
        return best_placement

    def _create_ilot_geometry(self, center: Point, spec: Dict[str, Any]) -> Polygon:
        """Create îlot geometry from center point and specifications"""
        width = spec['dimensions']['width']
        height = spec['dimensions']['height']
        
        # Create rectangular îlot
        x, y = center.x, center.y
        coords = [
            (x - width/2, y - height/2),
            (x + width/2, y - height/2),
            (x + width/2, y + height/2),
            (x - width/2, y + height/2)
        ]
        
        return Polygon(coords)

    def _is_valid_placement(self, ilot_geom: Polygon, area: Polygon, 
                          existing_ilots: List[PlacedIlot], 
                          config: IlotConfiguration) -> bool:
        """Check if îlot placement is valid"""
        
        # Check if îlot is within placeable area
        if not area.contains(ilot_geom):
            return False
        
        # Check minimum spacing from existing îlots
        min_spacing_mm = config.min_spacing * 1000
        
        for existing in existing_ilots:
            distance = ilot_geom.distance(existing.geometry)
            if distance < min_spacing_mm:
                return False
        
        return True

    def _simulate_physics_optimization(self, ilots: List[PlacedIlot], 
                                     spatial_data: Dict[str, Any], 
                                     config: IlotConfiguration) -> List[PlacedIlot]:
        """Simulate physics-based optimization"""
        # Simplified physics simulation
        # In production, this would include force calculations, momentum, etc.
        
        optimized_ilots = ilots.copy()
        
        for iteration in range(min(config.max_iterations // 10, 50)):
            for i, ilot in enumerate(optimized_ilots):
                # Calculate forces from other îlots and boundaries
                force_x, force_y = 0, 0
                
                # Repulsion from other îlots
                for j, other in enumerate(optimized_ilots):
                    if i != j:
                        dx = ilot.center.x - other.center.x
                        dy = ilot.center.y - other.center.y
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            repulsion = 1000 / max(distance, 100)  # Avoid division by zero
                            force_x += repulsion * dx / distance
                            force_y += repulsion * dy / distance
                
                # Apply small movement
                movement_factor = 0.1
                new_x = ilot.center.x + force_x * movement_factor
                new_y = ilot.center.y + force_y * movement_factor
                
                new_center = Point(new_x, new_y)
                new_geometry = self._create_ilot_geometry(new_center, {
                    'dimensions': {
                        'width': ilot.geometry.bounds[2] - ilot.geometry.bounds[0],
                        'height': ilot.geometry.bounds[3] - ilot.geometry.bounds[1]
                    }
                })
                
                # Check if new position is valid
                valid = False
                for area in spatial_data['placeable_areas']:
                    if area.contains(new_geometry):
                        # Check spacing from other îlots
                        min_spacing = config.min_spacing * 1000
                        spacing_ok = True
                        
                        for k, other in enumerate(optimized_ilots):
                            if i != k:
                                if new_geometry.distance(other.geometry) < min_spacing:
                                    spacing_ok = False
                                    break
                        
                        if spacing_ok:
                            valid = True
                            break
                
                if valid:
                    optimized_ilots[i] = PlacedIlot(
                        id=ilot.id,
                        geometry=new_geometry,
                        size_category=ilot.size_category,
                        area=ilot.area,
                        center=new_center,
                        placement_score=ilot.placement_score
                    )
        
        return optimized_ilots

    def _score_placement(self, placement: List[PlacedIlot], 
                        spatial_data: Dict[str, Any], 
                        config: IlotConfiguration) -> float:
        """Score a placement solution"""
        if not placement:
            return 0.0
        
        score = 0.0
        
        # Coverage score (how much space is utilized)
        total_ilot_area = sum(ilot.area for ilot in placement)
        total_available_area = sum(area.area for area in spatial_data['placeable_areas']) / 1000000  # Convert to m²
        
        if total_available_area > 0:
            coverage_score = min(total_ilot_area / total_available_area, 1.0) * 50
            score += coverage_score
        
        # Spacing score (uniform distribution)
        spacing_scores = []
        for i, ilot1 in enumerate(placement):
            min_distance = float('inf')
            for j, ilot2 in enumerate(placement):
                if i != j:
                    distance = ilot1.center.distance(ilot2.center)
                    min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                target_spacing = config.min_spacing * 1000
                spacing_score = max(0, 1 - abs(min_distance - target_spacing) / target_spacing) * 30
                spacing_scores.append(spacing_score)
        
        if spacing_scores:
            score += np.mean(spacing_scores)
        
        # Count score (number of successfully placed îlots)
        count_score = len(placement) * 0.5
        score += count_score
        
        return score

    def _optimize_placement(self, ilots: List[PlacedIlot], 
                          spatial_data: Dict[str, Any], 
                          config: IlotConfiguration) -> List[PlacedIlot]:
        """Post-processing optimization"""
        # Simple optimization: ensure all îlots are optimally spaced
        optimized = ilots.copy()
        
        # Calculate placement scores
        for ilot in optimized:
            ilot.placement_score = self._calculate_individual_score(ilot, optimized, spatial_data)
        
        return optimized

    def _calculate_individual_score(self, target_ilot: PlacedIlot, 
                                  all_ilots: List[PlacedIlot], 
                                  spatial_data: Dict[str, Any]) -> float:
        """Calculate individual placement score"""
        score = 1.0
        
        # Check if îlot is in optimal position relative to others
        distances = []
        for other in all_ilots:
            if other.id != target_ilot.id:
                distance = target_ilot.center.distance(other.center)
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            # Prefer medium distances (not too close, not too far)
            optimal_distance = 2000  # 2m in mm
            distance_score = max(0, 1 - abs(avg_distance - optimal_distance) / optimal_distance)
            score *= distance_score
        
        return score

    def _calculate_placement_metrics(self, placed_ilots: List[PlacedIlot], 
                                   spatial_data: Dict[str, Any], 
                                   config: IlotConfiguration) -> Dict[str, Any]:
        """Calculate comprehensive placement metrics"""
        
        metrics = {
            'total_ilots_placed': len(placed_ilots),
            'total_ilot_area': sum(ilot.area for ilot in placed_ilots),
            'size_distribution_actual': {},
            'space_utilization': 0.0,
            'average_spacing': 0.0,
            'placement_efficiency': 0.0,
            'constraint_satisfaction': 0.0
        }
        
        if not placed_ilots:
            return metrics
        
        # Calculate actual size distribution
        for size_cat in IlotSize:
            count = sum(1 for ilot in placed_ilots if ilot.size_category == size_cat)
            metrics['size_distribution_actual'][size_cat.value] = count
        
        # Calculate space utilization
        total_available_area = sum(area.area for area in spatial_data['placeable_areas']) / 1000000  # m²
        if total_available_area > 0:
            metrics['space_utilization'] = metrics['total_ilot_area'] / total_available_area
        
        # Calculate average spacing
        spacings = []
        for i, ilot1 in enumerate(placed_ilots):
            for j, ilot2 in enumerate(placed_ilots[i+1:], i+1):
                distance = ilot1.center.distance(ilot2.center) / 1000  # Convert to meters
                spacings.append(distance)
        
        if spacings:
            metrics['average_spacing'] = np.mean(spacings)
        
        # Calculate placement efficiency
        target_count = len(config.size_distribution) * 10  # Rough target
        efficiency = min(len(placed_ilots) / max(target_count, 1), 1.0)
        metrics['placement_efficiency'] = efficiency
        
        # Calculate constraint satisfaction
        satisfied_constraints = sum(1 for ilot in placed_ilots if ilot.constraints_satisfied)
        if placed_ilots:
            metrics['constraint_satisfaction'] = satisfied_constraints / len(placed_ilots)
        
        return metrics

    def _generate_placement_result(self, placed_ilots: List[PlacedIlot], 
                                 metrics: Dict[str, Any], 
                                 processing_time: float, 
                                 strategy: PlacementStrategy, 
                                 config: IlotConfiguration) -> Dict[str, Any]:
        """Generate comprehensive placement result"""
        
        # Convert placed îlots to visualization format
        ilots_data = []
        for ilot in placed_ilots:
            coords = list(ilot.geometry.exterior.coords)
            ilot_data = {
                'id': ilot.id,
                'coordinates': [[x, y] for x, y in coords],
                'center': [ilot.center.x, ilot.center.y],
                'size_category': ilot.size_category.value,
                'area': ilot.area,
                'color': self.size_colors[ilot.size_category],
                'placement_score': ilot.placement_score
            }
            ilots_data.append(ilot_data)
        
        result = {
            'success': True,
            'placed_ilots': ilots_data,
            'placement_metrics': metrics,
            'processing_info': {
                'processing_time': processing_time,
                'strategy_used': strategy.value,
                'total_ilots_requested': sum(config.size_distribution.values()),
                'total_ilots_placed': len(placed_ilots),
                'placement_success_rate': len(placed_ilots) / max(sum(config.size_distribution.values()), 1)
            },
            'configuration_used': {
                'size_distribution': {k.value: v for k, v in config.size_distribution.items()},
                'min_spacing': config.min_spacing,
                'wall_clearance': config.wall_clearance,
                'utilization_target': config.utilization_target,
                'strategy': strategy.value
            }
        }
        
        return result

    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with reason"""
        return {
            'success': False,
            'placed_ilots': [],
            'placement_metrics': {
                'total_ilots_placed': 0,
                'space_utilization': 0.0
            },
            'processing_info': {
                'processing_time': 0.1,
                'reason': reason
            }
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'success': False,
            'error': error_message,
            'placed_ilots': [],
            'placement_metrics': {
                'total_ilots_placed': 0,
                'space_utilization': 0.0
            }
        }

# Create global instance
advanced_ilot_placer = AdvancedIlotPlacer()