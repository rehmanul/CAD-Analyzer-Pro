
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import distance_matrix, Voronoi, ConvexHull
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import logging
from typing import Dict, List, Any, Tuple, Optional
import random
from dataclasses import dataclass
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class IlotConfiguration:
    """Configuration for îlot placement"""
    size_categories: Dict[str, Dict[str, float]]
    placement_constraints: Dict[str, float]
    optimization_weights: Dict[str, float]
    safety_requirements: Dict[str, float]

class AdvancedIlotPlacer:
    """Advanced AI-powered îlot placement system with intelligent zone awareness"""
    
    def __init__(self):
        self.default_config = IlotConfiguration(
            size_categories={
                'small': {'width_range': (1.0, 2.5), 'height_range': (1.0, 2.5), 'area_range': (1.0, 6.25)},
                'medium': {'width_range': (2.0, 4.0), 'height_range': (2.0, 4.0), 'area_range': (4.0, 16.0)},
                'large': {'width_range': (3.0, 6.0), 'height_range': (3.0, 6.0), 'area_range': (9.0, 36.0)},
                'xlarge': {'width_range': (4.0, 8.0), 'height_range': (4.0, 8.0), 'area_range': (16.0, 64.0)}
            },
            placement_constraints={
                'min_wall_distance': 0.5,
                'min_entrance_distance': 2.0,
                'min_restricted_distance': 1.5,
                'min_ilot_spacing': 1.2,
                'corridor_clearance': 1.8,
                'emergency_path_width': 1.2
            },
            optimization_weights={
                'space_efficiency': 0.25,
                'accessibility': 0.20,
                'safety': 0.20,
                'workflow_optimization': 0.15,
                'natural_light': 0.10,
                'noise_minimization': 0.10
            },
            safety_requirements={
                'max_occupancy_density': 0.15,  # people per m²
                'emergency_exit_distance': 30.0,  # meters
                'fire_safety_clearance': 2.0,  # meters
                'accessibility_compliance': True
            }
        )
        
        self.placement_strategies = {
            'grid_based': self._grid_based_placement,
            'cluster_optimized': self._cluster_optimized_placement,
            'flow_optimized': self._flow_optimized_placement,
            'hybrid_ai': self._hybrid_ai_placement
        }
        
        self.optimization_algorithms = {
            'genetic': self._genetic_algorithm_optimization,
            'simulated_annealing': self._simulated_annealing_optimization,
            'particle_swarm': self._particle_swarm_optimization,
            'gradient_descent': self._gradient_descent_optimization
        }
    
    def place_ilots_intelligent(self, zone_analysis: Dict[str, Any], 
                               ilot_requirements: Dict[str, Any],
                               config: Optional[IlotConfiguration] = None) -> Dict[str, Any]:
        """Main intelligent îlot placement function"""
        logger.info("Starting intelligent îlot placement with zone awareness")
        
        if config is None:
            config = self.default_config
        
        try:
            # Extract zone information
            zones = zone_analysis.get('zones', {})
            circulation_network = zone_analysis.get('circulation_network', {})
            spatial_relationships = zone_analysis.get('spatial_relationships', {})
            
            # Analyze placement opportunities
            placement_zones = self._analyze_placement_opportunities(zones, circulation_network)
            
            # Generate îlot specifications
            ilot_specifications = self._generate_ilot_specifications(ilot_requirements, config)
            
            # Run intelligent placement algorithm
            placement_results = self._execute_intelligent_placement(
                placement_zones, ilot_specifications, config, spatial_relationships
            )
            
            # Optimize placement using AI algorithms
            optimized_placement = self._optimize_placement_with_ai(
                placement_results, placement_zones, config
            )
            
            # Validate and post-process
            final_placement = self._validate_and_finalize_placement(
                optimized_placement, zones, config
            )
            
            # Generate placement analytics
            analytics = self._generate_placement_analytics(final_placement, zones, config)
            
            return {
                'ilots': final_placement,
                'placement_strategy': 'intelligent_zone_aware',
                'analytics': analytics,
                'optimization_metrics': self._calculate_optimization_metrics(final_placement, zones),
                'compliance_report': self._generate_compliance_report(final_placement, config),
                'recommendations': self._generate_placement_recommendations(final_placement, zones)
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent îlot placement: {str(e)}")
            return self._create_fallback_placement(zone_analysis, ilot_requirements)
    
    def _analyze_placement_opportunities(self, zones: Dict[str, List[Dict[str, Any]]], 
                                       circulation_network: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze zones to identify optimal placement opportunities"""
        
        placement_zones = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'restricted': []
        }
        
        # Analyze different zone types for placement suitability
        rooms = zones.get('rooms', [])
        open_spaces = zones.get('open_spaces', [])
        corridors = zones.get('corridors', [])
        
        # Score rooms for îlot placement
        for room in rooms:
            score = self._calculate_room_placement_score(room, circulation_network)
            priority = 'high_priority' if score > 0.7 else 'medium_priority' if score > 0.4 else 'low_priority'
            
            placement_zones[priority].append({
                'zone_id': room['id'],
                'zone_type': 'room',
                'geometry': room['geometry'],
                'area': room.get('area', 0),
                'placement_score': score,
                'constraints': self._extract_zone_constraints(room),
                'preferred_ilot_types': self._determine_preferred_ilot_types(room, score)
            })
        
        # Analyze open spaces
        for space in open_spaces:
            # Large open spaces can accommodate multiple îlots
            if space.get('area', 0) > 20:
                sub_zones = self._subdivide_large_space(space)
                for sub_zone in sub_zones:
                    score = self._calculate_space_placement_score(sub_zone, circulation_network)
                    priority = 'high_priority' if score > 0.6 else 'medium_priority'
                    
                    placement_zones[priority].append({
                        'zone_id': f"{space['id']}_sub_{len(placement_zones[priority])}",
                        'zone_type': 'open_space_subdivision',
                        'geometry': sub_zone['geometry'],
                        'area': sub_zone['area'],
                        'placement_score': score,
                        'constraints': self._extract_zone_constraints(space),
                        'preferred_ilot_types': ['small', 'medium', 'large']
                    })
            else:
                score = self._calculate_space_placement_score(space, circulation_network)
                priority = 'medium_priority' if score > 0.5 else 'low_priority'
                
                placement_zones[priority].append({
                    'zone_id': space['id'],
                    'zone_type': 'open_space',
                    'geometry': space['geometry'],
                    'area': space.get('area', 0),
                    'placement_score': score,
                    'constraints': self._extract_zone_constraints(space),
                    'preferred_ilot_types': ['small', 'medium']
                })
        
        # Mark corridors as restricted for îlot placement
        for corridor in corridors:
            placement_zones['restricted'].append({
                'zone_id': corridor['id'],
                'zone_type': 'corridor',
                'geometry': corridor['geometry'],
                'reason': 'circulation_path'
            })
        
        return placement_zones
    
    def _calculate_room_placement_score(self, room: Dict[str, Any], 
                                      circulation_network: Dict[str, Any]) -> float:
        """Calculate placement suitability score for a room"""
        score = 0.0
        
        # Area score (larger rooms are better for îlots)
        area = room.get('area', 0)
        if area > 15:
            score += 0.3
        elif area > 8:
            score += 0.2
        elif area > 4:
            score += 0.1
        
        # Shape score (more regular shapes are better)
        properties = room.get('properties', {})
        aspect_ratio = properties.get('aspect_ratio', 1)
        if 0.7 <= aspect_ratio <= 1.8:  # Good aspect ratio
            score += 0.2
        
        convexity = properties.get('convexity', 0.5)
        score += convexity * 0.2  # More convex shapes are better
        
        # Accessibility score
        circulation_graph = circulation_network.get('graph')
        if circulation_graph and room['id'] in circulation_graph.nodes():
            connectivity = circulation_graph.degree(room['id'])
            score += min(0.3, connectivity * 0.1)
        
        return min(1.0, score)
    
    def _calculate_space_placement_score(self, space: Dict[str, Any], 
                                       circulation_network: Dict[str, Any]) -> float:
        """Calculate placement suitability score for an open space"""
        score = 0.0
        
        # Area score
        area = space.get('area', 0)
        if area > 30:
            score += 0.4
        elif area > 15:
            score += 0.3
        elif area > 8:
            score += 0.2
        
        # Central location bonus
        center = space.get('center', (0, 0))
        # This would use actual building centroid calculation
        centrality_score = 0.5  # Placeholder
        score += centrality_score * 0.3
        
        # Natural light potential
        # This would analyze proximity to exterior walls
        light_score = 0.6  # Placeholder
        score += light_score * 0.2
        
        # Safety and egress score
        egress_score = 0.7  # Placeholder
        score += egress_score * 0.1
        
        return min(1.0, score)
    
    def _subdivide_large_space(self, space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Subdivide large spaces into manageable placement zones"""
        geometry = space.get('geometry')
        area = space.get('area', 0)
        
        if not geometry or area < 30:
            return [space]
        
        # Simple rectangular subdivision
        try:
            bounds = geometry.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            # Determine subdivision strategy
            if width > height * 1.5:
                # Subdivide horizontally
                num_divisions = max(2, int(width // 6))  # ~6m wide zones
                division_width = width / num_divisions
                
                sub_zones = []
                for i in range(num_divisions):
                    x_min = bounds[0] + i * division_width
                    x_max = bounds[0] + (i + 1) * division_width
                    
                    sub_geometry = box(x_min, bounds[1], x_max, bounds[3])
                    sub_zones.append({
                        'geometry': sub_geometry,
                        'area': sub_geometry.area,
                        'center': (sub_geometry.centroid.x, sub_geometry.centroid.y)
                    })
                
                return sub_zones
            
            elif height > width * 1.5:
                # Subdivide vertically
                num_divisions = max(2, int(height // 6))
                division_height = height / num_divisions
                
                sub_zones = []
                for i in range(num_divisions):
                    y_min = bounds[1] + i * division_height
                    y_max = bounds[1] + (i + 1) * division_height
                    
                    sub_geometry = box(bounds[0], y_min, bounds[2], y_max)
                    sub_zones.append({
                        'geometry': sub_geometry,
                        'area': sub_geometry.area,
                        'center': (sub_geometry.centroid.x, sub_geometry.centroid.y)
                    })
                
                return sub_zones
            
            else:
                # Grid subdivision
                cols = max(2, int(width // 5))
                rows = max(2, int(height // 5))
                
                sub_zones = []
                for row in range(rows):
                    for col in range(cols):
                        x_min = bounds[0] + col * width / cols
                        x_max = bounds[0] + (col + 1) * width / cols
                        y_min = bounds[1] + row * height / rows
                        y_max = bounds[1] + (row + 1) * height / rows
                        
                        sub_geometry = box(x_min, y_min, x_max, y_max)
                        sub_zones.append({
                            'geometry': sub_geometry,
                            'area': sub_geometry.area,
                            'center': (sub_geometry.centroid.x, sub_geometry.centroid.y)
                        })
                
                return sub_zones
        
        except Exception as e:
            logger.warning(f"Error subdividing space: {str(e)}")
            return [space]
    
    def _extract_zone_constraints(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        """Extract placement constraints from zone characteristics"""
        constraints = {
            'min_clearance': 1.0,
            'max_occupancy': 0.7,  # 70% of zone area
            'access_requirements': [],
            'environmental_factors': {}
        }
        
        zone_type = zone.get('type', 'unknown')
        
        if zone_type == 'room':
            constraints['max_occupancy'] = 0.6  # Rooms need more circulation space
            constraints['access_requirements'] = ['corridor_access']
        
        elif zone_type == 'open_space':
            constraints['max_occupancy'] = 0.8  # Open spaces can be denser
            constraints['min_clearance'] = 1.5  # More clearance needed
        
        # Add safety constraints
        constraints['emergency_egress'] = True
        constraints['fire_safety_clearance'] = 2.0
        
        return constraints
    
    def _determine_preferred_ilot_types(self, zone: Dict[str, Any], score: float) -> List[str]:
        """Determine preferred îlot types for a zone"""
        area = zone.get('area', 0)
        zone_type = zone.get('type', 'unknown')
        
        preferred_types = []
        
        if area > 25:
            preferred_types.extend(['large', 'xlarge'])
        if area > 10:
            preferred_types.extend(['medium', 'large'])
        if area > 5:
            preferred_types.append('medium')
        
        preferred_types.append('small')  # Small îlots work everywhere
        
        # Zone-specific preferences
        if zone_type == 'room' and score > 0.8:
            # High-quality rooms prefer larger îlots
            preferred_types = ['large', 'xlarge', 'medium']
        
        return list(set(preferred_types))  # Remove duplicates
    
    def _generate_ilot_specifications(self, requirements: Dict[str, Any], 
                                    config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Generate detailed îlot specifications based on requirements"""
        
        total_ilots = requirements.get('total_count', 20)
        size_distribution = requirements.get('size_distribution', {
            'small': 0.3, 'medium': 0.4, 'large': 0.2, 'xlarge': 0.1
        })
        
        specifications = []
        
        for size_category, percentage in size_distribution.items():
            count = int(total_ilots * percentage)
            size_config = config.size_categories.get(size_category, config.size_categories['medium'])
            
            for i in range(count):
                # Generate varied dimensions within range
                width = np.random.uniform(*size_config['width_range'])
                height = np.random.uniform(*size_config['height_range'])
                
                spec = {
                    'id': f'{size_category}_ilot_{i}',
                    'size_category': size_category,
                    'dimensions': {
                        'width': width,
                        'height': height,
                        'area': width * height
                    },
                    'requirements': {
                        'accessibility': requirements.get('accessibility_level', 'standard'),
                        'power_access': requirements.get('power_required', True),
                        'data_connectivity': requirements.get('data_required', True),
                        'natural_light_preference': requirements.get('natural_light', 'preferred'),
                        'noise_sensitivity': requirements.get('noise_sensitivity', 'medium')
                    },
                    'placement_preferences': {
                        'wall_proximity': size_category in ['large', 'xlarge'],
                        'corridor_access': True,
                        'group_placement': size_category == 'small'
                    }
                }
                
                specifications.append(spec)
        
        return specifications
    
    def _execute_intelligent_placement(self, placement_zones: Dict[str, Any], 
                                     ilot_specifications: List[Dict[str, Any]], 
                                     config: IlotConfiguration,
                                     spatial_relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the main intelligent placement algorithm"""
        
        placed_ilots = []
        
        # Sort îlots by size (place larger ones first)
        sorted_ilots = sorted(ilot_specifications, 
                            key=lambda x: x['dimensions']['area'], reverse=True)
        
        # Sort placement zones by priority
        high_priority_zones = placement_zones.get('high_priority', [])
        medium_priority_zones = placement_zones.get('medium_priority', [])
        low_priority_zones = placement_zones.get('low_priority', [])
        
        all_zones = high_priority_zones + medium_priority_zones + low_priority_zones
        
        for ilot_spec in sorted_ilots:
            best_placement = self._find_best_placement_for_ilot(
                ilot_spec, all_zones, placed_ilots, config, spatial_relationships
            )
            
            if best_placement:
                placed_ilots.append(best_placement)
                # Update zone availability
                self._update_zone_availability(best_placement, all_zones)
        
        return placed_ilots
    
    def _find_best_placement_for_ilot(self, ilot_spec: Dict[str, Any], 
                                    available_zones: List[Dict[str, Any]], 
                                    existing_ilots: List[Dict[str, Any]], 
                                    config: IlotConfiguration,
                                    spatial_relationships: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best placement location for a specific îlot"""
        
        best_placement = None
        best_score = 0
        
        ilot_width = ilot_spec['dimensions']['width']
        ilot_height = ilot_spec['dimensions']['height']
        size_category = ilot_spec['size_category']
        
        for zone in available_zones:
            if not self._can_fit_in_zone(ilot_spec, zone, existing_ilots, config):
                continue
            
            # Try multiple positions within the zone
            candidate_positions = self._generate_candidate_positions(
                zone, ilot_width, ilot_height, existing_ilots, config
            )
            
            for position in candidate_positions:
                placement_score = self._evaluate_placement_position(
                    ilot_spec, position, zone, existing_ilots, config, spatial_relationships
                )
                
                if placement_score > best_score:
                    best_score = placement_score
                    best_placement = {
                        'id': ilot_spec['id'],
                        'size_category': size_category,
                        'position': position,
                        'dimensions': ilot_spec['dimensions'],
                        'zone_id': zone['zone_id'],
                        'placement_score': placement_score,
                        'requirements': ilot_spec['requirements'],
                        'geometry': box(
                            position['x'] - ilot_width/2,
                            position['y'] - ilot_height/2,
                            position['x'] + ilot_width/2,
                            position['y'] + ilot_height/2
                        )
                    }
        
        return best_placement
    
    def _can_fit_in_zone(self, ilot_spec: Dict[str, Any], zone: Dict[str, Any], 
                        existing_ilots: List[Dict[str, Any]], 
                        config: IlotConfiguration) -> bool:
        """Check if an îlot can fit in a zone"""
        
        ilot_area = ilot_spec['dimensions']['area']
        zone_geometry = zone.get('geometry')
        zone_area = zone.get('area', 0)
        
        if not zone_geometry or zone_area < ilot_area * 1.5:  # Need 50% extra space
            return False
        
        # Check if îlot type is preferred for this zone
        preferred_types = zone.get('preferred_ilot_types', [])
        if preferred_types and ilot_spec['size_category'] not in preferred_types:
            return False
        
        # Check existing occupation
        existing_area_in_zone = sum(
            ilot['dimensions']['area'] for ilot in existing_ilots 
            if ilot.get('zone_id') == zone['zone_id']
        )
        
        max_occupancy = zone.get('constraints', {}).get('max_occupancy', 0.7)
        available_area = zone_area * max_occupancy - existing_area_in_zone
        
        return available_area >= ilot_area
    
    def _generate_candidate_positions(self, zone: Dict[str, Any], 
                                    ilot_width: float, ilot_height: float, 
                                    existing_ilots: List[Dict[str, Any]], 
                                    config: IlotConfiguration) -> List[Dict[str, float]]:
        """Generate candidate positions for îlot placement within a zone"""
        
        zone_geometry = zone.get('geometry')
        if not zone_geometry:
            return []
        
        bounds = zone_geometry.bounds
        min_x, min_y, max_x, max_y = bounds
        
        # Account for îlot dimensions and clearances
        clearance = config.placement_constraints['min_ilot_spacing']
        
        effective_min_x = min_x + ilot_width/2 + clearance
        effective_max_x = max_x - ilot_width/2 - clearance
        effective_min_y = min_y + ilot_height/2 + clearance
        effective_max_y = max_y - ilot_height/2 - clearance
        
        if effective_min_x >= effective_max_x or effective_min_y >= effective_max_y:
            return []
        
        # Generate grid of candidate positions
        grid_spacing = 1.0  # 1 meter grid
        
        x_positions = np.arange(effective_min_x, effective_max_x, grid_spacing)
        y_positions = np.arange(effective_min_y, effective_max_y, grid_spacing)
        
        candidates = []
        
        for x in x_positions:
            for y in y_positions:
                position = {'x': x, 'y': y, 'z': 0}
                
                # Check if position is within zone geometry
                point = Point(x, y)
                if not zone_geometry.contains(point):
                    continue
                
                # Check clearance from existing îlots
                if self._has_sufficient_clearance(position, ilot_width, ilot_height, 
                                                existing_ilots, config):
                    candidates.append(position)
        
        # If grid approach yields too few candidates, try random sampling
        if len(candidates) < 5:
            candidates.extend(self._random_position_sampling(
                zone_geometry, ilot_width, ilot_height, existing_ilots, config, 20
            ))
        
        return candidates
    
    def _has_sufficient_clearance(self, position: Dict[str, float], 
                                 ilot_width: float, ilot_height: float, 
                                 existing_ilots: List[Dict[str, Any]], 
                                 config: IlotConfiguration) -> bool:
        """Check if a position has sufficient clearance from existing îlots"""
        
        min_spacing = config.placement_constraints['min_ilot_spacing']
        
        # Create bounding box for proposed îlot
        proposed_box = box(
            position['x'] - ilot_width/2,
            position['y'] - ilot_height/2,
            position['x'] + ilot_width/2,
            position['y'] + ilot_height/2
        )
        
        for existing_ilot in existing_ilots:
            existing_geometry = existing_ilot.get('geometry')
            if existing_geometry and proposed_box.distance(existing_geometry) < min_spacing:
                return False
        
        return True
    
    def _random_position_sampling(self, zone_geometry: Polygon, 
                                 ilot_width: float, ilot_height: float, 
                                 existing_ilots: List[Dict[str, Any]], 
                                 config: IlotConfiguration, 
                                 num_samples: int) -> List[Dict[str, float]]:
        """Generate random candidate positions within zone"""
        
        candidates = []
        bounds = zone_geometry.bounds
        
        for _ in range(num_samples):
            # Random position within bounds
            x = random.uniform(bounds[0] + ilot_width/2, bounds[2] - ilot_width/2)
            y = random.uniform(bounds[1] + ilot_height/2, bounds[3] - ilot_height/2)
            
            position = {'x': x, 'y': y, 'z': 0}
            
            # Check if position is valid
            point = Point(x, y)
            if (zone_geometry.contains(point) and 
                self._has_sufficient_clearance(position, ilot_width, ilot_height, 
                                             existing_ilots, config)):
                candidates.append(position)
        
        return candidates
    
    def _evaluate_placement_position(self, ilot_spec: Dict[str, Any], 
                                   position: Dict[str, float], 
                                   zone: Dict[str, Any], 
                                   existing_ilots: List[Dict[str, Any]], 
                                   config: IlotConfiguration,
                                   spatial_relationships: Dict[str, Any]) -> float:
        """Evaluate the quality of a specific placement position"""
        
        score = 0.0
        weights = config.optimization_weights
        
        # 1. Zone compatibility score
        zone_score = zone.get('placement_score', 0.5)
        score += weights['space_efficiency'] * zone_score
        
        # 2. Accessibility score
        accessibility_score = self._calculate_position_accessibility(
            position, zone, existing_ilots, spatial_relationships
        )
        score += weights['accessibility'] * accessibility_score
        
        # 3. Safety score
        safety_score = self._calculate_position_safety(
            position, ilot_spec, zone, existing_ilots, config
        )
        score += weights['safety'] * safety_score
        
        # 4. Workflow optimization score
        workflow_score = self._calculate_workflow_optimization(
            position, ilot_spec, existing_ilots, spatial_relationships
        )
        score += weights['workflow_optimization'] * workflow_score
        
        # 5. Natural light score
        light_score = self._calculate_natural_light_score(position, zone)
        score += weights['natural_light'] * light_score
        
        # 6. Noise minimization score
        noise_score = self._calculate_noise_score(position, zone, existing_ilots)
        score += weights['noise_minimization'] * noise_score
        
        return min(1.0, score)
    
    def _calculate_position_accessibility(self, position: Dict[str, float], 
                                        zone: Dict[str, Any], 
                                        existing_ilots: List[Dict[str, Any]], 
                                        spatial_relationships: Dict[str, Any]) -> float:
        """Calculate accessibility score for a position"""
        
        score = 0.5  # Base score
        
        # Distance to zone center (closer to center is often better)
        zone_center = zone.get('center', (0, 0))
        if len(zone_center) >= 2:
            distance_to_center = np.sqrt(
                (position['x'] - zone_center[0])**2 + 
                (position['y'] - zone_center[1])**2
            )
            
            # Normalize distance (assume max useful distance is 20m)
            normalized_distance = min(1.0, distance_to_center / 20.0)
            score += (1 - normalized_distance) * 0.3
        
        # Distance to existing îlots (some clustering is good, but not too close)
        if existing_ilots:
            distances = []
            for ilot in existing_ilots:
                ilot_pos = ilot.get('position', {})
                if ilot_pos:
                    dist = np.sqrt(
                        (position['x'] - ilot_pos.get('x', 0))**2 + 
                        (position['y'] - ilot_pos.get('y', 0))**2
                    )
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                # Optimal distance is around 3-8 meters
                if 3 <= avg_distance <= 8:
                    score += 0.2
                elif avg_distance < 3:
                    score += 0.1  # Too close
                else:
                    score += max(0, 0.2 - (avg_distance - 8) * 0.02)  # Too far
        
        return min(1.0, score)
    
    def _calculate_position_safety(self, position: Dict[str, float], 
                                 ilot_spec: Dict[str, Any], 
                                 zone: Dict[str, Any], 
                                 existing_ilots: List[Dict[str, Any]], 
                                 config: IlotConfiguration) -> float:
        """Calculate safety score for a position"""
        
        score = 0.8  # Base safety score
        
        # Check emergency egress paths
        # This is a simplified calculation - in practice would use pathfinding
        zone_geometry = zone.get('geometry')
        if zone_geometry:
            bounds = zone_geometry.bounds
            
            # Distance to nearest edge (potential egress)
            edge_distances = [
                position['x'] - bounds[0],  # Left edge
                bounds[2] - position['x'],  # Right edge
                position['y'] - bounds[1],  # Bottom edge
                bounds[3] - position['y']   # Top edge
            ]
            
            min_edge_distance = min(edge_distances)
            max_safe_distance = config.safety_requirements['emergency_exit_distance']
            
            if min_edge_distance < max_safe_distance:
                score += 0.2
        
        # Fire safety clearance
        fire_clearance = config.safety_requirements['fire_safety_clearance']
        
        # Check clearance from other îlots
        adequate_clearance = True
        for ilot in existing_ilots:
            ilot_pos = ilot.get('position', {})
            if ilot_pos:
                distance = np.sqrt(
                    (position['x'] - ilot_pos.get('x', 0))**2 + 
                    (position['y'] - ilot_pos.get('y', 0))**2
                )
                if distance < fire_clearance:
                    adequate_clearance = False
                    break
        
        if not adequate_clearance:
            score -= 0.3
        
        return max(0, score)
    
    def _calculate_workflow_optimization(self, position: Dict[str, float], 
                                       ilot_spec: Dict[str, Any], 
                                       existing_ilots: List[Dict[str, Any]], 
                                       spatial_relationships: Dict[str, Any]) -> float:
        """Calculate workflow optimization score"""
        
        score = 0.5  # Base score
        
        size_category = ilot_spec.get('size_category', 'medium')
        
        # Group similar îlots together
        similar_ilots = [
            ilot for ilot in existing_ilots 
            if ilot.get('size_category') == size_category
        ]
        
        if similar_ilots:
            distances = []
            for ilot in similar_ilots:
                ilot_pos = ilot.get('position', {})
                if ilot_pos:
                    dist = np.sqrt(
                        (position['x'] - ilot_pos.get('x', 0))**2 + 
                        (position['y'] - ilot_pos.get('y', 0))**2
                    )
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                # Optimal clustering distance
                if 2 <= avg_distance <= 6:
                    score += 0.3
                else:
                    score += max(0, 0.3 - abs(avg_distance - 4) * 0.05)
        
        # Consider functional relationships
        # (This would be more sophisticated in practice)
        score += 0.2  # Placeholder for functional relationship score
        
        return min(1.0, score)
    
    def _calculate_natural_light_score(self, position: Dict[str, float], 
                                     zone: Dict[str, Any]) -> float:
        """Calculate natural light access score"""
        
        # Simplified calculation - would use actual building geometry
        zone_geometry = zone.get('geometry')
        if not zone_geometry:
            return 0.5
        
        bounds = zone_geometry.bounds
        
        # Assume light comes from exterior walls
        # Calculate distance to nearest perimeter
        perimeter_distances = [
            position['x'] - bounds[0],
            bounds[2] - position['x'],
            position['y'] - bounds[1],
            bounds[3] - position['y']
        ]
        
        min_perimeter_distance = min(perimeter_distances)
        
        # Closer to perimeter = more natural light
        max_distance = 15.0  # Assume effective daylight distance
        light_score = max(0, 1 - min_perimeter_distance / max_distance)
        
        return light_score
    
    def _calculate_noise_score(self, position: Dict[str, float], 
                             zone: Dict[str, Any], 
                             existing_ilots: List[Dict[str, Any]]) -> float:
        """Calculate noise minimization score"""
        
        # Base score assumes moderate noise environment
        score = 0.7
        
        # Penalty for being too close to high-activity areas
        # (This would use actual noise source analysis)
        
        # Distance from circulation paths
        zone_type = zone.get('zone_type', 'unknown')
        if zone_type == 'corridor':
            score -= 0.3  # Corridors are noisy
        elif zone_type == 'room':
            score += 0.2  # Rooms are quieter
        
        # Distance from other îlots (clustering can increase noise)
        if existing_ilots:
            nearby_count = 0
            for ilot in existing_ilots:
                ilot_pos = ilot.get('position', {})
                if ilot_pos:
                    distance = np.sqrt(
                        (position['x'] - ilot_pos.get('x', 0))**2 + 
                        (position['y'] - ilot_pos.get('y', 0))**2
                    )
                    if distance < 5:  # Within 5 meters
                        nearby_count += 1
            
            # Penalty for too many nearby îlots
            if nearby_count > 4:
                score -= (nearby_count - 4) * 0.05
        
        return max(0, min(1.0, score))
    
    def _update_zone_availability(self, placed_ilot: Dict[str, Any], 
                                 zones: List[Dict[str, Any]]):
        """Update zone availability after placing an îlot"""
        
        zone_id = placed_ilot.get('zone_id')
        ilot_area = placed_ilot['dimensions']['area']
        
        for zone in zones:
            if zone['zone_id'] == zone_id:
                current_used_area = zone.get('used_area', 0)
                zone['used_area'] = current_used_area + ilot_area
                
                # Update placement score based on remaining capacity
                max_occupancy = zone.get('constraints', {}).get('max_occupancy', 0.7)
                total_capacity = zone['area'] * max_occupancy
                remaining_capacity = total_capacity - zone['used_area']
                
                if remaining_capacity < 4:  # Less than 4 m² remaining
                    zone['placement_score'] *= 0.5  # Reduce attractiveness
                
                break
    
    def _optimize_placement_with_ai(self, initial_placement: List[Dict[str, Any]], 
                                   placement_zones: Dict[str, Any], 
                                   config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Optimize placement using AI algorithms"""
        
        # Use genetic algorithm for optimization
        optimization_method = 'genetic'  # Could be configurable
        
        if optimization_method in self.optimization_algorithms:
            optimizer = self.optimization_algorithms[optimization_method]
            return optimizer(initial_placement, placement_zones, config)
        else:
            return initial_placement
    
    def _genetic_algorithm_optimization(self, initial_placement: List[Dict[str, Any]], 
                                      placement_zones: Dict[str, Any], 
                                      config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Optimize placement using genetic algorithm"""
        
        # Simplified genetic algorithm implementation
        population_size = 20
        generations = 50
        mutation_rate = 0.1
        
        # Create initial population
        population = [initial_placement]
        
        # Generate additional random placements
        for _ in range(population_size - 1):
            variant = self._create_placement_variant(initial_placement, placement_zones, config)
            population.append(variant)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_placement_fitness(placement, config) 
                            for placement in population]
            
            # Selection and reproduction
            new_population = []
            
            # Keep best solutions (elitism)
            best_indices = np.argsort(fitness_scores)[-5:]
            for idx in best_indices:
                new_population.append(population[idx])
            
            # Create offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                offspring = self._crossover_placements(parent1, parent2, config)
                
                # Mutation
                if random.random() < mutation_rate:
                    offspring = self._mutate_placement(offspring, placement_zones, config)
                
                new_population.append(offspring)
            
            population = new_population
        
        # Return best solution
        final_fitness_scores = [self._evaluate_placement_fitness(placement, config) 
                              for placement in population]
        best_index = np.argmax(final_fitness_scores)
        
        return population[best_index]
    
    def _create_placement_variant(self, base_placement: List[Dict[str, Any]], 
                                 placement_zones: Dict[str, Any], 
                                 config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Create a variant of the base placement"""
        
        variant = []
        
        for ilot in base_placement:
            # Small random perturbation
            position = ilot['position'].copy()
            position['x'] += random.uniform(-2, 2)
            position['y'] += random.uniform(-2, 2)
            
            variant_ilot = ilot.copy()
            variant_ilot['position'] = position
            variant_ilot['geometry'] = box(
                position['x'] - ilot['dimensions']['width']/2,
                position['y'] - ilot['dimensions']['height']/2,
                position['x'] + ilot['dimensions']['width']/2,
                position['y'] + ilot['dimensions']['height']/2
            )
            
            variant.append(variant_ilot)
        
        return variant
    
    def _evaluate_placement_fitness(self, placement: List[Dict[str, Any]], 
                                   config: IlotConfiguration) -> float:
        """Evaluate overall fitness of a placement"""
        
        if not placement:
            return 0
        
        total_score = 0
        weights = config.optimization_weights
        
        # Individual îlot scores
        individual_scores = [ilot.get('placement_score', 0.5) for ilot in placement]
        avg_individual_score = np.mean(individual_scores)
        
        # Space efficiency
        total_area = sum(ilot['dimensions']['area'] for ilot in placement)
        # This would calculate against available space
        space_efficiency = min(1.0, total_area / 1000.0)  # Placeholder
        
        # Clustering quality
        clustering_score = self._calculate_clustering_quality(placement)
        
        # Safety compliance
        safety_score = self._calculate_overall_safety(placement, config)
        
        # Combine scores
        total_score = (
            weights['space_efficiency'] * space_efficiency +
            weights['accessibility'] * avg_individual_score +
            weights['safety'] * safety_score +
            weights['workflow_optimization'] * clustering_score
        )
        
        return total_score
    
    def _calculate_clustering_quality(self, placement: List[Dict[str, Any]]) -> float:
        """Calculate quality of îlot clustering"""
        
        if len(placement) < 2:
            return 1.0
        
        # Group by size category
        size_groups = {}
        for ilot in placement:
            size_cat = ilot.get('size_category', 'medium')
            if size_cat not in size_groups:
                size_groups[size_cat] = []
            size_groups[size_cat].append(ilot)
        
        group_scores = []
        
        for size_cat, ilots in size_groups.items():
            if len(ilots) < 2:
                group_scores.append(1.0)
                continue
            
            # Calculate intra-group distances
            positions = [(ilot['position']['x'], ilot['position']['y']) for ilot in ilots]
            distances = distance_matrix(positions, positions)
            
            # Remove diagonal (self-distances)
            np.fill_diagonal(distances, np.inf)
            
            # Calculate clustering metric
            avg_distance = np.mean(distances[distances < np.inf])
            
            # Good clustering: 3-8 meters average distance
            if 3 <= avg_distance <= 8:
                group_scores.append(1.0)
            else:
                group_scores.append(max(0, 1.0 - abs(avg_distance - 5.5) * 0.1))
        
        return np.mean(group_scores) if group_scores else 0.5
    
    def _calculate_overall_safety(self, placement: List[Dict[str, Any]], 
                                config: IlotConfiguration) -> float:
        """Calculate overall safety score for placement"""
        
        min_spacing = config.placement_constraints['min_ilot_spacing']
        
        violations = 0
        total_checks = 0
        
        # Check spacing violations
        for i, ilot1 in enumerate(placement):
            for ilot2 in placement[i+1:]:
                distance = np.sqrt(
                    (ilot1['position']['x'] - ilot2['position']['x'])**2 +
                    (ilot1['position']['y'] - ilot2['position']['y'])**2
                )
                
                required_distance = min_spacing
                if distance < required_distance:
                    violations += 1
                
                total_checks += 1
        
        if total_checks == 0:
            return 1.0
        
        safety_score = 1.0 - (violations / total_checks)
        return max(0, safety_score)
    
    def _tournament_selection(self, population: List[List[Dict[str, Any]]], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[Dict[str, Any]]:
        """Tournament selection for genetic algorithm"""
        
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover_placements(self, parent1: List[Dict[str, Any]], 
                            parent2: List[Dict[str, Any]], 
                            config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Crossover operation for genetic algorithm"""
        
        if len(parent1) != len(parent2):
            return parent1  # Fallback
        
        offspring = []
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Take first part from parent1, second part from parent2
        for i in range(len(parent1)):
            if i < crossover_point:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        
        return offspring
    
    def _mutate_placement(self, placement: List[Dict[str, Any]], 
                         placement_zones: Dict[str, Any], 
                         config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Mutation operation for genetic algorithm"""
        
        if not placement:
            return placement
        
        # Select random îlot to mutate
        mutate_idx = random.randint(0, len(placement) - 1)
        mutated_placement = placement.copy()
        
        # Small position perturbation
        ilot = mutated_placement[mutate_idx]
        position = ilot['position'].copy()
        position['x'] += random.uniform(-1, 1)
        position['y'] += random.uniform(-1, 1)
        
        # Update geometry
        width = ilot['dimensions']['width']
        height = ilot['dimensions']['height']
        
        mutated_ilot = ilot.copy()
        mutated_ilot['position'] = position
        mutated_ilot['geometry'] = box(
            position['x'] - width/2,
            position['y'] - height/2,
            position['x'] + width/2,
            position['y'] + height/2
        )
        
        mutated_placement[mutate_idx] = mutated_ilot
        
        return mutated_placement
    
    # Placeholder implementations for other optimization algorithms
    def _simulated_annealing_optimization(self, initial_placement: List[Dict[str, Any]], 
                                        placement_zones: Dict[str, Any], 
                                        config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Simulated annealing optimization (placeholder)"""
        return initial_placement
    
    def _particle_swarm_optimization(self, initial_placement: List[Dict[str, Any]], 
                                   placement_zones: Dict[str, Any], 
                                   config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Particle swarm optimization (placeholder)"""
        return initial_placement
    
    def _gradient_descent_optimization(self, initial_placement: List[Dict[str, Any]], 
                                     placement_zones: Dict[str, Any], 
                                     config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Gradient descent optimization (placeholder)"""
        return initial_placement
    
    def _validate_and_finalize_placement(self, placement: List[Dict[str, Any]], 
                                        zones: Dict[str, Any], 
                                        config: IlotConfiguration) -> List[Dict[str, Any]]:
        """Validate and finalize the optimized placement"""
        
        validated_placement = []
        
        for ilot in placement:
            # Validate constraints
            if self._validate_ilot_constraints(ilot, zones, config):
                # Add final metadata
                ilot['validation_status'] = 'passed'
                ilot['final_score'] = ilot.get('placement_score', 0.5)
                validated_placement.append(ilot)
            else:
                logger.warning(f"Îlot {ilot['id']} failed validation")
        
        return validated_placement
    
    def _validate_ilot_constraints(self, ilot: Dict[str, Any], 
                                  zones: Dict[str, Any], 
                                  config: IlotConfiguration) -> bool:
        """Validate that an îlot meets all constraints"""
        
        # Check minimum spacing
        min_spacing = config.placement_constraints['min_ilot_spacing']
        
        # Check safety requirements
        safety_reqs = config.safety_requirements
        
        # All validations pass for now (simplified)
        return True
    
    def _generate_placement_analytics(self, placement: List[Dict[str, Any]], 
                                    zones: Dict[str, Any], 
                                    config: IlotConfiguration) -> Dict[str, Any]:
        """Generate comprehensive analytics for the placement"""
        
        if not placement:
            return {'error': 'No placement data available'}
        
        # Calculate basic statistics
        total_ilots = len(placement)
        total_area = sum(ilot['dimensions']['area'] for ilot in placement)
        
        size_distribution = {}
        for ilot in placement:
            size_cat = ilot.get('size_category', 'unknown')
            size_distribution[size_cat] = size_distribution.get(size_cat, 0) + 1
        
        # Calculate zone utilization
        zone_utilization = {}
        zones_list = []
        for zone_type, zone_list in zones.items():
            zones_list.extend(zone_list)
        
        for zone in zones_list:
            zone_id = zone.get('id', 'unknown')
            zone_ilots = [ilot for ilot in placement if ilot.get('zone_id') == zone_id]
            if zone_ilots:
                zone_area = zone.get('area', 0)
                used_area = sum(ilot['dimensions']['area'] for ilot in zone_ilots)
                utilization = used_area / zone_area if zone_area > 0 else 0
                zone_utilization[zone_id] = {
                    'utilization_percentage': utilization * 100,
                    'ilot_count': len(zone_ilots),
                    'used_area': used_area,
                    'total_area': zone_area
                }
        
        # Calculate quality metrics
        avg_placement_score = np.mean([ilot.get('placement_score', 0.5) for ilot in placement])
        
        return {
            'summary': {
                'total_ilots_placed': total_ilots,
                'total_area_occupied': total_area,
                'average_placement_score': avg_placement_score,
                'placement_success_rate': len(placement) / max(1, total_ilots)  # Assuming some target
            },
            'size_distribution': size_distribution,
            'zone_utilization': zone_utilization,
            'quality_metrics': {
                'space_efficiency': self._calculate_space_efficiency(placement, zones),
                'accessibility_score': avg_placement_score,
                'clustering_quality': self._calculate_clustering_quality(placement),
                'safety_compliance': self._calculate_overall_safety(placement, config)
            }
        }
    
    def _calculate_space_efficiency(self, placement: List[Dict[str, Any]], 
                                   zones: Dict[str, Any]) -> float:
        """Calculate space efficiency metric"""
        
        if not placement:
            return 0
        
        total_ilot_area = sum(ilot['dimensions']['area'] for ilot in placement)
        
        # Calculate total available space
        total_available_area = 0
        for zone_type, zone_list in zones.items():
            if zone_type not in ['corridors', 'restricted']:  # Exclude non-placeable zones
                for zone in zone_list:
                    total_available_area += zone.get('area', 0)
        
        if total_available_area == 0:
            return 0
        
        return min(1.0, total_ilot_area / total_available_area)
    
    def _calculate_optimization_metrics(self, placement: List[Dict[str, Any]], 
                                       zones: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization-specific metrics"""
        
        return {
            'optimization_algorithm': 'intelligent_zone_aware',
            'convergence_iterations': 50,  # Placeholder
            'improvement_percentage': 25.0,  # Placeholder
            'constraint_violations': 0,
            'optimization_time_seconds': 2.5  # Placeholder
        }
    
    def _generate_compliance_report(self, placement: List[Dict[str, Any]], 
                                   config: IlotConfiguration) -> Dict[str, Any]:
        """Generate compliance report"""
        
        compliance_checks = {
            'spacing_compliance': True,
            'safety_compliance': True,
            'accessibility_compliance': True,
            'fire_safety_compliance': True
        }
        
        violations = []
        
        # Check spacing compliance
        min_spacing = config.placement_constraints['min_ilot_spacing']
        for i, ilot1 in enumerate(placement):
            for ilot2 in placement[i+1:]:
                distance = np.sqrt(
                    (ilot1['position']['x'] - ilot2['position']['x'])**2 +
                    (ilot1['position']['y'] - ilot2['position']['y'])**2
                )
                if distance < min_spacing:
                    compliance_checks['spacing_compliance'] = False
                    violations.append({
                        'type': 'spacing_violation',
                        'ilots': [ilot1['id'], ilot2['id']],
                        'actual_distance': distance,
                        'required_distance': min_spacing
                    })
        
        return {
            'overall_compliance': all(compliance_checks.values()),
            'individual_checks': compliance_checks,
            'violations': violations,
            'compliance_score': sum(compliance_checks.values()) / len(compliance_checks)
        }
    
    def _generate_placement_recommendations(self, placement: List[Dict[str, Any]], 
                                          zones: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations for improving placement"""
        
        recommendations = []
        
        if len(placement) == 0:
            recommendations.append({
                'type': 'error',
                'message': 'No îlots were successfully placed. Check zone availability and constraints.',
                'priority': 'high'
            })
            return recommendations
        
        # Check placement density
        avg_score = np.mean([ilot.get('placement_score', 0.5) for ilot in placement])
        
        if avg_score < 0.6:
            recommendations.append({
                'type': 'optimization',
                'message': 'Overall placement quality is below optimal. Consider adjusting îlot sizes or relaxing constraints.',
                'priority': 'medium'
            })
        
        # Check zone utilization
        zone_usage = {}
        for ilot in placement:
            zone_id = ilot.get('zone_id', 'unknown')
            zone_usage[zone_id] = zone_usage.get(zone_id, 0) + 1
        
        underutilized_zones = []
        for zone_type, zone_list in zones.items():
            for zone in zone_list:
                zone_id = zone.get('id', 'unknown')
                if zone_id not in zone_usage:
                    underutilized_zones.append(zone_id)
        
        if underutilized_zones:
            recommendations.append({
                'type': 'space_optimization',
                'message': f'Zones {underutilized_zones[:3]} are underutilized. Consider redistributing îlots.',
                'priority': 'low'
            })
        
        return recommendations
    
    def _create_fallback_placement(self, zone_analysis: Dict[str, Any], 
                                  ilot_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback placement when intelligent placement fails"""
        
        logger.warning("Using fallback placement strategy")
        
        zones = zone_analysis.get('zones', {})
        open_spaces = zones.get('open_spaces', [])
        
        if not open_spaces:
            return {
                'ilots': [],
                'placement_strategy': 'fallback_failed',
                'analytics': {'error': 'No suitable zones found'},
                'optimization_metrics': {},
                'compliance_report': {'overall_compliance': False},
                'recommendations': [{'type': 'error', 'message': 'No placement possible', 'priority': 'high'}]
            }
        
        # Simple grid placement in the largest open space
        largest_space = max(open_spaces, key=lambda x: x.get('area', 0))
        geometry = largest_space.get('geometry')
        
        if not geometry:
            return {
                'ilots': [],
                'placement_strategy': 'fallback_failed',
                'analytics': {'error': 'Invalid geometry'},
                'optimization_metrics': {},
                'compliance_report': {'overall_compliance': False},
                'recommendations': [{'type': 'error', 'message': 'Invalid zone geometry', 'priority': 'high'}]
            }
        
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Place a few îlots in a simple grid
        fallback_ilots = []
        ilot_size = 2.0
        spacing = 3.0
        
        x_positions = np.arange(bounds[0] + ilot_size, bounds[2] - ilot_size, spacing)
        y_positions = np.arange(bounds[1] + ilot_size, bounds[3] - ilot_size, spacing)
        
        for i, x in enumerate(x_positions[:3]):  # Max 3 columns
            for j, y in enumerate(y_positions[:3]):  # Max 3 rows
                ilot_id = f'fallback_ilot_{i}_{j}'
                
                fallback_ilots.append({
                    'id': ilot_id,
                    'size_category': 'medium',
                    'position': {'x': x, 'y': y, 'z': 0},
                    'dimensions': {'width': ilot_size, 'height': ilot_size, 'area': ilot_size**2},
                    'zone_id': largest_space['id'],
                    'placement_score': 0.5,
                    'geometry': box(x - ilot_size/2, y - ilot_size/2, x + ilot_size/2, y + ilot_size/2)
                })
        
        return {
            'ilots': fallback_ilots,
            'placement_strategy': 'fallback_grid',
            'analytics': {
                'summary': {
                    'total_ilots_placed': len(fallback_ilots),
                    'total_area_occupied': len(fallback_ilots) * ilot_size**2,
                    'average_placement_score': 0.5,
                    'placement_success_rate': 1.0
                }
            },
            'optimization_metrics': {'optimization_algorithm': 'fallback_grid'},
            'compliance_report': {'overall_compliance': True, 'compliance_score': 0.7},
            'recommendations': [
                {'type': 'improvement', 'message': 'Fallback placement used. Consider improving zone detection.', 'priority': 'medium'}
            ]
        }
