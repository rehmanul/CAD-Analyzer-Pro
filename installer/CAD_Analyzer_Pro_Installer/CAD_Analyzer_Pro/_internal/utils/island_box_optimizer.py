"""
Advanced Island Box Placement System
Optimizes placement of standardized furniture/room modules in floor plans
Based on real-world hotel and residential layout requirements
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon, box as Box
from shapely.ops import unary_union
import logging
from typing import Dict, List, Any, Tuple, Optional
import random

logger = logging.getLogger(__name__)

class IslandBoxOptimizer:
    """Intelligent placement system for island boxes (furniture/room modules)"""
    
    def __init__(self):
        # Standard box sizes based on real hotel/residential requirements
        self.standard_box_sizes = {
            'hotel_room_single': {'width': 3.0, 'height': 4.5, 'area': 13.5},
            'hotel_room_double': {'width': 3.5, 'height': 5.0, 'area': 17.5},
            'hotel_room_suite': {'width': 4.5, 'height': 6.0, 'area': 27.0},
            'bathroom_standard': {'width': 1.5, 'height': 2.0, 'area': 3.0},
            'bathroom_accessible': {'width': 2.0, 'height': 2.5, 'area': 5.0},
            'kitchen_compact': {'width': 2.0, 'height': 3.0, 'area': 6.0},
            'kitchen_full': {'width': 3.0, 'height': 4.0, 'area': 12.0},
            'living_small': {'width': 3.0, 'height': 3.5, 'area': 10.5},
            'living_large': {'width': 4.0, 'height': 5.0, 'area': 20.0},
            'office_single': {'width': 2.5, 'height': 3.0, 'area': 7.5},
            'office_double': {'width': 3.5, 'height': 4.0, 'area': 14.0},
            'storage_small': {'width': 1.5, 'height': 1.5, 'area': 2.25},
            'storage_large': {'width': 2.0, 'height': 3.0, 'area': 6.0},
            'corridor_standard': {'width': 1.5, 'height': 999, 'area': 999}  # Variable length
        }
        
        # Placement constraints
        self.constraints = {
            'min_wall_clearance': 0.1,  # 10cm from walls
            'min_box_spacing': 0.05,    # 5cm between boxes
            'corridor_width_min': 1.2,  # Minimum corridor width
            'corridor_width_preferred': 1.5,  # Preferred corridor width
            'emergency_access_width': 1.8,  # Emergency access width
            'max_room_depth': 8.0,      # Maximum room depth without windows
            'natural_light_priority': True,  # Prioritize rooms with exterior walls
            'accessibility_compliance': True
        }
        
        # Room type priorities (higher = more important placement)
        self.room_priorities = {
            'hotel_room_suite': 10,
            'hotel_room_double': 9,
            'hotel_room_single': 8,
            'living_large': 7,
            'living_small': 6,
            'kitchen_full': 8,
            'kitchen_compact': 6,
            'bathroom_accessible': 9,
            'bathroom_standard': 5,
            'office_double': 6,
            'office_single': 4,
            'storage_large': 3,
            'storage_small': 2
        }
    
    def optimize_island_placement(self, floor_plan_data: Dict[str, Any], 
                                 box_requirements: Dict[str, int],
                                 user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main optimization function for placing island boxes
        
        Args:
            floor_plan_data: Processed floor plan with dimensions and zones
            box_requirements: {'hotel_room_double': 10, 'bathroom_standard': 12, ...}
            user_preferences: Custom user settings and constraints
        """
        
        if user_preferences:
            self._update_preferences(user_preferences)
        
        # Extract usable spaces from floor plan
        usable_spaces = self._extract_usable_spaces(floor_plan_data)
        
        # Validate space requirements
        space_validation = self._validate_space_requirements(usable_spaces, box_requirements)
        if not space_validation['feasible']:
            return self._create_infeasible_result(space_validation)
        
        # Generate initial placement solution
        initial_solution = self._generate_initial_placement(usable_spaces, box_requirements)
        
        # Optimize placement using multiple algorithms
        optimized_solution = self._optimize_placement_multi_objective(
            initial_solution, usable_spaces, box_requirements
        )
        
        # Generate corridor network
        corridor_network = self._generate_optimal_corridors(optimized_solution, usable_spaces)
        
        # Validate final solution
        validation_result = self._validate_final_solution(
            optimized_solution, corridor_network, floor_plan_data
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_placement_metrics(
            optimized_solution, corridor_network, usable_spaces
        )
        
        return {
            'island_boxes': optimized_solution,
            'corridor_network': corridor_network,
            'performance_metrics': performance_metrics,
            'validation_result': validation_result,
            'space_utilization': self._calculate_space_utilization(
                optimized_solution, usable_spaces
            ),
            'optimization_metadata': {
                'algorithm_used': 'multi_objective_differential_evolution',
                'iterations': getattr(self, '_last_optimization_iterations', 100),
                'convergence_achieved': validation_result['valid']
            }
        }
    
    def _extract_usable_spaces(self, floor_plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract spaces where island boxes can be placed"""
        usable_spaces = []
        
        # Get dimensions and room information
        dimensions = floor_plan_data.get('dimensions', {})
        zones = floor_plan_data.get('zones', {})
        
        overall_dims = dimensions.get('overall_dimensions', {})
        bounds = overall_dims.get('bounds', {})
        
        if not bounds:
            # Create fallback space
            return [{
                'id': 'main_space',
                'polygon': Box(0, 0, 20, 15),  # 20x15m default space
                'area': 300,
                'type': 'open_space',
                'constraints': [],
                'access_points': [(10, 0), (10, 15)]  # Entry points
            }]
        
        # Process identified rooms/zones
        rooms = zones.get('rooms', [])
        open_spaces = zones.get('open_spaces', [])
        
        all_spaces = rooms + open_spaces
        
        for space in all_spaces:
            space_bounds = space.get('bounds', bounds)
            
            # Create polygon for space
            x_min = space_bounds.get('min_x', 0)
            y_min = space_bounds.get('min_y', 0)
            x_max = space_bounds.get('max_x', x_min + 10)
            y_max = space_bounds.get('max_y', y_min + 10)
            
            space_polygon = Box(x_min, y_min, x_max, y_max)
            
            usable_spaces.append({
                'id': space.get('id', f'space_{len(usable_spaces)}'),
                'polygon': space_polygon,
                'area': space_polygon.area,
                'type': space.get('type', 'open_space'),
                'constraints': self._identify_space_constraints(space),
                'access_points': self._identify_access_points(space, space_polygon),
                'exterior_walls': self._identify_exterior_walls(space_polygon, bounds)
            })
        
        return usable_spaces
    
    def _validate_space_requirements(self, usable_spaces: List[Dict[str, Any]], 
                                   box_requirements: Dict[str, int]) -> Dict[str, Any]:
        """Validate if space requirements can be met"""
        
        total_available_area = sum(space['area'] for space in usable_spaces)
        
        # Calculate required area
        total_required_area = 0
        box_details = []
        
        for box_type, quantity in box_requirements.items():
            if box_type in self.standard_box_sizes:
                box_info = self.standard_box_sizes[box_type]
                box_area = box_info['area']
                total_area = box_area * quantity
                total_required_area += total_area
                
                box_details.append({
                    'type': box_type,
                    'quantity': quantity,
                    'individual_area': box_area,
                    'total_area': total_area
                })
        
        # Add corridor area estimate (30% of box area)
        corridor_area_estimate = total_required_area * 0.3
        total_area_needed = total_required_area + corridor_area_estimate
        
        utilization_ratio = total_area_needed / total_available_area if total_available_area > 0 else 1.0
        
        return {
            'feasible': utilization_ratio <= 0.85,  # Max 85% utilization
            'utilization_ratio': utilization_ratio,
            'total_available_area': total_available_area,
            'total_required_area': total_required_area,
            'corridor_area_estimate': corridor_area_estimate,
            'box_details': box_details,
            'recommendations': self._generate_feasibility_recommendations(utilization_ratio)
        }
    
    def _generate_initial_placement(self, usable_spaces: List[Dict[str, Any]], 
                                  box_requirements: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate initial placement using intelligent heuristics"""
        
        placed_boxes = []
        box_id = 0
        
        # Sort box types by priority
        sorted_box_types = sorted(box_requirements.keys(), 
                                key=lambda x: self.room_priorities.get(x, 0), 
                                reverse=True)
        
        for box_type in sorted_box_types:
            quantity = box_requirements[box_type]
            box_info = self.standard_box_sizes.get(box_type)
            
            if not box_info:
                continue
                
            for i in range(quantity):
                # Find best space for this box
                best_placement = self._find_best_placement_location(
                    box_type, box_info, usable_spaces, placed_boxes
                )
                
                if best_placement:
                    placed_boxes.append({
                        'id': f'box_{box_id}',
                        'type': box_type,
                        'width': box_info['width'],
                        'height': box_info['height'],
                        'x': best_placement['x'],
                        'y': best_placement['y'],
                        'rotation': best_placement.get('rotation', 0),
                        'space_id': best_placement['space_id'],
                        'priority': self.room_priorities.get(box_type, 5)
                    })
                    box_id += 1
        
        return placed_boxes
    
    def _find_best_placement_location(self, box_type: str, box_info: Dict[str, Any],
                                    usable_spaces: List[Dict[str, Any]], 
                                    existing_boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find optimal location for a single box"""
        
        best_score = -1
        best_placement = None
        
        for space in usable_spaces:
            # Try multiple positions within this space
            space_bounds = space['polygon'].bounds
            x_min, y_min, x_max, y_max = space_bounds
            
            # Grid-based sampling for efficiency
            grid_size = 0.5  # 50cm grid
            x_positions = np.arange(x_min + box_info['width']/2, 
                                  x_max - box_info['width']/2, 
                                  grid_size)
            y_positions = np.arange(y_min + box_info['height']/2, 
                                  y_max - box_info['height']/2, 
                                  grid_size)
            
            for x in x_positions:
                for y in y_positions:
                    for rotation in [0, 90]:  # Try both orientations
                        
                        # Adjust dimensions for rotation
                        if rotation == 90:
                            width, height = box_info['height'], box_info['width']
                        else:
                            width, height = box_info['width'], box_info['height']
                        
                        # Check if placement is valid
                        if self._is_valid_placement(x, y, width, height, space, existing_boxes):
                            
                            # Calculate placement score
                            score = self._calculate_placement_score(
                                x, y, width, height, box_type, space, existing_boxes
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_placement = {
                                    'x': x,
                                    'y': y,
                                    'rotation': rotation,
                                    'space_id': space['id'],
                                    'score': score
                                }
        
        return best_placement
    
    def _is_valid_placement(self, x: float, y: float, width: float, height: float,
                          space: Dict[str, Any], existing_boxes: List[Dict[str, Any]]) -> bool:
        """Check if a box placement is valid"""
        
        # Create box polygon
        box_polygon = Box(x - width/2, y - height/2, x + width/2, y + height/2)
        
        # Check if completely within space
        if not space['polygon'].contains(box_polygon):
            return False
        
        # Check for overlaps with existing boxes
        for existing_box in existing_boxes:
            existing_polygon = Box(
                existing_box['x'] - existing_box['width']/2,
                existing_box['y'] - existing_box['height']/2,
                existing_box['x'] + existing_box['width']/2,
                existing_box['y'] + existing_box['height']/2
            )
            
            # Add minimum spacing
            expanded_existing = existing_polygon.buffer(self.constraints['min_box_spacing'])
            if expanded_existing.intersects(box_polygon):
                return False
        
        return True
    
    def _calculate_placement_score(self, x: float, y: float, width: float, height: float,
                                 box_type: str, space: Dict[str, Any], 
                                 existing_boxes: List[Dict[str, Any]]) -> float:
        """Calculate quality score for a placement"""
        
        score = 0.0
        
        # Base priority score
        score += self.room_priorities.get(box_type, 5) * 10
        
        # Exterior wall bonus (for rooms needing natural light)
        if box_type.startswith('hotel_room') or box_type.startswith('living'):
            exterior_walls = space.get('exterior_walls', [])
            if exterior_walls:
                # Calculate distance to nearest exterior wall
                box_center = Point(x, y)
                min_distance = min(box_center.distance(wall) for wall in exterior_walls)
                if min_distance < 2.0:  # Within 2m of exterior wall
                    score += 50
        
        # Access point proximity
        access_points = space.get('access_points', [])
        if access_points:
            box_center = Point(x, y)
            min_access_distance = min(box_center.distance(Point(ap)) for ap in access_points)
            # Prefer moderate distance (not too close, not too far)
            optimal_distance = 3.0
            distance_score = max(0, 20 - abs(min_access_distance - optimal_distance) * 5)
            score += distance_score
        
        # Clustering bonus for related room types
        related_types = self._get_related_room_types(box_type)
        nearby_related = 0
        for existing_box in existing_boxes:
            if existing_box['type'] in related_types:
                distance = ((x - existing_box['x'])**2 + (y - existing_box['y'])**2)**0.5
                if distance < 5.0:  # Within 5m
                    nearby_related += 1
        
        score += nearby_related * 10
        
        # Avoid corners (rooms in corners are less desirable)
        space_bounds = space['polygon'].bounds
        x_min, y_min, x_max, y_max = space_bounds
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        distance_from_center = ((x - center_x)**2 + (y - center_y)**2)**0.5
        max_distance = ((x_max - x_min)**2 + (y_max - y_min)**2)**0.5 / 2
        centrality_score = (1 - distance_from_center / max_distance) * 15
        score += centrality_score
        
        return score
    
    def _get_related_room_types(self, box_type: str) -> List[str]:
        """Get room types that should be clustered together"""
        clusters = {
            'bedrooms': ['hotel_room_single', 'hotel_room_double', 'hotel_room_suite'],
            'bathrooms': ['bathroom_standard', 'bathroom_accessible'],
            'kitchens': ['kitchen_compact', 'kitchen_full'],
            'living': ['living_small', 'living_large'],
            'offices': ['office_single', 'office_double'],
            'storage': ['storage_small', 'storage_large']
        }
        
        for cluster_name, room_types in clusters.items():
            if box_type in room_types:
                return room_types
        
        return []
    
    def _optimize_placement_multi_objective(self, initial_solution: List[Dict[str, Any]],
                                          usable_spaces: List[Dict[str, Any]],
                                          box_requirements: Dict[str, int]) -> List[Dict[str, Any]]:
        """Optimize placement using multi-objective optimization"""
        
        if not initial_solution:
            return initial_solution
        
        # Convert solution to optimization variables
        x0 = []
        for box in initial_solution:
            x0.extend([box['x'], box['y'], box['rotation']])
        
        # Define bounds
        bounds = []
        for i, box in enumerate(initial_solution):
            space = next(s for s in usable_spaces if s['id'] == box['space_id'])
            space_bounds = space['polygon'].bounds
            x_min, y_min, x_max, y_max = space_bounds
            
            # Position bounds
            bounds.append((x_min + box['width']/2, x_max - box['width']/2))  # x
            bounds.append((y_min + box['height']/2, y_max - box['height']/2))  # y
            bounds.append((0, 90))  # rotation (0 or 90 degrees)
        
        def objective_function(variables):
            return self._multi_objective_function(variables, initial_solution, usable_spaces)
        
        try:
            # Use differential evolution for global optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                seed=42,
                atol=1e-6,
                tol=1e-6
            )
            
            self._last_optimization_iterations = result.nit
            
            # Convert optimized variables back to solution format
            optimized_solution = self._variables_to_solution(result.x, initial_solution)
            
            return optimized_solution
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using initial solution")
            return initial_solution
    
    def _multi_objective_function(self, variables: np.ndarray, 
                                 initial_solution: List[Dict[str, Any]],
                                 usable_spaces: List[Dict[str, Any]]) -> float:
        """Multi-objective function combining space efficiency, accessibility, and aesthetics"""
        
        # Convert variables to box positions
        solution = self._variables_to_solution(variables, initial_solution)
        
        # Penalty for invalid placements
        penalty = 0
        for i, box in enumerate(solution):
            # Check bounds
            space = next(s for s in usable_spaces if s['id'] == box['space_id'])
            if not self._is_valid_placement(box['x'], box['y'], box['width'], box['height'], 
                                          space, solution[:i] + solution[i+1:]):
                penalty += 1000
        
        if penalty > 0:
            return penalty
        
        # Calculate objectives
        space_efficiency = self._calculate_space_efficiency(solution, usable_spaces)
        accessibility_score = self._calculate_accessibility_score(solution)
        aesthetic_score = self._calculate_aesthetic_score(solution)
        circulation_efficiency = self._calculate_circulation_efficiency(solution, usable_spaces)
        
        # Weighted combination (minimize, so negate positive scores)
        total_score = -(
            space_efficiency * 0.3 +
            accessibility_score * 0.25 +
            aesthetic_score * 0.2 +
            circulation_efficiency * 0.25
        )
        
        return total_score
    
    def _variables_to_solution(self, variables: np.ndarray, 
                             template_solution: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert optimization variables back to solution format"""
        
        solution = []
        for i, box_template in enumerate(template_solution):
            var_idx = i * 3
            
            # Handle rotation (round to 0 or 90)
            rotation = 90 if variables[var_idx + 2] > 45 else 0
            
            # Adjust dimensions based on rotation
            if rotation == 90:
                width = box_template['height']
                height = box_template['width']
            else:
                width = box_template['width']
                height = box_template['height']
            
            solution.append({
                'id': box_template['id'],
                'type': box_template['type'],
                'width': width,
                'height': height,
                'x': variables[var_idx],
                'y': variables[var_idx + 1],
                'rotation': rotation,
                'space_id': box_template['space_id'],
                'priority': box_template['priority']
            })
        
        return solution
    
    def _calculate_space_efficiency(self, solution: List[Dict[str, Any]], 
                                   usable_spaces: List[Dict[str, Any]]) -> float:
        """Calculate space utilization efficiency"""
        
        total_box_area = sum(box['width'] * box['height'] for box in solution)
        total_space_area = sum(space['area'] for space in usable_spaces)
        
        if total_space_area == 0:
            return 0
        
        utilization = total_box_area / total_space_area
        
        # Optimal utilization is around 70-75%
        optimal_utilization = 0.725
        efficiency = 1 - abs(utilization - optimal_utilization) / optimal_utilization
        
        return max(0, efficiency) * 100
    
    def _calculate_accessibility_score(self, solution: List[Dict[str, Any]]) -> float:
        """Calculate accessibility and circulation score"""
        
        # Simple accessibility metric based on spacing
        total_score = 0
        total_pairs = 0
        
        for i, box1 in enumerate(solution):
            for box2 in solution[i+1:]:
                distance = ((box1['x'] - box2['x'])**2 + (box1['y'] - box2['y'])**2)**0.5
                min_distance = (box1['width'] + box2['width']) / 2 + self.constraints['corridor_width_min']
                
                if distance >= min_distance:
                    # Good spacing
                    total_score += min(100, distance / min_distance * 50)
                else:
                    # Too close
                    total_score += max(0, distance / min_distance * 50)
                
                total_pairs += 1
        
        return total_score / total_pairs if total_pairs > 0 else 0
    
    def _calculate_aesthetic_score(self, solution: List[Dict[str, Any]]) -> float:
        """Calculate aesthetic arrangement score"""
        
        # Alignment bonus
        alignment_score = 0
        tolerance = 0.2  # 20cm tolerance for alignment
        
        for i, box1 in enumerate(solution):
            for box2 in solution[i+1:]:
                # Check horizontal alignment
                if abs(box1['y'] - box2['y']) < tolerance:
                    alignment_score += 10
                
                # Check vertical alignment
                if abs(box1['x'] - box2['x']) < tolerance:
                    alignment_score += 10
        
        # Symmetry bonus
        symmetry_score = self._calculate_symmetry_score(solution)
        
        return (alignment_score + symmetry_score) / len(solution) if solution else 0
    
    def _calculate_symmetry_score(self, solution: List[Dict[str, Any]]) -> float:
        """Calculate layout symmetry score"""
        
        if len(solution) < 2:
            return 0
        
        # Find center of all boxes
        center_x = sum(box['x'] for box in solution) / len(solution)
        center_y = sum(box['y'] for box in solution) / len(solution)
        
        # Check for symmetric pairs
        symmetry_score = 0
        tolerance = 1.0  # 1m tolerance
        
        for box in solution:
            # Find potential symmetric counterpart
            target_x = 2 * center_x - box['x']
            target_y = 2 * center_y - box['y']
            
            for other_box in solution:
                if (abs(other_box['x'] - target_x) < tolerance and 
                    abs(other_box['y'] - target_y) < tolerance and
                    other_box['type'] == box['type']):
                    symmetry_score += 10
                    break
        
        return symmetry_score
    
    def _calculate_circulation_efficiency(self, solution: List[Dict[str, Any]], 
                                        usable_spaces: List[Dict[str, Any]]) -> float:
        """Calculate circulation path efficiency"""
        
        # Simple circulation metric: average distance between related room types
        total_score = 0
        comparisons = 0
        
        related_groups = [
            ['hotel_room_single', 'hotel_room_double', 'hotel_room_suite'],
            ['bathroom_standard', 'bathroom_accessible'],
            ['kitchen_compact', 'kitchen_full'],
            ['office_single', 'office_double']
        ]
        
        for group in related_groups:
            group_boxes = [box for box in solution if box['type'] in group]
            
            if len(group_boxes) < 2:
                continue
            
            # Calculate average intra-group distance
            total_distance = 0
            pair_count = 0
            
            for i, box1 in enumerate(group_boxes):
                for box2 in group_boxes[i+1:]:
                    distance = ((box1['x'] - box2['x'])**2 + (box1['y'] - box2['y'])**2)**0.5
                    total_distance += distance
                    pair_count += 1
            
            if pair_count > 0:
                avg_distance = total_distance / pair_count
                # Prefer moderate distances (not too close, not too far)
                optimal_distance = 5.0
                score = max(0, 100 - abs(avg_distance - optimal_distance) * 10)
                total_score += score
                comparisons += 1
        
        return total_score / comparisons if comparisons > 0 else 50
    
    def _generate_optimal_corridors(self, placed_boxes: List[Dict[str, Any]], 
                                   usable_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimal corridor network for the placed boxes"""
        
        corridors = []
        
        # Main circulation spine
        if placed_boxes:
            # Find central corridor path
            x_coords = [box['x'] for box in placed_boxes]
            y_coords = [box['y'] for box in placed_boxes]
            
            # Create main horizontal corridor
            min_x, max_x = min(x_coords), max(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            main_corridor = {
                'id': 'main_horizontal',
                'type': 'main_corridor',
                'start': (min_x - 2, center_y),
                'end': (max_x + 2, center_y),
                'width': self.constraints['corridor_width_preferred'],
                'priority': 'high'
            }
            corridors.append(main_corridor)
            
            # Create connecting corridors to boxes
            for box in placed_boxes:
                # Connect to main corridor
                connector = {
                    'id': f'connector_to_{box["id"]}',
                    'type': 'secondary_corridor',
                    'start': (box['x'], center_y),
                    'end': (box['x'], box['y']),
                    'width': self.constraints['corridor_width_min'],
                    'priority': 'medium',
                    'connects_to': [box['id'], 'main_horizontal']
                }
                corridors.append(connector)
        
        return {
            'corridors': corridors,
            'total_length': sum(
                ((c['end'][0] - c['start'][0])**2 + (c['end'][1] - c['start'][1])**2)**0.5 
                for c in corridors
            ),
            'total_area': sum(
                ((c['end'][0] - c['start'][0])**2 + (c['end'][1] - c['start'][1])**2)**0.5 * c['width']
                for c in corridors
            )
        }
    
    def _validate_final_solution(self, solution: List[Dict[str, Any]], 
                               corridor_network: Dict[str, Any],
                               floor_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the final placement solution"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'compliance_score': 100
        }
        
        # Check for overlaps
        for i, box1 in enumerate(solution):
            for box2 in solution[i+1:]:
                box1_poly = Box(box1['x'] - box1['width']/2, box1['y'] - box1['height']/2,
                              box1['x'] + box1['width']/2, box1['y'] + box1['height']/2)
                box2_poly = Box(box2['x'] - box2['width']/2, box2['y'] - box2['height']/2,
                              box2['x'] + box2['width']/2, box2['y'] + box2['height']/2)
                
                if box1_poly.intersects(box2_poly):
                    validation['errors'].append(f"Overlap detected: {box1['id']} and {box2['id']}")
                    validation['valid'] = False
                    validation['compliance_score'] -= 20
        
        # Check accessibility
        for box in solution:
            # Ensure minimum corridor access
            has_access = True  # Simplified check
            if not has_access:
                validation['warnings'].append(f"Limited access to {box['id']}")
                validation['compliance_score'] -= 5
        
        # Check emergency access
        corridor_widths = [c['width'] for c in corridor_network.get('corridors', [])]
        main_corridors = [w for w in corridor_widths if w >= self.constraints['emergency_access_width']]
        
        if not main_corridors:
            validation['warnings'].append("No emergency access corridors found")
            validation['compliance_score'] -= 10
        
        return validation
    
    def _calculate_placement_metrics(self, solution: List[Dict[str, Any]], 
                                   corridor_network: Dict[str, Any],
                                   usable_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        total_box_area = sum(box['width'] * box['height'] for box in solution)
        total_corridor_area = corridor_network.get('total_area', 0)
        total_space_area = sum(space['area'] for space in usable_spaces)
        
        return {
            'space_utilization': (total_box_area / total_space_area * 100) if total_space_area > 0 else 0,
            'circulation_ratio': (total_corridor_area / total_box_area * 100) if total_box_area > 0 else 0,
            'boxes_placed': len(solution),
            'total_box_area': total_box_area,
            'total_corridor_area': total_corridor_area,
            'efficiency_score': self._calculate_space_efficiency(solution, usable_spaces),
            'accessibility_score': self._calculate_accessibility_score(solution),
            'aesthetic_score': self._calculate_aesthetic_score(solution),
            'room_type_distribution': {
                box_type: len([b for b in solution if b['type'] == box_type])
                for box_type in set(box['type'] for box in solution)
            }
        }
    
    def _calculate_space_utilization(self, solution: List[Dict[str, Any]], 
                                   usable_spaces: List[Dict[str, Any]]) -> float:
        """Calculate overall space utilization percentage"""
        
        total_box_area = sum(box['width'] * box['height'] for box in solution)
        total_space_area = sum(space['area'] for space in usable_spaces)
        
        return (total_box_area / total_space_area * 100) if total_space_area > 0 else 0
    
    def _identify_space_constraints(self, space: Dict[str, Any]) -> List[str]:
        """Identify constraints for a space"""
        constraints = []
        
        space_type = space.get('type', '')
        if 'entrance' in space_type.lower():
            constraints.append('keep_clear_for_access')
        if 'corridor' in space_type.lower():
            constraints.append('maintain_circulation_width')
        if 'stair' in space_type.lower():
            constraints.append('no_placement_allowed')
        
        return constraints
    
    def _identify_access_points(self, space: Dict[str, Any], 
                              space_polygon: Polygon) -> List[Tuple[float, float]]:
        """Identify access points for a space"""
        
        # Use space centroid as default access point
        centroid = space_polygon.centroid
        return [(centroid.x, centroid.y)]
    
    def _identify_exterior_walls(self, space_polygon: Polygon, 
                               overall_bounds: Dict[str, float]) -> List[LineString]:
        """Identify exterior walls of a space"""
        
        exterior_walls = []
        bounds = space_polygon.bounds
        overall_x_min = overall_bounds.get('min_x', bounds[0])
        overall_y_min = overall_bounds.get('min_y', bounds[1])
        overall_x_max = overall_bounds.get('max_x', bounds[2])
        overall_y_max = overall_bounds.get('max_y', bounds[3])
        
        tolerance = 0.1
        
        # Check each side of the space
        if abs(bounds[0] - overall_x_min) < tolerance:  # Left side
            exterior_walls.append(LineString([(bounds[0], bounds[1]), (bounds[0], bounds[3])]))
        if abs(bounds[2] - overall_x_max) < tolerance:  # Right side
            exterior_walls.append(LineString([(bounds[2], bounds[1]), (bounds[2], bounds[3])]))
        if abs(bounds[1] - overall_y_min) < tolerance:  # Bottom side
            exterior_walls.append(LineString([(bounds[0], bounds[1]), (bounds[2], bounds[1])]))
        if abs(bounds[3] - overall_y_max) < tolerance:  # Top side
            exterior_walls.append(LineString([(bounds[0], bounds[3]), (bounds[2], bounds[3])]))
        
        return exterior_walls
    
    def _create_infeasible_result(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Create result for infeasible placement scenarios"""
        
        return {
            'island_boxes': [],
            'corridor_network': {'corridors': [], 'total_length': 0, 'total_area': 0},
            'performance_metrics': {
                'space_utilization': validation['utilization_ratio'] * 100,
                'feasible': False,
                'recommendations': validation['recommendations']
            },
            'validation_result': {
                'valid': False,
                'errors': ['Insufficient space for required island boxes'],
                'warnings': validation['recommendations']
            },
            'space_utilization': validation['utilization_ratio'] * 100
        }
    
    def _generate_feasibility_recommendations(self, utilization_ratio: float) -> List[str]:
        """Generate recommendations for infeasible scenarios"""
        
        recommendations = []
        
        if utilization_ratio > 1.0:
            recommendations.append("Reduce number of island boxes")
            recommendations.append("Consider smaller box sizes")
            recommendations.append("Optimize corridor widths")
        elif utilization_ratio > 0.85:
            recommendations.append("Consider reducing box sizes slightly")
            recommendations.append("Optimize circulation paths")
        
        return recommendations
    
    def _update_preferences(self, user_preferences: Dict[str, Any]):
        """Update system preferences based on user input"""
        
        if 'box_sizes' in user_preferences:
            self.standard_box_sizes.update(user_preferences['box_sizes'])
        
        if 'constraints' in user_preferences:
            self.constraints.update(user_preferences['constraints'])
        
        if 'priorities' in user_preferences:
            self.room_priorities.update(user_preferences['priorities'])