import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import random
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class IlotPlacer:
    """Intelligent îlot placement system"""

    def __init__(self):
        self.placement_algorithms = {
            'grid': self._grid_placement,
            'random': self._random_placement,
            'optimized': self._optimized_placement,
            'cluster': self._cluster_placement
        }

        self.default_spacing = 1.5
        self.wall_clearance = 1.0
        self.accessibility_clearance = 0.8

    def place_ilots(self, analysis_results: Dict[str, Any], 
                   configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Place îlots based on analysis results and configuration"""
        logger.info("Starting îlot placement")

        # Extract configuration
        size_distribution = configuration.get('size_distribution', {'small': 30, 'medium': 50, 'large': 20})
        dimensions = configuration.get('dimensions', {'small': 2.0, 'medium': 4.0, 'large': 8.0})
        spacing = configuration.get('spacing', {})

        self.default_spacing = spacing.get('min_spacing', 1.5)
        self.wall_clearance = spacing.get('wall_clearance', 1.0)

        # Get available spaces
        open_spaces = analysis_results.get('open_spaces', [])
        walls = analysis_results.get('walls', [])
        restricted_areas = analysis_results.get('restricted_areas', [])

        if not open_spaces:
            logger.warning("No open spaces found for îlot placement")
            return []

        # Generate îlot requirements
        ilot_requirements = self._generate_ilot_requirements(size_distribution, dimensions)

        # Create placement constraints
        constraints = self._create_placement_constraints(walls, restricted_areas, open_spaces)

        # Place îlots using optimized algorithm
        placed_ilots = self._optimized_placement(ilot_requirements, constraints, open_spaces)

        # Post-process and validate placements
        validated_ilots = self._validate_placements(placed_ilots, constraints)

        logger.info(f"Successfully placed {len(validated_ilots)} îlots")
        return validated_ilots

    def _generate_ilot_requirements(self, size_distribution: Dict[str, int], 
                                  dimensions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate îlot requirements based on size distribution"""
        requirements = []
        total_ilots = 20  # Default number, could be configurable

        for size_category, percentage in size_distribution.items():
            count = int(total_ilots * percentage / 100)
            size_dim = dimensions.get(size_category, 2.0)

            for i in range(count):
                requirements.append({
                    'id': f"{size_category}_{i}",
                    'size_category': size_category,
                    'dimensions': {
                        'width': size_dim,
                        'height': size_dim,
                        'area': size_dim ** 2
                    },
                    'required_clearance': self.default_spacing,
                    'priority': self._get_size_priority(size_category)
                })

        return requirements

    def _get_size_priority(self, size_category: str) -> int:
        """Get placement priority for size category"""
        priority_map = {'large': 1, 'medium': 2, 'small': 3}
        return priority_map.get(size_category, 2)

    def _create_placement_constraints(self, walls: List[Dict[str, Any]], 
                                    restricted_areas: List[Dict[str, Any]],
                                    open_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create placement constraints"""
        # Create obstacle geometries
        obstacles = []

        # Add walls with clearance buffer
        for wall in walls:
            wall_geom = wall.get('geometry')
            if wall_geom:
                buffered = wall_geom.buffer(self.wall_clearance)
                obstacles.append(buffered)

        # Add restricted areas
        for area in restricted_areas:
            area_geom = area.get('geometry')
            if area_geom:
                obstacles.append(area_geom)

        # Create combined obstacle geometry
        if obstacles:
            combined_obstacles = unary_union(obstacles)
        else:
            combined_obstacles = None

        # Create available space geometry
        available_spaces = []
        for space in open_spaces:
            space_geom = space.get('geometry')
            if space_geom:
                # Subtract obstacles from available space
                if combined_obstacles:
                    try:
                        clean_space = space_geom.difference(combined_obstacles)
                        if not clean_space.is_empty:
                            available_spaces.append(clean_space)
                    except:
                        available_spaces.append(space_geom)
                else:
                    available_spaces.append(space_geom)

        return {
            'obstacles': combined_obstacles,
            'available_spaces': available_spaces,
            'walls': walls,
            'restricted_areas': restricted_areas,
            'min_spacing': self.default_spacing,
            'wall_clearance': self.wall_clearance
        }

    def _grid_placement(self, requirements: List[Dict[str, Any]], 
                       constraints: Dict[str, Any],
                       open_spaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Grid-based placement algorithm"""
        placed_ilots = []
        available_spaces = constraints['available_spaces']

        if not available_spaces:
            return []

        # Use the largest available space for grid placement
        largest_space = max(available_spaces, key=lambda x: x.area)
        bounds = largest_space.bounds

        # Calculate grid parameters
        grid_spacing = 3.0  # Base grid spacing
        x_min, y_min, x_max, y_max = bounds

        # Generate grid points
        x_points = np.arange(x_min + 1, x_max - 1, grid_spacing)
        y_points = np.arange(y_min + 1, y_max - 1, grid_spacing)

        # Sort requirements by priority (large first)
        sorted_requirements = sorted(requirements, key=lambda x: x['priority'])

        placed_count = 0
        for req in sorted_requirements:
            if placed_count >= len(x_points) * len(y_points):
                break

            # Find next grid position
            grid_idx = placed_count
            x_idx = grid_idx % len(x_points)
            y_idx = grid_idx // len(x_points)

            if y_idx >= len(y_points):
                break

            x = x_points[x_idx]
            y = y_points[y_idx]
            position = Point(x, y)

            # Check if position is valid
            if self._is_valid_position(position, req, constraints, placed_ilots):
                ilot = self._create_ilot(req, position)
                placed_ilots.append(ilot)
                placed_count += 1

        return placed_ilots

    def _random_placement(self, requirements: List[Dict[str, Any]], 
                         constraints: Dict[str, Any],
                         open_spaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Random placement with collision avoidance"""
        placed_ilots = []
        available_spaces = constraints['available_spaces']

        if not available_spaces:
            return []

        max_attempts = 1000

        for req in requirements:
            attempts = 0
            placed = False

            while attempts < max_attempts and not placed:
                # Choose random space
                space = random.choice(available_spaces)

                # Generate random position within space
                bounds = space.bounds
                x = random.uniform(bounds[0] + 1, bounds[2] - 1)
                y = random.uniform(bounds[1] + 1, bounds[3] - 1)
                position = Point(x, y)

                # Check if position is valid
                if self._is_valid_position(position, req, constraints, placed_ilots):
                    ilot = self._create_ilot(req, position)
                    placed_ilots.append(ilot)
                    placed = True

                attempts += 1

            if not placed:
                logger.warning(f"Could not place îlot {req['id']} after {max_attempts} attempts")

        return placed_ilots

    def _optimized_placement(self, requirements: List[Dict[str, Any]], 
                           constraints: Dict[str, Any],
                           open_spaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized placement using space efficiency and accessibility"""
        placed_ilots = []
        available_spaces = constraints['available_spaces']

        if not available_spaces:
            return []

        # Sort requirements by priority and size
        sorted_requirements = sorted(requirements, 
                                   key=lambda x: (x['priority'], -x['dimensions']['area']))

        for req in sorted_requirements:
            best_position = None
            best_score = -1

            # Evaluate potential positions
            for space in available_spaces:
                candidate_positions = self._generate_candidate_positions(space, req)

                for position in candidate_positions:
                    if self._is_valid_position(position, req, constraints, placed_ilots):
                        score = self._calculate_position_score(position, req, constraints, placed_ilots)

                        if score > best_score:
                            best_score = score
                            best_position = position

            if best_position:
                ilot = self._create_ilot(req, best_position)
                ilot['placement_score'] = best_score
                placed_ilots.append(ilot)
            else:
                logger.warning(f"Could not find valid position for îlot {req['id']}")

        return placed_ilots

    def _cluster_placement(self, requirements: List[Dict[str, Any]], 
                          constraints: Dict[str, Any],
                          open_spaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster-based placement for efficient grouping"""
        placed_ilots = []
        available_spaces = constraints['available_spaces']

        if not available_spaces:
            return []

        # Determine number of clusters based on space size and îlot count
        num_clusters = min(3, len(requirements) // 5 + 1)

        # Generate cluster centers
        cluster_centers = []
        for space in available_spaces[:num_clusters]:
            center = space.centroid
            cluster_centers.append((center.x, center.y))

        if not cluster_centers:
            return self._random_placement(requirements, constraints, open_spaces)

        # Assign îlots to clusters
        cluster_assignments = self._assign_ilots_to_clusters(requirements, cluster_centers)

        # Place îlots within each cluster
        for cluster_id, cluster_ilots in cluster_assignments.items():
            if cluster_id < len(cluster_centers):
                center_x, center_y = cluster_centers[cluster_id]
                cluster_placed = self._place_cluster(cluster_ilots, center_x, center_y, 
                                                   constraints, placed_ilots)
                placed_ilots.extend(cluster_placed)

        return placed_ilots

    def _generate_candidate_positions(self, space: Polygon, 
                                    requirement: Dict[str, Any]) -> List[Point]:
        """Generate candidate positions within a space"""
        bounds = space.bounds
        x_min, y_min, x_max, y_max = bounds

        # Account for îlot dimensions
        width = requirement['dimensions']['width']
        height = requirement['dimensions']['height']

        # Adjust bounds to ensure îlot fits within space
        x_min += width / 2 + 0.5
        y_min += height / 2 + 0.5
        x_max -= width / 2 + 0.5
        y_max -= height / 2 + 0.5

        if x_min >= x_max or y_min >= y_max:
            return []

        # Generate candidate grid
        x_step = max(1.0, (x_max - x_min) / 10)
        y_step = max(1.0, (y_max - y_min) / 10)

        candidates = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                point = Point(x, y)
                if space.contains(point):
                    candidates.append(point)
                y += y_step
            x += x_step

        return candidates

    def _is_valid_position(self, position: Point, requirement: Dict[str, Any], 
                          constraints: Dict[str, Any], 
                          existing_ilots: List[Dict[str, Any]]) -> bool:
        """Check if position is valid for îlot placement"""
        width = requirement['dimensions']['width']
        height = requirement['dimensions']['height']
        clearance = requirement['required_clearance']

        # Create îlot footprint
        ilot_box = box(position.x - width/2, position.y - height/2,
                       position.x + width/2, position.y + height/2)

        # Check collision with obstacles
        obstacles = constraints.get('obstacles')
        if obstacles and ilot_box.intersects(obstacles):
            return False

        # Check if within available spaces
        available_spaces = constraints.get('available_spaces', [])
        within_available = False
        for space in available_spaces:
            if space.contains(ilot_box):
                within_available = True
                break

        if not within_available:
            return False

        # Check collision with existing îlots
        for existing in existing_ilots:
            existing_pos = existing['position']
            existing_dims = existing['dimensions']

            existing_box = box(existing_pos['x'] - existing_dims['width']/2,
                             existing_pos['y'] - existing_dims['height']/2,
                             existing_pos['x'] + existing_dims['width']/2,
                             existing_pos['y'] + existing_dims['height']/2)

            # Check if too close (including clearance)
            distance_between = position.distance(Point(existing_pos['x'], existing_pos['y']))
            min_distance = (width + existing_dims['width'])/2 + clearance

            if distance_between < min_distance:
                return False

        return True

    def _calculate_position_score(self, position: Point, requirement: Dict[str, Any], 
                                constraints: Dict[str, Any], 
                                existing_ilots: List[Dict[str, Any]]) -> float:
        """Calculate placement score for position"""
        score = 0.0

        # Distance from walls (prefer some distance but not too far)
        walls = constraints.get('walls', [])
        if walls:
            min_wall_distance = float('inf')
            for wall in walls:
                wall_geom = wall.get('geometry')
                if wall_geom:
                    dist = position.distance(wall_geom)
                    min_wall_distance = min(min_wall_distance, dist)

            # Optimal distance is 2-3 meters from walls
            if min_wall_distance != float('inf'):
                if 2.0 <= min_wall_distance <= 3.0:
                    score += 0.3
                elif min_wall_distance > 1.0:
                    score += 0.1

        # Clustering bonus (prefer to be near other îlots but not too close)
        if existing_ilots:
            distances = []
            for existing in existing_ilots:
                existing_pos = Point(existing['position']['x'], existing['position']['y'])
                dist = position.distance(existing_pos)
                distances.append(dist)

            avg_distance = sum(distances) / len(distances)
            if 3.0 <= avg_distance <= 6.0:  # Optimal clustering distance
                score += 0.3
            elif avg_distance <= 10.0:
                score += 0.1

        # Center preference (prefer positions closer to space center)
        available_spaces = constraints.get('available_spaces', [])
        for space in available_spaces:
            if space.contains(position):
                center = space.centroid
                distance_from_center = position.distance(center)
                max_distance = max(space.bounds[2] - space.bounds[0], 
                                 space.bounds[3] - space.bounds[1]) / 2

                center_score = 1.0 - (distance_from_center / max_distance)
                score += center_score * 0.2
                break

        # Size category bonus
        size_category = requirement['size_category']
        if size_category == 'large':
            score += 0.1  # Large îlots get slight preference

        return score

    def _assign_ilots_to_clusters(self, requirements: List[Dict[str, Any]], 
                                cluster_centers: List[Tuple[float, float]]) -> Dict[int, List[Dict[str, Any]]]:
        """Assign îlots to clusters"""
        if not cluster_centers:
            return {0: requirements}

        # Simple round-robin assignment
        assignments = {i: [] for i in range(len(cluster_centers))}

        for i, req in enumerate(requirements):
            cluster_id = i % len(cluster_centers)
            assignments[cluster_id].append(req)

        return assignments

    def _place_cluster(self, cluster_ilots: List[Dict[str, Any]], 
                      center_x: float, center_y: float,
                      constraints: Dict[str, Any], 
                      existing_ilots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Place îlots within a cluster"""
        placed = []

        # Use spiral placement around cluster center
        angle_step = 2 * np.pi / max(8, len(cluster_ilots))
        radius = 2.0

        for i, req in enumerate(cluster_ilots):
            attempts = 0
            max_attempts = 20
            current_radius = radius

            while attempts < max_attempts:
                angle = i * angle_step + (attempts * 0.1)  # Add small offset for attempts
                x = center_x + current_radius * np.cos(angle)
                y = center_y + current_radius * np.sin(angle)
                position = Point(x, y)

                if self._is_valid_position(position, req, constraints, existing_ilots + placed):
                    ilot = self._create_ilot(req, position)
                    placed.append(ilot)
                    break

                attempts += 1
                if attempts % 5 == 0:
                    current_radius += 1.0  # Expand radius if no valid position found

        return placed

    def _create_ilot(self, requirement: Dict[str, Any], position: Point) -> Dict[str, Any]:
        """Create îlot object from requirement and position"""
        return {
            'id': requirement['id'],
            'size_category': requirement['size_category'],
            'dimensions': requirement['dimensions'],
            'position': {'x': position.x, 'y': position.y, 'z': 0},
            'rotation': 0,  # Default rotation
            'area': requirement['dimensions']['area'],
            'properties': {
                'algorithm': 'optimized_placement',
                'priority': requirement['priority']
            },
            'accessibility_score': 0.8,  # Default accessibility score
            'placement_score': 0.75  # Default placement score
        }

    def _validate_placements(self, placed_ilots: List[Dict[str, Any]], 
                           constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and post-process placed îlots"""
        validated = []

        for ilot in placed_ilots:
            # Additional validation checks
            position = Point(ilot['position']['x'], ilot['position']['y'])

            # Check minimum spacing
            valid = True
            for other in validated:
                other_pos = Point(other['position']['x'], other['position']['y'])
                distance = position.distance(other_pos)
                min_distance = (ilot['dimensions']['width'] + other['dimensions']['width'])/2 + 1.0

                if distance < min_distance:
                    valid = False
                    break

            if valid:
                # Calculate final accessibility score
                ilot['accessibility_score'] = self._calculate_accessibility_score(ilot, constraints)
                validated.append(ilot)

        return validated

    def _calculate_accessibility_score(self, ilot: Dict[str, Any], 
                                     constraints: Dict[str, Any]) -> float:
        """Calculate accessibility score for îlot"""
        position = Point(ilot['position']['x'], ilot['position']['y'])

        # Base score
        score = 0.5

        # Distance to walls (closer to entrance areas is better)
        walls = constraints.get('walls', [])
        if walls:
            min_wall_distance = float('inf')
            for wall in walls:
                wall_geom = wall.get('geometry')
                if wall_geom:
                    dist = position.distance(wall_geom)
                    min_wall_distance = min(min_wall_distance, dist)

            if min_wall_distance != float('inf'):
                # Score based on distance (1-4m is optimal for accessibility)
                if 1.0 <= min_wall_distance <= 4.0:
                    score += 0.3
                elif min_wall_distance <= 6.0:
                    score += 0.2

        # Size category accessibility
        size_category = ilot['size_category']
        if size_category == 'small':
            score += 0.1  # Small îlots are more accessible
        elif size_category == 'large':
            score -= 0.1  # Large îlots are less accessible

        return min(1.0, score)