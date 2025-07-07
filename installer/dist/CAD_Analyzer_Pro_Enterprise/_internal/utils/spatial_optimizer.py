import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import distance_matrix, Voronoi
from sklearn.cluster import DBSCAN, KMeans
import random
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class SpatialOptimizer:
    """Advanced spatial optimization for îlot placement"""

    def __init__(self):
        self.optimization_methods = {
            'genetic': self._genetic_optimization,
            'simulated_annealing': self._simulated_annealing,
            'particle_swarm': self._particle_swarm_optimization,
            'gradient_descent': self._gradient_descent_optimization
        }

        self.objective_weights = {
            'space_utilization': 0.3,
            'accessibility': 0.25,
            'safety': 0.2,
            'efficiency': 0.15,
            'aesthetics': 0.1
        }

    def optimize_placement(self, initial_placement: List[Dict[str, Any]], 
                          analysis_results: Dict[str, Any],
                          method: str = 'genetic',
                          max_iterations: int = 100) -> List[Dict[str, Any]]:
        """Optimize îlot placement using specified method"""
        logger.info(f"Starting spatial optimization using {method}")

        if not initial_placement:
            logger.warning("No initial placement provided")
            return []

        # Extract constraints from analysis results
        constraints = self._extract_optimization_constraints(analysis_results)

        # Prepare optimization data
        optimization_data = self._prepare_optimization_data(initial_placement, constraints)

        # Run optimization
        optimization_func = self.optimization_methods.get(method, self._genetic_optimization)
        optimized_data = optimization_func(optimization_data, constraints, max_iterations)

        # Convert back to îlot format
        optimized_placement = self._convert_optimization_result(optimized_data, initial_placement)

        # Validate and post-process
        final_placement = self._post_process_optimization(optimized_placement, constraints)

        logger.info(f"Optimization completed: {len(final_placement)} îlots optimized")
        return final_placement

    def _extract_optimization_constraints(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract constraints for optimization"""
        walls = analysis_results.get('walls', [])
        restricted_areas = analysis_results.get('restricted_areas', [])
        open_spaces = analysis_results.get('open_spaces', [])
        entrances = analysis_results.get('entrances', [])

        # Create constraint geometries
        obstacle_geometries = []

        # Add wall buffers
        for wall in walls:
            wall_geom = wall.get('geometry')
            if wall_geom:
                buffered = wall_geom.buffer(1.0)  # 1m clearance from walls
                obstacle_geometries.append(buffered)

        # Add restricted areas
        for area in restricted_areas:
            area_geom = area.get('geometry')
            if area_geom:
                obstacle_geometries.append(area_geom)

        # Combine obstacles
        if obstacle_geometries:
            combined_obstacles = unary_union(obstacle_geometries)
        else:
            combined_obstacles = None

        # Create available space
        available_geometries = []
        for space in open_spaces:
            space_geom = space.get('geometry')
            if space_geom:
                if combined_obstacles:
                    clean_space = space_geom.difference(combined_obstacles)
                    if not clean_space.is_empty:
                        available_geometries.append(clean_space)
                else:
                    available_geometries.append(space_geom)

        # Calculate bounds
        if available_geometries:
            all_bounds = [geom.bounds for geom in available_geometries]
            min_x = min(b[0] for b in all_bounds)
            min_y = min(b[1] for b in all_bounds)
            max_x = max(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
        else:
            min_x = min_y = 0
            max_x = max_y = 100

        return {
            'obstacles': combined_obstacles,
            'available_spaces': available_geometries,
            'bounds': {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y},
            'walls': walls,
            'entrances': entrances,
            'min_spacing': 1.5,
            'wall_clearance': 1.0
        }

    def _prepare_optimization_data(self, placement: List[Dict[str, Any]], 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for optimization algorithms"""
        # Extract position and dimension data
        positions = []
        dimensions = []
        ilot_ids = []

        for ilot in placement:
            pos = ilot.get('position', {})
            dims = ilot.get('dimensions', {})

            positions.append([pos.get('x', 0), pos.get('y', 0)])
            dimensions.append([dims.get('width', 2.0), dims.get('height', 2.0)])
            ilot_ids.append(ilot.get('id', ''))

        return {
            'positions': np.array(positions),
            'dimensions': np.array(dimensions),
            'ilot_ids': ilot_ids,
            'bounds': constraints['bounds'],
            'num_ilots': len(placement)
        }

    def _genetic_optimization(self, optimization_data: Dict[str, Any], 
                            constraints: Dict[str, Any], 
                            max_iterations: int) -> Dict[str, Any]:
        """Genetic algorithm optimization"""
        logger.info("Running genetic algorithm optimization")

        positions = optimization_data['positions']
        dimensions = optimization_data['dimensions']
        bounds = constraints['bounds']

        # Define bounds for optimization variables (x, y for each îlot)
        variable_bounds = []
        for i in range(len(positions)):
            width, height = dimensions[i]
            # Ensure îlot fits within bounds
            min_x = bounds['min_x'] + width/2
            max_x = bounds['max_x'] - width/2
            min_y = bounds['min_y'] + height/2
            max_y = bounds['max_y'] - height/2

            variable_bounds.extend([(min_x, max_x), (min_y, max_y)])

        # Initial solution (flatten positions)
        initial_solution = positions.flatten()

        # Objective function
        def objective(solution):
            return -self._calculate_fitness(solution, optimization_data, constraints)

        # Run differential evolution (genetic algorithm variant)
        result = differential_evolution(
            objective,
            variable_bounds,
            maxiter=max_iterations,
            popsize=15,
            seed=42
        )

        # Reshape result back to positions
        optimized_positions = result.x.reshape(-1, 2)

        updated_data = optimization_data.copy()
        updated_data['positions'] = optimized_positions
        updated_data['fitness'] = -result.fun

        return updated_data

    def _simulated_annealing(self, optimization_data: Dict[str, Any], 
                           constraints: Dict[str, Any], 
                           max_iterations: int) -> Dict[str, Any]:
        """Simulated annealing optimization"""
        logger.info("Running simulated annealing optimization")

        positions = optimization_data['positions'].copy()
        dimensions = optimization_data['dimensions']
        bounds = constraints['bounds']

        current_solution = positions.copy()
        current_fitness = self._calculate_fitness(current_solution.flatten(), optimization_data, constraints)

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        # Simulated annealing parameters
        initial_temp = 100.0
        cooling_rate = 0.95
        temperature = initial_temp

        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor_solution(current_solution, bounds, dimensions)
            neighbor_fitness = self._calculate_fitness(neighbor_solution.flatten(), optimization_data, constraints)

            # Accept or reject neighbor
            if neighbor_fitness > current_fitness:
                # Better solution - always accept
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness

                if neighbor_fitness > best_fitness:
                    best_solution = neighbor_solution.copy()
                    best_fitness = neighbor_fitness
            else:
                # Worse solution - accept with probability
                delta = current_fitness - neighbor_fitness
                probability = np.exp(-delta / temperature)

                if random.random() < probability:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness

            # Cool down
            temperature *= cooling_rate

            if iteration % 20 == 0:
                logger.debug(f"SA Iteration {iteration}: Best fitness = {best_fitness:.3f}")

        updated_data = optimization_data.copy()
        updated_data['positions'] = best_solution
        updated_data['fitness'] = best_fitness

        return updated_data

    def _particle_swarm_optimization(self, optimization_data: Dict[str, Any], 
                                   constraints: Dict[str, Any], 
                                   max_iterations: int) -> Dict[str, Any]:
        """Particle Swarm Optimization"""
        logger.info("Running particle swarm optimization")

        positions = optimization_data['positions']
        dimensions = optimization_data['dimensions']
        bounds = constraints['bounds']
        num_particles = 20
        num_dimensions = positions.size

        # Initialize particles
        particles = []
        for _ in range(num_particles):
            particle_positions = []
            for i in range(len(positions)):
                width, height = dimensions[i]
                min_x = bounds['min_x'] + width/2
                max_x = bounds['max_x'] - width/2
                min_y = bounds['min_y'] + height/2
                max_y = bounds['max_y'] - height/2

                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                particle_positions.extend([x, y])

            particles.append({
                'position': np.array(particle_positions),
                'velocity': np.random.uniform(-1, 1, num_dimensions),
                'best_position': np.array(particle_positions),
                'best_fitness': -float('inf')
            })

        # Global best
        global_best_position = positions.flatten().copy()
        global_best_fitness = self._calculate_fitness(global_best_position, optimization_data, constraints)

        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter

        for iteration in range(max_iterations):
            for particle in particles:
                # Evaluate fitness
                fitness = self._calculate_fitness(particle['position'], optimization_data, constraints)

                # Update personal best
                if fitness > particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'].copy()

                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle['position'].copy()

                # Update velocity
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (particle['best_position'] - particle['position'])
                social = c2 * r2 * (global_best_position - particle['position'])
                particle['velocity'] = w * particle['velocity'] + cognitive + social

                # Update position
                particle['position'] += particle['velocity']

                # Enforce bounds
                particle['position'] = self._enforce_bounds(particle['position'], bounds, dimensions)

            if iteration % 20 == 0:
                logger.debug(f"PSO Iteration {iteration}: Best fitness = {global_best_fitness:.3f}")

        updated_data = optimization_data.copy()
        updated_data['positions'] = global_best_position.reshape(-1, 2)
        updated_data['fitness'] = global_best_fitness

        return updated_data

    def _gradient_descent_optimization(self, optimization_data: Dict[str, Any], 
                                     constraints: Dict[str, Any], 
                                     max_iterations: int) -> Dict[str, Any]:
        """Gradient descent optimization using scipy.minimize"""
        logger.info("Running gradient descent optimization")

        positions = optimization_data['positions']
        dimensions = optimization_data['dimensions']
        bounds = constraints['bounds']

        # Define bounds
        variable_bounds = []
        for i in range(len(positions)):
            width, height = dimensions[i]
            min_x = bounds['min_x'] + width/2
            max_x = bounds['max_x'] - width/2
            min_y = bounds['min_y'] + height/2
            max_y = bounds['max_y'] - height/2

            variable_bounds.extend([(min_x, max_x), (min_y, max_y)])

        # Objective function (minimize negative fitness)
        def objective(solution):
            return -self._calculate_fitness(solution, optimization_data, constraints)

        # Initial solution
        initial_solution = positions.flatten()

        # Run optimization
        result = minimize(
            objective,
            initial_solution,
            method='L-BFGS-B',
            bounds=variable_bounds,
            options={'maxiter': max_iterations}
        )

        updated_data = optimization_data.copy()
        updated_data['positions'] = result.x.reshape(-1, 2)
        updated_data['fitness'] = -result.fun

        return updated_data

    def _calculate_fitness(self, solution: np.ndarray, optimization_data: Dict[str, Any], 
                          constraints: Dict[str, Any]) -> float:
        """Calculate fitness score for a solution"""
        positions = solution.reshape(-1, 2)
        dimensions = optimization_data['dimensions']

        # Initialize fitness components
        fitness_components = {}

        # 1. Space utilization
        fitness_components['space_utilization'] = self._calculate_space_utilization(
            positions, dimensions, constraints
        )

        # 2. Accessibility
        fitness_components['accessibility'] = self._calculate_accessibility_score(
            positions, dimensions, constraints
        )

        # 3. Safety (collision avoidance)
        fitness_components['safety'] = self._calculate_safety_score(
            positions, dimensions, constraints
        )

        # 4. Efficiency (minimize travel distances)
        fitness_components['efficiency'] = self._calculate_efficiency_score(
            positions, constraints
        )

        # 5. Aesthetics (spacing uniformity)
        fitness_components['aesthetics'] = self._calculate_aesthetics_score(
            positions, dimensions
        )

        # Combine fitness components with weights
        total_fitness = 0
        for component, score in fitness_components.items():
            weight = self.objective_weights.get(component, 0)
            total_fitness += weight * score

        return total_fitness

    def _calculate_space_utilization(self, positions: np.ndarray, dimensions: np.ndarray, 
                                   constraints: Dict[str, Any]) -> float:
        """Calculate space utilization score"""
        # Calculate total îlot area
        total_ilot_area = np.sum(dimensions[:, 0] * dimensions[:, 1])

        # Calculate available space area
        available_spaces = constraints.get('available_spaces', [])
        total_available_area = sum(space.area for space in available_spaces)

        if total_available_area == 0:
            return 0.0

        # Base utilization ratio
        utilization_ratio = total_ilot_area / total_available_area

        # Penalty for positions outside available spaces
        valid_positions = 0
        for i, pos in enumerate(positions):
            width, height = dimensions[i]
            ilot_box = box(pos[0] - width/2, pos[1] - height/2,
                          pos[0] + width/2, pos[1] + height/2)

            for space in available_spaces:
                if space.contains(ilot_box):
                    valid_positions += 1
                    break

        validity_ratio = valid_positions / len(positions) if len(positions) > 0 else 0

        return min(1.0, utilization_ratio) * validity_ratio

    def _calculate_accessibility_score(self, positions: np.ndarray, dimensions: np.ndarray, 
                                     constraints: Dict[str, Any]) -> float:
        """Calculate accessibility score"""
        entrances = constraints.get('entrances', [])

        if not entrances:
            return 0.5  # Neutral score if no entrances

        accessibility_scores = []

        for pos in positions:
            # Calculate distance to nearest entrance
            min_distance = float('inf')
            for entrance in entrances:
                entrance_pos = entrance.get('position', {})
                entrance_point = (entrance_pos.get('x', 0), entrance_pos.get('y', 0))

                distance = np.sqrt((pos[0] - entrance_point[0])**2 + 
                                 (pos[1] - entrance_point[1])**2)
                min_distance = min(min_distance, distance)

            # Score based on distance (closer is better, but diminishing returns)
            if min_distance == float('inf'):
                accessibility_scores.append(0)
            else:
                # Optimal distance is 5-15 meters
                if 5 <= min_distance <= 15:
                    score = 1.0
                elif min_distance < 5:
                    score = min_distance / 5.0
                else:
                    score = max(0, 1.0 - (min_distance - 15) / 20.0)

                accessibility_scores.append(score)

        return np.mean(accessibility_scores) if accessibility_scores else 0.0

    def _calculate_safety_score(self, positions: np.ndarray, dimensions: np.ndarray, 
                              constraints: Dict[str, Any]) -> float:
        """Calculate safety score (collision avoidance)"""
        min_spacing = constraints.get('min_spacing', 1.5)

        # Check îlot-îlot collisions
        collision_penalty = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)

                required_distance = (dimensions[i][0] + dimensions[j][0])/2 + min_spacing

                if distance < required_distance:
                    collision_penalty += (required_distance - distance) / required_distance

        # Check obstacle collisions
        obstacles = constraints.get('obstacles')
        obstacle_penalty = 0

        if obstacles:
            for i, pos in enumerate(positions):
                width, height = dimensions[i]
                ilot_box = box(pos[0] - width/2, pos[1] - height/2,
                              pos[0] + width/2, pos[1] + height/2)

                if obstacles.intersects(ilot_box):
                    obstacle_penalty += 1

        # Normalize penalties
        total_penalty = collision_penalty + obstacle_penalty
        max_possible_penalty = len(positions)  # Worst case scenario

        if max_possible_penalty == 0:
            return 1.0

        safety_score = max(0, 1.0 - total_penalty / max_possible_penalty)
        return safety_score

    def _calculate_efficiency_score(self, positions: np.ndarray, 
                                  constraints: Dict[str, Any]) -> float:
        """Calculate efficiency score based on travel distances"""
        if len(positions) < 2:
            return 1.0

        # Calculate distance matrix between all îlots
        distances = distance_matrix(positions, positions)

        # Score based on average distance (moderate distances are good)
        upper_triangle_indices = np.triu_indices(len(positions), k=1)
        pairwise_distances = distances[upper_triangle_indices]

        if len(pairwise_distances) == 0:
            return 1.0

        avg_distance = np.mean(pairwise_distances)

        # Optimal average distance is 4-8 meters
        if 4 <= avg_distance <= 8:
            return 1.0
        elif avg_distance < 4:
            return avg_distance / 4.0
        else:
            return max(0, 1.0 - (avg_distance - 8) / 10.0)

    def _calculate_aesthetics_score(self, positions: np.ndarray, 
                                  dimensions: np.ndarray) -> float:
        """Calculate aesthetics score based on spacing uniformity"""
        if len(positions) < 2:
            return 1.0

        # Calculate all pairwise distances
        distances = distance_matrix(positions, positions)
        upper_triangle_indices = np.triu_indices(len(positions), k=1)
        pairwise_distances = distances[upper_triangle_indices]

        if len(pairwise_distances) == 0:
            return 1.0

        # Score based on uniformity of spacing
        distance_std = np.std(pairwise_distances)
        distance_mean = np.mean(pairwise_distances)

        if distance_mean == 0:
            return 0.0

        # Coefficient of variation (lower is better for uniformity)
        cv = distance_std / distance_mean

        # Score: 1.0 for perfect uniformity, decreasing with higher variation
        aesthetics_score = max(0, 1.0 - cv)

        return aesthetics_score

    def _generate_neighbor_solution(self, current_solution: np.ndarray, 
                                  bounds: Dict[str, float], 
                                  dimensions: np.ndarray) -> np.ndarray:
        """Generate neighbor solution for simulated annealing"""
        neighbor = current_solution.copy()

        # Randomly select îlot to modify
        ilot_index = random.randint(0, len(neighbor) - 1)

        # Generate small random perturbation
        perturbation = np.random.normal(0, 1.0, 2)  # Standard deviation of 1 meter

        # Apply perturbation
        neighbor[ilot_index] += perturbation

        # Enforce bounds
        width, height = dimensions[ilot_index]
        neighbor[ilot_index][0] = np.clip(neighbor[ilot_index][0], 
                                        bounds['min_x'] + width/2, 
                                        bounds['max_x'] - width/2)
        neighbor[ilot_index][1] = np.clip(neighbor[ilot_index][1], 
                                        bounds['min_y'] + height/2, 
                                        bounds['max_y'] - height/2)

        return neighbor

    def _enforce_bounds(self, positions: np.ndarray, bounds: Dict[str, float], 
                       dimensions: np.ndarray) -> np.ndarray:
        """Enforce spatial bounds on positions"""
        bounded_positions = positions.copy()

        for i in range(0, len(positions), 2):
            ilot_idx = i // 2
            if ilot_idx < len(dimensions):
                width, height = dimensions[ilot_idx]

                # Enforce x bounds
                bounded_positions[i] = np.clip(bounded_positions[i], 
                                             bounds['min_x'] + width/2, 
                                             bounds['max_x'] - width/2)

                # Enforce y bounds
                bounded_positions[i+1] = np.clip(bounded_positions[i+1], 
                                               bounds['min_y'] + height/2, 
                                               bounds['max_y'] - height/2)

        return bounded_positions

    def _convert_optimization_result(self, optimized_data: Dict[str, Any], 
                                   original_placement: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert optimization result back to îlot format"""
        optimized_positions = optimized_data['positions']
        optimized_placement = []

        for i, ilot in enumerate(original_placement):
            if i < len(optimized_positions):
                optimized_ilot = ilot.copy()
                optimized_ilot['position'] = {
                    'x': optimized_positions[i][0],
                    'y': optimized_positions[i][1],
                    'z': 0
                }

                # Add optimization metadata
                optimized_ilot['optimization'] = {
                    'method': 'spatial_optimizer',
                    'fitness_score': optimized_data.get('fitness', 0),
                    'optimized': True
                }

                optimized_placement.append(optimized_ilot)

        return optimized_placement

    def _post_process_optimization(self, placement: List[Dict[str, Any]], 
                                 constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process optimized placement"""
        processed_placement = []

        for ilot in placement:
            # Validate placement
            if self._validate_ilot_placement(ilot, constraints):
                # Calculate final scores
                ilot['final_scores'] = self._calculate_final_scores(ilot, constraints)
                processed_placement.append(ilot)
            else:
                logger.warning(f"Îlot {ilot['id']} failed post-optimization validation")

        return processed_placement

    def _validate_ilot_placement(self, ilot: Dict[str, Any], 
                               constraints: Dict[str, Any]) -> bool:
        """Validate individual îlot placement"""
        position = ilot.get('position', {})
        dimensions = ilot.get('dimensions', {})

        x, y = position.get('x', 0), position.get('y', 0)
        width, height = dimensions.get('width', 2.0), dimensions.get('height', 2.0)

        # Create îlot geometry
        ilot_box = box(x - width/2, y - height/2, x + width/2, y + height/2)

        # Check if within available spaces
        available_spaces = constraints.get('available_spaces', [])
        within_available = False
        for space in available_spaces:
            if space.contains(ilot_box):
                within_available = True
                break

        if not within_available:
            return False

        # Check obstacle collision
        obstacles = constraints.get('obstacles')
        if obstacles and obstacles.intersects(ilot_box):
            return False

        return True

    def _calculate_final_scores(self, ilot: Dict[str, Any], 
                              constraints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate final scores for optimized îlot"""
        position = ilot.get('position', {})
        dimensions = ilot.get('dimensions', {})

        pos_array = np.array([[position.get('x', 0), position.get('y', 0)]])
        dim_array = np.array([[dimensions.get('width', 2.0), dimensions.get('height', 2.0)]])

        return {
            'space_utilization': self._calculate_space_utilization(pos_array, dim_array, constraints),
            'accessibility': self._calculate_accessibility_score(pos_array, dim_array, constraints),
            'safety': self._calculate_safety_score(pos_array, dim_array, constraints),
            'overall_fitness': self._calculate_fitness(pos_array.flatten(), 
                                                     {'positions': pos_array, 'dimensions': dim_array}, 
                                                     constraints)
        }