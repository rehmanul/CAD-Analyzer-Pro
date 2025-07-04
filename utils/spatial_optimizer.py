import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
import random
from collections import defaultdict
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialOptimizer:
    """Advanced spatial optimization for îlot placement using multi-objective optimization"""
    
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
            'circulation': 0.2,
            'safety': 0.15,
            'aesthetics': 0.1
        }
        
        self.constraints = {
            'min_spacing': 1.5,
            'wall_clearance': 1.0,
            'emergency_access': 2.0,
            'max_cluster_size': 5,
            'min_path_width': 1.2
        }
    
    def optimize_placement(self, ilot_results: List[Dict[str, Any]],
                          analysis_results: Dict[str, Any],
                          method: str = 'genetic') -> List[Dict[str, Any]]:
        """
        Optimize îlot placement using advanced spatial optimization
        
        Args:
            ilot_results: Initial îlot placement results
            analysis_results: Analysis results from zone detection
            method: Optimization method to use
            
        Returns:
            Optimized îlot placement results
        """
        logger.info(f"Starting spatial optimization using {method} method")
        
        try:
            # Extract spatial context
            spatial_context = self._extract_spatial_context(analysis_results)
            
            # Initialize optimization parameters
            optimization_params = self._initialize_optimization_params(
                ilot_results, spatial_context
            )
            
            # Select optimization method
            optimization_func = self.optimization_methods.get(method, self._genetic_optimization)
            
            # Perform optimization
            optimized_positions = optimization_func(
                optimization_params, spatial_context
            )
            
            # Apply optimized positions to îlots
            optimized_ilots = self._apply_optimized_positions(
                ilot_results, optimized_positions
            )
            
            # Validate and post-process results
            validated_ilots = self._validate_optimization_results(
                optimized_ilots, spatial_context
            )
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                ilot_results, validated_ilots, spatial_context
            )
            
            logger.info(f"Optimization completed with {improvement_metrics['overall_improvement']:.1f}% improvement")
            
            return validated_ilots
            
        except Exception as e:
            logger.error(f"Error during spatial optimization: {str(e)}")
            # Return original results if optimization fails
            return ilot_results
    
    def _extract_spatial_context(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial context for optimization"""
        return {
            'walls': analysis_results.get('walls', []),
            'restricted_areas': analysis_results.get('restricted_areas', []),
            'entrances': analysis_results.get('entrances', []),
            'open_spaces': analysis_results.get('open_spaces', []),
            'corridors': analysis_results.get('corridors', []),
            'bounds': self._calculate_optimization_bounds(analysis_results),
            'obstacles': self._identify_obstacles(analysis_results),
            'access_points': self._identify_access_points(analysis_results)
        }
    
    def _calculate_optimization_bounds(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimization bounds"""
        open_spaces = analysis_results.get('open_spaces', [])
        
        if not open_spaces:
            return {'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 100}
        
        all_bounds = []
        for space in open_spaces:
            if space.get('shapely_geom'):
                bounds = space['shapely_geom'].bounds
                all_bounds.append(bounds)
        
        if not all_bounds:
            return {'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 100}
        
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        
        return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}
    
    def _identify_obstacles(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify obstacles for optimization"""
        obstacles = []
        
        # Add walls as obstacles
        for wall in analysis_results.get('walls', []):
            wall_geom = wall.get('shapely_geom')
            if wall_geom:
                # Buffer wall to create obstacle area
                obstacle_geom = wall_geom.buffer(self.constraints['wall_clearance'])
                obstacles.append({
                    'type': 'wall',
                    'geometry': obstacle_geom,
                    'clearance': self.constraints['wall_clearance']
                })
        
        # Add restricted areas as obstacles
        for restricted in analysis_results.get('restricted_areas', []):
            restricted_geom = restricted.get('shapely_geom')
            if restricted_geom:
                obstacles.append({
                    'type': 'restricted',
                    'geometry': restricted_geom,
                    'clearance': 0.5
                })
        
        return obstacles
    
    def _identify_access_points(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify access points for optimization"""
        access_points = []
        
        # Add entrances as access points
        for entrance in analysis_results.get('entrances', []):
            position = entrance.get('position', {})
            access_points.append({
                'type': 'entrance',
                'position': (position.get('x', 0), position.get('y', 0)),
                'importance': 1.0,
                'access_width': entrance.get('width', 1.0)
            })
        
        # Add corridor intersections as access points
        corridors = analysis_results.get('corridors', [])
        if len(corridors) > 1:
            intersections = self._find_corridor_intersections(corridors)
            for intersection in intersections:
                access_points.append({
                    'type': 'intersection',
                    'position': intersection,
                    'importance': 0.7,
                    'access_width': 1.5
                })
        
        return access_points
    
    def _find_corridor_intersections(self, corridors: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Find corridor intersections"""
        intersections = []
        
        for i, corridor1 in enumerate(corridors):
            for corridor2 in corridors[i+1:]:
                geom1 = corridor1.get('shapely_geom')
                geom2 = corridor2.get('shapely_geom')
                
                if geom1 and geom2:
                    intersection = geom1.intersection(geom2)
                    if not intersection.is_empty:
                        if hasattr(intersection, 'coords'):
                            coords = list(intersection.coords)
                            if coords:
                                intersections.append((coords[0][0], coords[0][1]))
                        elif hasattr(intersection, 'centroid'):
                            centroid = intersection.centroid
                            intersections.append((centroid.x, centroid.y))
        
        return intersections
    
    def _initialize_optimization_params(self, ilot_results: List[Dict[str, Any]],
                                       spatial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize optimization parameters"""
        # Extract current positions
        current_positions = []
        ilot_properties = []
        
        for ilot in ilot_results:
            position = ilot.get('position', {})
            current_positions.append([position.get('x', 0), position.get('y', 0)])
            
            dimensions = ilot.get('dimensions', {})
            ilot_properties.append({
                'id': ilot.get('id', ''),
                'size_category': ilot.get('size_category', 'medium'),
                'width': dimensions.get('width', 2.0),
                'height': dimensions.get('height', 2.0),
                'area': dimensions.get('area', 4.0)
            })
        
        bounds = spatial_context['bounds']
        
        return {
            'current_positions': np.array(current_positions),
            'ilot_properties': ilot_properties,
            'bounds': bounds,
            'num_ilots': len(ilot_results),
            'search_space': {
                'x_min': bounds['min_x'] + 2,
                'x_max': bounds['max_x'] - 2,
                'y_min': bounds['min_y'] + 2,
                'y_max': bounds['max_y'] - 2
            }
        }
    
    def _genetic_optimization(self, params: Dict[str, Any], 
                             spatial_context: Dict[str, Any]) -> np.ndarray:
        """Genetic algorithm optimization"""
        logger.info("Running genetic algorithm optimization")
        
        # GA parameters
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population
        population = self._initialize_population(params, population_size)
        
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, params, spatial_context)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # Selection
            selected_population = self._tournament_selection(
                population, fitness_scores, population_size
            )
            
            # Crossover and mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i + 1, population_size - 1)]
                
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1, params)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2, params)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            
            # Log progress
            if generation % 20 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        return best_solution if best_solution is not None else params['current_positions']
    
    def _initialize_population(self, params: Dict[str, Any], population_size: int) -> List[np.ndarray]:
        """Initialize genetic algorithm population"""
        population = []
        search_space = params['search_space']
        num_ilots = params['num_ilots']
        
        for _ in range(population_size):
            individual = np.random.uniform(
                low=[search_space['x_min'], search_space['y_min']],
                high=[search_space['x_max'], search_space['y_max']],
                size=(num_ilots, 2)
            )
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, positions: np.ndarray, params: Dict[str, Any],
                         spatial_context: Dict[str, Any]) -> float:
        """Evaluate fitness of a solution"""
        # Calculate multiple objectives
        objectives = {
            'space_utilization': self._calculate_space_utilization_objective(
                positions, params, spatial_context
            ),
            'accessibility': self._calculate_accessibility_objective(
                positions, params, spatial_context
            ),
            'circulation': self._calculate_circulation_objective(
                positions, params, spatial_context
            ),
            'safety': self._calculate_safety_objective(
                positions, params, spatial_context
            ),
            'aesthetics': self._calculate_aesthetics_objective(
                positions, params, spatial_context
            )
        }
        
        # Apply constraint penalties
        constraint_penalty = self._calculate_constraint_penalty(
            positions, params, spatial_context
        )
        
        # Calculate weighted fitness
        fitness = sum(
            objectives[obj] * self.objective_weights[obj]
            for obj in objectives.keys()
        ) - constraint_penalty
        
        return fitness
    
    def _calculate_space_utilization_objective(self, positions: np.ndarray,
                                             params: Dict[str, Any],
                                             spatial_context: Dict[str, Any]) -> float:
        """Calculate space utilization objective"""
        # Calculate how well îlots fill the available space
        total_ilot_area = sum(prop['area'] for prop in params['ilot_properties'])
        
        # Calculate available space area
        available_area = 0
        for space in spatial_context['open_spaces']:
            if space.get('shapely_geom'):
                available_area += space['shapely_geom'].area
        
        if available_area == 0:
            return 0
        
        # Calculate utilization ratio
        utilization = min(1.0, total_ilot_area / available_area)
        
        # Bonus for even distribution
        distribution_bonus = self._calculate_distribution_bonus(positions)
        
        return utilization * 100 + distribution_bonus
    
    def _calculate_accessibility_objective(self, positions: np.ndarray,
                                         params: Dict[str, Any],
                                         spatial_context: Dict[str, Any]) -> float:
        """Calculate accessibility objective"""
        accessibility_score = 0
        access_points = spatial_context['access_points']
        
        if not access_points:
            return 50  # Neutral score if no access points
        
        for i, position in enumerate(positions):
            # Calculate distance to nearest access point
            min_distance = float('inf')
            for access_point in access_points:
                distance = np.sqrt(
                    (position[0] - access_point['position'][0])**2 +
                    (position[1] - access_point['position'][1])**2
                )
                weighted_distance = distance / access_point['importance']
                min_distance = min(min_distance, weighted_distance)
            
            # Convert distance to accessibility score (closer = better)
            if min_distance < 5:
                ilot_accessibility = 100 - min_distance * 10
            elif min_distance < 10:
                ilot_accessibility = 50 - (min_distance - 5) * 5
            else:
                ilot_accessibility = 0
            
            accessibility_score += max(0, ilot_accessibility)
        
        return accessibility_score / len(positions) if positions.size > 0 else 0
    
    def _calculate_circulation_objective(self, positions: np.ndarray,
                                       params: Dict[str, Any],
                                       spatial_context: Dict[str, Any]) -> float:
        """Calculate circulation objective"""
        if len(positions) < 2:
            return 100  # Perfect score for single îlot
        
        # Calculate average distance between îlots
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # Optimal distance is around 4-6 meters
        optimal_distance = 5.0
        distance_score = 100 - abs(avg_min_distance - optimal_distance) * 10
        
        # Penalty for overcrowding
        overcrowding_penalty = 0
        for distance in min_distances:
            if distance < self.constraints['min_spacing']:
                overcrowding_penalty += (self.constraints['min_spacing'] - distance) * 20
        
        return max(0, distance_score - overcrowding_penalty)
    
    def _calculate_safety_objective(self, positions: np.ndarray,
                                   params: Dict[str, Any],
                                   spatial_context: Dict[str, Any]) -> float:
        """Calculate safety objective"""
        safety_score = 100  # Start with perfect score
        
        # Check emergency access
        for i, position in enumerate(positions):
            ilot_geom = box(
                position[0] - params['ilot_properties'][i]['width'] / 2,
                position[1] - params['ilot_properties'][i]['height'] / 2,
                position[0] + params['ilot_properties'][i]['width'] / 2,
                position[1] + params['ilot_properties'][i]['height'] / 2
            )
            
            # Check distance to nearest emergency access
            min_emergency_distance = float('inf')
            for access_point in spatial_context['access_points']:
                if access_point['type'] == 'entrance':
                    distance = np.sqrt(
                        (position[0] - access_point['position'][0])**2 +
                        (position[1] - access_point['position'][1])**2
                    )
                    min_emergency_distance = min(min_emergency_distance, distance)
            
            # Penalty for being too far from emergency access
            if min_emergency_distance > 15:  # 15 meters max
                safety_score -= (min_emergency_distance - 15) * 2
        
        return max(0, safety_score)
    
    def _calculate_aesthetics_objective(self, positions: np.ndarray,
                                      params: Dict[str, Any],
                                      spatial_context: Dict[str, Any]) -> float:
        """Calculate aesthetics objective"""
        if len(positions) < 3:
            return 100  # Perfect score for small layouts
        
        # Calculate symmetry and balance
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        
        # Calculate deviation from center
        deviations = np.sqrt(
            (positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2
        )
        
        # Prefer balanced distribution
        balance_score = 100 - np.std(deviations) * 5
        
        # Prefer regular spacing
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        spacing_regularity = 100 - np.std(min_distances) * 10
        
        return max(0, (balance_score + spacing_regularity) / 2)
    
    def _calculate_distribution_bonus(self, positions: np.ndarray) -> float:
        """Calculate bonus for even distribution"""
        if len(positions) < 2:
            return 0
        
        # Calculate Voronoi-like distribution metric
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        
        min_distances = np.min(distances, axis=1)
        distribution_uniformity = 100 - np.std(min_distances) * 5
        
        return max(0, min(20, distribution_uniformity))  # Max 20 point bonus
    
    def _calculate_constraint_penalty(self, positions: np.ndarray,
                                    params: Dict[str, Any],
                                    spatial_context: Dict[str, Any]) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0
        
        # Check minimum spacing constraints
        if len(positions) > 1:
            distances = cdist(positions, positions)
            np.fill_diagonal(distances, np.inf)
            
            min_distances = np.min(distances, axis=1)
            for distance in min_distances:
                if distance < self.constraints['min_spacing']:
                    penalty += (self.constraints['min_spacing'] - distance) * 50
        
        # Check boundary constraints
        bounds = spatial_context['bounds']
        for i, position in enumerate(positions):
            ilot_props = params['ilot_properties'][i]
            half_width = ilot_props['width'] / 2
            half_height = ilot_props['height'] / 2
            
            if (position[0] - half_width < bounds['min_x'] or
                position[0] + half_width > bounds['max_x'] or
                position[1] - half_height < bounds['min_y'] or
                position[1] + half_height > bounds['max_y']):
                penalty += 100
        
        # Check obstacle constraints
        for i, position in enumerate(positions):
            ilot_props = params['ilot_properties'][i]
            ilot_geom = box(
                position[0] - ilot_props['width'] / 2,
                position[1] - ilot_props['height'] / 2,
                position[0] + ilot_props['width'] / 2,
                position[1] + ilot_props['height'] / 2
            )
            
            for obstacle in spatial_context['obstacles']:
                if ilot_geom.intersects(obstacle['geometry']):
                    penalty += 150
        
        return penalty
    
    def _tournament_selection(self, population: List[np.ndarray],
                             fitness_scores: List[float],
                             population_size: int) -> List[np.ndarray]:
        """Tournament selection for genetic algorithm"""
        selected = []
        tournament_size = 3
        
        for _ in range(population_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select best from tournament
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        
        return selected
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover operation for genetic algorithm"""
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        search_space = params['search_space']
        
        # Mutate random positions
        mutation_strength = 2.0  # meters
        for i in range(len(mutated)):
            if random.random() < 0.3:  # 30% chance to mutate each îlot
                mutated[i, 0] += random.gauss(0, mutation_strength)
                mutated[i, 1] += random.gauss(0, mutation_strength)
                
                # Clamp to bounds
                mutated[i, 0] = np.clip(mutated[i, 0], search_space['x_min'], search_space['x_max'])
                mutated[i, 1] = np.clip(mutated[i, 1], search_space['y_min'], search_space['y_max'])
        
        return mutated
    
    def _simulated_annealing(self, params: Dict[str, Any], 
                           spatial_context: Dict[str, Any]) -> np.ndarray:
        """Simulated annealing optimization"""
        logger.info("Running simulated annealing optimization")
        
        # Start with current solution
        current_solution = params['current_positions'].copy()
        current_fitness = self._evaluate_fitness(current_solution, params, spatial_context)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # SA parameters
        initial_temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 0.01
        max_iterations = 1000
        
        temperature = initial_temperature
        
        for iteration in range(max_iterations):
            if temperature < min_temperature:
                break
            
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, params)
            neighbor_fitness = self._evaluate_fitness(neighbor, params, spatial_context)
            
            # Accept or reject neighbor
            if neighbor_fitness > current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness
            else:
                # Accept worse solution with probability
                delta = neighbor_fitness - current_fitness
                probability = np.exp(delta / temperature)
                
                if random.random() < probability:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
            
            # Cool down
            temperature *= cooling_rate
            
            # Log progress
            if iteration % 200 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {best_fitness:.4f}, Temperature = {temperature:.4f}")
        
        return best_solution
    
    def _generate_neighbor(self, solution: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Generate neighbor solution for simulated annealing"""
        neighbor = solution.copy()
        search_space = params['search_space']
        
        # Randomly modify one îlot position
        ilot_index = random.randint(0, len(neighbor) - 1)
        
        # Add random perturbation
        perturbation = random.gauss(0, 1.0)  # 1 meter standard deviation
        axis = random.randint(0, 1)  # x or y axis
        
        neighbor[ilot_index, axis] += perturbation
        
        # Clamp to bounds
        neighbor[ilot_index, 0] = np.clip(
            neighbor[ilot_index, 0], search_space['x_min'], search_space['x_max']
        )
        neighbor[ilot_index, 1] = np.clip(
            neighbor[ilot_index, 1], search_space['y_min'], search_space['y_max']
        )
        
        return neighbor
    
    def _particle_swarm_optimization(self, params: Dict[str, Any], 
                                   spatial_context: Dict[str, Any]) -> np.ndarray:
        """Particle swarm optimization"""
        logger.info("Running particle swarm optimization")
        
        # PSO parameters
        swarm_size = 30
        max_iterations = 200
        w = 0.5  # inertia weight
        c1 = 1.5  # cognitive parameter
        c2 = 1.5  # social parameter
        
        # Initialize swarm
        search_space = params['search_space']
        num_ilots = params['num_ilots']
        
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(swarm_size):
            particle = np.random.uniform(
                low=[search_space['x_min'], search_space['y_min']],
                high=[search_space['x_max'], search_space['y_max']],
                size=(num_ilots, 2)
            )
            
            velocity = np.random.uniform(-1, 1, (num_ilots, 2))
            
            fitness = self._evaluate_fitness(particle, params, spatial_context)
            
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_fitness.append(fitness)
        
        # Find global best
        global_best_index = np.argmax(personal_best_fitness)
        global_best = personal_best[global_best_index].copy()
        global_best_fitness = personal_best_fitness[global_best_index]
        
        # PSO main loop
        for iteration in range(max_iterations):
            for i in range(swarm_size):
                # Update velocity
                r1 = np.random.random((num_ilots, 2))
                r2 = np.random.random((num_ilots, 2))
                
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Clamp to bounds
                particles[i][:, 0] = np.clip(
                    particles[i][:, 0], search_space['x_min'], search_space['x_max']
                )
                particles[i][:, 1] = np.clip(
                    particles[i][:, 1], search_space['y_min'], search_space['y_max']
                )
                
                # Evaluate fitness
                fitness = self._evaluate_fitness(particles[i], params, spatial_context)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
            
            # Log progress
            if iteration % 50 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {global_best_fitness:.4f}")
        
        return global_best
    
    def _gradient_descent_optimization(self, params: Dict[str, Any], 
                                     spatial_context: Dict[str, Any]) -> np.ndarray:
        """Gradient descent optimization"""
        logger.info("Running gradient descent optimization")
        
        # Convert to optimization format
        initial_positions = params['current_positions'].flatten()
        bounds = []
        search_space = params['search_space']
        
        for _ in range(params['num_ilots']):
            bounds.append((search_space['x_min'], search_space['x_max']))
            bounds.append((search_space['y_min'], search_space['y_max']))
        
        # Objective function for scipy
        def objective(x):
            positions = x.reshape(params['num_ilots'], 2)
            return -self._evaluate_fitness(positions, params, spatial_context)
        
        # Run optimization
        result = minimize(
            objective,
            initial_positions,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        
        # Return optimized positions
        return result.x.reshape(params['num_ilots'], 2)
    
    def _apply_optimized_positions(self, ilot_results: List[Dict[str, Any]],
                                  optimized_positions: np.ndarray) -> List[Dict[str, Any]]:
        """Apply optimized positions to îlot results"""
        optimized_ilots = []
        
        for i, ilot in enumerate(ilot_results):
            optimized_ilot = ilot.copy()
            
            # Update position
            new_position = {
                'x': float(optimized_positions[i, 0]),
                'y': float(optimized_positions[i, 1]),
                'z': 0
            }
            optimized_ilot['position'] = new_position
            
            # Update geometry
            dimensions = ilot.get('dimensions', {})
            width = dimensions.get('width', 2.0)
            height = dimensions.get('height', 2.0)
            
            optimized_ilot['geometry'] = box(
                new_position['x'] - width / 2,
                new_position['y'] - height / 2,
                new_position['x'] + width / 2,
                new_position['y'] + height / 2
            )
            
            # Mark as optimized
            optimized_ilot['properties'] = optimized_ilot.get('properties', {})
            optimized_ilot['properties']['optimized'] = True
            optimized_ilot['properties']['optimization_method'] = 'spatial_optimizer'
            
            optimized_ilots.append(optimized_ilot)
        
        return optimized_ilots
    
    def _validate_optimization_results(self, optimized_ilots: List[Dict[str, Any]],
                                     spatial_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate optimization results"""
        validated_ilots = []
        
        for ilot in optimized_ilots:
            if self._validate_ilot_placement(ilot, spatial_context):
                validated_ilots.append(ilot)
            else:
                logger.warning(f"Optimized îlot {ilot.get('id', 'unknown')} failed validation")
                # Keep original placement if optimization fails validation
                validated_ilots.append(ilot)
        
        return validated_ilots
    
    def _validate_ilot_placement(self, ilot: Dict[str, Any], 
                               spatial_context: Dict[str, Any]) -> bool:
        """Validate single îlot placement"""
        ilot_geom = ilot.get('geometry')
        if not ilot_geom:
            return False
        
        # Check obstacles
        for obstacle in spatial_context['obstacles']:
            if ilot_geom.intersects(obstacle['geometry']):
                return False
        
        # Check bounds
        bounds = spatial_context['bounds']
        ilot_bounds = ilot_geom.bounds
        
        if (ilot_bounds[0] < bounds['min_x'] or
            ilot_bounds[1] < bounds['min_y'] or
            ilot_bounds[2] > bounds['max_x'] or
            ilot_bounds[3] > bounds['max_y']):
            return False
        
        return True
    
    def _calculate_improvement_metrics(self, original_ilots: List[Dict[str, Any]],
                                     optimized_ilots: List[Dict[str, Any]],
                                     spatial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement metrics"""
        # Create dummy params for fitness evaluation
        original_positions = np.array([
            [ilot['position']['x'], ilot['position']['y']] 
            for ilot in original_ilots
        ])
        
        optimized_positions = np.array([
            [ilot['position']['x'], ilot['position']['y']] 
            for ilot in optimized_ilots
        ])
        
        params = {
            'current_positions': original_positions,
            'ilot_properties': [
                {
                    'id': ilot.get('id', ''),
                    'size_category': ilot.get('size_category', 'medium'),
                    'width': ilot.get('dimensions', {}).get('width', 2.0),
                    'height': ilot.get('dimensions', {}).get('height', 2.0),
                    'area': ilot.get('dimensions', {}).get('area', 4.0)
                }
                for ilot in original_ilots
            ],
            'bounds': spatial_context['bounds'],
            'num_ilots': len(original_ilots),
            'search_space': spatial_context['bounds']
        }
        
        # Calculate fitness scores
        original_fitness = self._evaluate_fitness(original_positions, params, spatial_context)
        optimized_fitness = self._evaluate_fitness(optimized_positions, params, spatial_context)
        
        # Calculate improvement
        improvement = ((optimized_fitness - original_fitness) / abs(original_fitness) * 100) if original_fitness != 0 else 0
        
        return {
            'original_fitness': original_fitness,
            'optimized_fitness': optimized_fitness,
            'overall_improvement': improvement,
            'fitness_improvement': optimized_fitness - original_fitness
        }
