"""
Smart Floor Plan Detector - Phase 1 Component
Intelligent detection of main floor plan from multi-view CAD files
Identifies the primary architectural drawing among layouts, elevations, and details
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
import time

from utils.enhanced_cad_parser import FloorPlanData, CADElement

@dataclass
class FloorPlanCandidate:
    """Candidate floor plan with quality metrics"""
    elements: List[CADElement]
    confidence_score: float
    geometric_metrics: Dict[str, float]
    spatial_bounds: Tuple[float, float, float, float]
    element_density: float
    wall_network_quality: float

class SmartFloorPlanDetector:
    """
    Intelligent detector that identifies the main architectural floor plan
    from complex CAD files containing multiple views and drawings
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Detection thresholds
        self.min_wall_count = 3
        self.min_plan_area = 10000  # mm²
        self.connectivity_threshold = 0.3
        self.density_optimal_range = (0.1, 0.8)  # elements per unit area
        
        # Spatial analysis parameters
        self.grid_size = 1000  # mm - for spatial analysis
        self.cluster_tolerance = 500  # mm
        
    def detect_main_floor_plan(self, floor_plan_data: FloorPlanData) -> FloorPlanData:
        """
        Detect and extract the main floor plan from CAD data
        
        Args:
            floor_plan_data: Raw CAD data with all elements
            
        Returns:
            Optimized FloorPlanData containing only main floor plan elements
        """
        start_time = time.time()
        
        if not floor_plan_data.walls:
            self.logger.warning("No walls found in CAD data")
            return floor_plan_data
        
        # Step 1: Identify floor plan candidates through spatial clustering
        candidates = self._identify_floor_plan_candidates(floor_plan_data)
        
        if not candidates:
            self.logger.warning("No valid floor plan candidates identified")
            return floor_plan_data
        
        # Step 2: Evaluate each candidate
        evaluated_candidates = []
        for candidate in candidates:
            metrics = self._evaluate_floor_plan_candidate(candidate, floor_plan_data)
            evaluated_candidates.append((candidate, metrics))
        
        # Step 3: Select best candidate
        best_candidate = self._select_best_candidate(evaluated_candidates)
        
        if not best_candidate:
            self.logger.warning("Could not determine best floor plan candidate")
            return floor_plan_data
        
        # Step 4: Extract optimized floor plan
        optimized_plan = self._extract_optimized_floor_plan(best_candidate, floor_plan_data)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Floor plan detection completed in {processing_time:.2f}s")
        
        return optimized_plan
    
    def _identify_floor_plan_candidates(self, floor_plan_data: FloorPlanData) -> List[FloorPlanCandidate]:
        """Identify potential floor plan regions through spatial analysis"""
        candidates = []
        
        # Method 1: Wall-based clustering
        wall_clusters = self._cluster_walls_spatially(floor_plan_data.walls)
        
        for cluster in wall_clusters:
            if len(cluster) >= self.min_wall_count:
                candidate = self._create_candidate_from_walls(cluster, floor_plan_data)
                if candidate:
                    candidates.append(candidate)
        
        # Method 2: Layer-based separation
        layer_candidates = self._identify_candidates_by_layer(floor_plan_data)
        candidates.extend(layer_candidates)
        
        # Method 3: Geometric density analysis
        density_candidates = self._identify_candidates_by_density(floor_plan_data)
        candidates.extend(density_candidates)
        
        return self._deduplicate_candidates(candidates)
    
    def _cluster_walls_spatially(self, walls: List[CADElement]) -> List[List[CADElement]]:
        """Cluster walls based on spatial proximity"""
        if not walls:
            return []
        
        # Extract wall centroids
        wall_points = []
        for wall in walls:
            try:
                centroid = wall.geometry.centroid
                wall_points.append((centroid.x, centroid.y, wall))
            except:
                continue
        
        if not wall_points:
            return []
        
        # Simple clustering based on distance
        clusters = []
        used_walls = set()
        
        for point in wall_points:
            if point[2] in used_walls:
                continue
            
            # Start new cluster
            cluster = [point[2]]
            used_walls.add(point[2])
            
            # Find nearby walls
            for other_point in wall_points:
                if other_point[2] in used_walls:
                    continue
                
                distance = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                if distance <= self.cluster_tolerance:
                    cluster.append(other_point[2])
                    used_walls.add(other_point[2])
            
            if len(cluster) >= self.min_wall_count:
                clusters.append(cluster)
        
        return clusters
    
    def _create_candidate_from_walls(self, walls: List[CADElement], 
                                   floor_plan_data: FloorPlanData) -> Optional[FloorPlanCandidate]:
        """Create floor plan candidate from wall cluster"""
        try:
            # Calculate spatial bounds
            all_coords = []
            for wall in walls:
                try:
                    coords = list(wall.geometry.coords)
                    all_coords.extend(coords)
                except:
                    continue
            
            if not all_coords:
                return None
            
            x_coords = [coord[0] for coord in all_coords]
            y_coords = [coord[1] for coord in all_coords]
            
            bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            
            if area < self.min_plan_area:
                return None
            
            # Find associated elements within bounds
            associated_elements = self._find_elements_in_bounds(bounds, floor_plan_data)
            
            # Calculate metrics
            wall_network_quality = self._calculate_wall_network_quality(walls)
            element_density = len(associated_elements) / area if area > 0 else 0
            
            # Calculate confidence score
            confidence = self._calculate_candidate_confidence(
                walls, associated_elements, area, wall_network_quality, element_density
            )
            
            return FloorPlanCandidate(
                elements=associated_elements,
                confidence_score=confidence,
                geometric_metrics={
                    'area': area,
                    'wall_count': len(walls),
                    'total_elements': len(associated_elements)
                },
                spatial_bounds=bounds,
                element_density=element_density,
                wall_network_quality=wall_network_quality
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating candidate from walls: {str(e)}")
            return None
    
    def _find_elements_in_bounds(self, bounds: Tuple[float, float, float, float], 
                               floor_plan_data: FloorPlanData) -> List[CADElement]:
        """Find all elements within spatial bounds"""
        elements = []
        
        # Check all element types
        all_elements = (
            floor_plan_data.walls + floor_plan_data.doors + floor_plan_data.windows +
            floor_plan_data.openings + floor_plan_data.text_annotations
        )
        
        for element in all_elements:
            try:
                # Check if element geometry intersects with bounds
                bounds_polygon = Polygon([
                    (bounds[0], bounds[1]), (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]), (bounds[0], bounds[3])
                ])
                
                if element.geometry.intersects(bounds_polygon):
                    elements.append(element)
                    
            except:
                continue
        
        return elements
    
    def _identify_candidates_by_layer(self, floor_plan_data: FloorPlanData) -> List[FloorPlanCandidate]:
        """Identify candidates based on layer organization"""
        candidates = []
        
        # Group elements by layer
        layer_elements = {}
        all_elements = (
            floor_plan_data.walls + floor_plan_data.doors + floor_plan_data.windows +
            floor_plan_data.openings + floor_plan_data.text_annotations
        )
        
        for element in all_elements:
            layer = element.layer or 'DEFAULT'
            if layer not in layer_elements:
                layer_elements[layer] = []
            layer_elements[layer].append(element)
        
        # Evaluate each layer as potential floor plan
        for layer, elements in layer_elements.items():
            walls_in_layer = [e for e in elements if e.element_type == 'wall']
            
            if len(walls_in_layer) >= self.min_wall_count:
                candidate = self._create_candidate_from_elements(elements)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _identify_candidates_by_density(self, floor_plan_data: FloorPlanData) -> List[FloorPlanCandidate]:
        """Identify candidates based on element density analysis"""
        candidates = []
        
        if not floor_plan_data.walls:
            return candidates
        
        # Create spatial grid for density analysis
        all_coords = []
        for wall in floor_plan_data.walls:
            try:
                coords = list(wall.geometry.coords)
                all_coords.extend(coords)
            except:
                continue
        
        if not all_coords:
            return candidates
        
        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Create grid cells
        grid_cells = []
        x_step = self.grid_size
        y_step = self.grid_size
        
        x = min_x
        while x < max_x:
            y = min_y
            while y < max_y:
                cell_bounds = (x, y, x + x_step, y + y_step)
                elements_in_cell = self._find_elements_in_bounds(cell_bounds, floor_plan_data)
                
                if len(elements_in_cell) > 0:
                    walls_in_cell = [e for e in elements_in_cell if e.element_type == 'wall']
                    if len(walls_in_cell) >= self.min_wall_count:
                        grid_cells.append((cell_bounds, elements_in_cell))
                
                y += y_step
            x += x_step
        
        # Merge adjacent high-density cells
        merged_regions = self._merge_adjacent_cells(grid_cells)
        
        for region_bounds, region_elements in merged_regions:
            candidate = self._create_candidate_from_elements(region_elements)
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _create_candidate_from_elements(self, elements: List[CADElement]) -> Optional[FloorPlanCandidate]:
        """Create candidate from list of elements"""
        walls = [e for e in elements if e.element_type == 'wall']
        
        if len(walls) < self.min_wall_count:
            return None
        
        return self._create_candidate_from_walls(walls, 
                                               type('FloorPlan', (), {
                                                   'walls': walls,
                                                   'doors': [e for e in elements if e.element_type == 'door'],
                                                   'windows': [e for e in elements if e.element_type == 'window'],
                                                   'openings': [e for e in elements if e.element_type in ['opening', 'door', 'window']],
                                                   'text_annotations': [e for e in elements if e.element_type == 'text']
                                               })())
    
    def _merge_adjacent_cells(self, grid_cells: List[Tuple]) -> List[Tuple]:
        """Merge adjacent grid cells with high element density"""
        # Simplified implementation - could be enhanced with more sophisticated clustering
        return grid_cells
    
    def _calculate_wall_network_quality(self, walls: List[CADElement]) -> float:
        """Calculate quality of wall network connectivity"""
        if len(walls) < 2:
            return 0.0
        
        connection_count = 0
        total_possible = len(walls) * (len(walls) - 1) // 2
        
        for i, wall1 in enumerate(walls):
            for wall2 in walls[i+1:]:
                if self._walls_are_connected(wall1, wall2):
                    connection_count += 1
        
        return connection_count / max(total_possible, 1)
    
    def _walls_are_connected(self, wall1: CADElement, wall2: CADElement) -> bool:
        """Check if two walls are geometrically connected"""
        try:
            # Check if walls share endpoints or intersect
            distance = wall1.geometry.distance(wall2.geometry)
            return distance <= 100  # 100mm tolerance
        except:
            return False
    
    def _calculate_candidate_confidence(self, walls: List[CADElement], elements: List[CADElement],
                                      area: float, wall_quality: float, density: float) -> float:
        """Calculate overall confidence score for candidate"""
        factors = []
        
        # Wall count factor
        wall_count = len(walls)
        if wall_count >= 10:
            factors.append(1.0)
        elif wall_count >= 5:
            factors.append(0.8)
        elif wall_count >= 3:
            factors.append(0.6)
        else:
            factors.append(0.3)
        
        # Area factor
        if area >= 100000:  # Large room/building
            factors.append(0.9)
        elif area >= 50000:  # Medium room
            factors.append(0.8)
        elif area >= self.min_plan_area:  # Minimum area
            factors.append(0.6)
        else:
            factors.append(0.3)
        
        # Wall network quality
        factors.append(wall_quality)
        
        # Element density (should be in optimal range)
        if self.density_optimal_range[0] <= density <= self.density_optimal_range[1]:
            factors.append(0.8)
        else:
            factors.append(0.4)
        
        # Element diversity factor
        element_types = set(e.element_type for e in elements)
        diversity_score = len(element_types) / 5.0  # Normalize by expected types
        factors.append(min(diversity_score, 1.0))
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _evaluate_floor_plan_candidate(self, candidate: FloorPlanCandidate, 
                                     floor_plan_data: FloorPlanData) -> Dict[str, float]:
        """Evaluate candidate with additional metrics"""
        metrics = {}
        
        # Geometric coherence
        metrics['geometric_coherence'] = self._calculate_geometric_coherence(candidate)
        
        # Architectural completeness
        metrics['architectural_completeness'] = self._calculate_architectural_completeness(candidate)
        
        # Scale appropriateness
        metrics['scale_appropriateness'] = self._calculate_scale_appropriateness(candidate)
        
        # Layout regularity
        metrics['layout_regularity'] = self._calculate_layout_regularity(candidate)
        
        return metrics
    
    def _calculate_geometric_coherence(self, candidate: FloorPlanCandidate) -> float:
        """Calculate how well elements form coherent geometric structure"""
        walls = [e for e in candidate.elements if e.element_type == 'wall']
        
        if len(walls) < 3:
            return 0.0
        
        # Check for closed regions
        try:
            # Create union of all wall geometries
            wall_union = unary_union([wall.geometry for wall in walls])
            
            # Simple coherence measure based on geometry complexity
            if hasattr(wall_union, 'bounds'):
                bounds = wall_union.bounds
                area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                perimeter = wall_union.length
                
                # Shape regularity (closer to square/rectangle is better)
                if area > 0 and perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    return min(compactness * 2, 1.0)  # Normalize
            
            return 0.5
            
        except:
            return 0.3
    
    def _calculate_architectural_completeness(self, candidate: FloorPlanCandidate) -> float:
        """Calculate architectural completeness (walls, openings, annotations)"""
        element_types = set(e.element_type for e in candidate.elements)
        
        expected_types = {'wall', 'door', 'window', 'text'}
        present_types = element_types.intersection(expected_types)
        
        return len(present_types) / len(expected_types)
    
    def _calculate_scale_appropriateness(self, candidate: FloorPlanCandidate) -> float:
        """Calculate if scale is appropriate for architectural floor plan"""
        area = candidate.geometric_metrics.get('area', 0)
        
        # Typical room/building areas (in mm²)
        if 10000 <= area <= 10000000:  # 0.01m² to 10,000m²
            return 1.0
        elif 5000 <= area <= 50000000:  # Extended range
            return 0.8
        else:
            return 0.3
    
    def _calculate_layout_regularity(self, candidate: FloorPlanCandidate) -> float:
        """Calculate layout regularity (orthogonal walls, regular spacing)"""
        walls = [e for e in candidate.elements if e.element_type == 'wall']
        
        if len(walls) < 2:
            return 0.0
        
        # Check wall angle distribution
        angles = []
        for wall in walls:
            try:
                coords = list(wall.geometry.coords)
                if len(coords) >= 2:
                    dx = coords[-1][0] - coords[0][0]
                    dy = coords[-1][1] - coords[0][1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    # Normalize to 0-90 degrees
                    angle = abs(angle) % 90
                    angles.append(angle)
            except:
                continue
        
        if not angles:
            return 0.5
        
        # Check for orthogonal dominance (angles near 0 or 90 degrees)
        orthogonal_count = sum(1 for angle in angles if angle < 10 or angle > 80)
        orthogonal_ratio = orthogonal_count / len(angles)
        
        return orthogonal_ratio
    
    def _select_best_candidate(self, evaluated_candidates: List[Tuple]) -> Optional[FloorPlanCandidate]:
        """Select the best floor plan candidate based on comprehensive scoring"""
        if not evaluated_candidates:
            return None
        
        best_candidate = None
        best_score = -1
        
        for candidate, metrics in evaluated_candidates:
            # Combine all scores
            total_score = (
                candidate.confidence_score * 0.4 +
                metrics.get('geometric_coherence', 0) * 0.2 +
                metrics.get('architectural_completeness', 0) * 0.2 +
                metrics.get('scale_appropriateness', 0) * 0.1 +
                metrics.get('layout_regularity', 0) * 0.1
            )
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
        
        return best_candidate
    
    def _extract_optimized_floor_plan(self, candidate: FloorPlanCandidate, 
                                    original_data: FloorPlanData) -> FloorPlanData:
        """Extract and optimize the selected floor plan"""
        # Create new FloorPlanData with selected elements
        optimized = FloorPlanData()
        
        # Separate elements by type
        for element in candidate.elements:
            if element.element_type == 'wall':
                optimized.walls.append(element)
            elif element.element_type == 'door':
                optimized.doors.append(element)
            elif element.element_type == 'window':
                optimized.windows.append(element)
            elif element.element_type in ['opening']:
                optimized.openings.append(element)
            elif element.element_type == 'text':
                optimized.text_annotations.append(element)
        
        # Copy metadata from original
        optimized.scale_factor = original_data.scale_factor
        optimized.units = original_data.units
        optimized.drawing_bounds = candidate.spatial_bounds
        
        # Update quality metrics
        optimized.element_count = len(candidate.elements)
        optimized.wall_connectivity = candidate.wall_network_quality
        optimized.processing_confidence = candidate.confidence_score
        
        return optimized
    
    def _deduplicate_candidates(self, candidates: List[FloorPlanCandidate]) -> List[FloorPlanCandidate]:
        """Remove duplicate or highly overlapping candidates"""
        if len(candidates) <= 1:
            return candidates
        
        unique_candidates = []
        
        for candidate in candidates:
            is_duplicate = False
            
            for existing in unique_candidates:
                # Check spatial overlap
                overlap = self._calculate_spatial_overlap(candidate, existing)
                if overlap > 0.8:  # 80% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _calculate_spatial_overlap(self, candidate1: FloorPlanCandidate, 
                                 candidate2: FloorPlanCandidate) -> float:
        """Calculate spatial overlap between two candidates"""
        try:
            bounds1 = candidate1.spatial_bounds
            bounds2 = candidate2.spatial_bounds
            
            # Calculate intersection area
            x_overlap = max(0, min(bounds1[2], bounds2[2]) - max(bounds1[0], bounds2[0]))
            y_overlap = max(0, min(bounds1[3], bounds2[3]) - max(bounds1[1], bounds2[1]))
            intersection_area = x_overlap * y_overlap
            
            # Calculate union area
            area1 = (bounds1[2] - bounds1[0]) * (bounds1[3] - bounds1[1])
            area2 = (bounds2[2] - bounds2[0]) * (bounds2[3] - bounds2[1])
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
            
        except:
            return 0.0