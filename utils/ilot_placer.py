import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import logging
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IlotPlacer:
    """Intelligent îlot placement with spatial optimization"""
    
    def __init__(self):
        self.placement_algorithms = {
            'grid': self._place_grid_pattern,
            'random': self._place_random_pattern,
            'optimized': self._place_optimized_pattern,
            'cluster': self._place_cluster_pattern
        }
        
        self.size_categories = {
            'small': {'min_size': 1.0, 'max_size': 3.0},
            'medium': {'min_size': 3.0, 'max_size': 6.0},
            'large': {'min_size': 6.0, 'max_size': 12.0}
        }
    
    def place_ilots(self, analysis_results: Dict[str, Any], 
                    configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Place îlots based on analysis results and configuration
        
        Args:
            analysis_results: Results from zone analysis
            configuration: Îlot configuration parameters
            
        Returns:
            List of placed îlots with positions and properties
        """
        logger.info("Starting îlot placement")
        
        try:
            # Extract placement parameters
            size_distribution = configuration.get('size_distribution', {})
            dimensions = configuration.get('dimensions', {})
            spacing = configuration.get('spacing', {})
            
            # Get available spaces
            available_spaces = self._get_available_spaces(analysis_results)
            
            if not available_spaces:
                logger.warning("No available spaces found for îlot placement")
                return []
            
            # Calculate îlot requirements
            ilot_requirements = self._calculate_ilot_requirements(
                available_spaces, size_distribution, dimensions
            )
            
            # Place îlots using optimized algorithm
            placed_ilots = self._place_optimized_pattern(
                available_spaces, ilot_requirements, spacing
            )
            
            # Validate placements
            validated_ilots = self._validate_placements(
                placed_ilots, analysis_results, spacing
            )
            
            logger.info(f"Successfully placed {len(validated_ilots)} îlots")
            return validated_ilots
            
        except Exception as e:
            logger.error(f"Error during îlot placement: {str(e)}")
            raise
    
    def _get_available_spaces(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract available spaces for îlot placement"""
        open_spaces = analysis_results.get('open_spaces', [])
        restricted_areas = analysis_results.get('restricted_areas', [])
        
        available_spaces = []
        
        for space in open_spaces:
            space_geom = space.get('shapely_geom')
            if not space_geom:
                continue
            
            # Subtract restricted areas from open space
            available_geom = space_geom
            
            for restricted in restricted_areas:
                restricted_geom = restricted.get('shapely_geom')
                if restricted_geom and space_geom.intersects(restricted_geom):
                    available_geom = available_geom.difference(restricted_geom)
            
            # Only consider spaces with sufficient area
            if available_geom.area > 4.0:  # Minimum 4 square meters
                available_spaces.append({
                    'id': space['id'],
                    'original_space': space,
                    'available_geom': available_geom,
                    'available_area': available_geom.area,
                    'bounds': available_geom.bounds,
                    'center': {
                        'x': available_geom.centroid.x,
                        'y': available_geom.centroid.y,
                        'z': 0
                    }
                })
        
        return available_spaces
    
    def _calculate_ilot_requirements(self, available_spaces: List[Dict[str, Any]],
                                   size_distribution: Dict[str, int],
                                   dimensions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate îlot requirements based on configuration"""
        total_available_area = sum(space['available_area'] for space in available_spaces)
        
        # Calculate target coverage (typically 60-70% of available space)
        target_coverage = total_available_area * 0.65
        
        # Calculate number of îlots for each size category
        ilot_requirements = []
        
        for size_category, percentage in size_distribution.items():
            if percentage <= 0:
                continue
            
            # Calculate area for this size category
            category_area = target_coverage * (percentage / 100)
            
            # Get îlot dimensions
            ilot_size = dimensions.get(size_category, 2.0)
            ilot_area = ilot_size * ilot_size
            
            # Calculate number of îlots
            num_ilots = max(1, int(category_area / ilot_area))
            
            for i in range(num_ilots):
                ilot_requirements.append({
                    'id': f"{size_category}_{i}",
                    'size_category': size_category,
                    'dimensions': {
                        'width': ilot_size,
                        'height': ilot_size,
                        'area': ilot_area
                    },
                    'placed': False
                })
        
        return ilot_requirements
    
    def _place_optimized_pattern(self, available_spaces: List[Dict[str, Any]],
                               ilot_requirements: List[Dict[str, Any]],
                               spacing: Dict[str, float]) -> List[Dict[str, Any]]:
        """Place îlots using optimized spatial distribution"""
        placed_ilots = []
        min_spacing = spacing.get('min_spacing', 1.5)
        wall_clearance = spacing.get('wall_clearance', 1.0)
        
        # Sort îlots by size (largest first)
        sorted_ilots = sorted(ilot_requirements, 
                            key=lambda x: x['dimensions']['area'], 
                            reverse=True)
        
        # Sort spaces by area (largest first)
        sorted_spaces = sorted(available_spaces, 
                             key=lambda x: x['available_area'], 
                             reverse=True)
        
        for ilot in sorted_ilots:
            best_position = self._find_best_position(
                ilot, sorted_spaces, placed_ilots, min_spacing, wall_clearance
            )
            
            if best_position:
                placed_ilot = self._create_placed_ilot(ilot, best_position)
                placed_ilots.append(placed_ilot)
        
        return placed_ilots
    
    def _find_best_position(self, ilot: Dict[str, Any],
                          available_spaces: List[Dict[str, Any]],
                          placed_ilots: List[Dict[str, Any]],
                          min_spacing: float,
                          wall_clearance: float) -> Optional[Dict[str, Any]]:
        """Find the best position for an îlot"""
        best_position = None
        best_score = -1
        
        ilot_width = ilot['dimensions']['width']
        ilot_height = ilot['dimensions']['height']
        
        for space in available_spaces:
            space_geom = space['available_geom']
            bounds = space['bounds']
            
            # Generate candidate positions within the space
            candidates = self._generate_candidate_positions(
                bounds, ilot_width, ilot_height, wall_clearance
            )
            
            for candidate in candidates:
                # Create îlot geometry at candidate position
                ilot_geom = box(
                    candidate['x'] - ilot_width / 2,
                    candidate['y'] - ilot_height / 2,
                    candidate['x'] + ilot_width / 2,
                    candidate['y'] + ilot_height / 2
                )
                
                # Check if îlot fits within space
                if not space_geom.contains(ilot_geom):
                    continue
                
                # Check spacing constraints with other îlots
                if not self._check_spacing_constraints(
                    ilot_geom, placed_ilots, min_spacing
                ):
                    continue
                
                # Calculate position score
                score = self._calculate_position_score(
                    candidate, ilot_geom, space, placed_ilots
                )
                
                if score > best_score:
                    best_score = score
                    best_position = {
                        'position': candidate,
                        'space_id': space['id'],
                        'geometry': ilot_geom,
                        'score': score
                    }
        
        return best_position
    
    def _generate_candidate_positions(self, bounds: Tuple[float, float, float, float],
                                    width: float, height: float,
                                    wall_clearance: float) -> List[Dict[str, float]]:
        """Generate candidate positions within space bounds"""
        min_x, min_y, max_x, max_y = bounds
        
        # Adjust bounds for wall clearance and îlot size
        adjusted_min_x = min_x + wall_clearance + width / 2
        adjusted_min_y = min_y + wall_clearance + height / 2
        adjusted_max_x = max_x - wall_clearance - width / 2
        adjusted_max_y = max_y - wall_clearance - height / 2
        
        if adjusted_min_x >= adjusted_max_x or adjusted_min_y >= adjusted_max_y:
            return []
        
        # Generate grid of candidate positions
        candidates = []
        
        # Grid resolution based on space size
        grid_resolution = min(2.0, (adjusted_max_x - adjusted_min_x) / 10)
        
        x = adjusted_min_x
        while x <= adjusted_max_x:
            y = adjusted_min_y
            while y <= adjusted_max_y:
                candidates.append({'x': x, 'y': y, 'z': 0})
                y += grid_resolution
            x += grid_resolution
        
        return candidates
    
    def _check_spacing_constraints(self, ilot_geom: Polygon,
                                 placed_ilots: List[Dict[str, Any]],
                                 min_spacing: float) -> bool:
        """Check if îlot meets spacing constraints"""
        for placed in placed_ilots:
            placed_geom = placed.get('geometry')
            if placed_geom and ilot_geom.distance(placed_geom) < min_spacing:
                return False
        
        return True
    
    def _calculate_position_score(self, position: Dict[str, float],
                                ilot_geom: Polygon,
                                space: Dict[str, Any],
                                placed_ilots: List[Dict[str, Any]]) -> float:
        """Calculate score for îlot position"""
        score = 0.0
        
        # Distance from space center (prefer central locations)
        space_center = space['center']
        distance_to_center = np.sqrt(
            (position['x'] - space_center['x'])**2 +
            (position['y'] - space_center['y'])**2
        )
        
        # Normalize distance score (closer to center is better)
        max_distance = max(
            space['bounds'][2] - space['bounds'][0],
            space['bounds'][3] - space['bounds'][1]
        )
        
        if max_distance > 0:
            distance_score = 1.0 - (distance_to_center / max_distance)
            score += distance_score * 30  # Weight: 30%
        
        # Accessibility score (distance to entrances)
        # This would be calculated based on entrance positions
        accessibility_score = 0.8  # Placeholder
        score += accessibility_score * 20  # Weight: 20%
        
        # Distribution score (avoid clustering)
        if placed_ilots:
            min_distance_to_existing = min(
                ilot_geom.distance(placed['geometry'])
                for placed in placed_ilots
                if placed.get('geometry')
            )
            
            # Prefer positions with reasonable distance to existing îlots
            if min_distance_to_existing > 3.0:
                distribution_score = 1.0
            elif min_distance_to_existing > 1.5:
                distribution_score = 0.7
            else:
                distribution_score = 0.3
            
            score += distribution_score * 30  # Weight: 30%
        else:
            score += 30  # First îlot gets full distribution score
        
        # Space utilization score
        space_utilization = ilot_geom.area / space['available_area']
        utilization_score = min(1.0, space_utilization * 5)  # Favor efficient use
        score += utilization_score * 20  # Weight: 20%
        
        return score
    
    def _create_placed_ilot(self, ilot: Dict[str, Any], 
                          position: Dict[str, Any]) -> Dict[str, Any]:
        """Create a placed îlot with all properties"""
        pos = position['position']
        
        return {
            'id': ilot['id'],
            'size_category': ilot['size_category'],
            'dimensions': ilot['dimensions'],
            'position': pos,
            'geometry': position['geometry'],
            'space_id': position['space_id'],
            'placement_score': position['score'],
            'rotation': 0,  # Default rotation
            'area': ilot['dimensions']['area'],
            'accessibility_score': self._calculate_accessibility_score(pos),
            'properties': {
                'placed_at': 'current_time',
                'algorithm': 'optimized',
                'validated': True
            }
        }
    
    def _calculate_accessibility_score(self, position: Dict[str, float]) -> float:
        """Calculate accessibility score for position"""
        # This would be calculated based on distance to entrances,
        # corridors, and other accessibility factors
        return 85.0  # Placeholder
    
    def _validate_placements(self, placed_ilots: List[Dict[str, Any]],
                           analysis_results: Dict[str, Any],
                           spacing: Dict[str, float]) -> List[Dict[str, Any]]:
        """Validate all îlot placements"""
        validated_ilots = []
        
        walls = analysis_results.get('walls', [])
        restricted_areas = analysis_results.get('restricted_areas', [])
        
        for ilot in placed_ilots:
            if self._validate_single_placement(ilot, walls, restricted_areas, spacing):
                validated_ilots.append(ilot)
            else:
                logger.warning(f"Îlot {ilot['id']} failed validation")
        
        return validated_ilots
    
    def _validate_single_placement(self, ilot: Dict[str, Any],
                                 walls: List[Dict[str, Any]],
                                 restricted_areas: List[Dict[str, Any]],
                                 spacing: Dict[str, float]) -> bool:
        """Validate a single îlot placement"""
        ilot_geom = ilot.get('geometry')
        if not ilot_geom:
            return False
        
        wall_clearance = spacing.get('wall_clearance', 1.0)
        
        # Check wall clearance
        for wall in walls:
            wall_geom = wall.get('shapely_geom')
            if wall_geom and ilot_geom.distance(wall_geom) < wall_clearance:
                return False
        
        # Check restricted areas
        for restricted in restricted_areas:
            restricted_geom = restricted.get('shapely_geom')
            if restricted_geom and ilot_geom.intersects(restricted_geom):
                return False
        
        return True
    
    def _place_grid_pattern(self, available_spaces: List[Dict[str, Any]],
                          ilot_requirements: List[Dict[str, Any]],
                          spacing: Dict[str, float]) -> List[Dict[str, Any]]:
        """Place îlots in a grid pattern"""
        # Implementation for grid pattern placement
        return []
    
    def _place_random_pattern(self, available_spaces: List[Dict[str, Any]],
                            ilot_requirements: List[Dict[str, Any]],
                            spacing: Dict[str, float]) -> List[Dict[str, Any]]:
        """Place îlots randomly with constraints"""
        # Implementation for random pattern placement
        return []
    
    def _place_cluster_pattern(self, available_spaces: List[Dict[str, Any]],
                             ilot_requirements: List[Dict[str, Any]],
                             spacing: Dict[str, float]) -> List[Dict[str, Any]]:
        """Place îlots in cluster patterns"""
        # Implementation for cluster pattern placement
        return []
