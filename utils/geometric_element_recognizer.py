"""
Geometric Element Recognizer - Phase 1 Component
Advanced geometric analysis for architectural element recognition
Detects wall thickness, opening types, connectivity patterns, and spatial relationships
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
import time

from utils.enhanced_cad_parser import FloorPlanData, CADElement

@dataclass
class GeometricAnalysis:
    """Results of geometric element analysis"""
    enhanced_walls: List[CADElement]
    wall_thickness_map: Dict[str, float]
    connectivity_graph: Dict[str, List[str]]
    opening_associations: Dict[str, List[str]]
    spatial_relationships: Dict[str, Any]
    quality_score: float

class GeometricElementRecognizer:
    """
    Advanced geometric analyzer that recognizes architectural patterns
    and enhances element detection with spatial relationship analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.wall_thickness_tolerance = 200  # mm
        self.connection_tolerance = 50  # mm
        self.parallel_tolerance = 5  # degrees
        self.perpendicular_tolerance = 5  # degrees
        
        # Pattern recognition thresholds
        self.min_room_area = 5000  # mm²
        self.typical_door_width = (600, 1200)  # mm
        self.typical_window_width = (800, 2000)  # mm

    def analyze_geometric_elements(self, floor_plan_data: FloorPlanData) -> GeometricAnalysis:
        """
        Perform comprehensive geometric analysis of floor plan elements
        
        Args:
            floor_plan_data: Input floor plan data
            
        Returns:
            GeometricAnalysis with enhanced element information
        """
        start_time = time.time()
        
        # Step 1: Analyze wall geometry and thickness
        enhanced_walls, thickness_map = self._analyze_wall_geometry(floor_plan_data.walls)
        
        # Step 2: Build connectivity graph
        connectivity = self._build_connectivity_graph(enhanced_walls)
        
        # Step 3: Associate openings with walls
        opening_associations = self._associate_openings_with_walls(
            enhanced_walls, floor_plan_data.doors + floor_plan_data.windows
        )
        
        # Step 4: Analyze spatial relationships
        spatial_relationships = self._analyze_spatial_relationships(floor_plan_data)
        
        # Step 5: Calculate quality score
        quality_score = self._calculate_analysis_quality(
            enhanced_walls, connectivity, opening_associations
        )
        
        processing_time = time.time() - start_time
        self.logger.info(f"Geometric analysis completed in {processing_time:.2f}s")
        
        return GeometricAnalysis(
            enhanced_walls=enhanced_walls,
            wall_thickness_map=thickness_map,
            connectivity_graph=connectivity,
            opening_associations=opening_associations,
            spatial_relationships=spatial_relationships,
            quality_score=quality_score
        )
    
    def _analyze_wall_geometry(self, walls: List[CADElement]) -> Tuple[List[CADElement], Dict[str, float]]:
        """Analyze wall geometry to detect thickness and enhance properties"""
        enhanced_walls = []
        thickness_map = {}
        
        for i, wall in enumerate(walls):
            wall_id = f"wall_{i}"
            
            # Detect wall thickness by analyzing nearby parallel walls
            thickness = self._detect_wall_thickness(wall, walls)
            
            # Enhance wall element with additional properties
            enhanced_wall = CADElement(
                element_type=wall.element_type,
                geometry=wall.geometry,
                layer=wall.layer,
                color=wall.color,
                thickness=thickness,
                properties={
                    **wall.properties,
                    'wall_id': wall_id,
                    'detected_thickness': thickness,
                    'geometric_analysis': self._analyze_wall_shape(wall),
                    'structural_type': self._classify_wall_type(wall, thickness)
                }
            )
            
            enhanced_walls.append(enhanced_wall)
            thickness_map[wall_id] = thickness
        
        return enhanced_walls, thickness_map
    
    def _detect_wall_thickness(self, target_wall: CADElement, all_walls: List[CADElement]) -> float:
        """Detect wall thickness by finding parallel walls"""
        try:
            target_coords = list(target_wall.geometry.coords)
            if len(target_coords) < 2:
                return 150  # Default thickness
            
            # Calculate wall direction
            target_direction = np.array([
                target_coords[-1][0] - target_coords[0][0],
                target_coords[-1][1] - target_coords[0][1]
            ])
            target_length = np.linalg.norm(target_direction)
            
            if target_length == 0:
                return 150
            
            target_direction = target_direction / target_length
            
            # Find parallel walls
            parallel_distances = []
            
            for i, other_wall in enumerate(all_walls):
                if other_wall is target_wall:
                    continue
                
                try:
                    other_coords = list(other_wall.geometry.coords)
                    if len(other_coords) < 2:
                        continue
                    
                    # Calculate other wall direction
                    other_direction = np.array([
                        other_coords[-1][0] - other_coords[0][0],
                        other_coords[-1][1] - other_coords[0][1]
                    ])
                    other_length = np.linalg.norm(other_direction)
                    
                    if other_length == 0:
                        continue
                    
                    other_direction = other_direction / other_length
                    
                    # Check if walls are parallel
                    dot_product = abs(np.dot(target_direction, other_direction))
                    angle = np.arccos(np.clip(dot_product, 0, 1)) * 180 / np.pi
                    
                    if angle <= self.parallel_tolerance or angle >= (180 - self.parallel_tolerance):
                        # Calculate distance between parallel walls
                        distance = target_wall.geometry.distance(other_wall.geometry)
                        if 50 <= distance <= self.wall_thickness_tolerance:
                            parallel_distances.append(distance)
                
                except:
                    continue
            
            # Use median distance as thickness estimate
            if parallel_distances:
                return np.median(parallel_distances)
            else:
                return 150  # Default thickness
                
        except Exception as e:
            self.logger.warning(f"Error detecting wall thickness: {str(e)}")
            return 150
    
    def _analyze_wall_shape(self, wall: CADElement) -> Dict[str, Any]:
        """Analyze wall shape and geometric properties"""
        try:
            coords = list(wall.geometry.coords)
            
            analysis = {
                'length': wall.geometry.length,
                'endpoint_count': len(coords),
                'is_straight': len(coords) == 2,
                'is_curved': len(coords) > 2,
                'bounding_box': wall.geometry.bounds
            }
            
            if len(coords) >= 2:
                # Calculate wall angle
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                analysis['angle'] = angle
                
                # Determine orientation
                if abs(angle) <= 10 or abs(angle) >= 170:
                    analysis['orientation'] = 'horizontal'
                elif 80 <= abs(angle) <= 100:
                    analysis['orientation'] = 'vertical'
                else:
                    analysis['orientation'] = 'diagonal'
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing wall shape: {str(e)}")
            return {'length': 0, 'is_straight': True}
    
    def _classify_wall_type(self, wall: CADElement, thickness: float) -> str:
        """Classify wall type based on thickness and properties"""
        if thickness >= 200:
            return 'structural'
        elif thickness >= 100:
            return 'partition'
        else:
            return 'light_partition'
    
    def _build_connectivity_graph(self, walls: List[CADElement]) -> Dict[str, List[str]]:
        """Build connectivity graph showing which walls connect to each other"""
        connectivity = {}
        
        for wall in walls:
            wall_id = wall.properties.get('wall_id', 'unknown')
            connectivity[wall_id] = []
        
        # Check each pair of walls for connections
        for i, wall1 in enumerate(walls):
            wall1_id = wall1.properties.get('wall_id', f'wall_{i}')
            
            for j, wall2 in enumerate(walls):
                if i >= j:  # Avoid duplicate checks
                    continue
                
                wall2_id = wall2.properties.get('wall_id', f'wall_{j}')
                
                if self._walls_are_connected(wall1, wall2):
                    connectivity[wall1_id].append(wall2_id)
                    connectivity[wall2_id].append(wall1_id)
        
        return connectivity
    
    def _walls_are_connected(self, wall1: CADElement, wall2: CADElement) -> bool:
        """Check if two walls are geometrically connected"""
        try:
            # Check if walls share endpoints or intersect
            distance = wall1.geometry.distance(wall2.geometry)
            
            if distance <= self.connection_tolerance:
                return True
            
            # Check for endpoint proximity
            coords1 = list(wall1.geometry.coords)
            coords2 = list(wall2.geometry.coords)
            
            if len(coords1) < 2 or len(coords2) < 2:
                return False
            
            endpoints1 = [coords1[0], coords1[-1]]
            endpoints2 = [coords2[0], coords2[-1]]
            
            for ep1 in endpoints1:
                for ep2 in endpoints2:
                    dist = np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                    if dist <= self.connection_tolerance:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking wall connection: {str(e)}")
            return False
    
    def _associate_openings_with_walls(self, walls: List[CADElement], 
                                     openings: List[CADElement]) -> Dict[str, List[str]]:
        """Associate doors and windows with their corresponding walls"""
        associations = {}
        
        for wall in walls:
            wall_id = wall.properties.get('wall_id', 'unknown')
            associations[wall_id] = []
        
        for i, opening in enumerate(openings):
            opening_id = f"{opening.element_type}_{i}"
            
            # Find closest wall to opening
            closest_wall = None
            min_distance = float('inf')
            
            for wall in walls:
                try:
                    distance = opening.geometry.distance(wall.geometry)
                    if distance < min_distance:
                        min_distance = distance
                        closest_wall = wall
                except:
                    continue
            
            # Associate if close enough
            if closest_wall and min_distance <= 500:  # 500mm threshold
                wall_id = closest_wall.properties.get('wall_id', 'unknown')
                if wall_id in associations:
                    associations[wall_id].append(opening_id)
        
        return associations
    
    def _analyze_spatial_relationships(self, floor_plan_data: FloorPlanData) -> Dict[str, Any]:
        """Analyze spatial relationships between elements"""
        relationships = {
            'room_boundaries': [],
            'circulation_paths': [],
            'structural_grid': {},
            'accessibility_analysis': {}
        }
        
        try:
            # Detect room boundaries
            if floor_plan_data.walls:
                relationships['room_boundaries'] = self._detect_room_boundaries(floor_plan_data.walls)
            
            # Analyze circulation paths
            if floor_plan_data.doors:
                relationships['circulation_paths'] = self._analyze_circulation_paths(
                    floor_plan_data.walls, floor_plan_data.doors
                )
            
            # Detect structural grid patterns
            relationships['structural_grid'] = self._detect_structural_grid(floor_plan_data.walls)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing spatial relationships: {str(e)}")
        
        return relationships
    
    def _detect_room_boundaries(self, walls: List[CADElement]) -> List[Dict[str, Any]]:
        """Detect room boundaries from wall network"""
        rooms = []
        
        try:
            # Simple room detection based on wall connectivity
            # This is a simplified implementation - could be enhanced with more sophisticated algorithms
            
            if len(walls) >= 4:  # Minimum walls for a room
                # Find potential room centers and analyze surrounding walls
                all_coords = []
                for wall in walls:
                    coords = list(wall.geometry.coords)
                    all_coords.extend(coords)
                
                if all_coords:
                    x_coords = [c[0] for c in all_coords]
                    y_coords = [c[1] for c in all_coords]
                    
                    # Simple room detection: create a basic rectangular room
                    room = {
                        'bounds': (min(x_coords), min(y_coords), max(x_coords), max(y_coords)),
                        'area': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)),
                        'wall_count': len(walls),
                        'type': 'detected_space'
                    }
                    
                    if room['area'] >= self.min_room_area:
                        rooms.append(room)
            
        except Exception as e:
            self.logger.warning(f"Error detecting room boundaries: {str(e)}")
        
        return rooms
    
    def _analyze_circulation_paths(self, walls: List[CADElement], 
                                 doors: List[CADElement]) -> List[Dict[str, Any]]:
        """Analyze circulation paths based on door locations"""
        paths = []
        
        try:
            for door in doors:
                # Find walls adjacent to door
                adjacent_walls = []
                for wall in walls:
                    if door.geometry.distance(wall.geometry) <= 100:  # 100mm threshold
                        adjacent_walls.append(wall)
                
                if adjacent_walls:
                    path = {
                        'door_location': door.geometry.centroid.coords[0],
                        'adjacent_wall_count': len(adjacent_walls),
                        'access_type': door.element_type,
                        'circulation_importance': min(len(adjacent_walls), 5) / 5.0
                    }
                    paths.append(path)
                    
        except Exception as e:
            self.logger.warning(f"Error analyzing circulation paths: {str(e)}")
        
        return paths
    
    def _detect_structural_grid(self, walls: List[CADElement]) -> Dict[str, Any]:
        """Detect structural grid patterns in wall layout"""
        grid_info = {
            'has_regular_grid': False,
            'grid_spacing': None,
            'primary_direction': None,
            'orthogonal_ratio': 0.0
        }
        
        try:
            if len(walls) < 4:
                return grid_info
            
            # Analyze wall angles
            angles = []
            for wall in walls:
                coords = list(wall.geometry.coords)
                if len(coords) >= 2:
                    dx = coords[-1][0] - coords[0][0]
                    dy = coords[-1][1] - coords[0][1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    # Normalize to 0-180 range
                    angle = angle % 180
                    angles.append(angle)
            
            if angles:
                # Check for orthogonal dominance
                orthogonal_count = 0
                for angle in angles:
                    if abs(angle) <= 10 or abs(angle - 90) <= 10 or abs(angle - 180) <= 10:
                        orthogonal_count += 1
                
                grid_info['orthogonal_ratio'] = orthogonal_count / len(angles)
                grid_info['has_regular_grid'] = grid_info['orthogonal_ratio'] >= 0.7
                
                # Determine primary direction
                horizontal_count = sum(1 for a in angles if abs(a) <= 10 or abs(a - 180) <= 10)
                vertical_count = sum(1 for a in angles if abs(a - 90) <= 10)
                
                if horizontal_count > vertical_count:
                    grid_info['primary_direction'] = 'horizontal'
                elif vertical_count > horizontal_count:
                    grid_info['primary_direction'] = 'vertical'
                else:
                    grid_info['primary_direction'] = 'balanced'
                    
        except Exception as e:
            self.logger.warning(f"Error detecting structural grid: {str(e)}")
        
        return grid_info
    
    def _calculate_analysis_quality(self, walls: List[CADElement], 
                                  connectivity: Dict[str, List[str]], 
                                  opening_associations: Dict[str, List[str]]) -> float:
        """Calculate overall quality score for geometric analysis"""
        quality_factors = []
        
        # Wall count factor
        wall_count = len(walls)
        if wall_count >= 10:
            quality_factors.append(1.0)
        elif wall_count >= 5:
            quality_factors.append(0.8)
        elif wall_count >= 2:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Connectivity factor
        if connectivity:
            total_connections = sum(len(connections) for connections in connectivity.values())
            avg_connections = total_connections / len(connectivity) if connectivity else 0
            connectivity_score = min(avg_connections / 4.0, 1.0)  # Normalize by expected connections
            quality_factors.append(connectivity_score)
        else:
            quality_factors.append(0.0)
        
        # Opening association factor
        if opening_associations:
            associated_count = sum(1 for assoc in opening_associations.values() if assoc)
            association_ratio = associated_count / len(opening_associations) if opening_associations else 0
            quality_factors.append(association_ratio)
        else:
            quality_factors.append(0.5)  # Neutral score if no openings
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0