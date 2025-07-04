import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, voronoi_diagram
from scipy.spatial import ConvexHull, distance_matrix
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN, KMeans
import logging
from typing import Dict, List, Any, Optional, Tuple
import cv2
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometricAnalyzer:
    """Advanced geometric analysis for floor plan entities"""
    
    def __init__(self):
        self.color_thresholds = {
            'wall': [0, 0, 0],  # Black
            'restricted': [0, 0, 255],  # Blue
            'entrance': [255, 0, 0],  # Red
            'open': [255, 255, 255]  # White
        }
        
        self.min_wall_length = 0.5  # meters
        self.min_area_threshold = 1.0  # square meters
        self.wall_thickness_threshold = 0.3  # meters
    
    def analyze_zones(self, floor_plan_data: Dict[str, Any], 
                     wall_threshold: float = 0.1,
                     restricted_threshold: float = 0.8,
                     entrance_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Comprehensive zone analysis of floor plan
        
        Args:
            floor_plan_data: Parsed floor plan data
            wall_threshold: Threshold for wall detection
            restricted_threshold: Threshold for restricted area detection
            entrance_threshold: Threshold for entrance detection
            
        Returns:
            Analysis results with detected zones
        """
        logger.info("Starting comprehensive zone analysis")
        
        try:
            entities = floor_plan_data.get('entities', [])
            bounds = floor_plan_data.get('bounds', {})
            
            # Initialize analysis results
            analysis_results = {
                'walls': [],
                'restricted_areas': [],
                'entrances': [],
                'open_spaces': [],
                'connections': [],
                'metadata': {
                    'total_entities': len(entities),
                    'analysis_bounds': bounds,
                    'thresholds': {
                        'wall': wall_threshold,
                        'restricted': restricted_threshold,
                        'entrance': entrance_threshold
                    }
                }
            }
            
            # Step 1: Classify entities by color and geometry
            classified_entities = self._classify_entities(entities)
            
            # Step 2: Detect and analyze walls
            analysis_results['walls'] = self._detect_walls(
                classified_entities, wall_threshold
            )
            
            # Step 3: Detect restricted areas
            analysis_results['restricted_areas'] = self._detect_restricted_areas(
                classified_entities, restricted_threshold
            )
            
            # Step 4: Detect entrances and exits
            analysis_results['entrances'] = self._detect_entrances(
                classified_entities, entrance_threshold
            )
            
            # Step 5: Analyze open spaces
            analysis_results['open_spaces'] = self._analyze_open_spaces(
                entities, analysis_results['walls']
            )
            
            # Step 6: Detect connections between spaces
            analysis_results['connections'] = self._detect_connections(
                analysis_results['open_spaces'],
                analysis_results['entrances']
            )
            
            # Step 7: Calculate spatial metrics
            analysis_results['spatial_metrics'] = self._calculate_spatial_metrics(
                analysis_results
            )
            
            logger.info("Zone analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during zone analysis: {str(e)}")
            raise
    
    def _classify_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify entities based on color and geometric properties"""
        classified = {
            'walls': [],
            'restricted': [],
            'entrances': [],
            'text': [],
            'other': []
        }
        
        for entity in entities:
            entity_color = entity.get('color', [0, 0, 0])
            geometry = entity.get('geometry', {})
            
            # Color-based classification
            if self._is_color_similar(entity_color, self.color_thresholds['wall'], 50):
                classified['walls'].append(entity)
            elif self._is_color_similar(entity_color, self.color_thresholds['restricted'], 50):
                classified['restricted'].append(entity)
            elif self._is_color_similar(entity_color, self.color_thresholds['entrance'], 50):
                classified['entrances'].append(entity)
            elif geometry.get('type') == 'text':
                classified['text'].append(entity)
            else:
                classified['other'].append(entity)
        
        return classified
    
    def _is_color_similar(self, color1: List[int], color2: List[int], threshold: int) -> bool:
        """Check if two colors are similar within threshold"""
        if len(color1) != 3 or len(color2) != 3:
            return False
        
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
        return distance <= threshold
    
    def _detect_walls(self, classified_entities: Dict[str, List], threshold: float) -> List[Dict[str, Any]]:
        """Detect and analyze wall structures"""
        wall_entities = classified_entities.get('walls', [])
        walls = []
        
        for entity in wall_entities:
            geometry = entity.get('geometry', {})
            
            if geometry.get('type') == 'line':
                wall = self._analyze_wall_line(entity, geometry)
                if wall and wall['length'] >= self.min_wall_length:
                    walls.append(wall)
            
            elif geometry.get('type') in ['polyline', 'polygon']:
                wall_segments = self._analyze_wall_polyline(entity, geometry)
                walls.extend(wall_segments)
        
        # Group parallel and connected walls
        grouped_walls = self._group_walls(walls)
        
        # Calculate wall properties
        for wall in grouped_walls:
            wall.update(self._calculate_wall_properties(wall))
        
        return grouped_walls
    
    def _analyze_wall_line(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single wall line"""
        return {
            'id': f"wall_{id(entity)}",
            'type': 'wall_line',
            'start': geometry['start'],
            'end': geometry['end'],
            'length': geometry['length'],
            'angle': geometry['angle'],
            'thickness': entity.get('lineweight', 1) * 0.01,  # Convert to meters
            'layer': entity.get('layer', 'default'),
            'shapely_geom': geometry.get('shapely_geom'),
            'entity_ref': entity
        }
    
    def _analyze_wall_polyline(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze wall polyline into segments"""
        points = geometry.get('points', [])
        segments = []
        
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            
            length = np.sqrt((end_point[0] - start_point[0])**2 + 
                           (end_point[1] - start_point[1])**2)
            
            if length >= self.min_wall_length:
                angle = np.degrees(np.arctan2(
                    end_point[1] - start_point[1],
                    end_point[0] - start_point[0]
                ))
                
                segment = {
                    'id': f"wall_{id(entity)}_{i}",
                    'type': 'wall_segment',
                    'start': {'x': start_point[0], 'y': start_point[1], 'z': 0},
                    'end': {'x': end_point[0], 'y': end_point[1], 'z': 0},
                    'length': length,
                    'angle': angle,
                    'thickness': entity.get('lineweight', 1) * 0.01,
                    'layer': entity.get('layer', 'default'),
                    'shapely_geom': LineString([start_point, end_point]),
                    'entity_ref': entity
                }
                
                segments.append(segment)
        
        return segments
    
    def _group_walls(self, walls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group parallel and connected walls"""
        if not walls:
            return []
        
        # Create distance matrix between wall endpoints
        wall_points = []
        for wall in walls:
            wall_points.append([wall['start']['x'], wall['start']['y']])
            wall_points.append([wall['end']['x'], wall['end']['y']])
        
        # Group walls using clustering
        grouped_walls = []
        
        # Simple grouping by proximity and angle similarity
        for wall in walls:
            # For now, keep walls separate
            # Advanced grouping logic would go here
            grouped_walls.append(wall)
        
        return grouped_walls
    
    def _calculate_wall_properties(self, wall: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional wall properties"""
        return {
            'area': wall['length'] * wall['thickness'],
            'midpoint': {
                'x': (wall['start']['x'] + wall['end']['x']) / 2,
                'y': (wall['start']['y'] + wall['end']['y']) / 2,
                'z': 0
            },
            'orientation': self._get_wall_orientation(wall['angle']),
            'endpoints': [
                {'x': wall['start']['x'], 'y': wall['start']['y']},
                {'x': wall['end']['x'], 'y': wall['end']['y']}
            ]
        }
    
    def _get_wall_orientation(self, angle: float) -> str:
        """Get wall orientation (horizontal, vertical, diagonal)"""
        normalized_angle = abs(angle % 90)
        
        if normalized_angle < 10 or normalized_angle > 80:
            return 'horizontal' if abs(angle % 180) < 10 else 'vertical'
        else:
            return 'diagonal'
    
    def _detect_restricted_areas(self, classified_entities: Dict[str, List], threshold: float) -> List[Dict[str, Any]]:
        """Detect restricted areas (typically blue zones)"""
        restricted_entities = classified_entities.get('restricted', [])
        restricted_areas = []
        
        for entity in restricted_entities:
            geometry = entity.get('geometry', {})
            
            if geometry.get('type') == 'polygon':
                area = self._analyze_restricted_polygon(entity, geometry)
                if area and area['area'] >= self.min_area_threshold:
                    restricted_areas.append(area)
            
            elif geometry.get('type') == 'circle':
                area = self._analyze_restricted_circle(entity, geometry)
                if area and area['area'] >= self.min_area_threshold:
                    restricted_areas.append(area)
        
        return restricted_areas
    
    def _analyze_restricted_polygon(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze restricted polygon area"""
        points = geometry.get('points', [])
        shapely_geom = geometry.get('shapely_geom')
        
        if not shapely_geom or not shapely_geom.is_valid:
            return None
        
        # Calculate centroid
        centroid = shapely_geom.centroid
        
        return {
            'id': f"restricted_{id(entity)}",
            'type': 'restricted_polygon',
            'points': points,
            'area': shapely_geom.area,
            'perimeter': shapely_geom.length,
            'center': {'x': centroid.x, 'y': centroid.y, 'z': 0},
            'bounds': shapely_geom.bounds,
            'layer': entity.get('layer', 'default'),
            'shapely_geom': shapely_geom,
            'entity_ref': entity
        }
    
    def _analyze_restricted_circle(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze restricted circular area"""
        center = geometry.get('center', {})
        radius = geometry.get('radius', 0)
        area = geometry.get('area', 0)
        
        return {
            'id': f"restricted_{id(entity)}",
            'type': 'restricted_circle',
            'center': center,
            'radius': radius,
            'area': area,
            'perimeter': 2 * np.pi * radius,
            'bounds': (
                center['x'] - radius, center['y'] - radius,
                center['x'] + radius, center['y'] + radius
            ),
            'layer': entity.get('layer', 'default'),
            'shapely_geom': geometry.get('shapely_geom'),
            'entity_ref': entity
        }
    
    def _detect_entrances(self, classified_entities: Dict[str, List], threshold: float) -> List[Dict[str, Any]]:
        """Detect entrances and exits (typically red zones)"""
        entrance_entities = classified_entities.get('entrances', [])
        entrances = []
        
        for entity in entrance_entities:
            geometry = entity.get('geometry', {})
            
            if geometry.get('type') == 'line':
                entrance = self._analyze_entrance_line(entity, geometry)
                if entrance:
                    entrances.append(entrance)
            
            elif geometry.get('type') == 'polygon':
                entrance = self._analyze_entrance_polygon(entity, geometry)
                if entrance:
                    entrances.append(entrance)
        
        return entrances
    
    def _analyze_entrance_line(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entrance line (opening in wall)"""
        return {
            'id': f"entrance_{id(entity)}",
            'type': 'entrance_line',
            'start': geometry['start'],
            'end': geometry['end'],
            'width': geometry['length'],
            'position': {
                'x': (geometry['start']['x'] + geometry['end']['x']) / 2,
                'y': (geometry['start']['y'] + geometry['end']['y']) / 2,
                'z': 0
            },
            'angle': geometry['angle'],
            'layer': entity.get('layer', 'default'),
            'shapely_geom': geometry.get('shapely_geom'),
            'entity_ref': entity
        }
    
    def _analyze_entrance_polygon(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entrance polygon area"""
        shapely_geom = geometry.get('shapely_geom')
        
        if not shapely_geom or not shapely_geom.is_valid:
            return None
        
        centroid = shapely_geom.centroid
        
        return {
            'id': f"entrance_{id(entity)}",
            'type': 'entrance_area',
            'area': shapely_geom.area,
            'perimeter': shapely_geom.length,
            'center': {'x': centroid.x, 'y': centroid.y, 'z': 0},
            'bounds': shapely_geom.bounds,
            'layer': entity.get('layer', 'default'),
            'shapely_geom': shapely_geom,
            'entity_ref': entity
        }
    
    def _analyze_open_spaces(self, entities: List[Dict[str, Any]], walls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze open spaces by subtracting walls from total area"""
        
        # Get overall bounds
        all_geometries = []
        for entity in entities:
            geometry = entity.get('geometry', {})
            shapely_geom = geometry.get('shapely_geom')
            if shapely_geom:
                all_geometries.append(shapely_geom)
        
        if not all_geometries:
            return []
        
        # Create overall boundary
        total_union = unary_union(all_geometries)
        
        # Get wall geometries
        wall_geometries = []
        for wall in walls:
            wall_geom = wall.get('shapely_geom')
            if wall_geom:
                # Buffer wall line to create area
                buffered_wall = wall_geom.buffer(wall.get('thickness', 0.1) / 2)
                wall_geometries.append(buffered_wall)
        
        # Calculate open spaces
        open_spaces = []
        
        if wall_geometries:
            walls_union = unary_union(wall_geometries)
            
            # Simple approach: use bounds to create rooms
            # In production, you'd use more sophisticated space detection
            if hasattr(total_union, 'bounds'):
                bounds = total_union.bounds
                
                # Create a simple rectangular open space
                open_space_geom = Polygon([
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3])
                ])
                
                # Subtract walls
                if walls_union:
                    open_space_geom = open_space_geom.difference(walls_union)
                
                if open_space_geom.area > self.min_area_threshold:
                    centroid = open_space_geom.centroid
                    
                    open_spaces.append({
                        'id': 'open_space_1',
                        'type': 'open_space',
                        'area': open_space_geom.area,
                        'usable_area': open_space_geom.area * 0.8,  # 80% usable
                        'perimeter': open_space_geom.length,
                        'center': {'x': centroid.x, 'y': centroid.y, 'z': 0},
                        'bounds': open_space_geom.bounds,
                        'shape': self._classify_space_shape(open_space_geom),
                        'shapely_geom': open_space_geom
                    })
        
        return open_spaces
    
    def _classify_space_shape(self, geometry) -> str:
        """Classify the shape of a space"""
        if not geometry or not hasattr(geometry, 'bounds'):
            return 'unknown'
        
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        aspect_ratio = width / height if height > 0 else 1
        
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        elif aspect_ratio > 1.5:
            return 'rectangular_horizontal'
        elif aspect_ratio < 0.67:
            return 'rectangular_vertical'
        else:
            return 'rectangular'
    
    def _detect_connections(self, open_spaces: List[Dict[str, Any]], entrances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect connections between open spaces through entrances"""
        connections = []
        
        for i, space1 in enumerate(open_spaces):
            for j, space2 in enumerate(open_spaces):
                if i >= j:
                    continue
                
                # Check if spaces are connected through entrances
                connecting_entrances = []
                
                for entrance in entrances:
                    entrance_point = Point(
                        entrance['position']['x'],
                        entrance['position']['y']
                    )
                    
                    # Check if entrance is near both spaces
                    space1_geom = space1.get('shapely_geom')
                    space2_geom = space2.get('shapely_geom')
                    
                    if (space1_geom and space2_geom and
                        space1_geom.distance(entrance_point) < 2.0 and
                        space2_geom.distance(entrance_point) < 2.0):
                        connecting_entrances.append(entrance)
                
                if connecting_entrances:
                    connections.append({
                        'id': f"connection_{i}_{j}",
                        'space1_id': space1['id'],
                        'space2_id': space2['id'],
                        'entrances': connecting_entrances,
                        'type': 'direct_connection',
                        'distance': self._calculate_space_distance(space1, space2)
                    })
        
        return connections
    
    def _calculate_space_distance(self, space1: Dict[str, Any], space2: Dict[str, Any]) -> float:
        """Calculate distance between two spaces"""
        center1 = space1.get('center', {})
        center2 = space2.get('center', {})
        
        return np.sqrt(
            (center1.get('x', 0) - center2.get('x', 0))**2 +
            (center1.get('y', 0) - center2.get('y', 0))**2
        )
    
    def _calculate_spatial_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive spatial metrics"""
        walls = analysis_results.get('walls', [])
        restricted_areas = analysis_results.get('restricted_areas', [])
        entrances = analysis_results.get('entrances', [])
        open_spaces = analysis_results.get('open_spaces', [])
        
        # Calculate areas
        total_wall_area = sum(wall.get('area', 0) for wall in walls)
        total_restricted_area = sum(area.get('area', 0) for area in restricted_areas)
        total_open_area = sum(space.get('area', 0) for space in open_spaces)
        total_area = total_wall_area + total_restricted_area + total_open_area
        
        # Calculate lengths
        total_wall_length = sum(wall.get('length', 0) for wall in walls)
        total_entrance_width = sum(entrance.get('width', 0) for entrance in entrances)
        
        # Calculate ratios
        wall_ratio = (total_wall_area / total_area * 100) if total_area > 0 else 0
        restricted_ratio = (total_restricted_area / total_area * 100) if total_area > 0 else 0
        open_ratio = (total_open_area / total_area * 100) if total_area > 0 else 0
        
        return {
            'total_area': total_area,
            'wall_area': total_wall_area,
            'restricted_area': total_restricted_area,
            'open_area': total_open_area,
            'wall_length': total_wall_length,
            'entrance_width': total_entrance_width,
            'area_ratios': {
                'wall_ratio': wall_ratio,
                'restricted_ratio': restricted_ratio,
                'open_ratio': open_ratio
            },
            'counts': {
                'walls': len(walls),
                'restricted_areas': len(restricted_areas),
                'entrances': len(entrances),
                'open_spaces': len(open_spaces)
            },
            'accessibility_score': self._calculate_accessibility_score(analysis_results),
            'circulation_efficiency': self._calculate_circulation_efficiency(analysis_results)
        }
    
    def _calculate_accessibility_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate accessibility score based on entrance distribution"""
        entrances = analysis_results.get('entrances', [])
        open_spaces = analysis_results.get('open_spaces', [])
        
        if not entrances or not open_spaces:
            return 0.0
        
        # Simple accessibility score based on entrance-to-space ratio
        entrance_density = len(entrances) / len(open_spaces)
        
        # Normalize to 0-100 scale
        score = min(100, entrance_density * 50)
        
        return score
    
    def _calculate_circulation_efficiency(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate circulation efficiency score"""
        connections = analysis_results.get('connections', [])
        open_spaces = analysis_results.get('open_spaces', [])
        
        if not open_spaces:
            return 0.0
        
        # Calculate connectivity ratio
        max_connections = len(open_spaces) * (len(open_spaces) - 1) / 2
        actual_connections = len(connections)
        
        if max_connections == 0:
            return 100.0
        
        connectivity_ratio = actual_connections / max_connections
        
        # Normalize to 0-100 scale
        score = min(100, connectivity_ratio * 100)
        
        return score
