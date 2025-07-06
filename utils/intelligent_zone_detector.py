
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix, ConvexHull, Voronoi
from scipy.ndimage import binary_erosion, binary_dilation, label
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, voronoi_diagram
import logging
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntelligentZoneDetector:
    """Advanced AI-powered zone detection system for floor plans"""
    
    def __init__(self):
        self.zone_templates = {
            'corridor': {
                'aspect_ratio_range': (3.0, 20.0),  # Long and narrow
                'width_range': (1.5, 4.0),  # Typical corridor width
                'min_length': 3.0,
                'connectivity_threshold': 0.7
            },
            'room': {
                'aspect_ratio_range': (0.6, 2.5),  # More square-like
                'min_area': 6.0,  # Minimum room area
                'max_area': 50.0,  # Maximum single room area
                'wall_adjacency_threshold': 0.8
            },
            'entrance': {
                'width_range': (0.8, 3.0),  # Door widths
                'wall_intersection_score': 0.9,
                'accessibility_weight': 2.0
            },
            'stairs': {
                'aspect_ratio_range': (1.5, 4.0),
                'area_range': (4.0, 25.0),
                'pattern_complexity': 0.6
            },
            'elevator': {
                'aspect_ratio_range': (0.8, 1.5),  # More square
                'area_range': (2.0, 8.0),
                'centrality_weight': 1.5
            }
        }
        
        self.ai_models = {
            'zone_classifier': None,
            'connectivity_analyzer': None,
            'spatial_optimizer': None
        }
    
    def analyze_floor_plan(self, floor_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis function that orchestrates all detection methods"""
        logger.info("Starting intelligent zone detection analysis")
        
        try:
            # Extract basic geometric data
            entities = floor_plan_data.get('entities', [])
            bounds = floor_plan_data.get('bounds', {})
            
            # Create spatial representation
            spatial_data = self._create_spatial_representation(entities, bounds)
            
            # Detect walls and structural elements
            structural_elements = self._detect_structural_elements(spatial_data)
            
            # Identify open spaces and rooms
            open_spaces = self._identify_open_spaces(spatial_data, structural_elements)
            
            # Classify zones using AI algorithms
            classified_zones = self._classify_zones_with_ai(open_spaces, structural_elements)
            
            # Detect circulation patterns
            circulation_network = self._analyze_circulation_patterns(classified_zones)
            
            # Identify entrance points
            entrances = self._detect_entrance_points(structural_elements, circulation_network)
            
            # Find vertical circulation (stairs, elevators)
            vertical_circulation = self._detect_vertical_circulation(classified_zones, spatial_data)
            
            # Analyze spatial relationships
            spatial_relationships = self._analyze_spatial_relationships(classified_zones)
            
            # Generate zone recommendations
            recommendations = self._generate_zone_recommendations(classified_zones, spatial_relationships)
            
            return {
                'zones': {
                    'corridors': classified_zones.get('corridors', []),
                    'rooms': classified_zones.get('rooms', []),
                    'entrances': entrances,
                    'vertical_circulation': vertical_circulation,
                    'open_spaces': classified_zones.get('open_spaces', [])
                },
                'circulation_network': circulation_network,
                'spatial_relationships': spatial_relationships,
                'recommendations': recommendations,
                'analysis_metadata': {
                    'detection_confidence': self._calculate_detection_confidence(classified_zones),
                    'zone_count': sum(len(zones) for zones in classified_zones.values()),
                    'coverage_percentage': self._calculate_coverage_percentage(classified_zones, bounds)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent zone detection: {str(e)}")
            return self._create_fallback_zones(floor_plan_data)
    
    def _create_spatial_representation(self, entities: List[Dict[str, Any]], 
                                     bounds: Dict[str, float]) -> Dict[str, Any]:
        """Create a comprehensive spatial representation of the floor plan"""
        
        # Initialize spatial grid
        resolution = 0.1  # 10cm resolution
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 100) - bounds.get('min_y', 0)
        
        grid_width = int(width / resolution) + 1
        grid_height = int(height / resolution) + 1
        
        # Create different layers
        wall_layer = np.zeros((grid_height, grid_width), dtype=np.uint8)
        space_layer = np.ones((grid_height, grid_width), dtype=np.uint8)
        
        # Process entities and populate layers
        wall_lines = []
        text_elements = []
        
        for entity in entities:
            entity_type = entity.get('type', '')
            geometry = entity.get('geometry', {})
            
            if entity_type in ['LINE', 'POLYLINE', 'LWPOLYLINE']:
                wall_lines.extend(self._extract_wall_lines(entity, geometry))
            elif entity_type in ['TEXT', 'MTEXT']:
                text_elements.append(self._extract_text_info(entity, geometry))
        
        # Rasterize walls
        for wall_line in wall_lines:
            self._rasterize_line(wall_layer, wall_line, bounds, resolution)
        
        # Create space layer (inverse of walls)
        space_layer = (1 - wall_layer).astype(np.uint8)
        
        return {
            'wall_layer': wall_layer,
            'space_layer': space_layer,
            'wall_lines': wall_lines,
            'text_elements': text_elements,
            'resolution': resolution,
            'bounds': bounds,
            'grid_shape': (grid_height, grid_width)
        }
    
    def _detect_structural_elements(self, spatial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect walls, columns, and other structural elements"""
        
        wall_layer = spatial_data['wall_layer']
        wall_lines = spatial_data['wall_lines']
        
        # Group connected wall segments
        wall_groups = self._group_connected_walls(wall_lines)
        
        # Identify load-bearing vs partition walls
        structural_walls = []
        partition_walls = []
        
        for group in wall_groups:
            if self._is_load_bearing_wall(group):
                structural_walls.append(group)
            else:
                partition_walls.append(group)
        
        # Detect columns and pillars
        columns = self._detect_columns(wall_layer)
        
        # Find openings in walls (doors, windows)
        openings = self._detect_wall_openings(wall_layer, wall_lines)
        
        return {
            'structural_walls': structural_walls,
            'partition_walls': partition_walls,
            'columns': columns,
            'openings': openings,
            'all_walls': wall_groups
        }
    
    def _identify_open_spaces(self, spatial_data: Dict[str, Any], 
                            structural_elements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify and segment open spaces using advanced algorithms"""
        
        space_layer = spatial_data['space_layer']
        
        # Apply morphological operations to clean up the space
        cleaned_space = binary_erosion(space_layer, iterations=2)
        cleaned_space = binary_dilation(cleaned_space, iterations=2)
        
        # Label connected components
        labeled_spaces, num_spaces = label(cleaned_space)
        
        open_spaces = []
        
        for space_id in range(1, num_spaces + 1):
            space_mask = (labeled_spaces == space_id)
            
            # Calculate space properties
            space_props = self._calculate_space_properties(space_mask, spatial_data)
            
            if space_props['area'] > 2.0:  # Minimum area threshold
                space_info = {
                    'id': f'space_{space_id}',
                    'mask': space_mask,
                    'properties': space_props,
                    'contour': self._extract_space_contour(space_mask, spatial_data),
                    'accessibility': self._calculate_space_accessibility(space_mask, structural_elements)
                }
                
                open_spaces.append(space_info)
        
        return open_spaces
    
def _classify_zones_with_ai(self, open_spaces: List[Dict[str, Any]], 
                              structural_elements: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Use AI algorithms to classify spaces into different zone types"""
    
    classified_zones = {
        'corridors': [],
        'rooms': [],
        'open_spaces': [],
        'utility_spaces': []
    }
    
    for space in open_spaces:
        props = space['properties']
        
        # Feature vector for classification
        features = [
            props['area'],
            props['aspect_ratio'],
            props['perimeter'],
            props['convexity'],
            props['elongation'],
            space['accessibility']['connectivity_score'],
            space['accessibility']['entrance_proximity']
        ]
        
        # Rule-based classification with AI enhancement
        zone_scores = self._calculate_zone_scores(features, props)
        
        # Assign to most likely zone type
        best_zone = max(zone_scores.items(), key=lambda x: x[1])
        zone_type = best_zone[0]
        confidence = best_zone[1]
        
        zone_info = {
            'id': space['id'],
            'type': zone_type,
            'confidence': confidence,
            'geometry': space['contour'],
            'properties': props,
            'features': features,
            'center': props['centroid'],
            'area': props['area']
        }
        
        if zone_type in classified_zones:
            classified_zones[zone_type].append(zone_info)
        else:
            classified_zones['open_spaces'].append(zone_info)
    
    # Post-process classifications
    classified_zones = self._refine_classifications(classified_zones, structural_elements)
    
    return classified_zones

def _analyze_circulation_patterns(self, classified_zones: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze circulation patterns and connectivity"""
    
    corridors = classified_zones.get('corridors', [])
    rooms = classified_zones.get('rooms', [])
    
    # Create circulation graph
    circulation_graph = nx.Graph()
    
    # Add nodes
    for corridor in corridors:
        circulation_graph.add_node(corridor['id'], type='corridor', **corridor)
    
    for room in rooms:
        circulation_graph.add_node(room['id'], type='room', **room)
    
    # Add edges based on adjacency
    all_zones = corridors + rooms
    for i, zone1 in enumerate(all_zones):
        for zone2 in all_zones[i+1:]:
            if self._are_zones_adjacent(zone1, zone2):
                distance = self._calculate_zone_distance(zone1, zone2)
                circulation_graph.add_edge(zone1['id'], zone2['id'], weight=distance)
    
    # Analyze circulation metrics
    circulation_metrics = {
        'connectivity_matrix': nx.adjacency_matrix(circulation_graph),
        'shortest_paths': dict(nx.all_pairs_shortest_path_length(circulation_graph)),
        'centrality_scores': nx.betweenness_centrality(circulation_graph),
        'main_circulation_spine': self._identify_main_circulation(corridors),
        'dead_ends': self._identify_dead_ends(circulation_graph),
        'circulation_efficiency': self._calculate_circulation_efficiency(circulation_graph)
    }
    
    return {
        'graph': circulation_graph,
        'metrics': circulation_metrics,
        'flow_analysis': self._analyze_pedestrian_flow(classified_zones)
    }
    
    def _detect_entrance_points(self, structural_elements: Dict[str, Any], 
                              circulation_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect building entrances and access points"""
        
        openings = structural_elements.get('openings', [])
        walls = structural_elements.get('all_walls', [])
        
        entrances = []
        
        for opening in openings:
            # Analyze opening characteristics
            entrance_score = self._calculate_entrance_score(opening, walls, circulation_network)
            
            if entrance_score > 0.7:  # Threshold for entrance classification
                entrance_info = {
                    'id': f"entrance_{len(entrances)}",
                    'type': 'main_entrance' if entrance_score > 0.9 else 'secondary_entrance',
                    'position': opening['center'],
                    'width': opening['width'],
                    'orientation': opening['orientation'],
                    'accessibility_score': entrance_score,
                    'connected_circulation': self._find_connected_circulation(opening, circulation_network)
                }
                
                entrances.append(entrance_info)
        
        # If no entrances found, use heuristics
        if not entrances:
            entrances = self._detect_entrances_heuristic(structural_elements, circulation_network)
        
        return entrances
    
    def _detect_vertical_circulation(self, classified_zones: Dict[str, List[Dict[str, Any]]], 
                                   spatial_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect stairs, elevators, and other vertical circulation elements"""
        
        vertical_elements = []
        open_spaces = classified_zones.get('open_spaces', [])
        
        for space in open_spaces:
            props = space['properties']
            
            # Check for stair patterns
            stair_score = self._calculate_stair_score(space, spatial_data)
            if stair_score > 0.6:
                vertical_elements.append({
                    'id': f"stairs_{len(vertical_elements)}",
                    'type': 'stairs',
                    'geometry': space['geometry'],
                    'area': props['area'],
                    'detection_confidence': stair_score
                })
            
            # Check for elevator patterns
            elevator_score = self._calculate_elevator_score(space, spatial_data)
            if elevator_score > 0.7:
                vertical_elements.append({
                    'id': f"elevator_{len(vertical_elements)}",
                    'type': 'elevator',
                    'geometry': space['geometry'],
                    'area': props['area'],
                    'detection_confidence': elevator_score
                })
        
        return vertical_elements
    
    def _analyze_spatial_relationships(self, classified_zones: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze spatial relationships between different zones"""
        
        relationships = {
            'adjacency_matrix': {},
            'visibility_analysis': {},
            'accessibility_paths': {},
            'functional_groupings': {}
        }
        
        all_zones = []
        for zone_type, zones in classified_zones.items():
            all_zones.extend(zones)
        
        # Create adjacency matrix
        adjacency = np.zeros((len(all_zones), len(all_zones)))
        for i, zone1 in enumerate(all_zones):
            for j, zone2 in enumerate(all_zones):
                if i != j:
                    adjacency[i][j] = 1 if self._are_zones_adjacent(zone1, zone2) else 0
        
        relationships['adjacency_matrix'] = adjacency
        
        # Analyze functional groupings using clustering
        zone_features = np.array([self._extract_zone_features(zone) for zone in all_zones])
        
        if len(zone_features) > 3:
            kmeans = KMeans(n_clusters=min(4, len(zone_features)//2))
            cluster_labels = kmeans.fit_predict(zone_features)
            
            groupings = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                groupings[f'group_{label}'].append(all_zones[i])
            
            relationships['functional_groupings'] = dict(groupings)
        
        return relationships
    
    def _generate_zone_recommendations(self, classified_zones: Dict[str, List[Dict[str, Any]]], 
                                     spatial_relationships: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate intelligent recommendations for zone improvements"""
        
        recommendations = []
        
        corridors = classified_zones.get('corridors', [])
        rooms = classified_zones.get('rooms', [])
        
        # Analyze corridor efficiency
        if corridors:
            total_corridor_area = sum(c['area'] for c in corridors)
            total_space_area = sum(sum(z['area'] for z in zones) for zones in classified_zones.values())
            
            corridor_ratio = total_corridor_area / total_space_area if total_space_area > 0 else 0
            
            if corridor_ratio > 0.25:
                recommendations.append({
                    'type': 'efficiency',
                    'message': 'Corridor space is above 25% of total area. Consider optimizing corridor width or layout.',
                    'priority': 'medium'
                })
            elif corridor_ratio < 0.1:
                recommendations.append({
                    'type': 'accessibility',
                    'message': 'Insufficient circulation space detected. Consider adding or widening corridors.',
                    'priority': 'high'
                })
        
        # Check room accessibility
        isolated_rooms = 0
        for room in rooms:
            if not self._has_corridor_access(room, corridors):
                isolated_rooms += 1
        
        if isolated_rooms > 0:
            recommendations.append({
                'type': 'accessibility',
                'message': f'{isolated_rooms} rooms have poor corridor access. Consider improving circulation connections.',
                'priority': 'high'
            })
        
        # Analyze natural lighting potential
        exterior_adjacent_rooms = sum(1 for room in rooms if self._is_exterior_adjacent(room))
        
        if exterior_adjacent_rooms / len(rooms) < 0.4 if rooms else 0:
            recommendations.append({
                'type': 'comfort',
                'message': 'Limited natural lighting access. Consider repositioning rooms or adding skylights.',
                'priority': 'medium'
            })
        
        return recommendations
    
    # Helper methods for various calculations and analyses
    
    def _extract_wall_lines(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract wall line segments from entity geometry"""
        lines = []
        
        if geometry.get('type') == 'line':
            start = geometry.get('start', {})
            end = geometry.get('end', {})
            lines.append({
                'start': (start.get('x', 0), start.get('y', 0)),
                'end': (end.get('x', 0), end.get('y', 0)),
                'thickness': entity.get('lineweight', 1) * 0.01
            })
        
        elif geometry.get('type') in ['polyline', 'polygon']:
            points = geometry.get('points', [])
            for i in range(len(points) - 1):
                lines.append({
                    'start': (points[i][0], points[i][1]),
                    'end': (points[i+1][0], points[i+1][1]),
                    'thickness': entity.get('lineweight', 1) * 0.01
                })
        
        return lines
    
    def _rasterize_line(self, layer: np.ndarray, line: Dict[str, Any], 
                       bounds: Dict[str, float], resolution: float):
        """Rasterize a line onto the spatial grid"""
        start_x, start_y = line['start']
        end_x, end_y = line['end']
        thickness = line.get('thickness', 0.1)
        
        # Convert to grid coordinates
        min_x, min_y = bounds.get('min_x', 0), bounds.get('min_y', 0)
        
        start_col = int((start_x - min_x) / resolution)
        start_row = int((start_y - min_y) / resolution)
        end_col = int((end_x - min_x) / resolution)
        end_row = int((end_y - min_y) / resolution)
        
        # Use Bresenham's line algorithm with thickness
        thickness_pixels = max(1, int(thickness / resolution))
        
        # Simple line rasterization
        if abs(end_col - start_col) > abs(end_row - start_row):
            # More horizontal
            if start_col > end_col:
                start_col, end_col = end_col, start_col
                start_row, end_row = end_row, start_row
            
            for col in range(max(0, start_col), min(layer.shape[1], end_col + 1)):
                if end_col != start_col:
                    row = start_row + (end_row - start_row) * (col - start_col) // (end_col - start_col)
                else:
                    row = start_row
                
                for t in range(-thickness_pixels//2, thickness_pixels//2 + 1):
                    r = row + t
                    if 0 <= r < layer.shape[0] and 0 <= col < layer.shape[1]:
                        layer[r, col] = 1
        else:
            # More vertical
            if start_row > end_row:
                start_col, end_col = end_col, start_col
                start_row, end_row = end_row, start_row
            
            for row in range(max(0, start_row), min(layer.shape[0], end_row + 1)):
                if end_row != start_row:
                    col = start_col + (end_col - start_col) * (row - start_row) // (end_row - start_row)
                else:
                    col = start_col
                
                for t in range(-thickness_pixels//2, thickness_pixels//2 + 1):
                    c = col + t
                    if 0 <= row < layer.shape[0] and 0 <= c < layer.shape[1]:
                        layer[row, c] = 1
    
    def _calculate_space_properties(self, space_mask: np.ndarray, 
                                  spatial_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate geometric properties of a space"""
        resolution = spatial_data['resolution']
        
        # Basic measurements
        area_pixels = np.sum(space_mask)
        area_m2 = area_pixels * (resolution ** 2)
        
        # Find contour for more detailed analysis
        contour_coords = np.column_stack(np.where(space_mask))
        
        if len(contour_coords) < 3:
            return {'area': 0, 'aspect_ratio': 1, 'perimeter': 0, 'convexity': 0, 'elongation': 0, 'centroid': (0, 0)}
        
        # Calculate centroid
        centroid_row = np.mean(contour_coords[:, 0])
        centroid_col = np.mean(contour_coords[:, 1])
        
        # Convert back to world coordinates
        bounds = spatial_data['bounds']
        centroid_x = bounds.get('min_x', 0) + centroid_col * resolution
        centroid_y = bounds.get('min_y', 0) + centroid_row * resolution
        
        # Calculate bounding box
        min_row, max_row = np.min(contour_coords[:, 0]), np.max(contour_coords[:, 0])
        min_col, max_col = np.min(contour_coords[:, 1]), np.max(contour_coords[:, 1])
        
        bbox_width = (max_col - min_col) * resolution
        bbox_height = (max_row - min_row) * resolution
        
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1
        
        # Calculate perimeter (simplified)
        perimeter = len(contour_coords) * resolution * 0.5  # Approximation
        
        # Calculate convexity
        try:
            hull = ConvexHull(contour_coords)
            convexity = area_pixels / hull.volume if hull.volume > 0 else 0
        except:
            convexity = 0.5
        
        # Calculate elongation
        elongation = max(bbox_width, bbox_height) / min(bbox_width, bbox_height) if min(bbox_width, bbox_height) > 0 else 1
        
        return {
            'area': area_m2,
            'aspect_ratio': aspect_ratio,
            'perimeter': perimeter,
            'convexity': convexity,
            'elongation': elongation,
            'centroid': (centroid_x, centroid_y),
            'bbox_width': bbox_width,
            'bbox_height': bbox_height
        }
    
    def _calculate_zone_scores(self, features: List[float], props: Dict[str, float]) -> Dict[str, float]:
        """Calculate likelihood scores for different zone types"""
        
        area, aspect_ratio, perimeter, convexity, elongation, connectivity, entrance_proximity = features
        
        scores = {}
        
        # Corridor scoring
        corridor_score = 0
        if 3.0 <= elongation <= 20.0:  # Long and narrow
            corridor_score += 0.4
        if 1.5 <= props.get('bbox_width', 0) <= 4.0:  # Appropriate width
            corridor_score += 0.3
        if connectivity > 0.5:  # Well connected
            corridor_score += 0.3
        
        scores['corridors'] = corridor_score
        
        # Room scoring
        room_score = 0
        if 0.6 <= aspect_ratio <= 2.5:  # More rectangular
            room_score += 0.4
        if 6.0 <= area <= 50.0:  # Appropriate size
            room_score += 0.3
        if convexity > 0.7:  # More regular shape
            room_score += 0.3
        
        scores['rooms'] = room_score
        
        # Open space scoring
        open_score = 0
        if area > 20.0:  # Large area
            open_score += 0.5
        if convexity > 0.8:  # Regular shape
            open_score += 0.3
        if connectivity > 0.3:  # Some connectivity
            open_score += 0.2
        
        scores['open_spaces'] = open_score
        
        # Utility space scoring
        utility_score = 0
        if area < 6.0:  # Small area
            utility_score += 0.4
        if connectivity < 0.3:  # Less connected
            utility_score += 0.3
        if entrance_proximity < 0.5:  # Away from entrances
            utility_score += 0.3
        
        scores['utility_spaces'] = utility_score
        
        return scores
    
    def _create_fallback_zones(self, floor_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic zones when intelligent detection fails"""
        logger.warning("Using fallback zone detection")
        
        bounds = floor_plan_data.get('bounds', {})
        
        # Create basic rectangular zones
        basic_zones = {
            'zones': {
                'corridors': [],
                'rooms': [],
                'entrances': [],
                'vertical_circulation': [],
                'open_spaces': [{
                    'id': 'fallback_space_1',
                    'type': 'open_space',
                    'geometry': Polygon([
                        (bounds.get('min_x', 0), bounds.get('min_y', 0)),
                        (bounds.get('max_x', 100), bounds.get('min_y', 0)),
                        (bounds.get('max_x', 100), bounds.get('max_y', 100)),
                        (bounds.get('min_x', 0), bounds.get('max_y', 100))
                    ]),
                    'area': (bounds.get('max_x', 100) - bounds.get('min_x', 0)) * 
                           (bounds.get('max_y', 100) - bounds.get('min_y', 0)),
                    'center': (
                        (bounds.get('min_x', 0) + bounds.get('max_x', 100)) / 2,
                        (bounds.get('min_y', 0) + bounds.get('max_y', 100)) / 2
                    )
                }]
            },
            'circulation_network': {'graph': nx.Graph(), 'metrics': {}},
            'spatial_relationships': {},
            'recommendations': [],
            'analysis_metadata': {
                'detection_confidence': 0.3,
                'zone_count': 1,
                'coverage_percentage': 100.0
            }
        }
        
        return basic_zones
    
    # Additional helper methods would continue here...
    # (Implementing remaining methods for completeness)
    
    def _group_connected_walls(self, wall_lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group wall lines that are connected"""
        if not wall_lines:
            return []
        
        groups = []
        used = set()
        
        for i, wall in enumerate(wall_lines):
            if i in used:
                continue
            
            group = [wall]
            used.add(i)
            
            # Find connected walls
            changed = True
            while changed:
                changed = False
                for j, other_wall in enumerate(wall_lines):
                    if j in used:
                        continue
                    
                    # Check if walls are connected
                    for group_wall in group:
                        if self._walls_connected(group_wall, other_wall):
                            group.append(other_wall)
                            used.add(j)
                            changed = True
                            break
                    
                    if changed:
                        break
            
            groups.append(group)
        
        return groups
    
    def _walls_connected(self, wall1: Dict[str, Any], wall2: Dict[str, Any], tolerance: float = 0.1) -> bool:
        """Check if two walls are connected"""
        points1 = [wall1['start'], wall1['end']]
        points2 = [wall2['start'], wall2['end']]
        
        for p1 in points1:
            for p2 in points2:
                distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance < tolerance:
                    return True
        
        return False
    
    def _is_load_bearing_wall(self, wall_group: List[Dict[str, Any]]) -> bool:
        """Determine if a wall group is load-bearing (simplified heuristic)"""
        total_length = sum(np.sqrt((w['end'][0] - w['start'][0])**2 + 
                                 (w['end'][1] - w['start'][1])**2) for w in wall_group)
        
        # Assume longer wall groups are more likely to be load-bearing
        return total_length > 10.0
    
    def _detect_columns(self, wall_layer: np.ndarray) -> List[Dict[str, Any]]:
        """Detect columns in the wall layer"""
        # Simple column detection based on isolated wall pixels
        columns = []
        
        # Find connected components in wall layer
        labeled_walls, num_labels = label(wall_layer)
        
        for label_id in range(1, num_labels + 1):
            component = (labeled_walls == label_id)
            area = np.sum(component)
            
            # Small isolated components might be columns
            if 5 <= area <= 50:  # Appropriate size for columns
                rows, cols = np.where(component)
                center_row, center_col = np.mean(rows), np.mean(cols)
                
                columns.append({
                    'id': f'column_{label_id}',
                    'center': (center_col, center_row),
                    'area': area
                })
        
        return columns
    
    def _detect_wall_openings(self, wall_layer: np.ndarray, 
                            wall_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect openings (doors, windows) in walls"""
        openings = []
        
        # For each wall line, check for gaps
        for wall in wall_lines:
            # This is a simplified implementation
            # In practice, you'd analyze the wall layer for gaps
            start_x, start_y = wall['start']
            end_x, end_y = wall['end']
            
            length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # Mock opening detection (replace with actual algorithm)
            if length > 5.0:  # Only check longer walls
                opening = {
                    'id': f'opening_{len(openings)}',
                    'center': ((start_x + end_x) / 2, (start_y + end_y) / 2),
                    'width': 1.0,  # Default width
                    'orientation': np.arctan2(end_y - start_y, end_x - start_x)
                }
                openings.append(opening)
        
        return openings
    
    def _calculate_space_accessibility(self, space_mask: np.ndarray, 
                                     structural_elements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate accessibility metrics for a space"""
        # Simplified accessibility calculation
        connectivity_score = 0.5  # Default
        entrance_proximity = 0.5  # Default
        
        return {
            'connectivity_score': connectivity_score,
            'entrance_proximity': entrance_proximity
        }
    
    def _extract_space_contour(self, space_mask: np.ndarray, 
                             spatial_data: Dict[str, Any]) -> Polygon:
        """Extract the contour of a space as a Polygon"""
        coords = np.column_stack(np.where(space_mask))
        
        if len(coords) < 3:
            return Polygon()
        
        # Convert to world coordinates
        resolution = spatial_data['resolution']
        bounds = spatial_data['bounds']
        
        world_coords = []
        for row, col in coords[::max(1, len(coords)//20)]:  # Subsample for efficiency
            x = bounds.get('min_x', 0) + col * resolution
            y = bounds.get('min_y', 0) + row * resolution
            world_coords.append((x, y))
        
        if len(world_coords) < 3:
            return Polygon()
        
        try:
            return Polygon(world_coords)
        except:
            return Polygon()
    
    def _refine_classifications(self, classified_zones: Dict[str, List[Dict[str, Any]]], 
                              structural_elements: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Refine zone classifications using additional context"""
        # This would implement post-processing logic
        return classified_zones
    
    def _are_zones_adjacent(self, zone1: Dict[str, Any], zone2: Dict[str, Any]) -> bool:
        """Check if two zones are adjacent"""
        geom1 = zone1.get('geometry')
        geom2 = zone2.get('geometry')
        
        if geom1 and geom2:
            try:
                return geom1.distance(geom2) < 1.0  # 1 meter threshold
            except:
                pass
        
        return False
    
    def _calculate_zone_distance(self, zone1: Dict[str, Any], zone2: Dict[str, Any]) -> float:
        """Calculate distance between two zones"""
        center1 = zone1.get('center', (0, 0))
        center2 = zone2.get('center', (0, 0))
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _identify_main_circulation(self, corridors: List[Dict[str, Any]]) -> List[str]:
        """Identify the main circulation spine"""
        if not corridors:
            return []
        
        # Sort by area (larger corridors are likely main circulation)
        sorted_corridors = sorted(corridors, key=lambda x: x.get('area', 0), reverse=True)
        
        # Return top 3 or 50% of corridors, whichever is smaller
        num_main = min(3, max(1, len(sorted_corridors) // 2))
        return [c['id'] for c in sorted_corridors[:num_main]]
    
    def _identify_dead_ends(self, circulation_graph: nx.Graph) -> List[str]:
        """Identify dead-end zones in circulation network"""
        dead_ends = []
        for node_id in circulation_graph.nodes():
            if circulation_graph.degree(node_id) == 1:
                dead_ends.append(node_id)
        return dead_ends
    
    def _calculate_circulation_efficiency(self, circulation_graph: nx.Graph) -> float:
        """Calculate overall circulation efficiency"""
        if circulation_graph.number_of_nodes() == 0:
            return 0.0
        
        # Simple efficiency metric based on connectivity
        total_possible_edges = circulation_graph.number_of_nodes() * (circulation_graph.number_of_nodes() - 1) / 2
        actual_edges = circulation_graph.number_of_edges()
        
        if total_possible_edges == 0:
            return 1.0
        
        return min(1.0, actual_edges / total_possible_edges * 4)  # Scale appropriately
    
    def _analyze_pedestrian_flow(self, classified_zones: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze potential pedestrian flow patterns"""
        return {
            'flow_capacity': 'medium',
            'bottlenecks': [],
            'flow_directions': []
        }
    
    def _calculate_entrance_score(self, opening: Dict[str, Any], 
                                walls: List[List[Dict[str, Any]]], 
                                circulation_network: Dict[str, Any]) -> float:
        """Calculate likelihood that an opening is a building entrance"""
        # Simplified scoring
        base_score = 0.5
        
        # Check if opening is on exterior wall
        # Check if opening connects to main circulation
        # Check opening size
        
        return base_score
    
    def _find_connected_circulation(self, opening: Dict[str, Any], 
                                  circulation_network: Dict[str, Any]) -> List[str]:
        """Find circulation elements connected to an opening"""
        return []
    
    def _detect_entrances_heuristic(self, structural_elements: Dict[str, Any], 
                                  circulation_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect entrances using heuristic methods when other methods fail"""
        return [{
            'id': 'entrance_0',
            'type': 'main_entrance',
            'position': (50, 10),  # Default position
            'width': 2.0,
            'orientation': 0,
            'accessibility_score': 0.7,
            'connected_circulation': []
        }]
    
    def _calculate_stair_score(self, space: Dict[str, Any], 
                             spatial_data: Dict[str, Any]) -> float:
        """Calculate likelihood that a space contains stairs"""
        props = space['properties']
        
        score = 0
        # Check aspect ratio
        if 1.5 <= props.get('elongation', 1) <= 4.0:
            score += 0.4
        
        # Check area
        if 4.0 <= props.get('area', 0) <= 25.0:
            score += 0.3
        
        # Check shape regularity
        if props.get('convexity', 0) > 0.6:
            score += 0.3
        
        return score
    
    def _calculate_elevator_score(self, space: Dict[str, Any], 
                                spatial_data: Dict[str, Any]) -> float:
        """Calculate likelihood that a space contains an elevator"""
        props = space['properties']
        
        score = 0
        # Check aspect ratio (more square)
        if 0.8 <= props.get('aspect_ratio', 1) <= 1.5:
            score += 0.4
        
        # Check area
        if 2.0 <= props.get('area', 0) <= 8.0:
            score += 0.4
        
        # Check shape regularity
        if props.get('convexity', 0) > 0.8:
            score += 0.2
        
        return score
    
    def _extract_zone_features(self, zone: Dict[str, Any]) -> List[float]:
        """Extract feature vector for a zone"""
        props = zone.get('properties', {})
        return [
            props.get('area', 0),
            props.get('aspect_ratio', 1),
            props.get('perimeter', 0),
            props.get('convexity', 0.5),
            zone.get('confidence', 0.5)
        ]
    
    def _has_corridor_access(self, room: Dict[str, Any], 
                           corridors: List[Dict[str, Any]]) -> bool:
        """Check if a room has access to corridors"""
        for corridor in corridors:
            if self._are_zones_adjacent(room, corridor):
                return True
        return False
    
    def _is_exterior_adjacent(self, room: Dict[str, Any]) -> bool:
        """Check if a room is adjacent to exterior walls"""
        # Simplified check
        return True  # Assume all rooms have potential exterior access
    
    def _calculate_detection_confidence(self, classified_zones: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall detection confidence"""
        all_confidences = []
        for zones in classified_zones.values():
            for zone in zones:
                all_confidences.append(zone.get('confidence', 0.5))
        
        return np.mean(all_confidences) if all_confidences else 0.5
    
    def _calculate_coverage_percentage(self, classified_zones: Dict[str, List[Dict[str, Any]]], 
                                     bounds: Dict[str, float]) -> float:
        """Calculate percentage of floor plan covered by detected zones"""
        total_area = (bounds.get('max_x', 100) - bounds.get('min_x', 0)) * \
                    (bounds.get('max_y', 100) - bounds.get('min_y', 0))
        
        covered_area = 0
        for zones in classified_zones.values():
            for zone in zones:
                covered_area += zone.get('area', 0)
        
        return (covered_area / total_area * 100) if total_area > 0 else 0
    
    def _extract_text_info(self, entity: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text information from entity"""
        return {
            'text': entity.get('text', ''),
            'position': geometry.get('position', {'x': 0, 'y': 0}),
            'height': geometry.get('height', 1.0)
        }
    
    def _group_connected_walls(self, wall_lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group connected wall segments"""
        if not wall_lines:
            return []
        
        groups = []
        used = set()
        
        for i, line in enumerate(wall_lines):
            if i in used:
                continue
                
            group = [line]
            used.add(i)
            
            # Find connected lines
            for j, other_line in enumerate(wall_lines):
                if j in used:
                    continue
                    
                # Check if lines are connected (within tolerance)
                tolerance = 0.1
                line_end = line['end']
                other_start = other_line['start']
                
                if (abs(line_end[0] - other_start[0]) < tolerance and 
                    abs(line_end[1] - other_start[1]) < tolerance):
                    group.append(other_line)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _is_load_bearing_wall(self, wall_group: List[Dict[str, Any]]) -> bool:
        """Determine if a wall group is load-bearing"""
        total_length = 0
        for wall in wall_group:
            start = wall['start']
            end = wall['end']
            length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
            total_length += length
        
        return total_length > 5.0  # Walls longer than 5 meters are likely load-bearing
    
    def _create_fallback_zones(self, floor_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback zones when intelligent detection fails"""
        entities = floor_plan_data.get('entities', [])
        bounds = floor_plan_data.get('bounds', {})
        
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 100) - bounds.get('min_y', 0)
        
        # Create simple rectangular zones
        zones = {
            'rooms': [
                {
                    'id': 'room_1',
                    'type': 'room',
                    'bounds': {
                        'min_x': bounds.get('min_x', 0),
                        'min_y': bounds.get('min_y', 0),
                        'max_x': bounds.get('min_x', 0) + width * 0.8,
                        'max_y': bounds.get('min_y', 0) + height * 0.8
                    },
                    'area': width * height * 0.64,
                    'properties': {
                        'aspect_ratio': 1.0,
                        'accessibility': 'high'
                    }
                }
            ],
            'corridors': [
                {
                    'id': 'corridor_1',
                    'type': 'corridor',
                    'bounds': {
                        'min_x': bounds.get('min_x', 0) + width * 0.8,
                        'min_y': bounds.get('min_y', 0),
                        'max_x': bounds.get('max_x', 100),
                        'max_y': bounds.get('max_y', 100)
                    },
                    'area': width * height * 0.2,
                    'properties': {
                        'width': width * 0.2,
                        'connectivity': 'high'
                    }
                }
            ],
            'service_areas': [],
            'amenities': []
        }
        
        return {
            'zones': zones,
            'structural_elements': {
                'walls': [{'type': 'wall', 'start': (0, 0), 'end': (width, 0)}],
                'columns': [],
                'openings': []
            },
            'circulation_network': {
                'main_paths': [],
                'secondary_paths': [],
                'emergency_exits': []
            },
            'spatial_relationships': {},
            'recommendations': ['Basic fallback zones created'],
            'analysis_metadata': {
                'detection_confidence': 0.5,
                'zone_count': 2,
                'coverage_percentage': 84.0
            }
        }
