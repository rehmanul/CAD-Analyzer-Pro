"""
Intelligent Zone Detection Module
Automatically detects and classifies zones in floor plans
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
import networkx as nx

class IntelligentZoneDetector:
    """Advanced zone detection using machine learning and spatial analysis"""
    
    def __init__(self):
        self.zone_types = {
            'entrance': {'color': '#FF0000', 'priority': 1},
            'corridor': {'color': '#FFB84D', 'priority': 2},
            'room': {'color': '#90EE90', 'priority': 3},
            'restricted': {'color': '#87CEEB', 'priority': 0},
            'open_space': {'color': '#F0F0F0', 'priority': 4},
            'emergency': {'color': '#FF6B6B', 'priority': 1}
        }
        
    def detect_zones(self, walls: List[Dict], entities: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Intelligently detect and classify zones based on spatial patterns
        """
        # Extract wall geometries
        wall_lines = self._extract_wall_lines(walls)
        
        # Detect rooms and spaces
        rooms = self._detect_rooms(wall_lines)
        
        # Detect entrances
        entrances = self._detect_entrances(rooms, wall_lines)
        
        # Detect corridors
        corridors = self._detect_corridors(rooms, entrances, wall_lines)
        
        # Detect restricted areas (stairs, elevators)
        restricted = self._detect_restricted_areas(entities, rooms)
        
        # Detect emergency routes
        emergency_routes = self._detect_emergency_routes(entrances, corridors)
        
        # Classify remaining spaces
        open_spaces = self._classify_open_spaces(rooms, corridors, restricted)
        
        return {
            'rooms': rooms,
            'entrances': entrances,
            'corridors': corridors,
            'restricted': restricted,
            'open_spaces': open_spaces,
            'emergency_routes': emergency_routes
        }
    
    def _extract_wall_lines(self, walls: List[Dict]) -> List[LineString]:
        """Extract wall geometries as LineString objects"""
        lines = []
        for wall in walls:
            if 'start' in wall and 'end' in wall:
                line = LineString([
                    (wall['start']['x'], wall['start']['y']),
                    (wall['end']['x'], wall['end']['y'])
                ])
                lines.append(line)
        return lines
    
    def _detect_rooms(self, wall_lines: List[LineString]) -> List[Dict]:
        """Detect enclosed spaces that form rooms"""
        rooms = []
        
        # Create a network graph from walls
        G = nx.Graph()
        
        # Add wall segments as edges
        for wall in wall_lines:
            coords = list(wall.coords)
            for i in range(len(coords) - 1):
                G.add_edge(coords[i], coords[i + 1])
        
        # Find cycles (enclosed spaces)
        cycles = nx.cycle_basis(G)
        
        for i, cycle in enumerate(cycles):
            if len(cycle) >= 3:  # Valid polygon
                polygon = Polygon(cycle)
                if polygon.is_valid and polygon.area > 10:  # Minimum area threshold
                    room = {
                        'id': f'room_{i}',
                        'type': 'room',
                        'geometry': polygon,
                        'area': polygon.area,
                        'centroid': polygon.centroid,
                        'bounds': polygon.bounds,
                        'perimeter': polygon.length,
                        'shape_factor': self._calculate_shape_factor(polygon)
                    }
                    
                    # Classify room type based on characteristics
                    room['subtype'] = self._classify_room_type(room)
                    rooms.append(room)
        
        return rooms
    
    def _detect_entrances(self, rooms: List[Dict], wall_lines: List[LineString]) -> List[Dict]:
        """Detect entrance points based on spatial patterns"""
        entrances = []
        
        # Find gaps in walls that could be doors
        wall_union = unary_union(wall_lines)
        
        # Analyze room connections
        for i, room in enumerate(rooms):
            boundary = room['geometry'].boundary
            
            # Check for openings in room boundaries
            intersections = []
            for wall in wall_lines:
                if boundary.intersects(wall):
                    intersection = boundary.intersection(wall)
                    if intersection.length < boundary.length * 0.1:  # Small gap
                        intersections.append(intersection)
            
            # Identify main entrances based on patterns
            if room['subtype'] in ['lobby', 'reception', 'entrance_hall']:
                # This is likely an entrance room
                entrance = {
                    'id': f'entrance_{i}',
                    'type': 'entrance',
                    'room_id': room['id'],
                    'geometry': room['geometry'],
                    'priority': 'main',
                    'width': self._estimate_entrance_width(room, wall_lines)
                }
                entrances.append(entrance)
        
        # Detect external entrances
        external_entrances = self._detect_external_entrances(rooms, wall_lines)
        entrances.extend(external_entrances)
        
        return entrances
    
    def _detect_corridors(self, rooms: List[Dict], entrances: List[Dict], 
                         wall_lines: List[LineString]) -> List[Dict]:
        """Detect corridors based on shape and connectivity"""
        corridors = []
        
        for i, room in enumerate(rooms):
            # Check if room has corridor characteristics
            if self._is_corridor_shape(room):
                corridor = {
                    'id': f'corridor_{i}',
                    'type': 'corridor',
                    'geometry': room['geometry'],
                    'width': self._calculate_corridor_width(room),
                    'length': self._calculate_corridor_length(room),
                    'connectivity': self._calculate_connectivity(room, rooms),
                    'traffic_flow': 'bidirectional'
                }
                
                # Classify corridor type
                corridor['subtype'] = self._classify_corridor_type(corridor, entrances)
                corridors.append(corridor)
        
        return corridors
    
    def _detect_restricted_areas(self, entities: List[Dict], rooms: List[Dict]) -> List[Dict]:
        """Detect restricted areas like stairs and elevators"""
        restricted = []
        
        # Detect based on entity attributes
        for entity in entities:
            if entity.get('type') in ['stairs', 'elevator', 'shaft', 'mechanical']:
                restricted_area = {
                    'id': f'restricted_{len(restricted)}',
                    'type': 'restricted',
                    'subtype': entity['type'],
                    'geometry': self._create_geometry_from_entity(entity),
                    'access_level': 'no_ilot'
                }
                restricted.append(restricted_area)
        
        # Detect based on room characteristics
        for room in rooms:
            if room['area'] < 20 and room['shape_factor'] < 0.5:
                # Small, irregular rooms might be utility spaces
                restricted_area = {
                    'id': f'restricted_{len(restricted)}',
                    'type': 'restricted',
                    'subtype': 'utility',
                    'geometry': room['geometry'],
                    'access_level': 'no_ilot'
                }
                restricted.append(restricted_area)
        
        return restricted
    
    def _detect_emergency_routes(self, entrances: List[Dict], 
                               corridors: List[Dict]) -> List[Dict]:
        """Detect emergency evacuation routes"""
        emergency_routes = []
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes for entrances and corridor intersections
        for entrance in entrances:
            G.add_node(entrance['id'], type='entrance', 
                      pos=entrance['geometry'].centroid.coords[0])
        
        for corridor in corridors:
            G.add_node(corridor['id'], type='corridor',
                      pos=corridor['geometry'].centroid.coords[0])
        
        # Connect nodes based on proximity and accessibility
        for entrance in entrances:
            for corridor in corridors:
                if self._are_connected(entrance['geometry'], corridor['geometry']):
                    G.add_edge(entrance['id'], corridor['id'])
        
        # Find shortest paths to exits
        for node in G.nodes():
            if G.nodes[node]['type'] == 'corridor':
                paths_to_exits = []
                for entrance in entrances:
                    try:
                        path = nx.shortest_path(G, node, entrance['id'])
                        paths_to_exits.append(path)
                    except nx.NetworkXNoPath:
                        continue
                
                if paths_to_exits:
                    route = {
                        'id': f'emergency_route_{len(emergency_routes)}',
                        'type': 'emergency_route',
                        'from': node,
                        'paths': paths_to_exits,
                        'primary_exit': min(paths_to_exits, key=len)
                    }
                    emergency_routes.append(route)
        
        return emergency_routes
    
    def _classify_open_spaces(self, rooms: List[Dict], corridors: List[Dict],
                            restricted: List[Dict]) -> List[Dict]:
        """Classify remaining open spaces"""
        open_spaces = []
        
        # Identify spaces not classified as rooms, corridors, or restricted
        all_classified = []
        for space_list in [rooms, corridors, restricted]:
            all_classified.extend([s['geometry'] for s in space_list])
        
        # Union all classified spaces
        if all_classified:
            classified_union = unary_union(all_classified)
            
            # Remaining spaces are open areas
            # This would require the total floor boundary
            # For now, we'll identify large rooms as potential open spaces
            for room in rooms:
                if room['area'] > 100 and room['shape_factor'] > 0.7:
                    open_space = {
                        'id': f'open_space_{len(open_spaces)}',
                        'type': 'open_space',
                        'geometry': room['geometry'],
                        'area': room['area'],
                        'capacity': self._estimate_capacity(room['area'])
                    }
                    open_spaces.append(open_space)
        
        return open_spaces
    
    def _calculate_shape_factor(self, polygon: Polygon) -> float:
        """Calculate shape regularity factor (0-1, 1 being most regular)"""
        area = polygon.area
        perimeter = polygon.length
        # Circularity measure
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity
    
    def _classify_room_type(self, room: Dict) -> str:
        """Classify room based on characteristics"""
        area = room['area']
        shape_factor = room['shape_factor']
        
        if area < 15:
            return 'small_office'
        elif area < 30:
            return 'office'
        elif area < 50:
            return 'meeting_room'
        elif area > 100 and shape_factor > 0.7:
            return 'open_office'
        elif area > 200:
            return 'hall'
        else:
            return 'general_room'
    
    def _is_corridor_shape(self, room: Dict) -> bool:
        """Check if room has corridor-like characteristics"""
        polygon = room['geometry']
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        aspect_ratio = max(width, height) / min(width, height)
        
        # Corridors are typically elongated
        return aspect_ratio > 3 and room['shape_factor'] > 0.6
    
    def _calculate_corridor_width(self, corridor: Dict) -> float:
        """Calculate average corridor width"""
        polygon = corridor['geometry']
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Return the smaller dimension as width
        return min(width, height)
    
    def _calculate_corridor_length(self, corridor: Dict) -> float:
        """Calculate corridor length"""
        polygon = corridor['geometry']
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Return the larger dimension as length
        return max(width, height)
    
    def _calculate_connectivity(self, room: Dict, all_rooms: List[Dict]) -> int:
        """Calculate how many other rooms this room connects to"""
        connections = 0
        for other_room in all_rooms:
            if other_room['id'] != room['id']:
                if room['geometry'].touches(other_room['geometry']):
                    connections += 1
        return connections
    
    def _classify_corridor_type(self, corridor: Dict, entrances: List[Dict]) -> str:
        """Classify corridor type based on location and connections"""
        # Check if corridor connects to entrance
        for entrance in entrances:
            if corridor['geometry'].intersects(entrance['geometry']):
                return 'main_corridor'
        
        if corridor['connectivity'] > 4:
            return 'central_corridor'
        elif corridor['width'] > 3:
            return 'wide_corridor'
        else:
            return 'secondary_corridor'
    
    def _estimate_entrance_width(self, room: Dict, wall_lines: List[LineString]) -> float:
        """Estimate entrance width based on gaps in walls"""
        # Simplified estimation
        return 2.0  # Default 2 meter entrance
    
    def _detect_external_entrances(self, rooms: List[Dict], 
                                  wall_lines: List[LineString]) -> List[Dict]:
        """Detect entrances from building exterior"""
        # This would analyze the building perimeter
        # For now, return empty list
        return []
    
    def _create_geometry_from_entity(self, entity: Dict) -> Polygon:
        """Create polygon geometry from entity data"""
        if 'points' in entity:
            return Polygon(entity['points'])
        elif 'bounds' in entity:
            b = entity['bounds']
            return Polygon([
                (b['min_x'], b['min_y']),
                (b['max_x'], b['min_y']),
                (b['max_x'], b['max_y']),
                (b['min_x'], b['max_y'])
            ])
        else:
            # Default small square
            return Polygon([
                (0, 0), (1, 0), (1, 1), (0, 1)
            ])
    
    def _are_connected(self, geom1: Polygon, geom2: Polygon) -> bool:
        """Check if two geometries are connected"""
        return geom1.touches(geom2) or geom1.intersects(geom2)
    
    def _estimate_capacity(self, area: float) -> int:
        """Estimate capacity based on area"""
        # Assuming 10 sqm per person for open spaces
        return int(area / 10)