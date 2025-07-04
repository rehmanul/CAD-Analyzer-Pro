import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiLineString
from shapely.ops import unary_union, snap
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorridorGenerator:
    """Intelligent corridor generation with advanced pathfinding algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'A-Star': self._astar_pathfinding,
            'Dijkstra': self._dijkstra_pathfinding,
            'Breadth-First': self._bfs_pathfinding
        }
        
        self.corridor_types = {
            'main': {'min_width': 2.0, 'max_width': 4.0, 'priority': 1},
            'secondary': {'min_width': 1.5, 'max_width': 2.5, 'priority': 2},
            'access': {'min_width': 1.0, 'max_width': 1.8, 'priority': 3},
            'emergency': {'min_width': 1.2, 'max_width': 2.0, 'priority': 1}
        }
    
    def generate_corridors(self, analysis_results: Dict[str, Any],
                         corridor_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate intelligent corridor network
        
        Args:
            analysis_results: Results from zone and îlot analysis
            corridor_config: Corridor configuration parameters
            
        Returns:
            List of generated corridors with properties
        """
        logger.info("Starting corridor generation")
        
        try:
            # Extract configuration
            width = corridor_config.get('width', 1.5)
            main_width = corridor_config.get('main_width', 2.0)
            algorithm = corridor_config.get('algorithm', 'A-Star')
            optimize_turns = corridor_config.get('optimize_turns', True)
            avoid_obstacles = corridor_config.get('avoid_obstacles', True)
            ensure_accessibility = corridor_config.get('ensure_accessibility', True)
            
            # Get spatial elements
            ilots = analysis_results.get('ilots', [])
            entrances = analysis_results.get('entrances', [])
            walls = analysis_results.get('walls', [])
            restricted_areas = analysis_results.get('restricted_areas', [])
            open_spaces = analysis_results.get('open_spaces', [])
            
            if not ilots:
                logger.warning("No îlots found for corridor generation")
                return []
            
            # Build navigation graph
            navigation_graph = self._build_navigation_graph(
                ilots, entrances, walls, restricted_areas, open_spaces
            )
            
            # Generate corridor network
            corridors = self._generate_corridor_network(
                navigation_graph, ilots, entrances, algorithm, width, main_width
            )
            
            # Optimize corridors
            if optimize_turns:
                corridors = self._optimize_corridor_turns(corridors)
            
            # Ensure accessibility compliance
            if ensure_accessibility:
                corridors = self._ensure_accessibility_compliance(corridors)
            
            # Validate and finalize corridors
            validated_corridors = self._validate_corridors(
                corridors, walls, restricted_areas, avoid_obstacles
            )
            
            logger.info(f"Generated {len(validated_corridors)} corridors")
            return validated_corridors
            
        except Exception as e:
            logger.error(f"Error during corridor generation: {str(e)}")
            raise
    
    def _build_navigation_graph(self, ilots: List[Dict[str, Any]],
                               entrances: List[Dict[str, Any]],
                               walls: List[Dict[str, Any]],
                               restricted_areas: List[Dict[str, Any]],
                               open_spaces: List[Dict[str, Any]]) -> nx.Graph:
        """Build navigation graph for pathfinding"""
        G = nx.Graph()
        
        # Add îlot nodes
        for ilot in ilots:
            pos = ilot.get('position', {})
            G.add_node(f"ilot_{ilot['id']}", 
                      pos=(pos.get('x', 0), pos.get('y', 0)),
                      type='ilot',
                      data=ilot)
        
        # Add entrance nodes
        for entrance in entrances:
            pos = entrance.get('position', {})
            G.add_node(f"entrance_{entrance['id']}", 
                      pos=(pos.get('x', 0), pos.get('y', 0)),
                      type='entrance',
                      data=entrance)
        
        # Add strategic waypoints in open spaces
        waypoints = self._generate_waypoints(open_spaces, walls, restricted_areas)
        for i, waypoint in enumerate(waypoints):
            G.add_node(f"waypoint_{i}", 
                      pos=(waypoint['x'], waypoint['y']),
                      type='waypoint',
                      data=waypoint)
        
        # Add edges between nodes
        self._add_navigation_edges(G, walls, restricted_areas)
        
        return G
    
    def _generate_waypoints(self, open_spaces: List[Dict[str, Any]],
                           walls: List[Dict[str, Any]],
                           restricted_areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate strategic waypoints for navigation"""
        waypoints = []
        
        for space in open_spaces:
            space_geom = space.get('shapely_geom')
            if not space_geom:
                continue
            
            bounds = space_geom.bounds
            center = space.get('center', {})
            
            # Generate waypoints along space perimeter and center
            waypoint_candidates = [
                # Center waypoint
                {'x': center.get('x', 0), 'y': center.get('y', 0), 'type': 'center'},
                
                # Corner waypoints (with clearance)
                {'x': bounds[0] + 1.0, 'y': bounds[1] + 1.0, 'type': 'corner'},
                {'x': bounds[2] - 1.0, 'y': bounds[1] + 1.0, 'type': 'corner'},
                {'x': bounds[2] - 1.0, 'y': bounds[3] - 1.0, 'type': 'corner'},
                {'x': bounds[0] + 1.0, 'y': bounds[3] - 1.0, 'type': 'corner'},
                
                # Edge midpoints
                {'x': (bounds[0] + bounds[2]) / 2, 'y': bounds[1] + 1.0, 'type': 'edge'},
                {'x': (bounds[0] + bounds[2]) / 2, 'y': bounds[3] - 1.0, 'type': 'edge'},
                {'x': bounds[0] + 1.0, 'y': (bounds[1] + bounds[3]) / 2, 'type': 'edge'},
                {'x': bounds[2] - 1.0, 'y': (bounds[1] + bounds[3]) / 2, 'type': 'edge'}
            ]
            
            # Filter waypoints that are within the space and not blocked
            for waypoint in waypoint_candidates:
                point = Point(waypoint['x'], waypoint['y'])
                
                if (space_geom.contains(point) and 
                    self._is_waypoint_valid(waypoint, walls, restricted_areas)):
                    waypoints.append(waypoint)
        
        return waypoints
    
    def _is_waypoint_valid(self, waypoint: Dict[str, Any],
                          walls: List[Dict[str, Any]],
                          restricted_areas: List[Dict[str, Any]]) -> bool:
        """Check if waypoint is valid (not blocked by obstacles)"""
        point = Point(waypoint['x'], waypoint['y'])
        clearance = 1.0  # Minimum clearance in meters
        
        # Check walls
        for wall in walls:
            wall_geom = wall.get('shapely_geom')
            if wall_geom and point.distance(wall_geom) < clearance:
                return False
        
        # Check restricted areas
        for restricted in restricted_areas:
            restricted_geom = restricted.get('shapely_geom')
            if restricted_geom and restricted_geom.contains(point):
                return False
        
        return True
    
    def _add_navigation_edges(self, G: nx.Graph, walls: List[Dict[str, Any]],
                            restricted_areas: List[Dict[str, Any]]):
        """Add edges between nodes in navigation graph"""
        nodes = list(G.nodes())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                pos1 = G.nodes[node1]['pos']
                pos2 = G.nodes[node2]['pos']
                
                # Calculate distance
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Check if direct path is clear
                if self._is_path_clear(pos1, pos2, walls, restricted_areas):
                    # Add edge weight based on distance and path type
                    weight = self._calculate_edge_weight(G.nodes[node1], G.nodes[node2], distance)
                    G.add_edge(node1, node2, weight=weight, distance=distance)
    
    def _is_path_clear(self, pos1: Tuple[float, float], pos2: Tuple[float, float],
                      walls: List[Dict[str, Any]], restricted_areas: List[Dict[str, Any]]) -> bool:
        """Check if path between two points is clear"""
        path_line = LineString([pos1, pos2])
        buffer_distance = 0.75  # Half corridor width
        
        # Check walls
        for wall in walls:
            wall_geom = wall.get('shapely_geom')
            if wall_geom and path_line.distance(wall_geom) < buffer_distance:
                return False
        
        # Check restricted areas
        for restricted in restricted_areas:
            restricted_geom = restricted.get('shapely_geom')
            if restricted_geom and path_line.intersects(restricted_geom):
                return False
        
        return True
    
    def _calculate_edge_weight(self, node1: Dict[str, Any], node2: Dict[str, Any], distance: float) -> float:
        """Calculate edge weight for pathfinding"""
        base_weight = distance
        
        # Penalty for connections between different types
        if node1['type'] != node2['type']:
            base_weight *= 1.2
        
        # Bonus for entrance connections
        if node1['type'] == 'entrance' or node2['type'] == 'entrance':
            base_weight *= 0.8
        
        # Penalty for long distances
        if distance > 10.0:
            base_weight *= 1.5
        
        return base_weight
    
    def _generate_corridor_network(self, G: nx.Graph, ilots: List[Dict[str, Any]],
                                 entrances: List[Dict[str, Any]], algorithm: str,
                                 width: float, main_width: float) -> List[Dict[str, Any]]:
        """Generate corridor network using specified algorithm"""
        corridors = []
        
        # Get pathfinding algorithm
        pathfinding_func = self.algorithms.get(algorithm, self._astar_pathfinding)
        
        # Generate main corridors (entrance to îlot connections)
        main_corridors = self._generate_main_corridors(
            G, ilots, entrances, pathfinding_func, main_width
        )
        corridors.extend(main_corridors)
        
        # Generate secondary corridors (îlot to îlot connections)
        secondary_corridors = self._generate_secondary_corridors(
            G, ilots, pathfinding_func, width
        )
        corridors.extend(secondary_corridors)
        
        # Generate access corridors for isolated îlots
        access_corridors = self._generate_access_corridors(
            G, ilots, corridors, pathfinding_func, width
        )
        corridors.extend(access_corridors)
        
        return corridors
    
    def _generate_main_corridors(self, G: nx.Graph, ilots: List[Dict[str, Any]],
                               entrances: List[Dict[str, Any]], pathfinding_func,
                               width: float) -> List[Dict[str, Any]]:
        """Generate main corridors from entrances to îlots"""
        main_corridors = []
        
        for entrance in entrances:
            entrance_node = f"entrance_{entrance['id']}"
            
            # Find closest îlots to this entrance
            closest_ilots = self._find_closest_ilots(G, entrance_node, ilots, max_count=3)
            
            for ilot_id in closest_ilots:
                ilot_node = f"ilot_{ilot_id}"
                
                # Find path
                path = pathfinding_func(G, entrance_node, ilot_node)
                
                if path:
                    corridor = self._create_corridor_from_path(
                        G, path, 'main', width, f"main_{entrance['id']}_{ilot_id}"
                    )
                    main_corridors.append(corridor)
        
        return main_corridors
    
    def _generate_secondary_corridors(self, G: nx.Graph, ilots: List[Dict[str, Any]],
                                    pathfinding_func, width: float) -> List[Dict[str, Any]]:
        """Generate secondary corridors between îlots"""
        secondary_corridors = []
        
        # Create minimum spanning tree for îlot connectivity
        ilot_positions = [(ilot['position']['x'], ilot['position']['y']) for ilot in ilots]
        
        if len(ilot_positions) < 2:
            return secondary_corridors
        
        # Calculate distance matrix
        distances = cdist(ilot_positions, ilot_positions)
        
        # Find key connections (not all pairs, just strategic ones)
        for i, ilot1 in enumerate(ilots):
            # Connect to 2-3 nearest neighbors
            nearest_indices = np.argsort(distances[i])[1:4]  # Skip self (index 0)
            
            for j in nearest_indices:
                if j < len(ilots) and distances[i][j] < 15.0:  # Max connection distance
                    ilot2 = ilots[j]
                    
                    ilot_node1 = f"ilot_{ilot1['id']}"
                    ilot_node2 = f"ilot_{ilot2['id']}"
                    
                    # Find path
                    path = pathfinding_func(G, ilot_node1, ilot_node2)
                    
                    if path:
                        corridor = self._create_corridor_from_path(
                            G, path, 'secondary', width, f"secondary_{ilot1['id']}_{ilot2['id']}"
                        )
                        secondary_corridors.append(corridor)
        
        return secondary_corridors
    
    def _generate_access_corridors(self, G: nx.Graph, ilots: List[Dict[str, Any]],
                                 existing_corridors: List[Dict[str, Any]],
                                 pathfinding_func, width: float) -> List[Dict[str, Any]]:
        """Generate access corridors for isolated îlots"""
        access_corridors = []
        
        # Find îlots that are not well connected
        connected_ilots = set()
        for corridor in existing_corridors:
            if corridor.get('start_node', '').startswith('ilot_'):
                connected_ilots.add(corridor['start_node'])
            if corridor.get('end_node', '').startswith('ilot_'):
                connected_ilots.add(corridor['end_node'])
        
        # Connect isolated îlots
        for ilot in ilots:
            ilot_node = f"ilot_{ilot['id']}"
            
            if ilot_node not in connected_ilots:
                # Find nearest connected node
                nearest_connected = self._find_nearest_connected_node(G, ilot_node, connected_ilots)
                
                if nearest_connected:
                    path = pathfinding_func(G, ilot_node, nearest_connected)
                    
                    if path:
                        corridor = self._create_corridor_from_path(
                            G, path, 'access', width, f"access_{ilot['id']}"
                        )
                        access_corridors.append(corridor)
        
        return access_corridors
    
    def _find_closest_ilots(self, G: nx.Graph, entrance_node: str,
                           ilots: List[Dict[str, Any]], max_count: int = 3) -> List[str]:
        """Find closest îlots to an entrance"""
        entrance_pos = G.nodes[entrance_node]['pos']
        
        distances = []
        for ilot in ilots:
            ilot_pos = (ilot['position']['x'], ilot['position']['y'])
            distance = np.sqrt((entrance_pos[0] - ilot_pos[0])**2 + 
                             (entrance_pos[1] - ilot_pos[1])**2)
            distances.append((distance, ilot['id']))
        
        # Sort by distance and return closest
        distances.sort()
        return [ilot_id for _, ilot_id in distances[:max_count]]
    
    def _find_nearest_connected_node(self, G: nx.Graph, target_node: str,
                                   connected_nodes: set) -> Optional[str]:
        """Find nearest connected node to target"""
        if not connected_nodes:
            return None
        
        target_pos = G.nodes[target_node]['pos']
        min_distance = float('inf')
        nearest_node = None
        
        for node in connected_nodes:
            if node in G.nodes:
                node_pos = G.nodes[node]['pos']
                distance = np.sqrt((target_pos[0] - node_pos[0])**2 + 
                                 (target_pos[1] - node_pos[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
        
        return nearest_node
    
    def _create_corridor_from_path(self, G: nx.Graph, path: List[str],
                                  corridor_type: str, width: float,
                                  corridor_id: str) -> Dict[str, Any]:
        """Create corridor object from path"""
        if len(path) < 2:
            return None
        
        # Get path coordinates
        coordinates = []
        for node in path:
            pos = G.nodes[node]['pos']
            coordinates.append([pos[0], pos[1]])
        
        # Create corridor geometry
        corridor_line = LineString(coordinates)
        
        # Calculate properties
        length = corridor_line.length
        start_pos = coordinates[0]
        end_pos = coordinates[-1]
        
        return {
            'id': corridor_id,
            'type': corridor_type,
            'path': path,
            'coordinates': coordinates,
            'start': {'x': start_pos[0], 'y': start_pos[1], 'z': 0},
            'end': {'x': end_pos[0], 'y': end_pos[1], 'z': 0},
            'length': length,
            'width': width,
            'area': length * width,
            'start_node': path[0],
            'end_node': path[-1],
            'shapely_geom': corridor_line,
            'accessibility': self._calculate_corridor_accessibility(corridor_line, width),
            'properties': {
                'created_at': 'current_time',
                'algorithm': 'pathfinding',
                'validated': False
            }
        }
    
    def _calculate_corridor_accessibility(self, corridor_line: LineString, width: float) -> str:
        """Calculate accessibility compliance of corridor"""
        if width >= 1.2:
            return 'compliant'
        elif width >= 1.0:
            return 'limited'
        else:
            return 'non-compliant'
    
    def _astar_pathfinding(self, G: nx.Graph, start: str, end: str) -> List[str]:
        """A* pathfinding algorithm"""
        try:
            path = nx.astar_path(G, start, end, heuristic=self._heuristic, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []
    
    def _dijkstra_pathfinding(self, G: nx.Graph, start: str, end: str) -> List[str]:
        """Dijkstra pathfinding algorithm"""
        try:
            path = nx.dijkstra_path(G, start, end, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []
    
    def _bfs_pathfinding(self, G: nx.Graph, start: str, end: str) -> List[str]:
        """Breadth-first search pathfinding"""
        try:
            path = nx.shortest_path(G, start, end)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def _heuristic(self, node1: str, node2: str) -> float:
        """Heuristic function for A* algorithm"""
        # This would normally be implemented with access to node positions
        # For now, return 0 (making it equivalent to Dijkstra)
        return 0.0
    
    def _optimize_corridor_turns(self, corridors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize corridor paths to minimize turns"""
        optimized_corridors = []
        
        for corridor in corridors:
            coordinates = corridor.get('coordinates', [])
            
            if len(coordinates) > 2:
                # Simplify path by removing unnecessary waypoints
                simplified_coords = self._simplify_path(coordinates)
                
                if simplified_coords != coordinates:
                    # Update corridor with simplified path
                    simplified_line = LineString(simplified_coords)
                    corridor['coordinates'] = simplified_coords
                    corridor['shapely_geom'] = simplified_line
                    corridor['length'] = simplified_line.length
                    corridor['area'] = simplified_line.length * corridor['width']
            
            optimized_corridors.append(corridor)
        
        return optimized_corridors
    
    def _simplify_path(self, coordinates: List[List[float]]) -> List[List[float]]:
        """Simplify path by removing collinear points"""
        if len(coordinates) <= 2:
            return coordinates
        
        simplified = [coordinates[0]]  # Always keep first point
        
        for i in range(1, len(coordinates) - 1):
            # Check if current point is collinear with previous and next
            if not self._is_collinear(coordinates[i-1], coordinates[i], coordinates[i+1]):
                simplified.append(coordinates[i])
        
        simplified.append(coordinates[-1])  # Always keep last point
        return simplified
    
    def _is_collinear(self, p1: List[float], p2: List[float], p3: List[float]) -> bool:
        """Check if three points are collinear"""
        # Calculate cross product
        cross_product = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                        (p2[1] - p1[1]) * (p3[0] - p1[0]))
        
        # Points are collinear if cross product is close to zero
        return abs(cross_product) < 0.1
    
    def _ensure_accessibility_compliance(self, corridors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure corridors meet accessibility requirements"""
        compliant_corridors = []
        
        for corridor in corridors:
            # Check minimum width requirements
            if corridor['width'] < 1.2:  # ADA minimum
                # Widen corridor if space allows
                corridor['width'] = 1.2
                corridor['area'] = corridor['length'] * corridor['width']
                corridor['accessibility'] = 'compliant'
            
            # Check for accessibility barriers
            # This would involve more complex analysis of path obstacles
            
            compliant_corridors.append(corridor)
        
        return compliant_corridors
    
    def _validate_corridors(self, corridors: List[Dict[str, Any]],
                           walls: List[Dict[str, Any]],
                           restricted_areas: List[Dict[str, Any]],
                           avoid_obstacles: bool) -> List[Dict[str, Any]]:
        """Validate corridor placements"""
        validated_corridors = []
        
        for corridor in corridors:
            if self._validate_single_corridor(corridor, walls, restricted_areas, avoid_obstacles):
                corridor['properties']['validated'] = True
                validated_corridors.append(corridor)
            else:
                logger.warning(f"Corridor {corridor['id']} failed validation")
        
        return validated_corridors
    
    def _validate_single_corridor(self, corridor: Dict[str, Any],
                                 walls: List[Dict[str, Any]],
                                 restricted_areas: List[Dict[str, Any]],
                                 avoid_obstacles: bool) -> bool:
        """Validate a single corridor"""
        corridor_geom = corridor.get('shapely_geom')
        if not corridor_geom:
            return False
        
        corridor_width = corridor.get('width', 1.5)
        corridor_buffer = corridor_geom.buffer(corridor_width / 2)
        
        if avoid_obstacles:
            # Check intersection with walls
            for wall in walls:
                wall_geom = wall.get('shapely_geom')
                if wall_geom and corridor_buffer.intersects(wall_geom):
                    return False
            
            # Check intersection with restricted areas
            for restricted in restricted_areas:
                restricted_geom = restricted.get('shapely_geom')
                if restricted_geom and corridor_buffer.intersects(restricted_geom):
                    return False
        
        # Check minimum length
        if corridor['length'] < 0.5:
            return False
        
        # Check reasonable width
        if corridor['width'] < 0.8 or corridor['width'] > 5.0:
            return False
        
        return True
