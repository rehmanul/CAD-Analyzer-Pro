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
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union
import heapq
from collections import defaultdict, deque
import logging
from typing import Dict, List, Any, Tuple, Optional, Set

logger = logging.getLogger(__name__)

class CorridorGenerator:
    """Advanced corridor network generation with pathfinding"""
    
    def __init__(self):
        self.pathfinding_algorithms = {
            'A-Star': self._astar_pathfinding,
            'Dijkstra': self._dijkstra_pathfinding,
            'Breadth-First': self._bfs_pathfinding
        }
        
        self.corridor_types = {
            'main': {'width': 2.0, 'priority': 1},
            'secondary': {'width': 1.5, 'priority': 2},
            'access': {'width': 1.0, 'priority': 3},
            'emergency': {'width': 1.2, 'priority': 1}
        }
        
        self.grid_resolution = 0.5  # Grid cell size for pathfinding
    
    def generate_corridors(self, analysis_results: Dict[str, Any], 
                          configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corridor network connecting îlots and entrances"""
        logger.info("Starting corridor generation")
        
        # Extract configuration
        corridor_width = configuration.get('width', 1.5)
        main_width = configuration.get('main_width', 2.0)
        algorithm = configuration.get('algorithm', 'A-Star')
        optimize_turns = configuration.get('optimize_turns', True)
        avoid_obstacles = configuration.get('avoid_obstacles', True)
        ensure_accessibility = configuration.get('ensure_accessibility', True)
        
        # Extract analysis data
        ilots = analysis_results.get('ilots', [])
        entrances = analysis_results.get('entrances', [])
        walls = analysis_results.get('walls', [])
        restricted_areas = analysis_results.get('restricted_areas', [])
        open_spaces = analysis_results.get('open_spaces', [])
        
        if not ilots or not open_spaces:
            logger.warning("Insufficient data for corridor generation")
            return []
        
        # Create navigation grid
        navigation_grid = self._create_navigation_grid(walls, restricted_areas, open_spaces)
        
        # Generate connection points
        connection_points = self._generate_connection_points(ilots, entrances)
        
        # Generate corridor network
        corridors = self._generate_corridor_network(
            connection_points, navigation_grid, algorithm, 
            corridor_width, main_width, optimize_turns
        )
        
        # Optimize corridor network
        optimized_corridors = self._optimize_corridor_network(corridors, ensure_accessibility)
        
        # Post-process corridors
        final_corridors = self._post_process_corridors(optimized_corridors, configuration)
        
        logger.info(f"Generated {len(final_corridors)} corridors")
        return final_corridors
    
    def _create_navigation_grid(self, walls: List[Dict[str, Any]], 
                               restricted_areas: List[Dict[str, Any]],
                               open_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create navigation grid for pathfinding"""
        # Calculate overall bounds
        all_bounds = []
        
        for space in open_spaces:
            if space.get('geometry'):
                bounds = space['geometry'].bounds
                all_bounds.append(bounds)
        
        if not all_bounds:
            return {'grid': {}, 'bounds': {}, 'resolution': self.grid_resolution}
        
        # Calculate unified bounds
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        
        # Create grid
        grid = {}
        resolution = self.grid_resolution
        
        # Initialize grid as navigable
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                grid_key = (round(x/resolution), round(y/resolution))
                point = Point(x, y)
                
                # Check if point is in open space
                in_open_space = False
                for space in open_spaces:
                    if space.get('geometry') and space['geometry'].contains(point):
                        in_open_space = True
                        break
                
                # Check if point is blocked by walls or restricted areas
                blocked = False
                
                # Check walls
                for wall in walls:
                    wall_geom = wall.get('geometry')
                    if wall_geom:
                        # Add buffer for wall thickness
                        buffered_wall = wall_geom.buffer(0.5)
                        if buffered_wall.contains(point):
                            blocked = True
                            break
                
                # Check restricted areas
                if not blocked:
                    for area in restricted_areas:
                        area_geom = area.get('geometry')
                        if area_geom and area_geom.contains(point):
                            blocked = True
                            break
                
                # Set grid cell value
                grid[grid_key] = {
                    'navigable': in_open_space and not blocked,
                    'cost': 1.0 if (in_open_space and not blocked) else float('inf'),
                    'position': (x, y)
                }
                
                y += resolution
            x += resolution
        
        return {
            'grid': grid,
            'bounds': {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y},
            'resolution': resolution
        }
    
    def _generate_connection_points(self, ilots: List[Dict[str, Any]], 
                                  entrances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate points that need to be connected by corridors"""
        connection_points = []
        
        # Add îlot connection points
        for ilot in ilots:
            position = ilot.get('position', {})
            connection_points.append({
                'id': f"ilot_{ilot['id']}",
                'type': 'ilot',
                'position': (position.get('x', 0), position.get('y', 0)),
                'size_category': ilot.get('size_category', 'medium'),
                'priority': self._get_connection_priority(ilot.get('size_category', 'medium'))
            })
        
        # Add entrance connection points
        for entrance in entrances:
            position = entrance.get('position', {})
            connection_points.append({
                'id': f"entrance_{entrance['id']}",
                'type': 'entrance',
                'position': (position.get('x', 0), position.get('y', 0)),
                'width': entrance.get('width', 0.9),
                'priority': 1  # High priority for entrances
            })
        
        return connection_points
    
    def _get_connection_priority(self, size_category: str) -> int:
        """Get connection priority based on size category"""
        priority_map = {'large': 1, 'medium': 2, 'small': 3}
        return priority_map.get(size_category, 2)
    
    def _generate_corridor_network(self, connection_points: List[Dict[str, Any]], 
                                 navigation_grid: Dict[str, Any],
                                 algorithm: str, corridor_width: float,
                                 main_width: float, optimize_turns: bool) -> List[Dict[str, Any]]:
        """Generate corridor network using pathfinding"""
        corridors = []
        
        if len(connection_points) < 2:
            return corridors
        
        # Create main corridors from entrances to largest îlots
        entrance_points = [p for p in connection_points if p['type'] == 'entrance']
        ilot_points = [p for p in connection_points if p['type'] == 'ilot']
        
        # Sort îlots by priority
        ilot_points.sort(key=lambda x: x['priority'])
        
        # Generate main corridors
        main_corridors = self._generate_main_corridors(
            entrance_points, ilot_points, navigation_grid, algorithm, main_width
        )
        corridors.extend(main_corridors)
        
        # Generate secondary corridors to connect remaining îlots
        secondary_corridors = self._generate_secondary_corridors(
            ilot_points, main_corridors, navigation_grid, algorithm, corridor_width
        )
        corridors.extend(secondary_corridors)
        
        # Generate access corridors for isolated îlots
        access_corridors = self._generate_access_corridors(
            connection_points, corridors, navigation_grid, algorithm, corridor_width
        )
        corridors.extend(access_corridors)
        
        return corridors
    
    def _generate_main_corridors(self, entrance_points: List[Dict[str, Any]], 
                               ilot_points: List[Dict[str, Any]],
                               navigation_grid: Dict[str, Any], 
                               algorithm: str, width: float) -> List[Dict[str, Any]]:
        """Generate main corridors from entrances"""
        main_corridors = []
        
        if not entrance_points:
            return main_corridors
        
        # Connect each entrance to nearest high-priority îlots
        for entrance in entrance_points:
            # Find 2-3 nearest high-priority îlots
            entrance_pos = entrance['position']
            
            # Calculate distances to all îlots
            distances = []
            for ilot in ilot_points:
                if ilot['priority'] <= 2:  # High and medium priority
                    ilot_pos = ilot['position']
                    dist = np.sqrt((entrance_pos[0] - ilot_pos[0])**2 + 
                                 (entrance_pos[1] - ilot_pos[1])**2)
                    distances.append((dist, ilot))
            
            # Sort by distance and take closest 2-3
            distances.sort(key=lambda x: x[0])
            targets = distances[:min(3, len(distances))]
            
            # Create corridors to targets
            for dist, target_ilot in targets:
                path = self._find_path(entrance_pos, target_ilot['position'], 
                                     navigation_grid, algorithm)
                
                if path:
                    corridor = self._create_corridor(
                        f"main_{entrance['id']}_to_{target_ilot['id']}",
                        'main', path, width
                    )
                    main_corridors.append(corridor)
        
        return main_corridors
    
    def _generate_secondary_corridors(self, ilot_points: List[Dict[str, Any]], 
                                    main_corridors: List[Dict[str, Any]],
                                    navigation_grid: Dict[str, Any], 
                                    algorithm: str, width: float) -> List[Dict[str, Any]]:
        """Generate secondary corridors between îlots"""
        secondary_corridors = []
        
        # Get îlots connected by main corridors
        connected_ilots = set()
        for corridor in main_corridors:
            # Extract connected points from corridor ID
            if 'to_ilot_' in corridor['id']:
                ilot_id = corridor['id'].split('to_ilot_')[-1]
                connected_ilots.add(ilot_id)
        
        # Find unconnected îlots
        unconnected_ilots = [ilot for ilot in ilot_points 
                           if ilot['id'].split('_')[-1] not in connected_ilots]
        
        # Connect unconnected îlots to nearest connected îlots or main corridors
        for unconnected in unconnected_ilots:
            best_connection = self._find_best_connection_point(
                unconnected, main_corridors, ilot_points, connected_ilots
            )
            
            if best_connection:
                path = self._find_path(unconnected['position'], best_connection, 
                                     navigation_grid, algorithm)
                
                if path:
                    corridor = self._create_corridor(
                        f"secondary_{unconnected['id']}_connection",
                        'secondary', path, width
                    )
                    secondary_corridors.append(corridor)
        
        return secondary_corridors
    
    def _generate_access_corridors(self, connection_points: List[Dict[str, Any]], 
                                 existing_corridors: List[Dict[str, Any]],
                                 navigation_grid: Dict[str, Any], 
                                 algorithm: str, width: float) -> List[Dict[str, Any]]:
        """Generate access corridors for isolated points"""
        access_corridors = []
        
        # Find points not connected by existing corridors
        connected_points = set()
        for corridor in existing_corridors:
            # Simple check - in a full implementation, you'd track connections properly
            if 'ilot_' in corridor['id']:
                connected_points.add(corridor['id'])
        
        # Create access corridors for isolated points
        isolated_points = [p for p in connection_points 
                         if p['id'] not in connected_points and len(existing_corridors) > 0]
        
        for isolated in isolated_points:
            # Find nearest existing corridor
            nearest_corridor_point = self._find_nearest_corridor_point(isolated, existing_corridors)
            
            if nearest_corridor_point:
                path = self._find_path(isolated['position'], nearest_corridor_point, 
                                     navigation_grid, algorithm)
                
                if path:
                    corridor = self._create_corridor(
                        f"access_{isolated['id']}",
                        'access', path, width * 0.8  # Narrower access corridors
                    )
                    access_corridors.append(corridor)
        
        return access_corridors
    
    def _find_best_connection_point(self, unconnected_ilot: Dict[str, Any], 
                                  main_corridors: List[Dict[str, Any]],
                                  all_ilots: List[Dict[str, Any]], 
                                  connected_ilots: Set[str]) -> Optional[Tuple[float, float]]:
        """Find best connection point for unconnected îlot"""
        unconnected_pos = unconnected_ilot['position']
        best_point = None
        min_distance = float('inf')
        
        # Check connection to main corridor midpoints
        for corridor in main_corridors:
            path = corridor.get('path', [])
            if len(path) >= 2:
                # Use midpoint of corridor
                mid_idx = len(path) // 2
                mid_point = path[mid_idx]
                
                dist = np.sqrt((unconnected_pos[0] - mid_point[0])**2 + 
                             (unconnected_pos[1] - mid_point[1])**2)
                
                if dist < min_distance:
                    min_distance = dist
                    best_point = mid_point
        
        # Check connection to connected îlots
        for ilot in all_ilots:
            if ilot['id'].split('_')[-1] in connected_ilots:
                ilot_pos = ilot['position']
                dist = np.sqrt((unconnected_pos[0] - ilot_pos[0])**2 + 
                             (unconnected_pos[1] - ilot_pos[1])**2)
                
                if dist < min_distance:
                    min_distance = dist
                    best_point = ilot_pos
        
        return best_point
    
    def _find_nearest_corridor_point(self, isolated_point: Dict[str, Any], 
                                   corridors: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        """Find nearest point on existing corridor network"""
        isolated_pos = isolated_point['position']
        nearest_point = None
        min_distance = float('inf')
        
        for corridor in corridors:
            path = corridor.get('path', [])
            
            for point in path:
                dist = np.sqrt((isolated_pos[0] - point[0])**2 + 
                             (isolated_pos[1] - point[1])**2)
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_point = point
        
        return nearest_point
    
    def _find_path(self, start: Tuple[float, float], end: Tuple[float, float], 
                   navigation_grid: Dict[str, Any], algorithm: str) -> Optional[List[Tuple[float, float]]]:
        """Find path between two points using specified algorithm"""
        pathfinding_func = self.pathfinding_algorithms.get(algorithm, self._astar_pathfinding)
        return pathfinding_func(start, end, navigation_grid)
    
    def _astar_pathfinding(self, start: Tuple[float, float], end: Tuple[float, float], 
                          navigation_grid: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
        """A* pathfinding algorithm"""
        grid = navigation_grid['grid']
        resolution = navigation_grid['resolution']
        
        # Convert coordinates to grid coordinates
        start_grid = (round(start[0]/resolution), round(start[1]/resolution))
        end_grid = (round(end[0]/resolution), round(end[1]/resolution))
        
        # Check if start and end are valid
        if start_grid not in grid or end_grid not in grid:
            return None
        
        if not grid[start_grid]['navigable'] or not grid[end_grid]['navigable']:
            return None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_grid] = 0
        
        f_score = defaultdict(lambda: float('inf'))
        f_score[start_grid] = self._heuristic(start_grid, end_grid)
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    grid_pos = grid[current]['position']
                    path.append(grid_pos)
                    current = came_from[current]
                
                grid_pos = grid[start_grid]['position']
                path.append(grid_pos)
                path.reverse()
                
                return path
            
            # Check neighbors
            for neighbor in self._get_neighbors(current, grid):
                tentative_g_score = g_score[current] + grid[neighbor]['cost']
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, end_grid)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _dijkstra_pathfinding(self, start: Tuple[float, float], end: Tuple[float, float], 
                             navigation_grid: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
        """Dijkstra pathfinding algorithm"""
        # Similar to A* but without heuristic
        grid = navigation_grid['grid']
        resolution = navigation_grid['resolution']
        
        start_grid = (round(start[0]/resolution), round(start[1]/resolution))
        end_grid = (round(end[0]/resolution), round(end[1]/resolution))
        
        if start_grid not in grid or end_grid not in grid:
            return None
        
        if not grid[start_grid]['navigable'] or not grid[end_grid]['navigable']:
            return None
        
        # Dijkstra algorithm
        distances = defaultdict(lambda: float('inf'))
        distances[start_grid] = 0
        came_from = {}
        pq = [(0, start_grid)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == end_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    grid_pos = grid[current]['position']
                    path.append(grid_pos)
                    current = came_from[current]
                
                grid_pos = grid[start_grid]['position']
                path.append(grid_pos)
                path.reverse()
                
                return path
            
            if current_dist > distances[current]:
                continue
            
            for neighbor in self._get_neighbors(current, grid):
                distance = current_dist + grid[neighbor]['cost']
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    came_from[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return None
    
    def _bfs_pathfinding(self, start: Tuple[float, float], end: Tuple[float, float], 
                        navigation_grid: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
        """Breadth-First Search pathfinding"""
        grid = navigation_grid['grid']
        resolution = navigation_grid['resolution']
        
        start_grid = (round(start[0]/resolution), round(start[1]/resolution))
        end_grid = (round(end[0]/resolution), round(end[1]/resolution))
        
        if start_grid not in grid or end_grid not in grid:
            return None
        
        if not grid[start_grid]['navigable'] or not grid[end_grid]['navigable']:
            return None
        
        # BFS algorithm
        queue = deque([start_grid])
        came_from = {start_grid: None}
        
        while queue:
            current = queue.popleft()
            
            if current == end_grid:
                # Reconstruct path
                path = []
                while current is not None:
                    grid_pos = grid[current]['position']
                    path.append(grid_pos)
                    current = came_from[current]
                
                path.reverse()
                return path
            
            for neighbor in self._get_neighbors(current, grid):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        return None
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Manhattan distance)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _get_neighbors(self, grid_pos: Tuple[int, int], 
                      grid: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Get navigable neighbors of a grid position"""
        x, y = grid_pos
        neighbors = []
        
        # 8-connected grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                neighbor = (x + dx, y + dy)
                if neighbor in grid and grid[neighbor]['navigable']:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _create_corridor(self, corridor_id: str, corridor_type: str, 
                        path: List[Tuple[float, float]], width: float) -> Dict[str, Any]:
        """Create corridor object from path"""
        if len(path) < 2:
            return {}
        
        # Calculate total length
        total_length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        # Create geometry
        line_geom = LineString(path)
        
        return {
            'id': corridor_id,
            'type': corridor_type,
            'path': path,
            'width': width,
            'length': total_length,
            'start': {'x': path[0][0], 'y': path[0][1]},
            'end': {'x': path[-1][0], 'y': path[-1][1]},
            'geometry': line_geom,
            'accessibility': 'compliant' if width >= 0.8 else 'non-compliant',
            'properties': {
                'algorithm': 'pathfinding',
                'turns': self._count_turns(path),
                'straightness': self._calculate_straightness(path)
            }
        }
    
    def _count_turns(self, path: List[Tuple[float, float]]) -> int:
        """Count number of turns in path"""
        if len(path) < 3:
            return 0
        
        turns = 0
        for i in range(1, len(path) - 1):
            # Calculate angles
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = np.arccos(cos_angle)
                
                # Consider significant direction changes as turns
                if angle > np.pi / 4:  # 45 degrees
                    turns += 1
        
        return turns
    
    def _calculate_straightness(self, path: List[Tuple[float, float]]) -> float:
        """Calculate straightness ratio of path"""
        if len(path) < 2:
            return 1.0
        
        # Direct distance
        direct_distance = np.sqrt((path[-1][0] - path[0][0])**2 + 
                                (path[-1][1] - path[0][1])**2)
        
        # Path distance
        path_distance = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_distance += np.sqrt(dx*dx + dy*dy)
        
        if path_distance == 0:
            return 1.0
        
        return direct_distance / path_distance
    
    def _optimize_corridor_network(self, corridors: List[Dict[str, Any]], 
                                 ensure_accessibility: bool) -> List[Dict[str, Any]]:
        """Optimize corridor network for efficiency"""
        if not corridors:
            return corridors
        
        optimized = []
        
        for corridor in corridors:
            # Optimize individual corridor
            optimized_corridor = self._optimize_corridor(corridor, ensure_accessibility)
            if optimized_corridor:
                optimized.append(optimized_corridor)
        
        # Remove redundant corridors
        optimized = self._remove_redundant_corridors(optimized)
        
        return optimized
    
    def _optimize_corridor(self, corridor: Dict[str, Any], 
                          ensure_accessibility: bool) -> Optional[Dict[str, Any]]:
        """Optimize individual corridor"""
        # Ensure minimum width for accessibility
        if ensure_accessibility and corridor['width'] < 0.8:
            corridor['width'] = 0.8
            corridor['accessibility'] = 'compliant'
        
        # Simplify path by removing unnecessary waypoints
        path = corridor['path']
        if len(path) > 2:
            simplified_path = self._simplify_path(path)
            if len(simplified_path) >= 2:
                corridor['path'] = simplified_path
                corridor['properties']['turns'] = self._count_turns(simplified_path)
                corridor['properties']['straightness'] = self._calculate_straightness(simplified_path)
        
        return corridor
    
    def _simplify_path(self, path: List[Tuple[float, float]], 
                      tolerance: float = 0.5) -> List[Tuple[float, float]]:
        """Simplify path using Douglas-Peucker algorithm"""
        if len(path) <= 2:
            return path
        
        # Find point with maximum distance from line between start and end
        start = path[0]
        end = path[-1]
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(path) - 1):
            distance = self._point_line_distance(path[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursive call
            left_simplified = self._simplify_path(path[:max_index + 1], tolerance)
            right_simplified = self._simplify_path(path[max_index:], tolerance)
            
            # Combine results (remove duplicate middle point)
            return left_simplified[:-1] + right_simplified
        else:
            # Return just start and end points
            return [start, end]
    
    def _point_line_distance(self, point: Tuple[float, float], 
                           line_start: Tuple[float, float], 
                           line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate distance using formula
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        if denominator == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        return numerator / denominator
    
    def _remove_redundant_corridors(self, corridors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant or overlapping corridors"""
        if len(corridors) <= 1:
            return corridors
        
        non_redundant = []
        
        for i, corridor in enumerate(corridors):
            is_redundant = False
            
            for j, other_corridor in enumerate(corridors):
                if i != j and self._are_corridors_redundant(corridor, other_corridor):
                    # Keep the better corridor (wider or shorter)
                    if (other_corridor['width'] > corridor['width'] or 
                        other_corridor['length'] < corridor['length']):
                        is_redundant = True
                        break
            
            if not is_redundant:
                non_redundant.append(corridor)
        
        return non_redundant
    
    def _are_corridors_redundant(self, corridor1: Dict[str, Any], 
                               corridor2: Dict[str, Any]]) -> bool:
        """Check if two corridors are redundant"""
        # Simple check based on path similarity
        path1 = corridor1['path']
        path2 = corridor2['path']
        
        if len(path1) < 2 or len(path2) < 2:
            return False
        
        # Check if start and end points are close
        start_distance = np.sqrt((path1[0][0] - path2[0][0])**2 + 
                               (path1[0][1] - path2[0][1])**2)
        end_distance = np.sqrt((path1[-1][0] - path2[-1][0])**2 + 
                             (path1[-1][1] - path2[-1][1])**2)
        
        # Consider redundant if both start and end are within 2 meters
        return start_distance < 2.0 and end_distance < 2.0
    
    def _post_process_corridors(self, corridors: List[Dict[str, Any]], 
                              configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process corridors with final adjustments"""
        processed = []
        
        for corridor in corridors:
            # Add configuration-specific properties
            corridor['configuration'] = {
                'algorithm': configuration.get('algorithm', 'A-Star'),
                'optimize_turns': configuration.get('optimize_turns', True),
                'avoid_obstacles': configuration.get('avoid_obstacles', True),
                'ensure_accessibility': configuration.get('ensure_accessibility', True)
            }
            
            # Calculate quality metrics
            corridor['quality_metrics'] = {
                'efficiency': self._calculate_corridor_efficiency(corridor),
                'accessibility_score': 1.0 if corridor['accessibility'] == 'compliant' else 0.5,
                'straightness_score': corridor['properties']['straightness'],
                'turn_penalty': max(0, 1.0 - corridor['properties']['turns'] * 0.1)
            }
            
            processed.append(corridor)
        
        return processed
    
    def _calculate_corridor_efficiency(self, corridor: Dict[str, Any]) -> float:
        """Calculate corridor efficiency score"""
        straightness = corridor['properties']['straightness']
        width_efficiency = min(1.0, corridor['width'] / 2.0)  # Normalize by 2m ideal width
        turn_efficiency = max(0, 1.0 - corridor['properties']['turns'] * 0.1)
        
        return (straightness + width_efficiency + turn_efficiency) / 3.0
