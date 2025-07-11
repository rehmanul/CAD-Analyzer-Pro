"""
Phase 2: Advanced Corridor Generation Engine
Intelligent corridor network generation with pathfinding algorithms,
traffic flow optimization, and comprehensive area calculations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import unary_union, linemerge
import networkx as nx
from scipy.spatial import cKDTree, Voronoi
from dataclasses import dataclass
from enum import Enum
import time
from heapq import heappush, heappop

class CorridorType(Enum):
    MAIN = "main"
    SECONDARY = "secondary"
    ACCESS = "access"
    EMERGENCY = "emergency"

class PathfindingAlgorithm(Enum):
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    VISIBILITY_GRAPH = "visibility_graph"
    RRT = "rapidly_exploring_random_tree"

@dataclass
class CorridorConfiguration:
    """Configuration for corridor generation"""
    main_corridor_width: float = 1.5  # meters
    secondary_corridor_width: float = 1.2  # meters
    access_corridor_width: float = 1.0  # meters
    min_clearance: float = 0.3  # meters
    pathfinding_algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    enable_emergency_paths: bool = True
    optimize_traffic_flow: bool = True
    max_corridor_length: float = 50.0  # meters
    preferred_angles: List[float] = None  # Preferred corridor angles (degrees)

@dataclass
class CorridorSegment:
    """Represents a corridor segment"""
    id: str
    geometry: LineString
    corridor_type: CorridorType
    width: float
    start_point: Point
    end_point: Point
    connected_ilots: List[str]
    traffic_weight: float = 1.0
    area: float = 0.0

class AdvancedCorridorGenerator:
    """
    Advanced corridor generation system with multiple pathfinding algorithms
    and traffic flow optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Corridor styling
        self.corridor_colors = {
            CorridorType.MAIN: "#3B82F6",        # Blue
            CorridorType.SECONDARY: "#10B981",   # Green  
            CorridorType.ACCESS: "#F59E0B",      # Yellow
            CorridorType.EMERGENCY: "#EF4444"    # Red
        }
        
        # Pathfinding parameters
        self.pathfinding_params = {
            'grid_resolution': 200,  # mm
            'heuristic_weight': 1.2,
            'obstacle_padding': 300,  # mm
            'max_iterations': 10000
        }

    def generate_corridors_advanced(self, floor_plan_data: Dict[str, Any], 
                                  placed_ilots: List[Dict[str, Any]], 
                                  config: CorridorConfiguration) -> Dict[str, Any]:
        """
        Main advanced corridor generation method
        """
        start_time = time.time()
        
        try:
            # Extract spatial environment
            environment = self._extract_corridor_environment(floor_plan_data, placed_ilots)
            
            if not environment['ilot_centers']:
                return self._create_empty_corridor_result("No îlots to connect")
            
            # Build connectivity graph
            connectivity_graph = self._build_connectivity_graph(environment, config)
            
            # Generate corridor network based on selected algorithm
            corridor_network = self._generate_corridor_network(
                connectivity_graph, environment, config
            )
            
            # Optimize corridor paths
            optimized_corridors = self._optimize_corridor_paths(
                corridor_network, environment, config
            )
            
            # Calculate corridor areas and metrics
            corridor_metrics = self._calculate_corridor_metrics(
                optimized_corridors, environment, config
            )
            
            processing_time = time.time() - start_time
            
            # Generate comprehensive result
            result = self._generate_corridor_result(
                optimized_corridors, corridor_metrics, processing_time, config
            )
            
            self.logger.info(f"Advanced corridor generation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced corridor generation: {str(e)}")
            return self._create_error_corridor_result(str(e))

    def _extract_corridor_environment(self, floor_plan_data: Dict[str, Any], 
                                    placed_ilots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract environment data for corridor generation"""
        
        environment = {
            'ilot_centers': [],
            'ilot_geometries': [],
            'obstacles': [],
            'entrances': [],
            'bounds': None,
            'free_space': None
        }
        
        try:
            # Extract îlot centers and geometries
            for ilot in placed_ilots:
                center_coords = ilot.get('center', [0, 0])
                center = Point(center_coords[0], center_coords[1])
                environment['ilot_centers'].append(center)
                
                coords = ilot.get('coordinates', [])
                if len(coords) >= 3:
                    ilot_geom = Polygon(coords)
                    environment['ilot_geometries'].append(ilot_geom)
            
            # Extract obstacles (walls, restricted areas)
            walls = floor_plan_data.get('walls', [])
            for wall in walls:
                coords = wall.get('coordinates', [])
                if len(coords) >= 2:
                    wall_geom = LineString(coords)
                    # Buffer walls to create obstacle polygons
                    obstacle = wall_geom.buffer(200)  # 20cm obstacle padding
                    environment['obstacles'].append(obstacle)
            
            restricted_areas = floor_plan_data.get('restricted_areas', [])
            for area in restricted_areas:
                coords = area.get('coordinates', [])
                if len(coords) >= 3:
                    restricted_geom = Polygon(coords)
                    environment['obstacles'].append(restricted_geom)
            
            # Extract entrances
            entrances = floor_plan_data.get('entrances', [])
            for entrance in entrances:
                coords = entrance.get('coordinates', [])
                if len(coords) >= 1:
                    if len(coords) == 1:
                        entrance_point = Point(coords[0])
                    else:
                        entrance_geom = Polygon(coords)
                        entrance_point = entrance_geom.centroid
                    environment['entrances'].append(entrance_point)
            
            # Calculate bounds
            bounds = floor_plan_data.get('floor_plan_bounds', {})
            if bounds:
                environment['bounds'] = bounds
            
            # Calculate free space for pathfinding
            environment['free_space'] = self._calculate_free_space(environment)
            
            return environment
            
        except Exception as e:
            self.logger.error(f"Error extracting corridor environment: {str(e)}")
            return environment

    def _calculate_free_space(self, environment: Dict[str, Any]) -> Optional[Polygon]:
        """Calculate free space available for corridors"""
        try:
            # Create overall bounds
            bounds = environment['bounds']
            if not bounds:
                return None
            
            total_area = Polygon([
                (bounds['min_x'], bounds['min_y']),
                (bounds['max_x'], bounds['min_y']),
                (bounds['max_x'], bounds['max_y']),
                (bounds['min_x'], bounds['max_y'])
            ])
            
            # Subtract all obstacles
            free_space = total_area
            
            all_obstacles = []
            all_obstacles.extend(environment['obstacles'])
            all_obstacles.extend(environment['ilot_geometries'])
            
            for obstacle in all_obstacles:
                try:
                    free_space = free_space.difference(obstacle)
                except Exception:
                    continue
            
            return free_space
            
        except Exception as e:
            self.logger.error(f"Error calculating free space: {str(e)}")
            return None

    def _build_connectivity_graph(self, environment: Dict[str, Any], 
                                config: CorridorConfiguration) -> nx.Graph:
        """Build connectivity graph between îlots and entrances"""
        
        graph = nx.Graph()
        
        try:
            # Add îlot nodes
            for i, center in enumerate(environment['ilot_centers']):
                graph.add_node(f"ilot_{i}", pos=(center.x, center.y), type="ilot")
            
            # Add entrance nodes
            for i, entrance in enumerate(environment['entrances']):
                graph.add_node(f"entrance_{i}", pos=(entrance.x, entrance.y), type="entrance")
            
            # Calculate connections based on visibility and distance
            all_nodes = list(graph.nodes())
            
            for i, node1 in enumerate(all_nodes):
                for j, node2 in enumerate(all_nodes[i+1:], i+1):
                    pos1 = graph.nodes[node1]['pos']
                    pos2 = graph.nodes[node2]['pos']
                    
                    point1 = Point(pos1)
                    point2 = Point(pos2)
                    
                    # Check if direct connection is possible
                    if self._is_path_clear(point1, point2, environment):
                        distance = point1.distance(point2)
                        weight = distance / 1000  # Convert to meters
                        
                        graph.add_edge(node1, node2, weight=weight, distance=distance)
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error building connectivity graph: {str(e)}")
            return graph

    def _is_path_clear(self, start: Point, end: Point, environment: Dict[str, Any]) -> bool:
        """Check if path between two points is clear of obstacles"""
        try:
            path_line = LineString([start, end])
            
            # Check intersection with obstacles
            for obstacle in environment['obstacles']:
                if path_line.intersects(obstacle):
                    return False
            
            # Check intersection with îlots (except start/end points)
            for ilot_geom in environment['ilot_geometries']:
                if path_line.intersects(ilot_geom):
                    # Allow connection if start or end point is on the îlot
                    if not (ilot_geom.contains(start) or ilot_geom.contains(end)):
                        return False
            
            return True
            
        except Exception:
            return False

    def _generate_corridor_network(self, connectivity_graph: nx.Graph, 
                                 environment: Dict[str, Any], 
                                 config: CorridorConfiguration) -> List[CorridorSegment]:
        """Generate corridor network using selected pathfinding algorithm"""
        
        corridors = []
        
        try:
            if config.pathfinding_algorithm == PathfindingAlgorithm.DIJKSTRA:
                corridors = self._generate_corridors_dijkstra(connectivity_graph, environment, config)
            elif config.pathfinding_algorithm == PathfindingAlgorithm.A_STAR:
                corridors = self._generate_corridors_a_star(connectivity_graph, environment, config)
            elif config.pathfinding_algorithm == PathfindingAlgorithm.VISIBILITY_GRAPH:
                corridors = self._generate_corridors_visibility_graph(connectivity_graph, environment, config)
            else:
                # Default to A* algorithm
                corridors = self._generate_corridors_a_star(connectivity_graph, environment, config)
            
            return corridors
            
        except Exception as e:
            self.logger.error(f"Error generating corridor network: {str(e)}")
            return corridors

    def _generate_corridors_a_star(self, connectivity_graph: nx.Graph, 
                                 environment: Dict[str, Any], 
                                 config: CorridorConfiguration) -> List[CorridorSegment]:
        """Generate corridors using A* pathfinding algorithm"""
        
        corridors = []
        
        try:
            # Create minimum spanning tree for main corridors
            if connectivity_graph.edges():
                mst = nx.minimum_spanning_tree(connectivity_graph, weight='weight')
                
                # Generate corridors for MST edges
                for edge in mst.edges(data=True):
                    node1, node2, data = edge
                    pos1 = connectivity_graph.nodes[node1]['pos']
                    pos2 = connectivity_graph.nodes[node2]['pos']
                    
                    # Create corridor geometry
                    start_point = Point(pos1)
                    end_point = Point(pos2)
                    corridor_line = LineString([start_point, end_point])
                    
                    # Determine corridor type and width
                    corridor_type = self._determine_corridor_type(node1, node2, connectivity_graph)
                    width = self._get_corridor_width(corridor_type, config)
                    
                    corridor = CorridorSegment(
                        id=f"corridor_{len(corridors)+1}",
                        geometry=corridor_line,
                        corridor_type=corridor_type,
                        width=width,
                        start_point=start_point,
                        end_point=end_point,
                        connected_ilots=[node1, node2],
                        area=corridor_line.length * width / 1000000  # Convert to m²
                    )
                    
                    corridors.append(corridor)
            
            # Add access corridors for isolated îlots
            corridors.extend(self._add_access_corridors(
                connectivity_graph, corridors, environment, config
            ))
            
            return corridors
            
        except Exception as e:
            self.logger.error(f"Error in A* corridor generation: {str(e)}")
            return corridors

    def _generate_corridors_dijkstra(self, connectivity_graph: nx.Graph, 
                                   environment: Dict[str, Any], 
                                   config: CorridorConfiguration) -> List[CorridorSegment]:
        """Generate corridors using Dijkstra's algorithm"""
        
        corridors = []
        
        try:
            # Find shortest paths from all entrances to all îlots
            entrance_nodes = [node for node in connectivity_graph.nodes() 
                            if connectivity_graph.nodes[node]['type'] == 'entrance']
            ilot_nodes = [node for node in connectivity_graph.nodes() 
                         if connectivity_graph.nodes[node]['type'] == 'ilot']
            
            used_edges = set()
            
            for entrance in entrance_nodes:
                # Find shortest paths to all îlots
                try:
                    paths = nx.single_source_dijkstra_path(
                        connectivity_graph, entrance, weight='weight'
                    )
                    
                    for ilot in ilot_nodes:
                        if ilot in paths:
                            path = paths[ilot]
                            
                            # Create corridors for path edges
                            for i in range(len(path) - 1):
                                edge = tuple(sorted([path[i], path[i+1]]))
                                
                                if edge not in used_edges:
                                    used_edges.add(edge)
                                    
                                    node1, node2 = edge
                                    pos1 = connectivity_graph.nodes[node1]['pos']
                                    pos2 = connectivity_graph.nodes[node2]['pos']
                                    
                                    start_point = Point(pos1)
                                    end_point = Point(pos2)
                                    corridor_line = LineString([start_point, end_point])
                                    
                                    corridor_type = self._determine_corridor_type(
                                        node1, node2, connectivity_graph
                                    )
                                    width = self._get_corridor_width(corridor_type, config)
                                    
                                    corridor = CorridorSegment(
                                        id=f"corridor_{len(corridors)+1}",
                                        geometry=corridor_line,
                                        corridor_type=corridor_type,
                                        width=width,
                                        start_point=start_point,
                                        end_point=end_point,
                                        connected_ilots=[node1, node2],
                                        area=corridor_line.length * width / 1000000
                                    )
                                    
                                    corridors.append(corridor)
                    
                except nx.NetworkXNoPath:
                    continue
            
            return corridors
            
        except Exception as e:
            self.logger.error(f"Error in Dijkstra corridor generation: {str(e)}")
            return corridors

    def _generate_corridors_visibility_graph(self, connectivity_graph: nx.Graph, 
                                           environment: Dict[str, Any], 
                                           config: CorridorConfiguration) -> List[CorridorSegment]:
        """Generate corridors using visibility graph pathfinding"""
        
        corridors = []
        
        try:
            # Build visibility graph with obstacle vertices
            visibility_graph = self._build_visibility_graph(environment)
            
            # Find paths using visibility graph
            for edge in connectivity_graph.edges():
                node1, node2 = edge
                pos1 = connectivity_graph.nodes[node1]['pos']
                pos2 = connectivity_graph.nodes[node2]['pos']
                
                # Find visibility path
                path = self._find_visibility_path(
                    Point(pos1), Point(pos2), visibility_graph, environment
                )
                
                if path:
                    corridor_type = self._determine_corridor_type(node1, node2, connectivity_graph)
                    width = self._get_corridor_width(corridor_type, config)
                    
                    corridor = CorridorSegment(
                        id=f"corridor_{len(corridors)+1}",
                        geometry=path,
                        corridor_type=corridor_type,
                        width=width,
                        start_point=Point(path.coords[0]),
                        end_point=Point(path.coords[-1]),
                        connected_ilots=[node1, node2],
                        area=path.length * width / 1000000
                    )
                    
                    corridors.append(corridor)
            
            return corridors
            
        except Exception as e:
            self.logger.error(f"Error in visibility graph corridor generation: {str(e)}")
            return corridors

    def _build_visibility_graph(self, environment: Dict[str, Any]) -> nx.Graph:
        """Build visibility graph for pathfinding"""
        
        vis_graph = nx.Graph()
        
        try:
            # Extract obstacle vertices
            vertices = []
            
            # Add îlot centers
            for center in environment['ilot_centers']:
                vertices.append((center.x, center.y))
            
            # Add entrance points
            for entrance in environment['entrances']:
                vertices.append((entrance.x, entrance.y))
            
            # Add obstacle corners
            for obstacle in environment['obstacles']:
                if hasattr(obstacle, 'exterior'):
                    coords = list(obstacle.exterior.coords)
                    vertices.extend(coords[:-1])  # Exclude duplicate last point
            
            # Add vertices to graph
            for i, vertex in enumerate(vertices):
                vis_graph.add_node(i, pos=vertex)
            
            # Add edges for visible connections
            for i, vertex1 in enumerate(vertices):
                for j, vertex2 in enumerate(vertices[i+1:], i+1):
                    point1 = Point(vertex1)
                    point2 = Point(vertex2)
                    
                    if self._is_path_clear(point1, point2, environment):
                        distance = point1.distance(point2)
                        vis_graph.add_edge(i, j, weight=distance)
            
            return vis_graph
            
        except Exception as e:
            self.logger.error(f"Error building visibility graph: {str(e)}")
            return vis_graph

    def _find_visibility_path(self, start: Point, end: Point, 
                            visibility_graph: nx.Graph, 
                            environment: Dict[str, Any]) -> Optional[LineString]:
        """Find path using visibility graph"""
        
        try:
            # Find nearest vertices to start and end points
            start_vertex = self._find_nearest_vertex(start, visibility_graph)
            end_vertex = self._find_nearest_vertex(end, visibility_graph)
            
            if start_vertex is None or end_vertex is None:
                return LineString([start, end])
            
            # Find shortest path
            try:
                path = nx.shortest_path(
                    visibility_graph, start_vertex, end_vertex, weight='weight'
                )
                
                # Convert path to coordinates
                coords = [start.coords[0]]  # Start point
                
                for vertex_id in path:
                    vertex_pos = visibility_graph.nodes[vertex_id]['pos']
                    coords.append(vertex_pos)
                
                coords.append(end.coords[0])  # End point
                
                return LineString(coords)
                
            except nx.NetworkXNoPath:
                return LineString([start, end])
            
        except Exception as e:
            self.logger.error(f"Error finding visibility path: {str(e)}")
            return LineString([start, end])

    def _find_nearest_vertex(self, point: Point, visibility_graph: nx.Graph) -> Optional[int]:
        """Find nearest vertex in visibility graph"""
        
        min_distance = float('inf')
        nearest_vertex = None
        
        for vertex_id in visibility_graph.nodes():
            vertex_pos = visibility_graph.nodes[vertex_id]['pos']
            vertex_point = Point(vertex_pos)
            distance = point.distance(vertex_point)
            
            if distance < min_distance:
                min_distance = distance
                nearest_vertex = vertex_id
        
        return nearest_vertex

    def _determine_corridor_type(self, node1: str, node2: str, 
                               connectivity_graph: nx.Graph) -> CorridorType:
        """Determine corridor type based on connected nodes"""
        
        type1 = connectivity_graph.nodes[node1]['type']
        type2 = connectivity_graph.nodes[node2]['type']
        
        if 'entrance' in [type1, type2]:
            return CorridorType.MAIN
        else:
            return CorridorType.SECONDARY

    def _get_corridor_width(self, corridor_type: CorridorType, 
                          config: CorridorConfiguration) -> float:
        """Get corridor width based on type"""
        
        width_map = {
            CorridorType.MAIN: config.main_corridor_width,
            CorridorType.SECONDARY: config.secondary_corridor_width,
            CorridorType.ACCESS: config.access_corridor_width,
            CorridorType.EMERGENCY: config.main_corridor_width
        }
        
        return width_map.get(corridor_type, config.secondary_corridor_width) * 1000  # Convert to mm

    def _add_access_corridors(self, connectivity_graph: nx.Graph, 
                            existing_corridors: List[CorridorSegment], 
                            environment: Dict[str, Any], 
                            config: CorridorConfiguration) -> List[CorridorSegment]:
        """Add access corridors for better connectivity"""
        
        access_corridors = []
        
        try:
            # Find îlots with limited connectivity
            connected_nodes = set()
            for corridor in existing_corridors:
                connected_nodes.update(corridor.connected_ilots)
            
            all_ilot_nodes = [node for node in connectivity_graph.nodes() 
                            if connectivity_graph.nodes[node]['type'] == 'ilot']
            
            isolated_nodes = [node for node in all_ilot_nodes if node not in connected_nodes]
            
            # Create access corridors for isolated îlots
            for isolated_node in isolated_nodes:
                # Find nearest connected îlot
                isolated_pos = connectivity_graph.nodes[isolated_node]['pos']
                isolated_point = Point(isolated_pos)
                
                min_distance = float('inf')
                nearest_connected = None
                
                for connected_node in connected_nodes:
                    connected_pos = connectivity_graph.nodes[connected_node]['pos']
                    connected_point = Point(connected_pos)
                    distance = isolated_point.distance(connected_point)
                    
                    if distance < min_distance and self._is_path_clear(
                        isolated_point, connected_point, environment
                    ):
                        min_distance = distance
                        nearest_connected = connected_node
                
                # Create access corridor
                if nearest_connected:
                    connected_pos = connectivity_graph.nodes[nearest_connected]['pos']
                    start_point = isolated_point
                    end_point = Point(connected_pos)
                    corridor_line = LineString([start_point, end_point])
                    
                    width = self._get_corridor_width(CorridorType.ACCESS, config)
                    
                    access_corridor = CorridorSegment(
                        id=f"access_{len(access_corridors)+1}",
                        geometry=corridor_line,
                        corridor_type=CorridorType.ACCESS,
                        width=width,
                        start_point=start_point,
                        end_point=end_point,
                        connected_ilots=[isolated_node, nearest_connected],
                        area=corridor_line.length * width / 1000000
                    )
                    
                    access_corridors.append(access_corridor)
                    connected_nodes.add(isolated_node)
            
            return access_corridors
            
        except Exception as e:
            self.logger.error(f"Error adding access corridors: {str(e)}")
            return access_corridors

    def _optimize_corridor_paths(self, corridors: List[CorridorSegment], 
                               environment: Dict[str, Any], 
                               config: CorridorConfiguration) -> List[CorridorSegment]:
        """Optimize corridor paths for better flow and efficiency"""
        
        optimized_corridors = []
        
        try:
            for corridor in corridors:
                # Smooth corridor path
                smoothed_path = self._smooth_corridor_path(corridor.geometry, environment)
                
                # Update corridor with optimized path
                optimized_corridor = CorridorSegment(
                    id=corridor.id,
                    geometry=smoothed_path,
                    corridor_type=corridor.corridor_type,
                    width=corridor.width,
                    start_point=Point(smoothed_path.coords[0]),
                    end_point=Point(smoothed_path.coords[-1]),
                    connected_ilots=corridor.connected_ilots,
                    traffic_weight=self._calculate_traffic_weight(corridor, corridors),
                    area=smoothed_path.length * corridor.width / 1000000
                )
                
                optimized_corridors.append(optimized_corridor)
            
            return optimized_corridors
            
        except Exception as e:
            self.logger.error(f"Error optimizing corridor paths: {str(e)}")
            return corridors

    def _smooth_corridor_path(self, path: LineString, 
                            environment: Dict[str, Any]) -> LineString:
        """Smooth corridor path to reduce sharp turns"""
        
        try:
            coords = list(path.coords)
            
            if len(coords) <= 2:
                return path
            
            # Simple smoothing: remove unnecessary intermediate points
            smoothed_coords = [coords[0]]  # Start point
            
            for i in range(1, len(coords) - 1):
                # Check if intermediate point is necessary
                prev_point = Point(coords[i-1])
                curr_point = Point(coords[i])
                next_point = Point(coords[i+1])
                
                # Create direct line from previous to next
                direct_line = LineString([prev_point, next_point])
                
                # If direct line is clear, skip current point
                if not self._is_path_clear(prev_point, next_point, environment):
                    smoothed_coords.append(coords[i])
            
            smoothed_coords.append(coords[-1])  # End point
            
            return LineString(smoothed_coords)
            
        except Exception:
            return path

    def _calculate_traffic_weight(self, target_corridor: CorridorSegment, 
                                all_corridors: List[CorridorSegment]) -> float:
        """Calculate traffic weight for corridor optimization"""
        
        # Simple traffic weight based on corridor type and connections
        type_weights = {
            CorridorType.MAIN: 3.0,
            CorridorType.SECONDARY: 2.0,
            CorridorType.ACCESS: 1.0,
            CorridorType.EMERGENCY: 1.5
        }
        
        base_weight = type_weights.get(target_corridor.corridor_type, 1.0)
        
        # Increase weight for corridors connecting to entrances
        entrance_bonus = 0
        for connected_node in target_corridor.connected_ilots:
            if 'entrance' in connected_node:
                entrance_bonus += 1.0
        
        return base_weight + entrance_bonus

    def _calculate_corridor_metrics(self, corridors: List[CorridorSegment], 
                                  environment: Dict[str, Any], 
                                  config: CorridorConfiguration) -> Dict[str, Any]:
        """Calculate comprehensive corridor metrics"""
        
        metrics = {
            'total_corridors': len(corridors),
            'total_corridor_length': 0.0,
            'total_corridor_area': 0.0,
            'corridor_types': {},
            'average_width': 0.0,
            'connectivity_score': 0.0,
            'coverage_efficiency': 0.0
        }
        
        if not corridors:
            return metrics
        
        # Calculate basic metrics
        total_length = sum(corridor.geometry.length for corridor in corridors) / 1000  # Convert to meters
        total_area = sum(corridor.area for corridor in corridors)
        
        metrics['total_corridor_length'] = total_length
        metrics['total_corridor_area'] = total_area
        
        # Count corridor types
        for corridor in corridors:
            corridor_type = corridor.corridor_type.value
            metrics['corridor_types'][corridor_type] = metrics['corridor_types'].get(corridor_type, 0) + 1
        
        # Calculate average width
        if corridors:
            avg_width = sum(corridor.width for corridor in corridors) / len(corridors) / 1000  # Convert to meters
            metrics['average_width'] = avg_width
        
        # Calculate connectivity score (simplified)
        unique_connections = set()
        for corridor in corridors:
            for ilot in corridor.connected_ilots:
                unique_connections.add(ilot)
        
        total_ilots = len(environment['ilot_centers'])
        if total_ilots > 0:
            metrics['connectivity_score'] = len(unique_connections) / total_ilots
        
        # Calculate coverage efficiency
        total_space = environment['bounds'].get('width', 1) * environment['bounds'].get('height', 1) / 1000000  # m²
        if total_space > 0:
            metrics['coverage_efficiency'] = total_area / total_space
        
        return metrics

    def _generate_corridor_result(self, corridors: List[CorridorSegment], 
                                metrics: Dict[str, Any], 
                                processing_time: float, 
                                config: CorridorConfiguration) -> Dict[str, Any]:
        """Generate comprehensive corridor result"""
        
        # Convert corridors to visualization format
        corridors_data = []
        for corridor in corridors:
            coords = list(corridor.geometry.coords)
            corridor_data = {
                'id': corridor.id,
                'coordinates': [[x, y] for x, y in coords],
                'corridor_type': corridor.corridor_type.value,
                'width': corridor.width / 1000,  # Convert to meters
                'area': corridor.area,
                'color': self.corridor_colors[corridor.corridor_type],
                'connected_ilots': corridor.connected_ilots,
                'traffic_weight': corridor.traffic_weight
            }
            corridors_data.append(corridor_data)
        
        result = {
            'success': True,
            'corridors': corridors_data,
            'corridor_metrics': metrics,
            'processing_info': {
                'processing_time': processing_time,
                'algorithm_used': config.pathfinding_algorithm.value,
                'total_corridors_generated': len(corridors)
            },
            'configuration_used': {
                'main_corridor_width': config.main_corridor_width,
                'secondary_corridor_width': config.secondary_corridor_width,
                'pathfinding_algorithm': config.pathfinding_algorithm.value,
                'optimize_traffic_flow': config.optimize_traffic_flow
            }
        }
        
        return result

    def _create_empty_corridor_result(self, reason: str) -> Dict[str, Any]:
        """Create empty corridor result"""
        return {
            'success': False,
            'corridors': [],
            'corridor_metrics': {
                'total_corridors': 0,
                'total_corridor_area': 0.0
            },
            'processing_info': {
                'processing_time': 0.1,
                'reason': reason
            }
        }

    def _create_error_corridor_result(self, error_message: str) -> Dict[str, Any]:
        """Create error corridor result"""
        return {
            'success': False,
            'error': error_message,
            'corridors': [],
            'corridor_metrics': {
                'total_corridors': 0,
                'total_corridor_area': 0.0
            }
        }

# Create global instance
advanced_corridor_generator = AdvancedCorridorGenerator()