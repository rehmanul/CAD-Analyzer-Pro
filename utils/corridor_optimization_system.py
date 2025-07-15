"""
Corridor Optimization System
Implements advanced pathfinding algorithms and corridor network generation
Following the detailed implementation plan for professional corridor visualization
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
from dataclasses import dataclass
import math
import heapq

@dataclass
class CorridorNode:
    """Node in the corridor network"""
    id: str
    position: Tuple[float, float]
    node_type: str  # 'ilot', 'junction', 'entrance'
    connections: List[str]
    accessibility_score: float

@dataclass
class CorridorSegment:
    """Segment of a corridor path"""
    id: str
    start_node: str
    end_node: str
    path: List[Tuple[float, float]]
    length: float
    width: float
    traffic_weight: float
    corridor_type: str  # 'main', 'secondary', 'access'

class CorridorOptimizationSystem:
    """
    Advanced corridor optimization system implementing the complete detailed plan
    Phase 4: Corridor Network Generation with pathfinding algorithms
    """
    
    def __init__(self):
        # Professional corridor specifications
        self.corridor_config = {
            'main_width': 2.0,
            'secondary_width': 1.5,
            'access_width': 1.2,
            'minimum_width': 1.0,
            'junction_radius': 1.0,
            'max_corridor_length': 50.0,
            'preferred_angle': 90.0,  # Prefer orthogonal corridors
            'smoothing_factor': 0.8
        }
        
        # Color coding for different corridor types
        self.corridor_colors = {
            'main': '#FF69B4',      # Pink main corridors
            'secondary': '#FFB6C1',  # Light pink secondary
            'access': '#FFC0CB',     # Pale pink access corridors
            'measurements': '#000000' # Black measurement text
        }
        
        # Traffic flow parameters
        self.traffic_config = {
            'base_flow': 1.0,
            'entrance_multiplier': 2.0,
            'junction_penalty': 0.5,
            'distance_weight': 0.3,
            'accessibility_weight': 0.4
        }
    
    def generate_corridor_network(self, floor_plan_data: Dict[str, Any],
                                ilots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4.1: Generate optimal corridor network using pathfinding algorithms
        Step 4.2: Create professional visualization with measurements
        """
        if not ilots:
            return []
        
        print("Generating corridor network...")
        
        # Step 4.1: Pathfinding Algorithm
        corridor_graph = self._create_corridor_graph(floor_plan_data, ilots)
        corridor_segments = self._optimize_corridor_paths(corridor_graph, floor_plan_data)
        
        # Step 4.2: Professional Visualization
        corridors_with_measurements = self._add_measurements_and_styling(corridor_segments, ilots)
        
        print(f"Generated {len(corridors_with_measurements)} corridor segments")
        
        return corridors_with_measurements
    
    def _create_corridor_graph(self, floor_plan_data: Dict[str, Any],
                             ilots: List[Dict[str, Any]]) -> nx.Graph:
        """Create graph representation of corridor network"""
        graph = nx.Graph()
        
        # Add îlot nodes
        for ilot in ilots:
            center_x = ilot['x'] + ilot['width'] / 2
            center_y = ilot['y'] + ilot['height'] / 2
            
            node_id = f"ilot_{ilot['id']}"
            graph.add_node(node_id, 
                         position=(center_x, center_y),
                         node_type='ilot',
                         accessibility_score=1.0,
                         ilot_data=ilot)
        
        # Add entrance nodes
        entrances = floor_plan_data.get('entrances', [])
        for i, entrance in enumerate(entrances):
            entrance_x = entrance.get('x', 0)
            entrance_y = entrance.get('y', 0)
            
            node_id = f"entrance_{i}"
            graph.add_node(node_id,
                         position=(entrance_x, entrance_y),
                         node_type='entrance',
                         accessibility_score=2.0)
        
        # Add junction nodes for optimal routing
        junction_nodes = self._generate_junction_nodes(floor_plan_data, ilots)
        for junction in junction_nodes:
            graph.add_node(junction['id'],
                         position=junction['position'],
                         node_type='junction',
                         accessibility_score=1.5)
        
        # Create edges based on visibility and distance
        self._create_corridor_edges(graph, floor_plan_data)
        
        return graph
    
    def _generate_junction_nodes(self, floor_plan_data: Dict[str, Any],
                               ilots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate strategic junction nodes for optimal routing"""
        bounds = floor_plan_data.get('bounds', {})
        junctions = []
        
        # Central junction
        center_x = (bounds.get('min_x', 0) + bounds.get('max_x', 100)) / 2
        center_y = (bounds.get('min_y', 0) + bounds.get('max_y', 100)) / 2
        
        junctions.append({
            'id': 'central_junction',
            'position': (center_x, center_y),
            'type': 'main_junction'
        })
        
        # Quadrant junctions for large spaces
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 100) - bounds.get('min_y', 0)
        
        if width > 50 and height > 50:
            # Quarter points
            quarter_points = [
                (bounds.get('min_x', 0) + width * 0.25, bounds.get('min_y', 0) + height * 0.25),
                (bounds.get('min_x', 0) + width * 0.75, bounds.get('min_y', 0) + height * 0.25),
                (bounds.get('min_x', 0) + width * 0.25, bounds.get('min_y', 0) + height * 0.75),
                (bounds.get('min_x', 0) + width * 0.75, bounds.get('min_y', 0) + height * 0.75)
            ]
            
            for i, point in enumerate(quarter_points):
                junctions.append({
                    'id': f'quad_junction_{i}',
                    'position': point,
                    'type': 'secondary_junction'
                })
        
        return junctions
    
    def _create_corridor_edges(self, graph: nx.Graph, floor_plan_data: Dict[str, Any]):
        """Create edges between nodes based on visibility and accessibility"""
        nodes = list(graph.nodes(data=True))
        
        for i, (node1_id, node1_data) in enumerate(nodes):
            for j, (node2_id, node2_data) in enumerate(nodes):
                if i >= j:
                    continue
                
                pos1 = node1_data['position']
                pos2 = node2_data['position']
                
                # Calculate distance
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Check if connection is valid
                if self._is_valid_connection(pos1, pos2, floor_plan_data):
                    # Calculate edge weight
                    weight = self._calculate_edge_weight(
                        node1_data, node2_data, distance, floor_plan_data
                    )
                    
                    graph.add_edge(node1_id, node2_id, 
                                 weight=weight,
                                 distance=distance,
                                 path=[pos1, pos2])
    
    def _is_valid_connection(self, pos1: Tuple[float, float], pos2: Tuple[float, float],
                           floor_plan_data: Dict[str, Any]) -> bool:
        """Check if direct connection between two points is valid"""
        # Check if line intersects with restricted areas
        line = LineString([pos1, pos2])
        
        restricted_areas = floor_plan_data.get('restricted_areas', [])
        for area in restricted_areas:
            if area.get('type') == 'circle':
                center = Point(area.get('x', 0), area.get('y', 0))
                radius = area.get('radius', 1.0)
                restricted_polygon = center.buffer(radius)
                
                if line.intersects(restricted_polygon):
                    return False
        
        return True
    
    def _calculate_edge_weight(self, node1_data: Dict, node2_data: Dict,
                             distance: float, floor_plan_data: Dict[str, Any]) -> float:
        """Calculate edge weight for pathfinding optimization"""
        base_weight = distance * self.traffic_config['distance_weight']
        
        # Accessibility bonus
        accessibility_bonus = (
            node1_data['accessibility_score'] + node2_data['accessibility_score']
        ) * self.traffic_config['accessibility_weight']
        
        # Entrance multiplier
        entrance_multiplier = 1.0
        if node1_data['node_type'] == 'entrance' or node2_data['node_type'] == 'entrance':
            entrance_multiplier = self.traffic_config['entrance_multiplier']
        
        # Junction penalty
        junction_penalty = 1.0
        if node1_data['node_type'] == 'junction' or node2_data['node_type'] == 'junction':
            junction_penalty = self.traffic_config['junction_penalty']
        
        weight = base_weight * entrance_multiplier * junction_penalty - accessibility_bonus
        
        return max(weight, 0.1)  # Minimum weight
    
    def _optimize_corridor_paths(self, graph: nx.Graph, floor_plan_data: Dict[str, Any]) -> List[CorridorSegment]:
        """Optimize corridor paths using advanced algorithms"""
        corridor_segments = []
        
        # Find minimum spanning tree for main corridors
        mst = nx.minimum_spanning_tree(graph)
        
        # Convert MST edges to corridor segments
        for edge in mst.edges(data=True):
            node1, node2, edge_data = edge
            
            segment = CorridorSegment(
                id=f"corridor_{node1}_{node2}",
                start_node=node1,
                end_node=node2,
                path=edge_data['path'],
                length=edge_data['distance'],
                width=self._determine_corridor_width(node1, node2, graph),
                traffic_weight=edge_data['weight'],
                corridor_type=self._determine_corridor_type(node1, node2, graph)
            )
            
            corridor_segments.append(segment)
        
        # Add additional connections for redundancy
        additional_segments = self._add_redundant_connections(graph, mst, floor_plan_data)
        corridor_segments.extend(additional_segments)
        
        # Smooth corridor paths
        corridor_segments = self._smooth_corridor_paths(corridor_segments)
        
        return corridor_segments
    
    def _determine_corridor_width(self, node1: str, node2: str, graph: nx.Graph) -> float:
        """Determine corridor width based on node types and traffic"""
        node1_data = graph.nodes[node1]
        node2_data = graph.nodes[node2]
        
        # Main corridors between entrances and junctions
        if (node1_data['node_type'] == 'entrance' or node2_data['node_type'] == 'entrance'):
            return self.corridor_config['main_width']
        
        # Secondary corridors between junctions and îlots
        if (node1_data['node_type'] == 'junction' or node2_data['node_type'] == 'junction'):
            return self.corridor_config['secondary_width']
        
        # Access corridors between îlots
        return self.corridor_config['access_width']
    
    def _determine_corridor_type(self, node1: str, node2: str, graph: nx.Graph) -> str:
        """Determine corridor type based on node types"""
        node1_data = graph.nodes[node1]
        node2_data = graph.nodes[node2]
        
        if (node1_data['node_type'] == 'entrance' or node2_data['node_type'] == 'entrance'):
            return 'main'
        elif (node1_data['node_type'] == 'junction' or node2_data['node_type'] == 'junction'):
            return 'secondary'
        else:
            return 'access'
    
    def _add_redundant_connections(self, graph: nx.Graph, mst: nx.Graph,
                                 floor_plan_data: Dict[str, Any]) -> List[CorridorSegment]:
        """Add redundant connections for better accessibility"""
        additional_segments = []
        
        # Find edges in original graph but not in MST
        mst_edges = set(mst.edges())
        
        for edge in graph.edges(data=True):
            node1, node2, edge_data = edge
            
            # Check if edge is not in MST and has low weight (good connection)
            if ((node1, node2) not in mst_edges and 
                (node2, node1) not in mst_edges and
                edge_data['weight'] < 10.0):
                
                segment = CorridorSegment(
                    id=f"redundant_{node1}_{node2}",
                    start_node=node1,
                    end_node=node2,
                    path=edge_data['path'],
                    length=edge_data['distance'],
                    width=self.corridor_config['access_width'],
                    traffic_weight=edge_data['weight'],
                    corridor_type='access'
                )
                
                additional_segments.append(segment)
        
        return additional_segments
    
    def _smooth_corridor_paths(self, segments: List[CorridorSegment]) -> List[CorridorSegment]:
        """Apply smoothing to corridor paths for better visualization"""
        smoothed_segments = []
        
        for segment in segments:
            if len(segment.path) > 2:
                # Apply smoothing algorithm
                smoothed_path = self._apply_path_smoothing(segment.path)
                
                segment.path = smoothed_path
                segment.length = self._calculate_path_length(smoothed_path)
            
            smoothed_segments.append(segment)
        
        return smoothed_segments
    
    def _apply_path_smoothing(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply smoothing algorithm to path points"""
        if len(path) <= 2:
            return path
        
        smoothed_path = [path[0]]  # Keep first point
        
        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # Apply smoothing factor
            smoothed_x = (prev_point[0] * (1 - self.corridor_config['smoothing_factor']) +
                         curr_point[0] * self.corridor_config['smoothing_factor'])
            smoothed_y = (prev_point[1] * (1 - self.corridor_config['smoothing_factor']) +
                         curr_point[1] * self.corridor_config['smoothing_factor'])
            
            smoothed_path.append((smoothed_x, smoothed_y))
        
        smoothed_path.append(path[-1])  # Keep last point
        
        return smoothed_path
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total length of a path"""
        total_length = 0.0
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += segment_length
        
        return total_length
    
    def _add_measurements_and_styling(self, segments: List[CorridorSegment],
                                    ilots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4.2: Add measurements and professional styling
        Convert corridor segments to visualization format
        """
        corridors_with_measurements = []
        
        for segment in segments:
            corridor_dict = {
                'id': segment.id,
                'type': segment.corridor_type,
                'path': segment.path,
                'width': segment.width,
                'length': segment.length,
                'color': self.corridor_colors[segment.corridor_type],
                'measurements': {
                    'length': f"{segment.length:.2f}m",
                    'width': f"{segment.width:.1f}m",
                    'area': f"{segment.length * segment.width:.2f}m²"
                }
            }
            
            corridors_with_measurements.append(corridor_dict)
        
        return corridors_with_measurements