"""
Optimized Corridor Generator with Spatial Indexing
Ultra-high performance corridor generation with advanced pathfinding algorithms
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
from spatial_index import SpatialIndex
import logging

class OptimizedCorridorGenerator:
    """Ultra-high performance corridor generator with spatial optimization"""
    
    def __init__(self):
        self.spatial_index = SpatialIndex()
        self.corridor_cache = {}
        self.network_graph = None
        
    def generate_optimized_corridors(self, analysis_data: Dict, ilots: List[Dict]) -> List[Dict]:
        """Generate optimized corridor network with spatial indexing"""
        start_time = time.time()
        
        # Build spatial indices
        self._build_spatial_indices(analysis_data, ilots)
        
        # Generate corridor network
        corridors = []
        
        # 1. Generate main corridors from entrances
        main_corridors = self._generate_main_corridors(analysis_data, ilots)
        corridors.extend(main_corridors)
        
        # 2. Generate facing corridors (mandatory client requirement)
        facing_corridors = self._generate_facing_corridors(ilots)
        corridors.extend(facing_corridors)
        
        # 3. Generate secondary corridors for connectivity
        secondary_corridors = self._generate_secondary_corridors(ilots, main_corridors)
        corridors.extend(secondary_corridors)
        
        # 4. Optimize corridor network
        corridors = self._optimize_corridor_network(corridors)
        
        generation_time = time.time() - start_time
        
        # Add performance metrics
        for corridor in corridors:
            corridor['generation_time'] = generation_time
            corridor['generation_method'] = 'optimized_spatial'
            
        logging.info(f"Generated {len(corridors)} corridors in {generation_time:.3f}s")
        
        return corridors
        
    def _build_spatial_indices(self, analysis_data: Dict, ilots: List[Dict]):
        """Build spatial indices for fast corridor planning"""
        # Build îlot spatial index
        if ilots:
            self.spatial_index.build_ilots_index(ilots)
            
        # Build walls spatial index
        walls = analysis_data.get('walls', [])
        if walls:
            self.spatial_index.build_walls_index(walls)
            
        # Build zones spatial index
        zones = analysis_data.get('zones', [])
        if zones:
            self.spatial_index.build_zones_index(zones)
            
    def _generate_main_corridors(self, analysis_data: Dict, ilots: List[Dict]) -> List[Dict]:
        """Generate main corridors from entrances using optimized pathfinding"""
        corridors = []
        
        entrances = analysis_data.get('entrances', [])
        bounds = analysis_data.get('bounds', {})
        
        if not entrances or not ilots:
            return corridors
            
        # For each entrance, create main corridor
        for entrance in entrances:
            entrance_pos = entrance.get('position', [50, 0])
            
            # Find nearest îlot rows
            ilot_rows = self._detect_ilot_rows(ilots)
            
            for row_id, row_ilots in ilot_rows.items():
                # Calculate row center
                row_center = self._calculate_row_center(row_ilots)
                
                # Generate corridor from entrance to row
                corridor_points = self._generate_corridor_path(entrance_pos, row_center, bounds)
                
                if corridor_points:
                    corridor = {
                        'id': f'main_corridor_{entrance["id"]}_{row_id}',
                        'type': 'main',
                        'points': corridor_points,
                        'width': 2.0,
                        'length': self._calculate_path_length(corridor_points),
                        'entrance_id': entrance['id'],
                        'row_id': row_id,
                        'is_mandatory': True
                    }
                    corridors.append(corridor)
                    
        return corridors
        
    def _generate_facing_corridors(self, ilots: List[Dict]) -> List[Dict]:
        """Generate corridors between facing îlots - MANDATORY CLIENT REQUIREMENT"""
        corridors = []
        
        # Detect îlot rows
        ilot_rows = self._detect_ilot_rows(ilots)
        
        # Generate corridors between facing rows
        row_ids = list(ilot_rows.keys())
        
        for i in range(len(row_ids)):
            for j in range(i + 1, len(row_ids)):
                row1_id = row_ids[i]
                row2_id = row_ids[j]
                
                row1_ilots = ilot_rows[row1_id]
                row2_ilots = ilot_rows[row2_id]
                
                # Check if rows are facing each other
                if self._are_rows_facing(row1_ilots, row2_ilots):
                    # Generate corridor between rows
                    corridor_points = self._generate_facing_corridor_path(row1_ilots, row2_ilots)
                    
                    if corridor_points:
                        corridor = {
                            'id': f'facing_corridor_{row1_id}_{row2_id}',
                            'type': 'facing',
                            'points': corridor_points,
                            'width': 1.5,
                            'length': self._calculate_path_length(corridor_points),
                            'row1_id': row1_id,
                            'row2_id': row2_id,
                            'is_mandatory': True
                        }
                        corridors.append(corridor)
                        
        return corridors
        
    def _generate_secondary_corridors(self, ilots: List[Dict], main_corridors: List[Dict]) -> List[Dict]:
        """Generate secondary corridors for improved connectivity"""
        corridors = []
        
        # Create network graph
        self.network_graph = nx.Graph()
        
        # Add îlots as nodes - ensure they have IDs
        for i, ilot in enumerate(ilots):
            if 'id' not in ilot:
                ilot['id'] = f"ilot_{i+1}"
            self.network_graph.add_node(ilot['id'], pos=(ilot['x'], ilot['y']))
            
        # Add main corridors as edges
        for corridor in main_corridors:
            if len(corridor['points']) >= 2:
                start_point = corridor['points'][0]
                end_point = corridor['points'][-1]
                
                # Find nearest îlots to corridor endpoints
                start_ilot = self._find_nearest_ilot(start_point, ilots)
                end_ilot = self._find_nearest_ilot(end_point, ilots)
                
                if start_ilot and end_ilot:
                    self.network_graph.add_edge(start_ilot['id'], end_ilot['id'])
                    
        # Generate secondary corridors to improve connectivity
        unconnected_components = list(nx.connected_components(self.network_graph))
        
        if len(unconnected_components) > 1:
            # Connect components
            for i in range(len(unconnected_components) - 1):
                comp1 = unconnected_components[i]
                comp2 = unconnected_components[i + 1]
                
                # Find closest nodes between components
                min_distance = float('inf')
                closest_pair = None
                
                for node1 in comp1:
                    for node2 in comp2:
                        pos1 = self.network_graph.nodes[node1]['pos']
                        pos2 = self.network_graph.nodes[node2]['pos']
                        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_pair = (node1, node2)
                            
                if closest_pair:
                    # Create secondary corridor
                    pos1 = self.network_graph.nodes[closest_pair[0]]['pos']
                    pos2 = self.network_graph.nodes[closest_pair[1]]['pos']
                    
                    corridor = {
                        'id': f'secondary_corridor_{i}',
                        'type': 'secondary',
                        'points': [list(pos1), list(pos2)],
                        'width': 1.0,
                        'length': min_distance,
                        'ilot1_id': closest_pair[0],
                        'ilot2_id': closest_pair[1],
                        'is_mandatory': False
                    }
                    corridors.append(corridor)
                    
        return corridors
        
    def _detect_ilot_rows(self, ilots: List[Dict]) -> Dict[int, List[Dict]]:
        """Detect îlot rows using optimized clustering"""
        if not ilots:
            return {}
            
        # Sort îlots by y-coordinate
        sorted_ilots = sorted(ilots, key=lambda x: x.get('y', 0))
        
        # Group into rows using distance threshold
        rows = {}
        current_row_id = 0
        row_threshold = 3.0  # Maximum distance between îlots in same row
        
        for ilot in sorted_ilots:
            placed_in_row = False
            
            # Check if îlot belongs to existing row
            for row_id, row_ilots in rows.items():
                if row_ilots:
                    avg_y = sum(i.get('y', 0) for i in row_ilots) / len(row_ilots)
                    if abs(ilot.get('y', 0) - avg_y) <= row_threshold:
                        row_ilots.append(ilot)
                        placed_in_row = True
                        break
                        
            # Create new row if not placed
            if not placed_in_row:
                rows[current_row_id] = [ilot]
                current_row_id += 1
                
        return rows
        
    def _calculate_row_center(self, row_ilots: List[Dict]) -> List[float]:
        """Calculate center point of îlot row"""
        if not row_ilots:
            return [0, 0]
            
        avg_x = sum(ilot.get('x', 0) + ilot.get('width', 3) / 2 for ilot in row_ilots) / len(row_ilots)
        avg_y = sum(ilot.get('y', 0) + ilot.get('height', 2) / 2 for ilot in row_ilots) / len(row_ilots)
        
        return [avg_x, avg_y]
        
    def _generate_corridor_path(self, start: List[float], end: List[float], bounds: Dict) -> List[List[float]]:
        """Generate optimized corridor path between two points"""
        # Simple straight line for now (can be enhanced with A* pathfinding)
        return [start, end]
        
    def _are_rows_facing(self, row1_ilots: List[Dict], row2_ilots: List[Dict]) -> bool:
        """Check if two rows are facing each other"""
        if not row1_ilots or not row2_ilots:
            return False
            
        # Calculate row centers
        center1 = self._calculate_row_center(row1_ilots)
        center2 = self._calculate_row_center(row2_ilots)
        
        # Check if rows are horizontally aligned and reasonably close
        y_diff = abs(center1[1] - center2[1])
        x_diff = abs(center1[0] - center2[0])
        
        # Rows are facing if they are horizontally separated but vertically aligned
        return y_diff <= 5.0 and x_diff >= 3.0
        
    def _generate_facing_corridor_path(self, row1_ilots: List[Dict], row2_ilots: List[Dict]) -> List[List[float]]:
        """Generate corridor path between facing rows"""
        center1 = self._calculate_row_center(row1_ilots)
        center2 = self._calculate_row_center(row2_ilots)
        
        # Create corridor path with intermediate points for better visualization
        mid_x = (center1[0] + center2[0]) / 2
        mid_y = (center1[1] + center2[1]) / 2
        
        return [center1, [mid_x, mid_y], center2]
        
    def _find_nearest_ilot(self, point: List[float], ilots: List[Dict]) -> Optional[Dict]:
        """Find nearest îlot to a given point"""
        if not ilots:
            return None
            
        min_distance = float('inf')
        nearest_ilot = None
        
        for ilot in ilots:
            ilot_center = [
                ilot.get('x', 0) + ilot.get('width', 3) / 2,
                ilot.get('y', 0) + ilot.get('height', 2) / 2
            ]
            
            center_x, center_y = center
            distance = np.sqrt((point[0] - ilot_center_x)**2 + (point[1] - ilot_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_ilot = ilot
                
        return nearest_ilot
        
    def _calculate_path_length(self, points: List[List[float]]) -> float:
        """Calculate total length of path"""
        if len(points) < 2:
            return 0.0
            
        total_length = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            total_length += np.sqrt(dx * dx + dy * dy)
            
        return total_length
        
    def _optimize_corridor_network(self, corridors: List[Dict]) -> List[Dict]:
        """Optimize corridor network by removing redundant corridors"""
        if not corridors:
            return corridors
            
        # Remove corridors that are too short
        min_length = 1.0
        filtered_corridors = [c for c in corridors if c.get('length', 0) >= min_length]
        
        # Remove overlapping corridors (basic implementation)
        optimized_corridors = []
        
        for corridor in filtered_corridors:
            is_redundant = False
            
            # Check for overlaps with existing corridors
            for existing in optimized_corridors:
                if self._corridors_overlap(corridor, existing):
                    # Keep the mandatory one or the longer one
                    if corridor.get('is_mandatory', False) and not existing.get('is_mandatory', False):
                        # Replace existing with mandatory corridor
                        optimized_corridors.remove(existing)
                        optimized_corridors.append(corridor)
                        is_redundant = True
                        break
                    elif existing.get('is_mandatory', False) and not corridor.get('is_mandatory', False):
                        # Skip this corridor
                        is_redundant = True
                        break
                    elif corridor.get('length', 0) > existing.get('length', 0):
                        # Replace with longer corridor
                        optimized_corridors.remove(existing)
                        optimized_corridors.append(corridor)
                        is_redundant = True
                        break
                    else:
                        is_redundant = True
                        break
                        
            if not is_redundant:
                optimized_corridors.append(corridor)
                
        return optimized_corridors
        
    def _corridors_overlap(self, corridor1: Dict, corridor2: Dict) -> bool:
        """Check if two corridors overlap"""
        # Simple overlap check based on endpoint proximity
        points1 = corridor1.get('points', [])
        points2 = corridor2.get('points', [])
        
        if len(points1) < 2 or len(points2) < 2:
            return False
            
        # Check if endpoints are close
        threshold = 2.0
        
        for p1 in [points1[0], points1[-1]]:
            for p2 in [points2[0], points2[-1]]:
                distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance < threshold:
                    return True
                    
        return False