"""
Building Outline Creator
Creates complete connected building perimeter from DXF wall segments
"""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, linemerge
from typing import List, Tuple, Dict, Any
import networkx as nx

class BuildingOutlineCreator:
    """Creates connected building outline from wall segments"""
    
    def __init__(self):
        self.connection_tolerance = 5.0  # Increased tolerance for DXF precision issues
    
    def create_building_outline(self, wall_segments: List[List[List[float]]]) -> List[List[List[float]]]:
        """Create connected building outline from individual wall segments"""
        if not wall_segments:
            return []
        
        # Convert to LineString objects
        lines = []
        for segment in wall_segments:
            if len(segment) >= 2:
                try:
                    line = LineString(segment)
                    if line.length > 0.1:  # Filter out tiny segments
                        lines.append(line)
                except:
                    continue
        
        if not lines:
            return wall_segments
        
        # Method 1: Try linemerge (combines touching LineStrings)
        try:
            merged = linemerge(lines)
            if hasattr(merged, 'geoms'):
                # Multiple LineStrings returned
                connected_walls = []
                for geom in merged.geoms:
                    if geom.length > 1.0:  # Only keep substantial walls
                        coords = list(geom.coords)
                        connected_walls.append(coords)
                
                if len(connected_walls) < len(wall_segments) * 0.8:  # Significant reduction
                    return connected_walls
            elif merged.length > 0:
                # Single LineString returned
                coords = list(merged.coords)
                return [coords]
        except:
            pass
        
        # Method 2: Graph-based connection
        return self._connect_via_graph(wall_segments)
    
    def _connect_via_graph(self, wall_segments: List[List[List[float]]]) -> List[List[List[float]]]:
        """Connect wall segments using graph algorithms"""
        
        # Build graph of connections
        G = nx.Graph()
        
        # Add all endpoints as nodes
        endpoints = []
        segment_map = {}
        
        for i, segment in enumerate(wall_segments):
            if len(segment) >= 2:
                start = tuple(segment[0])
                end = tuple(segment[-1])
                endpoints.extend([start, end])
                segment_map[i] = (start, end, segment)
        
        # Add nodes to graph
        for point in set(endpoints):
            G.add_node(point)
        
        # Connect nearby endpoints
        for i, p1 in enumerate(endpoints):
            for j, p2 in enumerate(endpoints[i+1:], i+1):
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist <= self.connection_tolerance:
                    G.add_edge(p1, p2, weight=dist)
        
        # Find connected components and build walls
        connected_walls = []
        components = list(nx.connected_components(G))
        
        for component in components:
            if len(component) >= 4:  # Need at least 2 segments (4 endpoints)
                # Build wall from this component
                wall_points = self._build_wall_from_component(component, segment_map)
                if len(wall_points) >= 4:  # Substantial wall
                    connected_walls.append(wall_points)
        
        # If graph method didn't work well, return simplified version
        if not connected_walls or len(connected_walls) > len(wall_segments) * 0.9:
            return self._create_simplified_outline(wall_segments)
        
        return connected_walls
    
    def _build_wall_from_component(self, component, segment_map):
        """Build continuous wall from connected component"""
        # Find segments that belong to this component
        relevant_segments = []
        for seg_id, (start, end, segment) in segment_map.items():
            if start in component or end in component:
                relevant_segments.append(segment)
        
        if not relevant_segments:
            return []
        
        # Sort segments to create continuous path
        wall_points = list(relevant_segments[0])
        used_segments = {0}
        
        while len(used_segments) < len(relevant_segments):
            last_point = wall_points[-1]
            best_segment = None
            best_dist = float('inf')
            best_idx = None
            reverse = False
            
            for i, segment in enumerate(relevant_segments):
                if i in used_segments:
                    continue
                
                # Check distance to start of segment
                dist_start = np.sqrt((last_point[0] - segment[0][0])**2 + (last_point[1] - segment[0][1])**2)
                if dist_start < best_dist and dist_start <= self.connection_tolerance:
                    best_dist = dist_start
                    best_segment = segment
                    best_idx = i
                    reverse = False
                
                # Check distance to end of segment (reversed)
                dist_end = np.sqrt((last_point[0] - segment[-1][0])**2 + (last_point[1] - segment[-1][1])**2)
                if dist_end < best_dist and dist_end <= self.connection_tolerance:
                    best_dist = dist_end
                    best_segment = segment
                    best_idx = i
                    reverse = True
            
            if best_segment is None:
                break
            
            # Add segment to wall
            if reverse:
                wall_points.extend(reversed(best_segment[:-1]))  # Skip last point to avoid duplication
            else:
                wall_points.extend(best_segment[1:])  # Skip first point to avoid duplication
            
            used_segments.add(best_idx)
        
        return wall_points
    
    def _create_simplified_outline(self, wall_segments):
        """Create simplified building outline when connection fails"""
        # Get overall bounds
        all_points = []
        for segment in wall_segments:
            all_points.extend(segment)
        
        if not all_points:
            return wall_segments
        
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        # Create rectangular outline
        outline = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y]  # Close the rectangle
        ]
        
        return [outline]
    
    def optimize_wall_display(self, walls):
        """Optimize walls for better display"""
        optimized = []
        
        for wall in walls:
            if len(wall) >= 2:
                # Simplify wall if it has too many points
                if len(wall) > 20:
                    # Sample points to reduce complexity
                    step = len(wall) // 15
                    simplified = wall[::step]
                    if wall[-1] not in simplified:
                        simplified.append(wall[-1])
                    optimized.append(simplified)
                else:
                    optimized.append(wall)
        
        return optimized