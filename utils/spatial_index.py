"""
Spatial Indexing System for Ultra-High Performance
R-tree spatial index for fast overlap and proximity queries
"""

import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
import logging

class SpatialIndex:
    """High-performance spatial indexing for geometry operations"""
    
    def __init__(self):
        self.zones_index = None
        self.walls_index = None
        self.ilots_index = None
        self.zones_data = []
        self.walls_data = []
        self.ilots_data = []
        self.geometry_cache = {}
        
    def build_zones_index(self, zones: List[Dict]):
        """Build spatial index for zones"""
        start_time = time.time()
        
        geometries = []
        self.zones_data = []
        
        for i, zone in enumerate(zones):
            if 'bounds' in zone:
                bounds = zone['bounds']
                geom = box(bounds['min_x'], bounds['min_y'], bounds['max_x'], bounds['max_y'])
                geometries.append(geom)
                self.zones_data.append((i, zone, geom))
        
        if geometries:
            self.zones_index = STRtree(geometries)
            
        build_time = time.time() - start_time
        logging.info(f"Built zones spatial index: {len(geometries)} zones in {build_time:.3f}s")
        
    def build_walls_index(self, walls: List[Dict]):
        """Build spatial index for walls"""
        start_time = time.time()
        
        geometries = []
        self.walls_data = []
        
        for i, wall in enumerate(walls):
            if 'coordinates' in wall:
                coords = wall['coordinates']
                if len(coords) >= 2:
                    # Create line buffer for wall
                    from shapely.geometry import LineString
                    line = LineString(coords)
                    geom = line.buffer(0.1)  # Small buffer for wall thickness
                    geometries.append(geom)
                    self.walls_data.append((i, wall, geom))
        
        if geometries:
            self.walls_index = STRtree(geometries)
            
        build_time = time.time() - start_time
        logging.info(f"Built walls spatial index: {len(geometries)} walls in {build_time:.3f}s")
        
    def build_ilots_index(self, ilots: List[Dict]):
        """Build spatial index for placed îlots"""
        start_time = time.time()
        
        geometries = []
        self.ilots_data = []
        
        for i, ilot in enumerate(ilots):
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 3.0)
            height = ilot.get('height', 2.0)
            
            geom = box(x, y, x + width, y + height)
            geometries.append(geom)
            self.ilots_data.append((i, ilot, geom))
        
        if geometries:
            self.ilots_index = STRtree(geometries)
            
        build_time = time.time() - start_time
        logging.info(f"Built îlots spatial index: {len(geometries)} îlots in {build_time:.3f}s")
        
    def find_overlapping_zones(self, point: Point) -> List[Dict]:
        """Find zones that overlap with a point"""
        if not self.zones_index:
            return []
            
        possible_matches = self.zones_index.query(point)
        overlapping = []
        
        for geom in possible_matches:
            for idx, zone, zone_geom in self.zones_data:
                if zone_geom == geom and zone_geom.contains(point):
                    overlapping.append(zone)
                    break
                    
        return overlapping
        
    def find_nearby_walls(self, point: Point, distance: float = 1.0) -> List[Dict]:
        """Find walls within distance of a point"""
        if not self.walls_index:
            return []
            
        search_area = point.buffer(distance)
        possible_matches = self.walls_index.query(search_area)
        nearby = []
        
        for geom in possible_matches:
            for idx, wall, wall_geom in self.walls_data:
                if wall_geom == geom and wall_geom.intersects(search_area):
                    nearby.append(wall)
                    break
                    
        return nearby
        
    def check_ilot_overlap(self, x: float, y: float, width: float, height: float) -> bool:
        """Check if îlot position overlaps with existing îlots"""
        if not self.ilots_index:
            return False
            
        test_box = box(x, y, x + width, y + height)
        possible_matches = self.ilots_index.query(test_box)
        
        for geom in possible_matches:
            if geom.intersects(test_box):
                return True
                
        return False
        
    def get_cached_geometry(self, cache_key: str) -> Optional[Any]:
        """Get cached geometry result"""
        return self.geometry_cache.get(cache_key)
        
    def cache_geometry(self, cache_key: str, geometry: Any):
        """Cache geometry result"""
        self.geometry_cache[cache_key] = geometry
        
    def clear_cache(self):
        """Clear geometry cache"""
        self.geometry_cache.clear()