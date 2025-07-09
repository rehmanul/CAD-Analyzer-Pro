
"""
Spatial Index for Ultra-High Performance Floor Plan Processing
Provides spatial indexing capabilities for fast geometric queries
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point, Polygon, LineString, box
from shapely.strtree import STRtree
import logging

class SpatialIndex:
    """Spatial indexing system for fast geometric queries"""
    
    def __init__(self):
        self.zones_tree = None
        self.walls_tree = None
        self.ilots_tree = None
        
        # Data stores
        self.zones_data = []
        self.walls_data = []
        self.ilots_data = []
        
        # Performance cache
        self.query_cache = {}
        
    def build_zones_index(self, zones: List[Dict]):
        """Build spatial index for zones"""
        try:
            if not zones:
                return
                
            geometries = []
            self.zones_data = []
            
            for zone in zones:
                try:
                    # Create geometry from zone data
                    if 'polygon' in zone:
                        geom = Polygon(zone['polygon'])
                    elif 'bounds' in zone:
                        bounds = zone['bounds']
                        geom = box(bounds.get('min_x', 0), bounds.get('min_y', 0),
                                  bounds.get('max_x', 10), bounds.get('max_y', 10))
                    else:
                        # Default geometry
                        geom = box(0, 0, 10, 10)
                    
                    if geom.is_valid:
                        geometries.append(geom)
                        self.zones_data.append(zone)
                        
                except Exception as e:
                    logging.warning(f"Error processing zone: {e}")
                    continue
            
            if geometries:
                self.zones_tree = STRtree(geometries)
                
        except Exception as e:
            logging.error(f"Error building zones index: {e}")
            
    def build_walls_index(self, walls: List[Dict]):
        """Build spatial index for walls"""
        try:
            if not walls:
                return
                
            geometries = []
            self.walls_data = []
            
            for wall in walls:
                try:
                    # Create geometry from wall data
                    if 'start' in wall and 'end' in wall:
                        geom = LineString([wall['start'], wall['end']])
                    elif 'coordinates' in wall:
                        coords = wall['coordinates']
                        if len(coords) >= 2:
                            geom = LineString(coords)
                        else:
                            continue
                    else:
                        continue
                    
                    if geom.is_valid:
                        geometries.append(geom)
                        self.walls_data.append(wall)
                        
                except Exception as e:
                    logging.warning(f"Error processing wall: {e}")
                    continue
            
            if geometries:
                self.walls_tree = STRtree(geometries)
                
        except Exception as e:
            logging.error(f"Error building walls index: {e}")
            
    def build_ilots_index(self, ilots: List[Dict]):
        """Build spatial index for îlots"""
        try:
            if not ilots:
                return
                
            geometries = []
            self.ilots_data = []
            
            for ilot in ilots:
                try:
                    # Create geometry from îlot data
                    x = ilot.get('x', 0)
                    y = ilot.get('y', 0)
                    width = ilot.get('width', 3)
                    height = ilot.get('height', 2)
                    
                    geom = box(x, y, x + width, y + height)
                    
                    if geom.is_valid:
                        geometries.append(geom)
                        self.ilots_data.append(ilot)
                        
                except Exception as e:
                    logging.warning(f"Error processing îlot: {e}")
                    continue
            
            if geometries:
                self.ilots_tree = STRtree(geometries)
                
        except Exception as e:
            logging.error(f"Error building îlots index: {e}")
            
    def find_nearby_zones(self, point: Point, distance: float = 1.0) -> List[Dict]:
        """Find zones near a point"""
        try:
            if not self.zones_tree:
                return []
                
            query_geom = point.buffer(distance)
            possible_matches = self.zones_tree.query(query_geom)
            
            nearby_zones = []
            for i, geom in enumerate(possible_matches):
                if geom.distance(point) <= distance:
                    if i < len(self.zones_data):
                        nearby_zones.append(self.zones_data[i])
                        
            return nearby_zones
            
        except Exception as e:
            logging.error(f"Error finding nearby zones: {e}")
            return []
            
    def find_nearby_walls(self, point: Point, distance: float = 1.0) -> List[Dict]:
        """Find walls near a point"""
        try:
            if not self.walls_tree:
                return []
                
            query_geom = point.buffer(distance)
            possible_matches = self.walls_tree.query(query_geom)
            
            nearby_walls = []
            for i, geom in enumerate(possible_matches):
                if geom.distance(point) <= distance:
                    if i < len(self.walls_data):
                        nearby_walls.append(self.walls_data[i])
                        
            return nearby_walls
            
        except Exception as e:
            logging.error(f"Error finding nearby walls: {e}")
            return []
            
    def find_nearby_ilots(self, point: Point, distance: float = 1.0) -> List[Dict]:
        """Find îlots near a point"""
        try:
            if not self.ilots_tree:
                return []
                
            query_geom = point.buffer(distance)
            possible_matches = self.ilots_tree.query(query_geom)
            
            nearby_ilots = []
            for i, geom in enumerate(possible_matches):
                if geom.distance(point) <= distance:
                    if i < len(self.ilots_data):
                        nearby_ilots.append(self.ilots_data[i])
                        
            return nearby_ilots
            
        except Exception as e:
            logging.error(f"Error finding nearby îlots: {e}")
            return []
            
    def check_ilot_overlap(self, x: float, y: float, width: float, height: float, 
                          tolerance: float = 0.1) -> bool:
        """Check if îlot placement would overlap with existing îlots"""
        try:
            if not self.ilots_tree:
                return False
                
            # Create proposed îlot geometry
            proposed_geom = box(x, y, x + width, y + height)
            
            # Query for potential overlaps
            possible_matches = self.ilots_tree.query(proposed_geom)
            
            for geom in possible_matches:
                if proposed_geom.intersects(geom):
                    # Check if intersection is more than tolerance
                    intersection = proposed_geom.intersection(geom)
                    if intersection.area > tolerance:
                        return True
                        
            return False
            
        except Exception as e:
            logging.error(f"Error checking îlot overlap: {e}")
            return False
            
    def check_wall_proximity(self, x: float, y: float, width: float, height: float,
                           min_distance: float = 0.5) -> bool:
        """Check if îlot placement respects minimum distance from walls"""
        try:
            if not self.walls_tree:
                return True  # No walls to check
                
            # Create îlot geometry
            ilot_geom = box(x, y, x + width, y + height)
            
            # Query for nearby walls
            possible_matches = self.walls_tree.query(ilot_geom)
            
            for geom in possible_matches:
                if ilot_geom.distance(geom) < min_distance:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error checking wall proximity: {e}")
            return True
            
    def find_best_zone_for_ilot(self, x: float, y: float, width: float, height: float) -> Optional[Dict]:
        """Find the best zone for îlot placement"""
        try:
            if not self.zones_tree:
                return None
                
            # Create îlot geometry
            ilot_geom = box(x, y, x + width, y + height)
            ilot_center = Point(x + width/2, y + height/2)
            
            # Query for containing zones
            possible_matches = self.zones_tree.query(ilot_geom)
            
            best_zone = None
            best_score = -1
            
            for i, geom in enumerate(possible_matches):
                if geom.contains(ilot_center):
                    zone_data = self.zones_data[i] if i < len(self.zones_data) else {}
                    
                    # Calculate score based on zone type
                    score = 0
                    zone_type = zone_data.get('type', 'unknown')
                    
                    if zone_type == 'buildable':
                        score = 100
                    elif zone_type == 'open':
                        score = 80
                    elif zone_type == 'circulation':
                        score = 60
                    else:
                        score = 40
                        
                    if score > best_score:
                        best_score = score
                        best_zone = zone_data
                        
            return best_zone
            
        except Exception as e:
            logging.error(f"Error finding best zone: {e}")
            return None
            
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        
    def get_statistics(self) -> Dict:
        """Get spatial index statistics"""
        return {
            'zones_count': len(self.zones_data),
            'walls_count': len(self.walls_data),
            'ilots_count': len(self.ilots_data),
            'zones_indexed': self.zones_tree is not None,
            'walls_indexed': self.walls_tree is not None,
            'ilots_indexed': self.ilots_tree is not None,
            'cache_size': len(self.query_cache)
        }
