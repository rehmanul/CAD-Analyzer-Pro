"""
Targeted Floor Plan Extractor
Extracts the specific floor plan section (bottom-left) from multi-view DXF files
"""

import ezdxf
from ezdxf import recover
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time

class TargetedFloorPlanExtractor:
    """Extract specific floor plan section from multi-view DXF files"""
    
    def __init__(self):
        self.wall_layers = ['WALLS', 'WALL', 'MUR', 'MURS', '0', 'DEFPOINTS', 'LAYER']
        self.door_layers = ['DOORS', 'DOOR', 'PORTE', 'PORTES']
        
    def process_dxf_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process DXF file and extract the bottom-left floor plan"""
        try:
            import io
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                print(f"Processing DXF file: {filename}")
                start_time = time.time()
                
                # Load DXF document
                doc, auditor = recover.readfile(tmp_file_path)
                
                # Get all entities and their bounds
                all_entities = list(doc.modelspace())
                
                # Find the target floor plan region (bottom-left quadrant)
                target_bounds = self._find_target_floor_plan_region(all_entities)
                print(f"Target floor plan bounds: {target_bounds}")
                
                # Extract entities from the target region
                walls = self._extract_walls_from_target_region(all_entities, target_bounds)
                doors = self._extract_doors_from_target_region(all_entities, target_bounds)
                restricted_areas = self._detect_restricted_areas(all_entities, target_bounds)
                entrances = self._detect_entrances(all_entities, target_bounds)
                
                processing_time = time.time() - start_time
                print(f"Target floor plan extraction completed in {processing_time:.2f}s")
                print(f"Extracted: {len(walls)} walls, {len(doors)} doors, {len(restricted_areas)} restricted areas, {len(entrances)} entrances")
                
                return {
                    'success': True,
                    'walls': walls,
                    'doors': doors,
                    'windows': [],
                    'boundaries': [],
                    'restricted_areas': restricted_areas,
                    'entrances': entrances,
                    'bounds': target_bounds,
                    'entity_count': len(walls) + len(doors),
                    'entities': [],
                    'processing_time': processing_time
                }
                
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error processing DXF file: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'walls': [],
                'restricted_areas': [],
                'entrances': [],
                'bounds': {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100},
                'entity_count': 0,
                'entities': []
            }
    
    def _find_target_floor_plan_region(self, entities) -> Dict[str, float]:
        """Find the bottom-left floor plan region from the 6-view layout"""
        try:
            # Get all entity bounds
            all_bounds = []
            for entity in entities:
                bbox = self._get_entity_bbox(entity)
                if bbox:
                    all_bounds.append(bbox)
            
            if not all_bounds:
                return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
            
            # Find overall document bounds
            doc_min_x = min(b['min_x'] for b in all_bounds)
            doc_max_x = max(b['max_x'] for b in all_bounds)
            doc_min_y = min(b['min_y'] for b in all_bounds)
            doc_max_y = max(b['max_y'] for b in all_bounds)
            
            # Calculate dimensions
            doc_width = doc_max_x - doc_min_x
            doc_height = doc_max_y - doc_min_y
            
            # The layout appears to be 2 rows x 3 columns
            # Bottom-left floor plan would be in the bottom-left section
            # Based on your image, it's roughly the left 1/3 and bottom 1/2
            
            target_min_x = doc_min_x
            target_max_x = doc_min_x + (doc_width * 0.35)  # Left 35% of width
            target_min_y = doc_min_y
            target_max_y = doc_min_y + (doc_height * 0.5)  # Bottom 50% of height
            
            # Look for the most dense region of architectural elements in this area
            region_entities = []
            for entity in entities:
                bbox = self._get_entity_bbox(entity)
                if bbox and self._is_entity_in_region(bbox, {
                    'min_x': target_min_x,
                    'max_x': target_max_x,
                    'min_y': target_min_y,
                    'max_y': target_max_y
                }):
                    if self._is_architectural_entity(entity):
                        region_entities.append(entity)
            
            if region_entities:
                # Refine bounds based on actual architectural entities
                arch_bounds = []
                for entity in region_entities:
                    bbox = self._get_entity_bbox(entity)
                    if bbox:
                        arch_bounds.append(bbox)
                
                if arch_bounds:
                    refined_min_x = min(b['min_x'] for b in arch_bounds)
                    refined_max_x = max(b['max_x'] for b in arch_bounds)
                    refined_min_y = min(b['min_y'] for b in arch_bounds)
                    refined_max_y = max(b['max_y'] for b in arch_bounds)
                    
                    # Add small padding
                    padding = min(doc_width, doc_height) * 0.02
                    
                    return {
                        'min_x': refined_min_x - padding,
                        'max_x': refined_max_x + padding,
                        'min_y': refined_min_y - padding,
                        'max_y': refined_max_y + padding
                    }
            
            # Fallback to calculated region
            return {
                'min_x': target_min_x,
                'max_x': target_max_x,
                'min_y': target_min_y,
                'max_y': target_max_y
            }
            
        except Exception as e:
            print(f"Error finding target floor plan region: {str(e)}")
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
    
    def _is_architectural_entity(self, entity) -> bool:
        """Check if entity is architectural (walls, doors, etc.)"""
        entity_type = entity.dxftype()
        
        # Lines that could be walls
        if entity_type == 'LINE':
            return True
        
        # Polylines that could be room boundaries
        if entity_type in ['LWPOLYLINE', 'POLYLINE']:
            return True
        
        # Arcs that could be door swings
        if entity_type == 'ARC':
            return True
        
        # Circles that could be fixtures
        if entity_type == 'CIRCLE':
            return True
        
        return False
    
    def _is_entity_in_region(self, bbox: Dict, region: Dict) -> bool:
        """Check if entity bbox overlaps with region"""
        return not (bbox['max_x'] < region['min_x'] or 
                   bbox['min_x'] > region['max_x'] or
                   bbox['max_y'] < region['min_y'] or
                   bbox['min_y'] > region['max_y'])
    
    def _extract_walls_from_target_region(self, entities, bounds: Dict[str, float]) -> List[Dict]:
        """Extract walls from the target region"""
        walls = []
        
        try:
            for entity in entities:
                if not self._is_entity_in_bounds(entity, bounds):
                    continue
                
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    
                    # Check if it's a substantial line (potential wall)
                    length = ((end.x - start.x)**2 + (end.y - start.y)**2)**0.5
                    if length > 0.5:  # Minimum wall length
                        wall = {
                            'type': 'LINE',
                            'points': [
                                (float(start.x), float(start.y)),
                                (float(end.x), float(end.y))
                            ],
                            'layer': entity.dxf.layer
                        }
                        walls.append(wall)
                
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    if len(points) >= 2:
                        wall = {
                            'type': 'POLYLINE',
                            'points': [(float(point[0]), float(point[1])) for point in points],
                            'layer': entity.dxf.layer
                        }
                        walls.append(wall)
        
        except Exception as e:
            print(f"Error extracting walls: {str(e)}")
        
        return walls
    
    def _extract_doors_from_target_region(self, entities, bounds: Dict[str, float]) -> List[Dict]:
        """Extract doors from the target region"""
        doors = []
        
        try:
            for entity in entities:
                if not self._is_entity_in_bounds(entity, bounds):
                    continue
                
                if entity.dxftype() == 'ARC':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    
                    # Check if it's a door-sized arc
                    if 0.3 <= radius <= 3.0:
                        door = {
                            'type': 'ARC',
                            'center': (float(center.x), float(center.y)),
                            'radius': float(radius),
                            'start_angle': float(entity.dxf.start_angle),
                            'end_angle': float(entity.dxf.end_angle),
                            'layer': entity.dxf.layer
                        }
                        doors.append(door)
        
        except Exception as e:
            print(f"Error extracting doors: {str(e)}")
        
        return doors
    
    def _detect_restricted_areas(self, entities, bounds: Dict[str, float]) -> List[Dict]:
        """Detect restricted areas (like stairs, elevators) in the target region"""
        restricted_areas = []
        
        try:
            # Look for rectangular polylines that could be restricted areas
            for entity in entities:
                if not self._is_entity_in_bounds(entity, bounds):
                    continue
                
                if entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    if len(points) >= 4:
                        # Check if it forms a closed rectangle
                        if self._is_closed_rectangle(points):
                            # Calculate area
                            area = self._calculate_polygon_area(points)
                            
                            # If it's a reasonable size for a restricted area
                            if 1.0 <= area <= 50.0:
                                restricted_area = {
                                    'type': 'POLYGON',
                                    'points': [(float(point[0]), float(point[1])) for point in points],
                                    'layer': entity.dxf.layer
                                }
                                restricted_areas.append(restricted_area)
        
        except Exception as e:
            print(f"Error detecting restricted areas: {str(e)}")
        
        return restricted_areas
    
    def _detect_entrances(self, entities, bounds: Dict[str, float]) -> List[Dict]:
        """Detect entrances in the target region"""
        entrances = []
        
        try:
            # Look for gaps in walls (entrances) or door symbols
            for entity in entities:
                if not self._is_entity_in_bounds(entity, bounds):
                    continue
                
                if entity.dxftype() == 'ARC':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    
                    # Door arcs are typically entrances
                    if 0.3 <= radius <= 3.0:
                        entrance = {
                            'type': 'ENTRANCE',
                            'position': (float(center.x), float(center.y)),
                            'width': float(radius * 2),
                            'layer': entity.dxf.layer
                        }
                        entrances.append(entrance)
        
        except Exception as e:
            print(f"Error detecting entrances: {str(e)}")
        
        return entrances
    
    def _is_entity_in_bounds(self, entity, bounds: Dict[str, float]) -> bool:
        """Check if entity is within bounds"""
        try:
            bbox = self._get_entity_bbox(entity)
            if not bbox:
                return False
            
            # Check if entity center is within bounds
            center_x = (bbox['min_x'] + bbox['max_x']) / 2
            center_y = (bbox['min_y'] + bbox['max_y']) / 2
            
            return (bounds['min_x'] <= center_x <= bounds['max_x'] and
                    bounds['min_y'] <= center_y <= bounds['max_y'])
        except:
            return False
    
    def _get_entity_bbox(self, entity) -> Optional[Dict]:
        """Get bounding box for an entity"""
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                return {
                    'min_x': min(start.x, end.x),
                    'max_x': max(start.x, end.x),
                    'min_y': min(start.y, end.y),
                    'max_y': max(start.y, end.y)
                }
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points())
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    return {
                        'min_x': min(xs),
                        'max_x': max(xs),
                        'min_y': min(ys),
                        'max_y': max(ys)
                    }
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                return {
                    'min_x': center.x - radius,
                    'max_x': center.x + radius,
                    'min_y': center.y - radius,
                    'max_y': center.y + radius
                }
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                return {
                    'min_x': center.x - radius,
                    'max_x': center.x + radius,
                    'min_y': center.y - radius,
                    'max_y': center.y + radius
                }
        except:
            pass
        return None
    
    def _is_closed_rectangle(self, points) -> bool:
        """Check if points form a closed rectangle"""
        try:
            if len(points) < 4:
                return False
            
            # Check if first and last points are close (closed)
            if len(points) >= 4:
                first = points[0]
                last = points[-1]
                if abs(first[0] - last[0]) < 0.1 and abs(first[1] - last[1]) < 0.1:
                    return True
            
            return False
        except:
            return False
    
    def _calculate_polygon_area(self, points) -> float:
        """Calculate area of a polygon"""
        try:
            if len(points) < 3:
                return 0.0
            
            area = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            
            return abs(area) / 2.0
        except:
            return 0.0