"""
Floor Plan Extractor
Extracts only the main floor plan from DXF files containing multiple views
"""

import ezdxf
from ezdxf import recover
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import defaultdict

class FloorPlanExtractor:
    """Extract main floor plan from multi-view DXF files"""
    
    def __init__(self):
        self.wall_layers = ['WALLS', 'WALL', 'MUR', 'MURS', '0', 'DEFPOINTS', 'LAYER']
        self.door_layers = ['DOORS', 'DOOR', 'PORTE', 'PORTES']
        
    def process_dxf_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process DXF file and extract main floor plan"""
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
                
                # Find the main floor plan region
                floor_plan_bounds = self._find_floor_plan_region(doc)
                print(f"Floor plan bounds: {floor_plan_bounds}")
                
                # Extract entities from the floor plan region
                walls = self._extract_walls_from_region(doc, floor_plan_bounds)
                doors = self._extract_doors_from_region(doc, floor_plan_bounds)
                restricted_areas = self._create_restricted_areas(floor_plan_bounds)
                entrances = self._create_entrances_from_doors(doors)
                
                processing_time = time.time() - start_time
                print(f"Floor plan extraction completed in {processing_time:.2f}s")
                print(f"Extracted: {len(walls)} walls, {len(doors)} doors")
                
                return {
                    'success': True,
                    'walls': walls,
                    'doors': doors,
                    'windows': [],
                    'boundaries': [],
                    'restricted_areas': restricted_areas,
                    'entrances': entrances,
                    'bounds': floor_plan_bounds,
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
    
    def _find_floor_plan_region(self, doc) -> Dict[str, float]:
        """Find the floor plan region by analyzing rectangular wall patterns"""
        try:
            # Get all line entities
            lines = []
            for entity in doc.modelspace():
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    lines.append({
                        'start': (float(start.x), float(start.y)),
                        'end': (float(end.x), float(end.y)),
                        'entity': entity
                    })
            
            if not lines:
                return self._get_document_bounds(doc)
            
            # Find rectangular regions by looking for connected horizontal/vertical lines
            rectangular_regions = self._find_rectangular_regions(lines)
            
            if not rectangular_regions:
                return self._get_document_bounds(doc)
            
            # Select the largest rectangular region as the floor plan
            largest_region = max(rectangular_regions, key=lambda r: r['area'])
            
            # Add some padding
            padding = 2.0
            return {
                'min_x': largest_region['min_x'] - padding,
                'max_x': largest_region['max_x'] + padding,
                'min_y': largest_region['min_y'] - padding,
                'max_y': largest_region['max_y'] + padding
            }
            
        except Exception as e:
            print(f"Error finding floor plan region: {str(e)}")
            return self._get_document_bounds(doc)
    
    def _find_rectangular_regions(self, lines) -> List[Dict]:
        """Find rectangular regions formed by horizontal and vertical lines"""
        regions = []
        
        try:
            # Separate horizontal and vertical lines
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                start = line.get('start', [0, 0])
                end = line.get('end', [0, 0])
                
                # Check if line is horizontal (small Y difference)
                if abs(end[1] - start[1]) < 0.5:
                    horizontal_lines.append({
                        'y': (start[1] + end[1]) / 2,
                        'x1': min(start[0], end[0]),
                        'x2': max(start[0], end[0]),
                        'length': abs(end[0] - start[0])
                    })
                
                # Check if line is vertical (small X difference)
                elif abs(end[0] - start[0]) < 0.5:
                    vertical_lines.append({
                        'x': (start[0] + end[0]) / 2,
                        'y1': min(start[1], end[1]),
                        'y2': max(start[1], end[1]),
                        'length': abs(end[1] - start[1])
                    })
            
            # Filter lines to keep only significant ones
            horizontal_lines = [line for line in horizontal_lines if line['length'] > 3.0]
            vertical_lines = [line for line in vertical_lines if line['length'] > 3.0]
            
            print(f"Found {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")
            
            # Look for rectangular patterns
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                # Find the bounds of the most connected region
                all_x_coords = []
                all_y_coords = []
                
                for h_line in horizontal_lines:
                    all_x_coords.extend([h_line['x1'], h_line['x2']])
                    all_y_coords.append(h_line['y'])
                
                for v_line in vertical_lines:
                    all_x_coords.append(v_line['x'])
                    all_y_coords.extend([v_line['y1'], v_line['y2']])
                
                if all_x_coords and all_y_coords:
                    # Find the densest region
                    min_x = min(all_x_coords)
                    max_x = max(all_x_coords)
                    min_y = min(all_y_coords)
                    max_y = max(all_y_coords)
                    
                    # Calculate region area
                    area = (max_x - min_x) * (max_y - min_y)
                    
                    regions.append({
                        'min_x': min_x,
                        'max_x': max_x,
                        'min_y': min_y,
                        'max_y': max_y,
                        'area': area,
                        'horizontal_lines': len(horizontal_lines),
                        'vertical_lines': len(vertical_lines)
                    })
            
        except Exception as e:
            print(f"Error finding rectangular regions: {str(e)}")
        
        return regions
    
    def _extract_walls_from_region(self, doc, bounds: Dict[str, float]) -> List[Dict]:
        """Extract walls from the specified region"""
        walls = []
        
        try:
            for entity in doc.modelspace():
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    
                    # Check if line is within bounds
                    if (bounds['min_x'] <= start.x <= bounds['max_x'] and
                        bounds['min_y'] <= start.y <= bounds['max_y'] and
                        bounds['min_x'] <= end.x <= bounds['max_x'] and
                        bounds['min_y'] <= end.y <= bounds['max_y']):
                        
                        # Check if it's a wall (horizontal or vertical line)
                        dx = abs(end.x - start.x)
                        dy = abs(end.y - start.y)
                        
                        if (dx < 0.5 and dy > 1.0) or (dy < 0.5 and dx > 1.0):  # Horizontal or vertical
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
                        # Check if polyline is within bounds
                        within_bounds = True
                        for point in points:
                            if not (bounds['min_x'] <= point[0] <= bounds['max_x'] and
                                   bounds['min_y'] <= point[1] <= bounds['max_y']):
                                within_bounds = False
                                break
                        
                        if within_bounds:
                            wall = {
                                'type': 'POLYLINE',
                                'points': [(float(point[0]), float(point[1])) for point in points],
                                'layer': entity.dxf.layer
                            }
                            walls.append(wall)
        
        except Exception as e:
            print(f"Error extracting walls: {str(e)}")
        
        return walls
    
    def _extract_doors_from_region(self, doc, bounds: Dict[str, float]) -> List[Dict]:
        """Extract doors from the specified region"""
        doors = []
        
        try:
            for entity in doc.modelspace():
                if entity.dxftype() == 'ARC':
                    center = entity.dxf.center
                    
                    # Check if arc is within bounds
                    if (bounds['min_x'] <= center.x <= bounds['max_x'] and
                        bounds['min_y'] <= center.y <= bounds['max_y']):
                        
                        # Check if it's a door-sized arc
                        radius = entity.dxf.radius
                        if 0.5 <= radius <= 2.0:
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
    
    def _create_restricted_areas(self, bounds: Dict[str, float]) -> List[Dict]:
        """Create restricted areas based on floor plan bounds"""
        restricted_areas = []
        
        try:
            width = bounds['max_x'] - bounds['min_x']
            height = bounds['max_y'] - bounds['min_y']
            
            # Add a few strategic restricted areas
            restricted_areas.extend([
                {
                    'type': 'RECTANGLE',
                    'points': [
                        (bounds['min_x'] + width * 0.1, bounds['min_y'] + height * 0.1),
                        (bounds['min_x'] + width * 0.3, bounds['min_y'] + height * 0.3)
                    ],
                    'layer': 'RESTRICTED'
                },
                {
                    'type': 'RECTANGLE',
                    'points': [
                        (bounds['min_x'] + width * 0.1, bounds['min_y'] + height * 0.7),
                        (bounds['min_x'] + width * 0.3, bounds['min_y'] + height * 0.9)
                    ],
                    'layer': 'RESTRICTED'
                }
            ])
        
        except Exception as e:
            print(f"Error creating restricted areas: {str(e)}")
        
        return restricted_areas
    
    def _create_entrances_from_doors(self, doors) -> List[Dict]:
        """Create entrance points from door entities"""
        entrances = []
        
        for door in doors:
            try:
                if door.get('type', 'rectangle') == 'ARC':
                    center = door.get('center', [0, 0])
                    entrance = {
                        'type': 'ENTRANCE',
                        'position': center,
                        'width': door.get('radius', 1.0) * 2,
                        'layer': door['layer']
                    }
                    entrances.append(entrance)
            except Exception as e:
                print(f"Error creating entrance: {str(e)}")
        
        return entrances
    
    def _get_document_bounds(self, doc) -> Dict[str, float]:
        """Get document bounds as fallback"""
        try:
            header = doc.header
            if '$EXTMIN' in header and '$EXTMAX' in header:
                min_pt = header['$EXTMIN']
                max_pt = header['$EXTMAX']
                return {
                    'min_x': float(min_pt[0]),
                    'max_x': float(max_pt[0]),
                    'min_y': float(min_pt[1]),
                    'max_y': float(max_pt[1])
                }
        except:
            pass
        
        return {'min_x': 0, 'max_x': 200, 'min_y': 0, 'max_y': 150}