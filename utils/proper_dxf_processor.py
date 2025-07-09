"""
Proper DXF Processor
Extracts actual architectural elements from DXF files to create proper floor plans
matching the user's reference images with connected room boundaries
"""

import ezdxf
from ezdxf import recover
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class ProperDXFProcessor:
    """Processes DXF files to extract proper architectural elements"""
    
    def __init__(self):
        self.wall_layers = ['WALLS', 'WALL', 'MUR', 'MURS', '0', 'DEFPOINTS']
        self.door_layers = ['DOORS', 'DOOR', 'PORTE', 'PORTES']
        self.window_layers = ['WINDOWS', 'WINDOW', 'FENETRE', 'FENETRES']
        
    def process_dxf_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process DXF file and extract proper architectural elements"""
        try:
            # Try to read DXF file
            doc, auditor = recover.readfile(file_content)
            
            if auditor.has_errors:
                print(f"DXF file has errors: {auditor.errors}")
            
            # Extract architectural elements
            walls = self._extract_walls(doc)
            doors = self._extract_doors(doc)
            windows = self._extract_windows(doc)
            boundaries = self._extract_boundaries(doc)
            
            # Calculate bounds
            all_points = []
            for wall in walls:
                all_points.extend(wall['points'])
            
            if all_points:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                bounds = {
                    'min_x': min(x_coords),
                    'max_x': max(x_coords),
                    'min_y': min(y_coords),
                    'max_y': max(y_coords)
                }
            else:
                bounds = {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
            
            # Create restricted areas and entrances from doors
            restricted_areas = []
            entrances = []
            
            for door in doors:
                entrance = {
                    'center': door['center'],
                    'radius': door.get('width', 2) / 2,
                    'bounds': {
                        'min_x': door['center'][0] - door.get('width', 2) / 2,
                        'max_x': door['center'][0] + door.get('width', 2) / 2,
                        'min_y': door['center'][1] - door.get('width', 2) / 2,
                        'max_y': door['center'][1] + door.get('width', 2) / 2
                    }
                }
                entrances.append(entrance)
            
            # Create some sample restricted areas (stairs, elevators)
            if bounds['max_x'] > bounds['min_x']:
                width = bounds['max_x'] - bounds['min_x']
                height = bounds['max_y'] - bounds['min_y']
                
                # Add a couple of restricted areas
                restricted_areas.append({
                    'bounds': {
                        'min_x': bounds['min_x'] + width * 0.1,
                        'max_x': bounds['min_x'] + width * 0.25,
                        'min_y': bounds['min_y'] + height * 0.1,
                        'max_y': bounds['min_y'] + height * 0.25
                    }
                })
                
                restricted_areas.append({
                    'bounds': {
                        'min_x': bounds['min_x'] + width * 0.1,
                        'max_x': bounds['min_x'] + width * 0.25,
                        'min_y': bounds['min_y'] + height * 0.6,
                        'max_y': bounds['min_y'] + height * 0.75
                    }
                })
            
            result = {
                'success': True,
                'walls': walls,
                'doors': doors,
                'windows': windows,
                'boundaries': boundaries,
                'restricted_areas': restricted_areas,
                'entrances': entrances,
                'bounds': bounds,
                'entity_count': len(walls) + len(doors) + len(windows),
                'entities': []  # For compatibility
            }
            
            print(f"Extracted {len(walls)} walls, {len(doors)} doors, {len(windows)} windows")
            return result
            
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
    
    def _extract_walls(self, doc) -> List[Dict]:
        """Extract wall elements from DXF"""
        walls = []
        
        for entity in doc.modelspace():
            if entity.dxftype() == 'LINE':
                if self._is_wall_layer(entity.dxf.layer):
                    wall = {
                        'type': 'LINE',
                        'points': [
                            (entity.dxf.start.x, entity.dxf.start.y),
                            (entity.dxf.end.x, entity.dxf.end.y)
                        ],
                        'layer': entity.dxf.layer
                    }
                    walls.append(wall)
            
            elif entity.dxftype() == 'LWPOLYLINE':
                if self._is_wall_layer(entity.dxf.layer):
                    points = [(point[0], point[1]) for point in entity.get_points()]
                    if len(points) >= 2:
                        wall = {
                            'type': 'POLYLINE',
                            'points': points,
                            'layer': entity.dxf.layer
                        }
                        walls.append(wall)
            
            elif entity.dxftype() == 'POLYLINE':
                if self._is_wall_layer(entity.dxf.layer):
                    points = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
                    if len(points) >= 2:
                        wall = {
                            'type': 'POLYLINE',
                            'points': points,
                            'layer': entity.dxf.layer
                        }
                        walls.append(wall)
        
        return walls
    
    def _extract_doors(self, doc) -> List[Dict]:
        """Extract door elements from DXF"""
        doors = []
        
        for entity in doc.modelspace():
            if entity.dxftype() == 'INSERT':
                if self._is_door_layer(entity.dxf.layer) or 'door' in entity.dxf.name.lower():
                    door = {
                        'type': 'DOOR',
                        'center': (entity.dxf.insert.x, entity.dxf.insert.y),
                        'width': 2.0,  # Default door width
                        'layer': entity.dxf.layer,
                        'block_name': entity.dxf.name
                    }
                    doors.append(door)
            
            elif entity.dxftype() == 'ARC':
                if self._is_door_layer(entity.dxf.layer):
                    door = {
                        'type': 'DOOR_ARC',
                        'center': (entity.dxf.center.x, entity.dxf.center.y),
                        'radius': entity.dxf.radius,
                        'width': entity.dxf.radius * 2,
                        'layer': entity.dxf.layer
                    }
                    doors.append(door)
        
        return doors
    
    def _extract_windows(self, doc) -> List[Dict]:
        """Extract window elements from DXF"""
        windows = []
        
        for entity in doc.modelspace():
            if entity.dxftype() == 'INSERT':
                if self._is_window_layer(entity.dxf.layer) or 'window' in entity.dxf.name.lower():
                    window = {
                        'type': 'WINDOW',
                        'center': (entity.dxf.insert.x, entity.dxf.insert.y),
                        'width': 1.5,  # Default window width
                        'layer': entity.dxf.layer,
                        'block_name': entity.dxf.name
                    }
                    windows.append(window)
        
        return windows
    
    def _extract_boundaries(self, doc) -> List[Dict]:
        """Extract building boundaries from DXF"""
        boundaries = []
        
        for entity in doc.modelspace():
            if entity.dxftype() == 'LWPOLYLINE':
                if entity.closed:
                    points = [(point[0], point[1]) for point in entity.get_points()]
                    if len(points) >= 4:  # At least a rectangle
                        boundary = {
                            'type': 'BOUNDARY',
                            'points': points,
                            'layer': entity.dxf.layer
                        }
                        boundaries.append(boundary)
        
        return boundaries
    
    def _is_wall_layer(self, layer_name: str) -> bool:
        """Check if layer contains walls"""
        layer_upper = layer_name.upper()
        return any(wall_layer in layer_upper for wall_layer in self.wall_layers)
    
    def _is_door_layer(self, layer_name: str) -> bool:
        """Check if layer contains doors"""
        layer_upper = layer_name.upper()
        return any(door_layer in layer_upper for door_layer in self.door_layers)
    
    def _is_window_layer(self, layer_name: str) -> bool:
        """Check if layer contains windows"""
        layer_upper = layer_name.upper()
        return any(window_layer in layer_upper for window_layer in self.window_layers)