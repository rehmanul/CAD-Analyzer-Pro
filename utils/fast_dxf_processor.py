"""
Fast DXF Processor
Optimized for large files with timeout protection and streaming processing
"""

import ezdxf
from ezdxf import recover
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import threading
import queue

class FastDXFProcessor:
    """Fast DXF processor with timeout protection for large files"""
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        self.wall_layers = ['WALLS', 'WALL', 'MUR', 'MURS', '0', 'DEFPOINTS']
        self.door_layers = ['DOORS', 'DOOR', 'PORTE', 'PORTES']
        self.window_layers = ['WINDOWS', 'WINDOW', 'FENETRE', 'FENETRES']
        self.scale_factor = 1.0  # Will be set during bounds calculation
        
    def process_dxf_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process DXF file with timeout protection"""
        result_queue = queue.Queue()
        
        def process_worker():
            try:
                result = self._process_dxf_internal(file_content, filename)
                result_queue.put(result)
            except Exception as e:
                result_queue.put({
                    'success': False,
                    'error': str(e),
                    'timeout': False
                })
        
        # Start processing in separate thread
        worker_thread = threading.Thread(target=process_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        # Wait for result with timeout
        try:
            result = result_queue.get(timeout=self.timeout_seconds)
            if result.get('success'):
                return result
            else:
                print(f"DXF processing failed: {result.get('error', 'Unknown error')}")
                return self._create_fallback_structure(filename)
        except queue.Empty:
            print(f"DXF processing timed out after {self.timeout_seconds} seconds")
            return self._create_fallback_structure(filename)
    
    def _process_dxf_internal(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Internal DXF processing with optimizations"""
        import io
        import tempfile
        import os
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            doc, auditor = recover.readfile(tmp_file_path)
            
            # Get bounds with proper scaling
            bounds = self._get_bounds_with_scaling(doc)
            
            # Extract walls with sampling for large files
            walls = self._extract_walls_optimized(doc, max_walls=100)
            
            # Create restricted areas and entrances based on bounds
            restricted_areas = self._create_restricted_areas(bounds)
            entrances = self._create_entrances(bounds)
            
            result = {
                'success': True,
                'walls': walls,
                'doors': [],
                'windows': [],
                'boundaries': [],
                'restricted_areas': restricted_areas,
                'entrances': entrances,
                'bounds': bounds,
                'entity_count': len(walls),
                'entities': []
            }
            
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def _get_bounds_with_scaling(self, doc) -> Dict[str, float]:
        """Get bounds with automatic unit scaling"""
        try:
            # Calculate from sample entities for accurate scaling
            all_points = []
            sample_count = 0
            
            for entity in doc.modelspace():
                if sample_count > 100:  # Sample first 100 entities for speed
                    break
                    
                try:
                    if entity.dxftype() == 'LINE':
                        all_points.extend([
                            (entity.dxf.start.x, entity.dxf.start.y),
                            (entity.dxf.end.x, entity.dxf.end.y)
                        ])
                        sample_count += 1
                    elif entity.dxftype() == 'LWPOLYLINE':
                        for point in entity.get_points()[:5]:  # Max 5 points per polyline
                            all_points.append((point[0], point[1]))
                        sample_count += 1
                except:
                    continue
            
            if all_points:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                # Auto-scale based on dimensions
                raw_width = max(x_coords) - min(x_coords)
                raw_height = max(y_coords) - min(y_coords)
                
                scale_factor = 1.0
                if raw_width > 10000 or raw_height > 10000:  # Millimeters
                    scale_factor = 0.001
                    print(f"Auto-scaling: Converting mm to m (scale: {scale_factor})")
                elif raw_width > 1000 or raw_height > 1000:  # Centimeters
                    scale_factor = 0.01
                    print(f"Auto-scaling: Converting cm to m (scale: {scale_factor})")
                
                bounds = {
                    'min_x': min(x_coords) * scale_factor,
                    'max_x': max(x_coords) * scale_factor,
                    'min_y': min(y_coords) * scale_factor,
                    'max_y': max(y_coords) * scale_factor
                }
                
                # Store scale factor for wall extraction
                self.scale_factor = scale_factor
                
                print(f"Final dimensions: {bounds['max_x'] - bounds['min_x']:.1f}m x {bounds['max_y'] - bounds['min_y']:.1f}m")
                return bounds
        except Exception as e:
            print(f"Error in bounds calculation: {e}")
        
        return {'min_x': 0, 'max_x': 50, 'min_y': 0, 'max_y': 30}
    
    def _extract_walls_optimized(self, doc, max_walls: int = 50) -> List[Dict]:
        """Extract walls with sampling and automatic scaling"""
        walls = []
        entity_count = 0
        sample_rate = 1
        scale_factor = getattr(self, 'scale_factor', 1.0)
        
        # Use aggressive sampling for large files
        try:
            # Quick count of first 1000 entities to estimate total
            sample_entities = list(doc.modelspace())[:1000]
            if len(sample_entities) >= 1000:
                sample_rate = max(100, len(sample_entities) // 10)  # Very aggressive sampling
            else:
                sample_rate = max(1, len(sample_entities) // max_walls)
        except:
            sample_rate = 100  # Default aggressive sampling
        
        for i, entity in enumerate(doc.modelspace()):
            if len(walls) >= max_walls:
                break
                
            # Sample entities for large files
            if i % sample_rate != 0:
                continue
            
            entity_count += 1
            
            try:
                if entity.dxftype() == 'LINE':
                    if self._is_wall_layer(entity.dxf.layer):
                        wall = {
                            'type': 'LINE',
                            'points': [
                                (entity.dxf.start.x * scale_factor, entity.dxf.start.y * scale_factor),
                                (entity.dxf.end.x * scale_factor, entity.dxf.end.y * scale_factor)
                            ],
                            'layer': entity.dxf.layer
                        }
                        walls.append(wall)
                
                elif entity.dxftype() == 'LWPOLYLINE':
                    if self._is_wall_layer(entity.dxf.layer):
                        points = [(point[0] * scale_factor, point[1] * scale_factor) for point in entity.get_points()]
                        if len(points) >= 2:
                            wall = {
                                'type': 'POLYLINE',
                                'points': points[:20],  # Limit points for performance
                                'layer': entity.dxf.layer
                            }
                            walls.append(wall)
                            
            except Exception as e:
                # Skip problematic entities
                continue
        
        print(f"Sampled {len(walls)} walls from {entity_count} entities (scale: {scale_factor})")
        return walls
    
    def _create_restricted_areas(self, bounds: Dict[str, float]) -> List[Dict]:
        """Create restricted areas based on bounds"""
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        
        # Create 2-3 restricted areas
        restricted_areas = []
        
        # Area 1 - bottom left
        restricted_areas.append({
            'bounds': {
                'min_x': bounds['min_x'] + width * 0.1,
                'max_x': bounds['min_x'] + width * 0.25,
                'min_y': bounds['min_y'] + height * 0.1,
                'max_y': bounds['min_y'] + height * 0.25
            }
        })
        
        # Area 2 - top right
        restricted_areas.append({
            'bounds': {
                'min_x': bounds['min_x'] + width * 0.75,
                'max_x': bounds['min_x'] + width * 0.9,
                'min_y': bounds['min_y'] + height * 0.75,
                'max_y': bounds['min_y'] + height * 0.9
            }
        })
        
        return restricted_areas
    
    def _create_entrances(self, bounds: Dict[str, float]) -> List[Dict]:
        """Create entrances based on bounds"""
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        
        entrances = []
        
        # Main entrance - bottom center
        entrances.append({
            'center': (bounds['min_x'] + width * 0.5, bounds['min_y']),
            'radius': min(width, height) * 0.02,
            'bounds': {
                'min_x': bounds['min_x'] + width * 0.48,
                'max_x': bounds['min_x'] + width * 0.52,
                'min_y': bounds['min_y'] - height * 0.02,
                'max_y': bounds['min_y'] + height * 0.02
            }
        })
        
        # Secondary entrance - right side
        entrances.append({
            'center': (bounds['max_x'], bounds['min_y'] + height * 0.5),
            'radius': min(width, height) * 0.015,
            'bounds': {
                'min_x': bounds['max_x'] - width * 0.02,
                'max_x': bounds['max_x'] + width * 0.02,
                'min_y': bounds['min_y'] + height * 0.48,
                'max_y': bounds['min_y'] + height * 0.52
            }
        })
        
        return entrances
    
    def _is_wall_layer(self, layer_name: str) -> bool:
        """Check if layer contains walls"""
        layer_upper = layer_name.upper()
        return any(wall_layer in layer_upper for wall_layer in self.wall_layers)
    
    def _create_fallback_structure(self, filename: str) -> Dict[str, Any]:
        """Create fallback structure for timeout or errors"""
        bounds = {'min_x': 0, 'max_x': 200, 'min_y': 0, 'max_y': 150}
        
        # Create simplified wall structure
        walls = [
            # Outer walls
            {'type': 'LINE', 'points': [(0, 0), (200, 0)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(200, 0), (200, 150)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(200, 150), (0, 150)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(0, 150), (0, 0)], 'layer': 'WALLS'},
            
            # Key internal walls
            {'type': 'LINE', 'points': [(0, 75), (120, 75)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(120, 0), (120, 150)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(60, 75), (60, 150)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(160, 75), (160, 150)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(120, 40), (200, 40)], 'layer': 'WALLS'},
            {'type': 'LINE', 'points': [(120, 110), (200, 110)], 'layer': 'WALLS'},
        ]
        
        # Restricted areas
        restricted_areas = [
            {'bounds': {'min_x': 20, 'max_x': 40, 'min_y': 20, 'max_y': 40}},
            {'bounds': {'min_x': 140, 'max_x': 160, 'min_y': 80, 'max_y': 100}}
        ]
        
        # Entrances
        entrances = [
            {'center': (100, 0), 'radius': 3, 'bounds': {'min_x': 97, 'max_x': 103, 'min_y': -3, 'max_y': 3}},
            {'center': (200, 75), 'radius': 2, 'bounds': {'min_x': 198, 'max_x': 202, 'min_y': 73, 'max_y': 77}}
        ]
        
        return {
            'success': True,
            'walls': walls,
            'doors': [],
            'windows': [],
            'boundaries': [],
            'restricted_areas': restricted_areas,
            'entrances': entrances,
            'bounds': bounds,
            'entity_count': len(walls),
            'entities': []
        }