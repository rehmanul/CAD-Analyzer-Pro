import time
import os
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

class UltraHighPerformanceAnalyzer:
    """Ultra-high performance CAD analyzer for floor plan processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_floor_plan(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Analyze floor plan from file content"""
        try:
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                result = self.analyze_cad_file(temp_file_path)
                return result
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            return self._create_empty_analysis_result(filename, f"Processing error: {str(e)}")

def _create_empty_analysis_result(filename: str, reason: str) -> Dict[str, Any]:
    """Create empty analysis result when no valid CAD data is found"""
    return {
        'filename': filename,
        'processing_time': 0.0,
        'status': 'no_valid_data',
        'reason': reason,
        'walls': [],
        'doors': [],
        'windows': [],
        'rooms': [],
        'openings': [],
        'text_annotations': [],
        'bounds': {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0},
        'scale_factor': 1.0,
        'units': 'meters',
        'element_counts': {'walls': 0, 'doors': 0, 'windows': 0, 'rooms': 0},
        'performance_metrics': {
            'enhancement_level': 'NO_DATA_DETECTED',
            'processing_speed_mbps': 0
        },
        'quality_metrics': {
            'overall_quality_score': 0.0,
            'geometric_accuracy': 0.0,
            'completeness_score': 0.0
        }
    }

def _extract_text_elements(self, doc):
        """Extract text elements from CAD document"""
        pass # Placeholder since the original code has this function, keeping it for completeness

def _create_analysis_result(self, walls, doors, windows, openings, filename):
        """Create analysis result from extracted elements"""
        # Calculate bounds from walls
        bounds = {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100}
        
        if walls:
            all_x = []
            all_y = []
            for wall in walls:
                if 'start' in wall and 'end' in wall:
                    all_x.extend([wall['start'][0], wall['end'][0]])
                    all_y.extend([wall['start'][1], wall['end'][1]])
            
            if all_x and all_y:
                bounds = {
                    "min_x": min(all_x),
                    "max_x": max(all_x),
                    "min_y": min(all_y),
                    "max_y": max(all_y)
                }
        
        return {
            'filename': filename,
            'walls': walls,
            'doors': doors,
            'windows': windows,
            'openings': openings,
            'bounds': bounds,
            'scale': 1.0,
            'units': 'mm',
            'processing_time': 0.1,
            'status': 'success'
        }

    def analyze_cad_file(self, filename: str) -> Dict[str, Any]:

        # Process DXF files with enhanced wall detection
        if filename.lower().endswith('.dxf'):
            try:
                import ezdxf
                doc = ezdxf.readfile(filename)

                walls = []
                doors = []
                windows = []

                # Check both modelspace and paperspace
                spaces = [doc.modelspace()]
                if hasattr(doc, 'paperspace'):
                    spaces.append(doc.paperspace())

                for space in spaces:
                    for entity in space:
                        try:
                            # Enhanced wall detection
                            if entity.dxftype() == 'LINE':
                                start = [entity.dxf.start.x, entity.dxf.start.y]
                                end = [entity.dxf.end.x, entity.dxf.end.y]

                                # Calculate length
                                length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

                                # Accept lines that could be walls (reduced threshold)
                                if length >= 50:  # Reduced from typical 100mm
                                    walls.append({
                                        'type': 'line',
                                        'start': start,
                                        'end': end,
                                        'layer': entity.dxf.layer,
                                        'length': length
                                    })

                            elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                                try:
                                    if hasattr(entity, 'get_points'):
                                        points = list(entity.get_points())
                                        if len(points) >= 2:
                                            # Convert polyline to wall segments
                                            for i in range(len(points) - 1):
                                                start = [points[i][0], points[i][1]]
                                                end = [points[i+1][0], points[i+1][1]]
                                                length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

                                                if length >= 50:
                                                    walls.append({
                                                        'type': 'polyline_segment',
                                                        'start': start,
                                                        'end': end,
                                                        'layer': entity.dxf.layer,
                                                        'length': length
                                                    })
                                except:
                                    continue

                            # Look for door/window blocks
                            elif entity.dxftype() == 'INSERT':
                                insert_point = [entity.dxf.insert.x, entity.dxf.insert.y]
                                block_name = entity.dxf.name.lower()

                                if any(door_keyword in block_name for door_keyword in ['door', 'porte', 'entry']):
                                    doors.append({
                                        'type': 'door',
                                        'position': insert_point,
                                        'block_name': block_name,
                                        'layer': entity.dxf.layer
                                    })
                                elif any(window_keyword in block_name for window_keyword in ['window', 'fenetre', 'wind']):
                                    windows.append({
                                        'type': 'window',
                                        'position': insert_point,
                                        'block_name': block_name,
                                        'layer': entity.dxf.layer
                                    })
                        except:
                            continue

                print(f"Enhanced DXF processing found: {len(walls)} walls, {len(doors)} doors, {len(windows)} windows")
                return self._create_analysis_result(walls, doors, windows, [], filename)

            except Exception as e:
                print(f"DXF processing error: {e}")
                return self._create_empty_analysis_result(filename, f"DXF parsing failed: {str(e)}")
        else:
            return self._create_empty_analysis_result(filename, "Unsupported file type")