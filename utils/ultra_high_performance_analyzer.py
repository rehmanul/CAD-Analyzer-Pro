import time
import os
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

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
    pass # Placeholder since the original code has this function, keeping it for completeness

def _create_analysis_result(walls, doors, windows, openings, filename):
    return {} # Placeholder implementation, replace with actual logic.

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