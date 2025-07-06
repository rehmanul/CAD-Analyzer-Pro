"""
Advanced dimension reading and extraction system for CAD files
Handles text annotations, dimension lines, and measurement extraction
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from shapely.geometry import Point, LineString
import logging

logger = logging.getLogger(__name__)

class DimensionReader:
    """Extract and interpret dimensions from CAD floor plans"""
    
    def __init__(self):
        self.dimension_patterns = [
            r'(\d+[.,]\d+)\s*m',        # 3.85m, 1.50m
            r'(\d+[.,]\d+)\s*mm',       # 1500mm
            r'(\d+[.,]\d+)\s*cm',       # 385cm
            r'(\d+[.,]\d+)',            # 3.85, 1.50 (assume meters)
            r'(\d+)\s*x\s*(\d+)',       # 3x4 room dimensions
            r'(\d+[.,]\d+)\s*×\s*(\d+[.,]\d+)', # 3.85×1.50
        ]
        
        self.room_keywords = [
            'room', 'bedroom', 'bathroom', 'kitchen', 'living', 'lobby',
            'corridor', 'hallway', 'suite', 'apartment', 'closet',
            'chambre', 'salle', 'cuisine', 'salon', 'couloir', 'entrée'
        ]
    
    def extract_dimensions_from_plan(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract all dimensional information from floor plan entities"""
        
        dimensions = {
            'room_dimensions': {},
            'overall_dimensions': {},
            'corridor_widths': [],
            'door_widths': [],
            'text_dimensions': [],
            'dimension_lines': [],
            'scale_info': None
        }
        
        # Process text entities for dimension annotations
        text_entities = [e for e in entities if e.get('type') == 'TEXT' or e.get('type') == 'MTEXT']
        dimensions['text_dimensions'] = self._extract_text_dimensions(text_entities)
        
        # Process dimension lines
        dim_entities = [e for e in entities if e.get('type') == 'DIMENSION']
        dimensions['dimension_lines'] = self._extract_dimension_lines(dim_entities)
        
        # Identify room dimensions by proximity to room labels
        dimensions['room_dimensions'] = self._identify_room_dimensions(
            text_entities, dimensions['text_dimensions']
        )
        
        # Extract corridor and door measurements
        dimensions['corridor_widths'] = self._extract_corridor_widths(dimensions['text_dimensions'])
        dimensions['door_widths'] = self._extract_door_widths(dimensions['text_dimensions'])
        
        # Determine overall plan dimensions
        dimensions['overall_dimensions'] = self._calculate_overall_dimensions(entities)
        
        # Detect scale information
        dimensions['scale_info'] = self._detect_scale_info(text_entities)
        
        return dimensions
    
    def _extract_text_dimensions(self, text_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract dimensional information from text annotations"""
        extracted_dims = []
        
        for entity in text_entities:
            text_content = entity.get('text', '').strip()
            position = entity.get('position', {})
            
            # Try each pattern
            for pattern in self.dimension_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            # Handle dimension pairs like "3x4" or "3.85×1.50"
                            width, height = match
                            extracted_dims.append({
                                'type': 'area_dimension',
                                'width': self._normalize_dimension(width),
                                'height': self._normalize_dimension(height),
                                'position': position,
                                'original_text': text_content,
                                'unit': self._detect_unit(text_content)
                            })
                        else:
                            # Single dimension
                            extracted_dims.append({
                                'type': 'linear_dimension',
                                'value': self._normalize_dimension(match),
                                'position': position,
                                'original_text': text_content,
                                'unit': self._detect_unit(text_content)
                            })
                    break
        
        return extracted_dims
    
    def _extract_dimension_lines(self, dim_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract information from CAD dimension lines"""
        dimension_lines = []
        
        for entity in dim_entities:
            dim_info = {
                'type': 'dimension_line',
                'measurement': entity.get('measurement', 0),
                'start_point': entity.get('start_point', (0, 0)),
                'end_point': entity.get('end_point', (0, 0)),
                'text_position': entity.get('text_position', (0, 0)),
                'dimension_type': entity.get('dimension_type', 'linear')
            }
            
            # Calculate length if not provided
            if dim_info['measurement'] == 0:
                start = np.array(dim_info['start_point'])
                end = np.array(dim_info['end_point'])
                dim_info['measurement'] = np.linalg.norm(end - start)
            
            dimension_lines.append(dim_info)
        
        return dimension_lines
    
    def _identify_room_dimensions(self, text_entities: List[Dict[str, Any]], 
                                 text_dimensions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Identify which dimensions belong to which rooms"""
        room_dimensions = {}
        
        # Find room labels
        room_labels = []
        for entity in text_entities:
            text = entity.get('text', '').lower()
            for keyword in self.room_keywords:
                if keyword in text:
                    room_labels.append({
                        'name': entity.get('text', ''),
                        'position': entity.get('position', {}),
                        'entity': entity
                    })
                    break
        
        # Associate dimensions with nearest room labels
        for room_label in room_labels:
            room_pos = Point(room_label['position'].get('x', 0), 
                           room_label['position'].get('y', 0))
            
            nearby_dimensions = []
            for dim in text_dimensions:
                dim_pos = Point(dim['position'].get('x', 0), 
                              dim['position'].get('y', 0))
                distance = room_pos.distance(dim_pos)
                
                if distance < 5.0:  # Within 5 units (adjust based on scale)
                    nearby_dimensions.append({
                        'dimension': dim,
                        'distance': distance
                    })
            
            # Sort by distance and take closest dimensions
            nearby_dimensions.sort(key=lambda x: x['distance'])
            
            if nearby_dimensions:
                room_dimensions[room_label['name']] = {
                    'position': room_label['position'],
                    'dimensions': [d['dimension'] for d in nearby_dimensions[:3]]  # Take up to 3 closest
                }
        
        return room_dimensions
    
    def _extract_corridor_widths(self, text_dimensions: List[Dict[str, Any]]) -> List[float]:
        """Extract corridor width measurements"""
        corridor_widths = []
        
        for dim in text_dimensions:
            if dim['type'] == 'linear_dimension':
                value = dim['value']
                # Typical corridor widths are 1.2-3.0 meters
                if 1.0 <= value <= 4.0:
                    corridor_widths.append(value)
        
        return corridor_widths
    
    def _extract_door_widths(self, text_dimensions: List[Dict[str, Any]]) -> List[float]:
        """Extract door width measurements"""
        door_widths = []
        
        for dim in text_dimensions:
            if dim['type'] == 'linear_dimension':
                value = dim['value']
                # Typical door widths are 0.6-1.2 meters
                if 0.5 <= value <= 1.5:
                    door_widths.append(value)
        
        return door_widths
    
    def _calculate_overall_dimensions(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall plan dimensions from all entities"""
        if not entities:
            return {'width': 0, 'height': 0, 'area': 0}
        
        # Extract all coordinates
        all_x = []
        all_y = []
        
        for entity in entities:
            coords = entity.get('coordinates', [])
            if coords:
                if isinstance(coords[0], (list, tuple)):
                    # Multiple points
                    for point in coords:
                        if len(point) >= 2:
                            all_x.append(point[0])
                            all_y.append(point[1])
                else:
                    # Single point
                    if len(coords) >= 2:
                        all_x.append(coords[0])
                        all_y.append(coords[1])
        
        if not all_x or not all_y:
            return {'width': 0, 'height': 0, 'area': 0}
        
        width = max(all_x) - min(all_x)
        height = max(all_y) - min(all_y)
        
        return {
            'width': width,
            'height': height,
            'area': width * height,
            'bounds': {
                'min_x': min(all_x),
                'max_x': max(all_x),
                'min_y': min(all_y),
                'max_y': max(all_y)
            }
        }
    
    def _detect_scale_info(self, text_entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect scale information from plan"""
        scale_patterns = [
            r'1:(\d+)',
            r'scale\s*1:(\d+)',
            r'échelle\s*1:(\d+)',
            r'(\d+)\s*mm\s*=\s*1\s*m'
        ]
        
        for entity in text_entities:
            text = entity.get('text', '').lower()
            for pattern in scale_patterns:
                match = re.search(pattern, text)
                if match:
                    scale_value = int(match.group(1))
                    return {
                        'scale_ratio': scale_value,
                        'scale_text': entity.get('text', ''),
                        'position': entity.get('position', {})
                    }
        
        return None
    
    def _normalize_dimension(self, dim_str: str) -> float:
        """Convert dimension string to standardized float value in meters"""
        try:
            # Replace comma with dot for decimal
            dim_str = str(dim_str).replace(',', '.')
            value = float(dim_str)
            
            # Convert to meters based on typical ranges
            if value > 1000:  # Likely millimeters
                return value / 1000
            elif value > 100:  # Likely centimeters
                return value / 100
            else:  # Likely meters
                return value
        except (ValueError, TypeError):
            return 0.0
    
    def _detect_unit(self, text: str) -> str:
        """Detect the unit from text"""
        text_lower = text.lower()
        if 'mm' in text_lower:
            return 'mm'
        elif 'cm' in text_lower:
            return 'cm'
        elif 'm' in text_lower:
            return 'm'
        else:
            return 'm'  # Default to meters
    
    def get_room_areas(self, room_dimensions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate areas for rooms with identified dimensions"""
        room_areas = {}
        
        for room_name, room_data in room_dimensions.items():
            dimensions = room_data.get('dimensions', [])
            
            # Look for area dimensions (width x height)
            area_dims = [d for d in dimensions if d['type'] == 'area_dimension']
            if area_dims:
                dim = area_dims[0]  # Take first area dimension
                area = dim['width'] * dim['height']
                room_areas[room_name] = area
            else:
                # Try to calculate from linear dimensions
                linear_dims = [d for d in dimensions if d['type'] == 'linear_dimension']
                if len(linear_dims) >= 2:
                    # Assume first two are width and height
                    area = linear_dims[0]['value'] * linear_dims[1]['value']
                    room_areas[room_name] = area
        
        return room_areas
    
    def validate_dimensions(self, dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted dimensions for reasonableness"""
        validation_report = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check overall dimensions
        overall = dimensions.get('overall_dimensions', {})
        width = overall.get('width', 0)
        height = overall.get('height', 0)
        
        if width > 1000 or height > 1000:
            validation_report['warnings'].append(
                f"Plan dimensions very large: {width:.1f}m x {height:.1f}m. Check scale."
            )
        
        if width < 1 or height < 1:
            validation_report['errors'].append(
                f"Plan dimensions too small: {width:.1f}m x {height:.1f}m"
            )
            validation_report['valid'] = False
        
        # Check room dimensions
        room_areas = self.get_room_areas(dimensions.get('room_dimensions', {}))
        for room_name, area in room_areas.items():
            if area > 200:  # Very large room
                validation_report['warnings'].append(
                    f"Room '{room_name}' very large: {area:.1f}m²"
                )
            elif area < 2:  # Very small room
                validation_report['warnings'].append(
                    f"Room '{room_name}' very small: {area:.1f}m²"
                )
        
        return validation_report