"""
Enhanced CAD Parser for Phase 1 Implementation
Multi-format support with advanced floor plan extraction and geometric element recognition
"""

import ezdxf
import fitz  # PyMuPDF
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import logging
from concurrent.futures import ThreadPoolExecutor
import re
from dataclasses import dataclass
from enum import Enum

class ElementType(Enum):
    WALL = "wall"
    DOOR = "door"
    WINDOW = "window"
    ROOM_BOUNDARY = "room_boundary"
    DIMENSION = "dimension"
    TEXT = "text"
    FURNITURE = "furniture"
    STRUCTURAL = "structural"

@dataclass
class CADElement:
    """Represents a geometric element extracted from CAD"""
    element_type: ElementType
    geometry: Any  # Shapely geometry object
    properties: Dict[str, Any]
    layer: str = ""
    color: str = ""
    line_weight: float = 1.0
    area: float = 0.0
    perimeter: float = 0.0

@dataclass
class FloorPlanData:
    """Complete floor plan data structure"""
    walls: List[CADElement]
    doors: List[CADElement]
    windows: List[CADElement]
    rooms: List[CADElement]
    restricted_areas: List[CADElement]
    entrances: List[CADElement]
    dimensions: List[CADElement]
    text_annotations: List[CADElement]
    scale_factor: float = 1.0
    units: str = "mm"
    bounds: Dict[str, float] = None
    metadata: Dict[str, Any] = None

class EnhancedCADParser:
    """
    Enhanced CAD Parser for Phase 1 - Advanced floor plan extraction
    Handles DXF, DWG, PDF files with intelligent geometric element recognition
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.dxf', '.dwg', '.pdf', '.png', '.jpg', '.jpeg']
        
        # CAD layer patterns for element classification
        self.layer_patterns = {
            ElementType.WALL: ['wall', 'mur', 'partition', 'structural', 'cloison'],
            ElementType.DOOR: ['door', 'porte', 'opening', 'ouverture'],
            ElementType.WINDOW: ['window', 'fenetre', 'vitrage', 'glazing'],
            ElementType.DIMENSION: ['dim', 'cote', 'measure', 'annotation'],
            ElementType.TEXT: ['text', 'texte', 'label', 'note'],
            ElementType.FURNITURE: ['furniture', 'mobilier', 'equipment', 'equipement']
        }
        
        # Geometric thresholds for element classification
        self.wall_thickness_range = (50, 300)  # mm
        self.door_width_range = (600, 1200)  # mm
        self.window_width_range = (400, 2000)  # mm
        self.min_room_area = 2.0  # m²

    def parse_cad_file(self, file_path: str) -> FloorPlanData:
        """
        Main entry point for parsing CAD files
        Returns structured floor plan data
        """
        try:
            file_ext = file_path.lower().split('.')[-1]
            
            if file_ext == 'dxf':
                return self._parse_dxf_file(file_path)
            elif file_ext in ['pdf']:
                return self._parse_pdf_file(file_path)
            elif file_ext in ['png', 'jpg', 'jpeg']:
                return self._parse_image_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.logger.error(f"Error parsing CAD file {file_path}: {str(e)}")
            return self._create_empty_floor_plan()

    def _parse_dxf_file(self, file_path: str) -> FloorPlanData:
        """Enhanced DXF parsing with layer-aware processing"""
        try:
            doc = ezdxf.readfile(file_path)
            modelspace = doc.modelspace()
            
            # Extract all geometric entities by layer
            raw_elements = self._extract_dxf_entities(modelspace)
            
            # Classify elements by type
            classified_elements = self._classify_elements(raw_elements)
            
            # Post-process and clean geometry
            floor_plan = self._post_process_elements(classified_elements)
            
            # Calculate scale and units
            floor_plan.scale_factor = self._detect_scale(floor_plan)
            floor_plan.units = self._detect_units(doc)
            
            # Calculate bounds
            floor_plan.bounds = self._calculate_bounds(floor_plan)
            
            return floor_plan
            
        except Exception as e:
            self.logger.error(f"Error parsing DXF file: {str(e)}")
            return self._create_empty_floor_plan()

    def _extract_dxf_entities(self, modelspace) -> List[Dict[str, Any]]:
        """Extract all entities from DXF modelspace with properties"""
        raw_elements = []
        
        for entity in modelspace:
            element_data = {
                'entity': entity,
                'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'default',
                'color': getattr(entity.dxf, 'color', 256),
                'linetype': getattr(entity.dxf, 'linetype', 'CONTINUOUS'),
                'geometry': None,
                'properties': {}
            }
            
            # Extract geometry based on entity type
            if entity.dxftype() == 'LINE':
                start = np.array([entity.dxf.start.x, entity.dxf.start.y])
                end = np.array([entity.dxf.end.x, entity.dxf.end.y])
                element_data['geometry'] = LineString([start, end])
                element_data['properties']['length'] = np.linalg.norm(end - start)
                
            elif entity.dxftype() == 'POLYLINE':
                points = [(vertex.dxf.location.x, vertex.dxf.location.y) 
                         for vertex in entity.vertices]
                if len(points) >= 2:
                    if entity.is_closed or (len(points) > 2 and points[0] == points[-1]):
                        element_data['geometry'] = Polygon(points)
                    else:
                        element_data['geometry'] = LineString(points)
                        
            elif entity.dxftype() == 'LWPOLYLINE':
                points = [(point[0], point[1]) for point in entity.get_points()]
                if len(points) >= 2:
                    if entity.closed or (len(points) > 2 and points[0] == points[-1]):
                        element_data['geometry'] = Polygon(points)
                    else:
                        element_data['geometry'] = LineString(points)
                        
            elif entity.dxftype() == 'CIRCLE':
                center = (entity.dxf.center.x, entity.dxf.center.y)
                radius = entity.dxf.radius
                element_data['geometry'] = Point(center).buffer(radius)
                element_data['properties']['radius'] = radius
                
            elif entity.dxftype() == 'ARC':
                # Convert arc to line string approximation
                center = np.array([entity.dxf.center.x, entity.dxf.center.y])
                radius = entity.dxf.radius
                start_angle = np.radians(entity.dxf.start_angle)
                end_angle = np.radians(entity.dxf.end_angle)
                
                # Create arc points
                angles = np.linspace(start_angle, end_angle, 20)
                points = [(center[0] + radius * np.cos(a), 
                          center[1] + radius * np.sin(a)) for a in angles]
                element_data['geometry'] = LineString(points)
                element_data['properties']['is_arc'] = True
                element_data['properties']['radius'] = radius
                
            elif entity.dxftype() == 'TEXT':
                pos = (entity.dxf.insert.x, entity.dxf.insert.y)
                element_data['geometry'] = Point(pos)
                element_data['properties']['text'] = entity.dxf.text
                element_data['properties']['height'] = entity.dxf.height
                
            if element_data['geometry'] is not None:
                raw_elements.append(element_data)
                
        return raw_elements

    def _classify_elements(self, raw_elements: List[Dict[str, Any]]) -> Dict[ElementType, List[CADElement]]:
        """Classify raw elements into semantic types"""
        classified = {element_type: [] for element_type in ElementType}
        
        for element_data in raw_elements:
            layer_name = element_data['layer'].lower()
            geometry = element_data['geometry']
            
            # Classify by layer name patterns
            element_type = self._classify_by_layer(layer_name)
            
            # If no layer match, classify by geometry
            if element_type is None:
                element_type = self._classify_by_geometry(geometry, element_data)
            
            if element_type:
                cad_element = CADElement(
                    element_type=element_type,
                    geometry=geometry,
                    properties=element_data['properties'],
                    layer=element_data['layer'],
                    color=str(element_data['color'])
                )
                classified[element_type].append(cad_element)
                
        return classified

    def _classify_by_layer(self, layer_name: str) -> Optional[ElementType]:
        """Classify element by layer name patterns"""
        for element_type, patterns in self.layer_patterns.items():
            if any(pattern in layer_name for pattern in patterns):
                return element_type
        return None

    def _classify_by_geometry(self, geometry, element_data: Dict[str, Any]) -> Optional[ElementType]:
        """Classify element by geometric properties"""
        if isinstance(geometry, LineString):
            length = geometry.length
            
            # Check if it's a wall (long lines with potential thickness)
            if length > 1000:  # > 1m
                return ElementType.WALL
            elif 600 <= length <= 1200:  # Door width range
                return ElementType.DOOR
            elif 400 <= length <= 2000:  # Window width range
                return ElementType.WINDOW
                
        elif isinstance(geometry, Polygon):
            area = geometry.area
            
            # Large polygons could be rooms
            if area > 2000000:  # > 2m² in mm²
                return ElementType.ROOM_BOUNDARY
            elif area < 500000:  # Small polygons could be furniture
                return ElementType.FURNITURE
                
        elif isinstance(geometry, Point):
            # Points with text are annotations
            if 'text' in element_data['properties']:
                return ElementType.TEXT
                
        return None

    def _post_process_elements(self, classified_elements: Dict[ElementType, List[CADElement]]) -> FloorPlanData:
        """Post-process and clean classified elements"""
        
        # Process walls - connect adjacent walls and detect thickness
        walls = self._process_walls(classified_elements[ElementType.WALL])
        
        # Process rooms - create room boundaries from walls
        rooms = self._detect_rooms(walls)
        
        # Process doors and windows - ensure they're on walls
        doors = self._process_openings(classified_elements[ElementType.DOOR], walls, 'door')
        windows = self._process_openings(classified_elements[ElementType.WINDOW], walls, 'window')
        
        # Detect restricted areas and entrances from geometry and annotations
        restricted_areas, entrances = self._detect_special_areas(rooms, classified_elements)
        
        return FloorPlanData(
            walls=walls,
            doors=doors,
            windows=windows,
            rooms=rooms,
            restricted_areas=restricted_areas,
            entrances=entrances,
            dimensions=classified_elements[ElementType.DIMENSION],
            text_annotations=classified_elements[ElementType.TEXT]
        )

    def _process_walls(self, wall_elements: List[CADElement]) -> List[CADElement]:
        """Process and connect wall elements"""
        processed_walls = []
        
        for wall in wall_elements:
            if isinstance(wall.geometry, LineString):
                # Calculate wall properties
                length = wall.geometry.length
                
                # Estimate wall thickness from nearby parallel lines
                thickness = self._estimate_wall_thickness(wall, wall_elements)
                
                wall.properties.update({
                    'length': length,
                    'thickness': thickness,
                    'is_load_bearing': thickness > 150  # Assume thick walls are load-bearing
                })
                
                processed_walls.append(wall)
                
        return processed_walls

    def _estimate_wall_thickness(self, wall: CADElement, all_walls: List[CADElement]) -> float:
        """Estimate wall thickness by finding parallel walls"""
        if not isinstance(wall.geometry, LineString):
            return 100  # Default thickness
            
        wall_coords = list(wall.geometry.coords)
        if len(wall_coords) < 2:
            return 100
            
        # Find parallel walls within reasonable distance
        for other_wall in all_walls:
            if other_wall == wall or not isinstance(other_wall.geometry, LineString):
                continue
                
            # Check if walls are parallel and close
            distance = wall.geometry.distance(other_wall.geometry)
            if 50 <= distance <= 300:  # Reasonable wall thickness range
                # Check if they're roughly parallel
                if self._are_lines_parallel(wall.geometry, other_wall.geometry, tolerance=15):
                    return distance
                    
        return 100  # Default thickness

    def _are_lines_parallel(self, line1: LineString, line2: LineString, tolerance: float = 10) -> bool:
        """Check if two lines are parallel within tolerance (degrees)"""
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)
        
        if len(coords1) < 2 or len(coords2) < 2:
            return False
            
        # Calculate angles
        v1 = np.array(coords1[-1]) - np.array(coords1[0])
        v2 = np.array(coords2[-1]) - np.array(coords2[0])
        
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        
        angle_diff = abs(np.degrees(angle1 - angle2))
        angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wraparound
        
        return angle_diff <= tolerance

    def _detect_rooms(self, walls: List[CADElement]) -> List[CADElement]:
        """Detect room boundaries from wall structure"""
        # This is a simplified implementation
        # In a full implementation, you would use polygon detection algorithms
        
        rooms = []
        
        # Create a simple bounding approach for now
        if walls:
            # Get all wall geometries
            wall_geoms = [wall.geometry for wall in walls if isinstance(wall.geometry, LineString)]
            
            if wall_geoms:
                # Create a union of all walls
                try:
                    wall_union = unary_union(wall_geoms)
                    bounds = wall_union.bounds
                    
                    # Create a simple rectangular room for demonstration
                    room_polygon = Polygon([
                        (bounds[0], bounds[1]),
                        (bounds[2], bounds[1]),
                        (bounds[2], bounds[3]),
                        (bounds[0], bounds[3])
                    ])
                    
                    room_element = CADElement(
                        element_type=ElementType.ROOM_BOUNDARY,
                        geometry=room_polygon,
                        properties={
                            'area': room_polygon.area,
                            'perimeter': room_polygon.length,
                            'room_id': 'main_room'
                        }
                    )
                    rooms.append(room_element)
                    
                except Exception as e:
                    self.logger.warning(f"Error detecting rooms: {str(e)}")
                    
        return rooms

    def _process_openings(self, opening_elements: List[CADElement], walls: List[CADElement], opening_type: str) -> List[CADElement]:
        """Process doors and windows, ensuring they're associated with walls"""
        processed_openings = []
        
        for opening in opening_elements:
            # Find the nearest wall
            nearest_wall = None
            min_distance = float('inf')
            
            for wall in walls:
                distance = opening.geometry.distance(wall.geometry)
                if distance < min_distance:
                    min_distance = distance
                    nearest_wall = wall
                    
            if nearest_wall and min_distance < 100:  # Within 10cm of a wall
                opening.properties.update({
                    'associated_wall': nearest_wall,
                    'distance_to_wall': min_distance,
                    'opening_type': opening_type
                })
                processed_openings.append(opening)
                
        return processed_openings

    def _detect_special_areas(self, rooms: List[CADElement], classified_elements: Dict[ElementType, List[CADElement]]) -> Tuple[List[CADElement], List[CADElement]]:
        """Detect restricted areas and entrances from annotations and geometry"""
        restricted_areas = []
        entrances = []
        
        # Look for text annotations that indicate special areas
        text_elements = classified_elements[ElementType.TEXT]
        
        for text_elem in text_elements:
            text_content = text_elem.properties.get('text', '').lower()
            
            if any(keyword in text_content for keyword in ['no entree', 'restricted', 'interdit']):
                # Create a restricted area around the text
                center = text_elem.geometry
                restricted_area = center.buffer(1000)  # 1m radius
                
                restricted_element = CADElement(
                    element_type=ElementType.ROOM_BOUNDARY,
                    geometry=restricted_area,
                    properties={
                        'area_type': 'restricted',
                        'description': text_content
                    }
                )
                restricted_areas.append(restricted_element)
                
            elif any(keyword in text_content for keyword in ['entree', 'sortie', 'entrance', 'exit']):
                # Create an entrance area
                center = text_elem.geometry
                entrance_area = center.buffer(800)  # 0.8m radius
                
                entrance_element = CADElement(
                    element_type=ElementType.ROOM_BOUNDARY,
                    geometry=entrance_area,
                    properties={
                        'area_type': 'entrance',
                        'description': text_content
                    }
                )
                entrances.append(entrance_element)
                
        return restricted_areas, entrances

    def _parse_pdf_file(self, file_path: str) -> FloorPlanData:
        """Parse PDF files for floor plan extraction"""
        try:
            doc = fitz.open(file_path)
            
            # Find the page with the floor plan (usually the largest drawing)
            best_page = None
            max_drawing_area = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                drawings = page.get_drawings()
                
                total_area = sum(self._calculate_drawing_area(drawing) for drawing in drawings)
                if total_area > max_drawing_area:
                    max_drawing_area = total_area
                    best_page = page
                    
            if best_page:
                # Extract geometric elements from the best page
                return self._extract_pdf_elements(best_page)
            else:
                return self._create_empty_floor_plan()
                
        except Exception as e:
            self.logger.error(f"Error parsing PDF file: {str(e)}")
            return self._create_empty_floor_plan()

    def _extract_pdf_elements(self, page) -> FloorPlanData:
        """Extract geometric elements from PDF page"""
        walls = []
        
        # Get all drawings (vector graphics)
        drawings = page.get_drawings()
        
        for drawing in drawings:
            for item in drawing.get('items', []):
                if item[0] == 'l':  # Line
                    start_point = item[1]
                    end_point = item[2]
                    
                    line_geom = LineString([start_point, end_point])
                    length = line_geom.length
                    
                    # Classify as wall if it's long enough
                    if length > 50:  # Minimum line length for walls
                        wall_element = CADElement(
                            element_type=ElementType.WALL,
                            geometry=line_geom,
                            properties={'length': length, 'thickness': 100}
                        )
                        walls.append(wall_element)
                        
        # Create basic floor plan structure
        return FloorPlanData(
            walls=walls,
            doors=[],
            windows=[],
            rooms=[],
            restricted_areas=[],
            entrances=[],
            dimensions=[],
            text_annotations=[]
        )

    def _parse_image_file(self, file_path: str) -> FloorPlanData:
        """Parse image files using computer vision"""
        try:
            # Load image
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect lines using HoughLines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            walls = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    line_geom = LineString([(x1, y1), (x2, y2)])
                    
                    wall_element = CADElement(
                        element_type=ElementType.WALL,
                        geometry=line_geom,
                        properties={'length': line_geom.length, 'thickness': 100}
                    )
                    walls.append(wall_element)
                    
            return FloorPlanData(
                walls=walls,
                doors=[],
                windows=[],
                rooms=[],
                restricted_areas=[],
                entrances=[],
                dimensions=[],
                text_annotations=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing image file: {str(e)}")
            return self._create_empty_floor_plan()

    def _calculate_drawing_area(self, drawing: Dict) -> float:
        """Calculate the area covered by a drawing"""
        if 'rect' in drawing:
            rect = drawing['rect']
            return (rect[2] - rect[0]) * (rect[3] - rect[1])
        return 0

    def _detect_scale(self, floor_plan: FloorPlanData) -> float:
        """Detect scale factor from dimension annotations"""
        scale_factor = 1.0
        
        # Look for dimension annotations to determine scale
        for dim in floor_plan.dimensions:
            text = dim.properties.get('text', '')
            
            # Look for dimension patterns like "5000" or "5.0m"
            dimension_match = re.search(r'(\d+\.?\d*)\s*(mm|cm|m)?', text)
            if dimension_match:
                value = float(dimension_match.group(1))
                unit = dimension_match.group(2) or 'mm'
                
                # Get the associated geometry length
                if hasattr(dim, 'geometry') and isinstance(dim.geometry, LineString):
                    geom_length = dim.geometry.length
                    
                    # Calculate scale factor
                    if unit == 'm':
                        actual_length = value * 1000  # Convert to mm
                    elif unit == 'cm':
                        actual_length = value * 10
                    else:
                        actual_length = value
                        
                    if geom_length > 0:
                        scale_factor = actual_length / geom_length
                        break
                        
        return scale_factor

    def _detect_units(self, doc) -> str:
        """Detect units from DXF header or content"""
        try:
            if hasattr(doc, 'header'):
                # Try to get units from DXF header
                units_var = doc.header.get('$INSUNITS', 1)
                unit_mapping = {
                    1: 'inches',
                    2: 'feet',
                    4: 'mm',
                    5: 'cm',
                    6: 'm'
                }
                return unit_mapping.get(units_var, 'mm')
        except:
            pass
            
        return 'mm'  # Default to millimeters

    def _calculate_bounds(self, floor_plan: FloorPlanData) -> Dict[str, float]:
        """Calculate overall bounds of the floor plan"""
        all_geometries = []
        
        for element_list in [floor_plan.walls, floor_plan.doors, floor_plan.windows, floor_plan.rooms]:
            for element in element_list:
                if element.geometry:
                    all_geometries.append(element.geometry)
                    
        if all_geometries:
            union_geom = unary_union(all_geometries)
            bounds = union_geom.bounds
            
            return {
                'min_x': bounds[0],
                'min_y': bounds[1],
                'max_x': bounds[2],
                'max_y': bounds[3],
                'width': bounds[2] - bounds[0],
                'height': bounds[3] - bounds[1]
            }
        else:
            return {
                'min_x': 0, 'min_y': 0, 'max_x': 1000, 'max_y': 1000,
                'width': 1000, 'height': 1000
            }

    def _create_empty_floor_plan(self) -> FloorPlanData:
        """Create an empty floor plan data structure"""
        return FloorPlanData(
            walls=[],
            doors=[],
            windows=[],
            rooms=[],
            restricted_areas=[],
            entrances=[],
            dimensions=[],
            text_annotations=[],
            scale_factor=1.0,
            units='mm',
            bounds={'min_x': 0, 'min_y': 0, 'max_x': 1000, 'max_y': 1000, 'width': 1000, 'height': 1000},
            metadata={}
        )

    def get_floor_plan_summary(self, floor_plan: FloorPlanData) -> Dict[str, Any]:
        """Generate a summary of the extracted floor plan"""
        return {
            'total_walls': len(floor_plan.walls),
            'total_doors': len(floor_plan.doors),
            'total_windows': len(floor_plan.windows),
            'total_rooms': len(floor_plan.rooms),
            'restricted_areas': len(floor_plan.restricted_areas),
            'entrances': len(floor_plan.entrances),
            'scale_factor': floor_plan.scale_factor,
            'units': floor_plan.units,
            'bounds': floor_plan.bounds,
            'total_area': sum(room.geometry.area for room in floor_plan.rooms if room.geometry),
            'wall_total_length': sum(wall.properties.get('length', 0) for wall in floor_plan.walls)
        }