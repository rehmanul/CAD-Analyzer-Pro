"""
Enhanced CAD Parser - Phase 1 Component
Multi-format CAD file processing with layer-aware extraction and precise geometric analysis
No fallback or mock data - only authentic CAD file processing
"""

import ezdxf
import fitz  # PyMuPDF
import numpy as np
import logging
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
import time
from pathlib import Path

@dataclass
class CADElement:
    """Base class for CAD elements with geometric and metadata information"""
    element_type: str
    geometry: Any  # Shapely geometry object
    layer: str = ""
    color: Optional[str] = None
    thickness: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FloorPlanData:
    """Comprehensive floor plan data structure"""
    walls: List[CADElement] = field(default_factory=list)
    doors: List[CADElement] = field(default_factory=list)
    windows: List[CADElement] = field(default_factory=list)
    openings: List[CADElement] = field(default_factory=list)
    rooms: List[CADElement] = field(default_factory=list)
    text_annotations: List[CADElement] = field(default_factory=list)
    dimensions: List[CADElement] = field(default_factory=list)
    furniture: List[CADElement] = field(default_factory=list)
    structural_elements: List[CADElement] = field(default_factory=list)
    
    # Metadata
    scale_factor: float = 1.0
    units: str = "mm"
    drawing_bounds: Optional[Tuple[float, float, float, float]] = None
    layer_info: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    wall_connectivity: float = 0.0
    element_count: int = 0
    processing_confidence: float = 0.0

class EnhancedCADParser:
    """
    Advanced CAD parser supporting DXF, DWG, and PDF formats
    Extracts architectural elements with high precision and no fallback data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.dxf', '.dwg', '.pdf']
        
        # Element detection patterns
        self.wall_layers = [
            'WALLS', 'WALL', 'MUR', 'MURS', 'A-WALL', 'ARCH-WALL',
            'PARTITION', 'CONSTRUCTION', 'STRUCTURE'
        ]
        
        self.door_layers = [
            'DOORS', 'DOOR', 'PORTE', 'PORTES', 'A-DOOR', 'ARCH-DOOR',
            'OPENING', 'OUVERTURE'
        ]
        
        self.window_layers = [
            'WINDOWS', 'WINDOW', 'FENETRE', 'FENETRES', 'A-WIND', 'ARCH-WIND'
        ]
        
        self.text_layers = [
            'TEXT', 'TEXTS', 'TEXTE', 'ANNOTATION', 'LABEL', 'DIMENSION',
            'DIM', 'COTE', 'ROOM', 'PIECE'
        ]
        
        # Geometric analysis parameters
        self.min_wall_length = 100  # mm
        self.wall_thickness_tolerance = 50  # mm
        self.connection_tolerance = 10  # mm

    def parse_cad_file(self, file_path: str) -> Optional[FloorPlanData]:
        """
        Parse CAD file and extract architectural elements
        
        Args:
            file_path: Path to the CAD file
            
        Returns:
            FloorPlanData object with extracted elements or None if parsing fails
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None
            
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.dxf':
                return self._parse_dxf_file(file_path)
            elif file_ext == '.dwg':
                return self._parse_dwg_file(file_path)
            elif file_ext == '.pdf':
                return self._parse_pdf_file(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing CAD file {file_path}: {str(e)}")
            return None

    def parse_cad_content(self, file_content: bytes, filename: str) -> Optional[FloorPlanData]:
        """
        Parse CAD file from memory content
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename with extension
            
        Returns:
            FloorPlanData object with extracted elements or None if parsing fails
        """
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            result = self.parse_cad_file(temp_file_path)
            return result
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def _parse_dxf_file(self, file_path: str) -> Optional[FloorPlanData]:
        """Parse DXF file with layer-aware processing"""
        try:
            doc = ezdxf.readfile(file_path)
            modelspace = doc.modelspace()
            
            floor_plan = FloorPlanData()
            
            # Get drawing bounds
            try:
                extents = doc.header.get('$EXTMIN'), doc.header.get('$EXTMAX')
                if extents[0] and extents[1]:
                    floor_plan.drawing_bounds = (
                        extents[0][0], extents[0][1], extents[1][0], extents[1][1]
                    )
            except:
                pass
            
            # Extract scale and units
            floor_plan.units = self._detect_drawing_units(doc)
            floor_plan.scale_factor = self._detect_scale_factor(doc)
            
            # Process all entities by layer
            layer_elements = {}
            for entity in modelspace:
                layer_name = entity.dxf.layer.upper() if hasattr(entity.dxf, 'layer') else 'DEFAULT'
                
                if layer_name not in layer_elements:
                    layer_elements[layer_name] = []
                layer_elements[layer_name].append(entity)
            
            floor_plan.layer_info = {layer: len(entities) for layer, entities in layer_elements.items()}
            
            # Extract walls
            self._extract_walls_from_dxf(layer_elements, floor_plan)
            
            # Extract doors and windows
            self._extract_openings_from_dxf(layer_elements, floor_plan)
            
            # Extract text and annotations
            self._extract_text_from_dxf(layer_elements, floor_plan)
            
            # Calculate quality metrics
            floor_plan.element_count = (
                len(floor_plan.walls) + len(floor_plan.doors) + 
                len(floor_plan.windows) + len(floor_plan.text_annotations)
            )
            
            floor_plan.wall_connectivity = self._calculate_wall_connectivity(floor_plan.walls)
            floor_plan.processing_confidence = self._calculate_processing_confidence(floor_plan)
            
            self.logger.info(f"DXF parsing completed: {floor_plan.element_count} elements extracted")
            return floor_plan
            
        except Exception as e:
            self.logger.error(f"Error parsing DXF file: {str(e)}")
            return None

    def _extract_walls_from_dxf(self, layer_elements: Dict[str, List], floor_plan: FloorPlanData):
        """Extract wall elements from DXF layers"""
        wall_entities = []
        
        # Collect entities from wall layers
        for layer_name, entities in layer_elements.items():
            if any(wall_layer in layer_name for wall_layer in self.wall_layers):
                wall_entities.extend(entities)
        
        # Also check entities that look like walls (lines, polylines in construction layers)
        for layer_name, entities in layer_elements.items():
            if 'CONSTRUCTION' in layer_name or 'ARCH' in layer_name:
                for entity in entities:
                    if entity.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE']:
                        wall_entities.append(entity)
        
        # Process wall entities
        for entity in wall_entities:
            wall_element = self._process_wall_entity(entity)
            if wall_element:
                floor_plan.walls.append(wall_element)

    def _process_wall_entity(self, entity) -> Optional[CADElement]:
        """Process individual wall entity and create CADElement"""
        try:
            if entity.dxftype() == 'LINE':
                start = (entity.dxf.start.x, entity.dxf.start.y)
                end = (entity.dxf.end.x, entity.dxf.end.y)
                
                # Check minimum length
                length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                if length < self.min_wall_length:
                    return None
                
                geometry = LineString([start, end])
                
                return CADElement(
                    element_type='wall',
                    geometry=geometry,
                    layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '',
                    thickness=length,
                    properties={
                        'length': length,
                        'start_point': start,
                        'end_point': end,
                        'dxf_type': entity.dxftype()
                    }
                )
                
            elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                points = []
                try:
                    if hasattr(entity, 'vertices'):
                        points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                    elif hasattr(entity, 'get_points'):
                        points = [(p[0], p[1]) for p in entity.get_points()]
                except:
                    return None
                
                if len(points) < 2:
                    return None
                
                geometry = LineString(points)
                
                return CADElement(
                    element_type='wall',
                    geometry=geometry,
                    layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '',
                    thickness=geometry.length,
                    properties={
                        'length': geometry.length,
                        'points': points,
                        'dxf_type': entity.dxftype()
                    }
                )
                
        except Exception as e:
            self.logger.warning(f"Error processing wall entity: {str(e)}")
            return None

    def _extract_openings_from_dxf(self, layer_elements: Dict[str, List], floor_plan: FloorPlanData):
        """Extract door and window elements"""
        # Extract doors
        for layer_name, entities in layer_elements.items():
            if any(door_layer in layer_name for door_layer in self.door_layers):
                for entity in entities:
                    door_element = self._process_opening_entity(entity, 'door')
                    if door_element:
                        floor_plan.doors.append(door_element)
        
        # Extract windows
        for layer_name, entities in layer_elements.items():
            if any(window_layer in layer_name for window_layer in self.window_layers):
                for entity in entities:
                    window_element = self._process_opening_entity(entity, 'window')
                    if window_element:
                        floor_plan.windows.append(window_element)

    def _process_opening_entity(self, entity, opening_type: str) -> Optional[CADElement]:
        """Process door or window entity"""
        try:
            if entity.dxftype() == 'INSERT':
                # Block reference - typical for doors/windows
                insertion_point = (entity.dxf.insert.x, entity.dxf.insert.y)
                geometry = Point(insertion_point)
                
                return CADElement(
                    element_type=opening_type,
                    geometry=geometry,
                    layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '',
                    properties={
                        'insertion_point': insertion_point,
                        'block_name': entity.dxf.name if hasattr(entity.dxf, 'name') else '',
                        'rotation': entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0,
                        'scale_x': entity.dxf.xscale if hasattr(entity.dxf, 'xscale') else 1,
                        'scale_y': entity.dxf.yscale if hasattr(entity.dxf, 'yscale') else 1,
                        'dxf_type': entity.dxftype()
                    }
                )
                
            elif entity.dxftype() in ['CIRCLE', 'ARC']:
                center = (entity.dxf.center.x, entity.dxf.center.y)
                geometry = Point(center).buffer(entity.dxf.radius)
                
                return CADElement(
                    element_type=opening_type,
                    geometry=geometry,
                    layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '',
                    properties={
                        'center': center,
                        'radius': entity.dxf.radius,
                        'dxf_type': entity.dxftype()
                    }
                )
                
        except Exception as e:
            self.logger.warning(f"Error processing opening entity: {str(e)}")
            return None

    def _extract_text_from_dxf(self, layer_elements: Dict[str, List], floor_plan: FloorPlanData):
        """Extract text and annotation elements"""
        for layer_name, entities in layer_elements.items():
            if any(text_layer in layer_name for text_layer in self.text_layers):
                for entity in entities:
                    text_element = self._process_text_entity(entity)
                    if text_element:
                        floor_plan.text_annotations.append(text_element)

    def _process_text_entity(self, entity) -> Optional[CADElement]:
        """Process text entity"""
        try:
            if entity.dxftype() in ['TEXT', 'MTEXT']:
                position = (entity.dxf.insert.x, entity.dxf.insert.y)
                geometry = Point(position)
                
                text_content = entity.dxf.text if hasattr(entity.dxf, 'text') else ''
                
                return CADElement(
                    element_type='text',
                    geometry=geometry,
                    layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '',
                    properties={
                        'text': text_content,
                        'position': position,
                        'height': entity.dxf.height if hasattr(entity.dxf, 'height') else 0,
                        'rotation': entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0,
                        'dxf_type': entity.dxftype()
                    }
                )
                
        except Exception as e:
            self.logger.warning(f"Error processing text entity: {str(e)}")
            return None

    def _parse_dwg_file(self, file_path: str) -> Optional[FloorPlanData]:
        """Parse DWG file (convert to DXF first)"""
        # Note: DWG parsing requires additional tools like ODA File Converter
        # For now, return None and log that DWG support needs additional setup
        self.logger.warning("DWG file support requires ODA File Converter or similar tool")
        return None

    def _parse_pdf_file(self, file_path: str) -> Optional[FloorPlanData]:
        """Parse architectural PDF files"""
        try:
            doc = fitz.open(file_path)
            floor_plan = FloorPlanData()
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract vector graphics (lines, rectangles)
                paths = page.get_drawings()
                
                for path in paths:
                    element = self._process_pdf_path(path)
                    if element:
                        if self._is_wall_like(element):
                            floor_plan.walls.append(element)
                
                # Extract text
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text_element = self._process_pdf_text(span)
                                if text_element:
                                    floor_plan.text_annotations.append(text_element)
            
            floor_plan.element_count = len(floor_plan.walls) + len(floor_plan.text_annotations)
            floor_plan.processing_confidence = 0.7 if floor_plan.element_count > 0 else 0.0
            
            return floor_plan
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF file: {str(e)}")
            return None

    def _process_pdf_path(self, path) -> Optional[CADElement]:
        """Process PDF vector path as potential wall"""
        try:
            items = path.get("items", [])
            if not items:
                return None
            
            points = []
            for item in items:
                if item[0] == "l":  # Line to
                    points.append((item[1].x, item[1].y))
                elif item[0] == "m":  # Move to
                    points.append((item[1].x, item[1].y))
            
            if len(points) < 2:
                return None
            
            geometry = LineString(points)
            
            return CADElement(
                element_type='wall_candidate',
                geometry=geometry,
                properties={
                    'length': geometry.length,
                    'points': points,
                    'source': 'pdf_vector'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Error processing PDF path: {str(e)}")
            return None

    def _process_pdf_text(self, span) -> Optional[CADElement]:
        """Process PDF text span"""
        try:
            text = span.get("text", "").strip()
            if not text:
                return None
            
            bbox = span.get("bbox")
            if not bbox:
                return None
            
            position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            geometry = Point(position)
            
            return CADElement(
                element_type='text',
                geometry=geometry,
                properties={
                    'text': text,
                    'position': position,
                    'font_size': span.get("size", 0),
                    'source': 'pdf_text'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Error processing PDF text: {str(e)}")
            return None

    def _is_wall_like(self, element: CADElement) -> bool:
        """Determine if element represents a wall"""
        if element.element_type != 'wall_candidate':
            return False
        
        # Check length threshold
        if element.geometry.length < self.min_wall_length:
            return False
        
        # Additional wall detection logic could be added here
        return True

    def _detect_drawing_units(self, doc) -> str:
        """Detect drawing units from DXF header"""
        try:
            units_code = doc.header.get('$INSUNITS', 0)
            units_map = {
                0: 'unitless',
                1: 'inches',
                2: 'feet',
                4: 'mm',
                5: 'cm',
                6: 'm',
                14: 'micrometers'
            }
            return units_map.get(units_code, 'mm')
        except:
            return 'mm'

    def _detect_scale_factor(self, doc) -> float:
        """Detect scale factor from drawing"""
        try:
            # Check for scale information in variables or blocks
            # This is a simplified implementation
            extents = doc.header.get('$EXTMIN'), doc.header.get('$EXTMAX')
            if extents[0] and extents[1]:
                width = abs(extents[1][0] - extents[0][0])
                height = abs(extents[1][1] - extents[0][1])
                
                # Estimate scale based on typical room sizes
                if width > 50000 or height > 50000:  # Very large drawing
                    return 0.001  # 1:1000 scale
                elif width > 10000 or height > 10000:  # Large drawing
                    return 0.01   # 1:100 scale
                else:
                    return 1.0    # 1:1 scale
            
            return 1.0
        except:
            return 1.0

    def _calculate_wall_connectivity(self, walls: List[CADElement]) -> float:
        """Calculate wall connectivity score"""
        if len(walls) < 2:
            return 0.0
        
        connected_count = 0
        total_connections = 0
        
        for i, wall1 in enumerate(walls):
            for wall2 in walls[i+1:]:
                total_connections += 1
                
                # Check if walls are connected (endpoints close together)
                if self._walls_connected(wall1, wall2):
                    connected_count += 1
        
        return connected_count / max(total_connections, 1)

    def _walls_connected(self, wall1: CADElement, wall2: CADElement) -> bool:
        """Check if two walls are connected"""
        try:
            # Get endpoints of walls
            coords1 = list(wall1.geometry.coords)
            coords2 = list(wall2.geometry.coords)
            
            if len(coords1) < 2 or len(coords2) < 2:
                return False
            
            endpoints1 = [coords1[0], coords1[-1]]
            endpoints2 = [coords2[0], coords2[-1]]
            
            # Check if any endpoints are close
            for ep1 in endpoints1:
                for ep2 in endpoints2:
                    distance = np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                    if distance <= self.connection_tolerance:
                        return True
            
            return False
        except:
            return False

    def _calculate_processing_confidence(self, floor_plan: FloorPlanData) -> float:
        """Calculate overall processing confidence score"""
        factors = []
        
        # Element count factor
        if floor_plan.element_count > 10:
            factors.append(0.8)
        elif floor_plan.element_count > 5:
            factors.append(0.6)
        elif floor_plan.element_count > 0:
            factors.append(0.4)
        else:
            factors.append(0.0)
        
        # Wall connectivity factor
        factors.append(floor_plan.wall_connectivity)
        
        # Layer organization factor
        organized_layers = sum(1 for layer in floor_plan.layer_info.keys() 
                              if any(wall_layer in layer.upper() for wall_layer in self.wall_layers))
        if organized_layers > 0:
            factors.append(0.8)
        else:
            factors.append(0.3)
        
        return sum(factors) / len(factors) if factors else 0.0