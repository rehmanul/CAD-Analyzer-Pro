import ezdxf
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import logging
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF for PDF parsing
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DXFParser:
    """Production-grade DXF/DWG/PDF parser for CAD floor plans"""

    def __init__(self):
        self.supported_entities = [
            'LINE', 'POLYLINE', 'LWPOLYLINE', 'CIRCLE', 'ARC', 
            'ELLIPSE', 'SPLINE', 'TEXT', 'MTEXT', 'INSERT', 'HATCH'
        ]
        self.color_mapping = {
            'black': [0, 0, 0],
            'red': [255, 0, 0],
            'blue': [0, 0, 255],
            'green': [0, 255, 0],
            'yellow': [255, 255, 0],
            'cyan': [0, 255, 255],
            'magenta': [255, 0, 255],
            'white': [255, 255, 255]
        }

    def parse_dxf(self, file_path: str) -> Dict[str, Any]:
        """Parse DXF file and extract all geometric entities"""
        try:
            logger.info(f"Starting DXF parsing for: {file_path}")

            # Load DXF document
            doc = ezdxf.readfile(file_path)

            # Initialize result structure
            result = {
                'entities': [],
                'layers': {},
                'blocks': {},
                'metadata': self._extract_metadata(doc),
                'bounds': None,
                'units': self._get_units(doc)
            }

            # Extract entities from model space
            msp = doc.modelspace()

            # Process each entity
            for entity in msp:
                parsed_entity = self._parse_entity(entity)
                if parsed_entity:
                    result['entities'].append(parsed_entity)

            # Extract layer information
            result['layers'] = self._extract_layers(doc)

            # Extract blocks
            result['blocks'] = self._extract_blocks(doc)

            # Calculate bounds
            result['bounds'] = self._calculate_bounds(result['entities'])

            logger.info(f"Successfully parsed {len(result['entities'])} entities")

            return result

        except Exception as e:
            logger.error(f"Error parsing DXF file: {str(e)}")
            raise Exception(f"Failed to parse DXF file: {str(e)}")

    def parse_dwg(self, file_path: str) -> Dict[str, Any]:
        """Parse DWG file (converted to DXF first)"""
        try:
            logger.info(f"Starting DWG parsing for: {file_path}")

            # Try to read as DXF first (some .dwg files are actually DXF)
            try:
                doc = ezdxf.readfile(file_path)
                logger.info("File successfully read as DXF format")
                return self.parse_dxf_document(doc)
            except ezdxf.DXFError as dxf_error:
                logger.warning(f"File is not in DXF format: {str(dxf_error)}")
                
                # Check if it's a binary DWG file
                with open(file_path, 'rb') as f:
                    header = f.read(6)
                    if header.startswith(b'AC10'):
                        dwg_version = "AutoCAD R10/R11"
                    elif header.startswith(b'AC12'):
                        dwg_version = "AutoCAD R12/R13/R14"
                    elif header.startswith(b'AC15'):
                        dwg_version = "AutoCAD 2000/2002"
                    elif header.startswith(b'AC18'):
                        dwg_version = "AutoCAD 2004/2005/2006"
                    elif header.startswith(b'AC21'):
                        dwg_version = "AutoCAD 2007/2008/2009"
                    elif header.startswith(b'AC24'):
                        dwg_version = "AutoCAD 2010/2011/2012"
                    elif header.startswith(b'AC27'):
                        dwg_version = "AutoCAD 2013/2014/2015/2016/2017"
                    elif header.startswith(b'AC32'):
                        dwg_version = "AutoCAD 2018/2019/2020/2021/2022/2023/2024"
                    else:
                        dwg_version = "Unknown version"
                
                raise Exception(
                    f"DWG file detected ({dwg_version}). "
                    f"Please convert to DXF format using AutoCAD, FreeCAD, or LibreCAD. "
                    f"Online converters are also available at cloudconvert.com or zamzar.com. "
                    f"Alternatively, try uploading the file in DXF or PDF format."
                )

        except Exception as e:
            logger.error(f"Error parsing DWG file: {str(e)}")
            if "DWG file detected" in str(e):
                raise e
            else:
                raise Exception(f"DWG parsing failed. Please convert to DXF format first: {str(e)}")

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF file and extract vector graphics"""
        try:
            logger.info(f"Starting PDF parsing for: {file_path}")

            doc = fitz.open(file_path)

            result = {
                'entities': [],
                'layers': {},
                'blocks': {},
                'metadata': {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'pages': doc.page_count
                },
                'bounds': None,
                'units': 'mm'  # Default for PDF
            }

            # Process each page
            for page_num in range(doc.page_count):
                page = doc[page_num]

                # Extract vector paths
                paths = page.get_drawings()

                for path in paths:
                    parsed_entity = self._parse_pdf_path(path, page_num)
                    if parsed_entity:
                        result['entities'].append(parsed_entity)

                # Extract text
                text_blocks = page.get_text("dict")
                for block in text_blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            parsed_text = self._parse_pdf_text(line, page_num)
                            if parsed_text:
                                result['entities'].append(parsed_text)

            doc.close()

            # Calculate bounds
            result['bounds'] = self._calculate_bounds(result['entities'])

            logger.info(f"Successfully parsed {len(result['entities'])} entities from PDF")

            return result

        except Exception as e:
            logger.error(f"Error parsing PDF file: {str(e)}")
            raise Exception(f"Failed to parse PDF file: {str(e)}")

    def _parse_entity(self, entity) -> Optional[Dict[str, Any]]:
        """Parse individual DXF entity"""
        try:
            entity_type = entity.dxftype()

            if entity_type not in self.supported_entities:
                return None

            # Base entity information
            parsed = {
                'type': entity_type,
                'layer': entity.dxf.layer,
                'color': self._get_entity_color(entity),
                'linetype': getattr(entity.dxf, 'linetype', 'CONTINUOUS'),
                'lineweight': getattr(entity.dxf, 'lineweight', 0),
                'geometry': None,
                'properties': {}
            }

            # Parse geometry based on entity type
            if entity_type == 'LINE':
                parsed['geometry'] = self._parse_line(entity)
            elif entity_type in ['POLYLINE', 'LWPOLYLINE']:
                parsed['geometry'] = self._parse_polyline(entity)
            elif entity_type == 'CIRCLE':
                parsed['geometry'] = self._parse_circle(entity)
            elif entity_type == 'ARC':
                parsed['geometry'] = self._parse_arc(entity)
            elif entity_type == 'ELLIPSE':
                parsed['geometry'] = self._parse_ellipse(entity)
            elif entity_type == 'SPLINE':
                parsed['geometry'] = self._parse_spline(entity)
            elif entity_type in ['TEXT', 'MTEXT']:
                parsed['geometry'] = self._parse_text(entity)
            elif entity_type == 'INSERT':
                parsed['geometry'] = self._parse_insert(entity)
            elif entity_type == 'HATCH':
                parsed['geometry'] = self._parse_hatch(entity)

            return parsed

        except Exception as e:
            logger.warning(f"Error parsing entity {entity.dxftype()}: {str(e)}")
            return None

    def _parse_line(self, entity) -> Dict[str, Any]:
        """Parse LINE entity"""
        start = entity.dxf.start
        end = entity.dxf.end

        return {
            'type': 'line',
            'start': {'x': start.x, 'y': start.y, 'z': start.z},
            'end': {'x': end.x, 'y': end.y, 'z': end.z},
            'length': (end - start).magnitude,
            'angle': np.degrees(np.arctan2(end.y - start.y, end.x - start.x)),
            'shapely_geom': LineString([(start.x, start.y), (end.x, end.y)])
        }

    def _parse_polyline(self, entity) -> Dict[str, Any]:
        """Parse POLYLINE/LWPOLYLINE entity"""
        try:
            points = []

            if entity.dxftype() == 'LWPOLYLINE':
                # Lightweight polyline
                for point in entity.get_points():
                    points.append([point[0], point[1]])
            else:
                # Regular polyline
                for vertex in entity.vertices:
                    points.append([vertex.dxf.location.x, vertex.dxf.location.y])

            if len(points) < 2:
                return None

            # Create shapely geometry
            if entity.is_closed and len(points) > 2:
                shapely_geom = Polygon(points)
                geom_type = 'polygon'
            else:
                shapely_geom = LineString(points)
                geom_type = 'polyline'

            return {
                'type': geom_type,
                'points': points,
                'is_closed': entity.is_closed,
                'area': shapely_geom.area if geom_type == 'polygon' else 0,
                'length': shapely_geom.length,
                'shapely_geom': shapely_geom
            }

        except Exception as e:
            logger.warning(f"Error parsing polyline: {str(e)}")
            return None

    def _parse_circle(self, entity) -> Dict[str, Any]:
        """Parse CIRCLE entity"""
        center = entity.dxf.center
        radius = entity.dxf.radius

        # Create shapely geometry
        shapely_geom = Point(center.x, center.y).buffer(radius)

        return {
            'type': 'circle',
            'center': {'x': center.x, 'y': center.y, 'z': center.z},
            'radius': radius,
            'area': np.pi * radius ** 2,
            'circumference': 2 * np.pi * radius,
            'shapely_geom': shapely_geom
        }

    def _parse_arc(self, entity) -> Dict[str, Any]:
        """Parse ARC entity"""
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = np.radians(entity.dxf.start_angle)
        end_angle = np.radians(entity.dxf.end_angle)

        # Calculate arc points
        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 2 * np.pi

        num_points = max(10, int(angle_diff * 20))  # Adaptive point count
        angles = np.linspace(start_angle, end_angle, num_points)

        points = []
        for angle in angles:
            x = center.x + radius * np.cos(angle)
            y = center.y + radius * np.sin(angle)
            points.append([x, y])

        shapely_geom = LineString(points)

        return {
            'type': 'arc',
            'center': {'x': center.x, 'y': center.y, 'z': center.z},
            'radius': radius,
            'start_angle': entity.dxf.start_angle,
            'end_angle': entity.dxf.end_angle,
            'points': points,
            'length': radius * angle_diff,
            'shapely_geom': shapely_geom
        }

    def _parse_ellipse(self, entity) -> Dict[str, Any]:
        """Parse ELLIPSE entity"""
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio

        # Calculate ellipse parameters
        major_radius = major_axis.magnitude
        minor_radius = major_radius * ratio

        # Generate ellipse points
        num_points = 50
        angles = np.linspace(0, 2 * np.pi, num_points)

        points = []
        for angle in angles:
            x = center.x + major_radius * np.cos(angle) * major_axis.x / major_axis.magnitude
            y = center.y + major_radius * np.sin(angle) * major_axis.y / major_axis.magnitude
            points.append([x, y])

        shapely_geom = Polygon(points)

        return {
            'type': 'ellipse',
            'center': {'x': center.x, 'y': center.y, 'z': center.z},
            'major_radius': major_radius,
            'minor_radius': minor_radius,
            'ratio': ratio,
            'points': points,
            'area': np.pi * major_radius * minor_radius,
            'shapely_geom': shapely_geom
        }

    def _parse_spline(self, entity) -> Dict[str, Any]:
        """Parse SPLINE entity"""
        try:
            # Get control points
            control_points = []
            for point in entity.control_points:
                control_points.append([point.x, point.y])

            if len(control_points) < 2:
                return None

            # For simplicity, create a polyline through control points
            # In production, you'd want to evaluate the actual spline curve
            shapely_geom = LineString(control_points)

            return {
                'type': 'spline',
                'control_points': control_points,
                'degree': entity.dxf.degree,
                'closed': entity.closed,
                'length': shapely_geom.length,
                'shapely_geom': shapely_geom
            }

        except Exception as e:
            logger.warning(f"Error parsing spline: {str(e)}")
            return None

    def _parse_text(self, entity) -> Dict[str, Any]:
        """Parse TEXT/MTEXT entity"""
        position = entity.dxf.insert

        return {
            'type': 'text',
            'text': entity.dxf.text,
            'position': {'x': position.x, 'y': position.y, 'z': position.z},
            'height': entity.dxf.height,
            'rotation': getattr(entity.dxf, 'rotation', 0),
            'style': getattr(entity.dxf, 'style', 'STANDARD'),
            'shapely_geom': Point(position.x, position.y)
        }

    def _parse_insert(self, entity) -> Dict[str, Any]:
        """Parse INSERT (block reference) entity"""
        position = entity.dxf.insert

        return {
            'type': 'insert',
            'block_name': entity.dxf.name,
            'position': {'x': position.x, 'y': position.y, 'z': position.z},
            'scale': {
                'x': entity.dxf.xscale,
                'y': entity.dxf.yscale,
                'z': entity.dxf.zscale
            },
            'rotation': entity.dxf.rotation,
            'shapely_geom': Point(position.x, position.y)
        }

    def _parse_hatch(self, entity) -> Dict[str, Any]:
        """Parse HATCH entity"""
        try:
            # Get hatch boundary paths
            paths = []
            for path in entity.paths:
                path_edges = []
                for edge in path.edges:
                    if edge.type == 'LineEdge':
                        path_edges.append([edge.start, edge.end])
                    elif edge.type == 'ArcEdge':
                        # Approximate arc with line segments
                        path_edges.append([edge.start, edge.end])
                paths.append(path_edges)

            if not paths:
                return None

            # Create shapely geometry from first path
            # In production, you'd want to handle multiple paths properly
            first_path = paths[0]
            if len(first_path) > 2:
                points = []
                for edge in first_path:
                    points.append([edge[0][0], edge[0][1]])

                shapely_geom = Polygon(points)

                return {
                    'type': 'hatch',
                    'pattern': entity.dxf.pattern_name,
                    'paths': paths,
                    'area': shapely_geom.area,
                    'shapely_geom': shapely_geom
                }

            return None

        except Exception as e:
            logger.warning(f"Error parsing hatch: {str(e)}")
            return None

    def _parse_pdf_path(self, path, page_num: int) -> Optional[Dict[str, Any]]:
        """Parse PDF vector path"""
        try:
            if 'items' not in path:
                return None

            points = []
            for item in path['items']:
                if item[0] == 'l':  # Line to
                    points.append([item[1].x, item[1].y])
                elif item[0] == 'm':  # Move to
                    points.append([item[1].x, item[1].y])
                elif item[0] == 'c':  # Curve to
                    # Approximate curve with end point
                    points.append([item[3].x, item[3].y])

            if len(points) < 2:
                return None

            # Determine if this is a closed path
            is_closed = (len(points) > 2 and 
                        abs(points[0][0] - points[-1][0]) < 1 and 
                        abs(points[0][1] - points[-1][1]) < 1)

            if is_closed and len(points) > 2:
                shapely_geom = Polygon(points)
                geom_type = 'polygon'
            else:
                shapely_geom = LineString(points)
                geom_type = 'polyline'

            return {
                'type': geom_type,
                'points': points,
                'is_closed': is_closed,
                'page': page_num,
                'color': path.get('color', [0, 0, 0]),
                'width': path.get('width', 1),
                'area': shapely_geom.area if geom_type == 'polygon' else 0,
                'length': shapely_geom.length,
                'shapely_geom': shapely_geom,
                'layer': f'page_{page_num}'
            }

        except Exception as e:
            logger.warning(f"Error parsing PDF path: {str(e)}")
            return None

    def _parse_pdf_text(self, line, page_num: int) -> Optional[Dict[str, Any]]:
        """Parse PDF text"""
        try:
            text_content = ""
            bbox = None

            for span in line.get("spans", []):
                text_content += span.get("text", "")
                if bbox is None:
                    bbox = span.get("bbox")

            if not text_content.strip():
                return None

            # Calculate text position from bbox
            if bbox:
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
            else:
                x, y = 0, 0

            return {
                'type': 'text',
                'text': text_content,
                'position': {'x': x, 'y': y, 'z': 0},
                'page': page_num,
                'bbox': bbox,
                'shapely_geom': Point(x, y),
                'layer': f'page_{page_num}'
            }

        except Exception as e:
            logger.warning(f"Error parsing PDF text: {str(e)}")
            return None

    def _get_entity_color(self, entity) -> List[int]:
        """Get entity color as RGB values"""
        try:
            color_index = entity.dxf.color

            # ACI color mapping (simplified)
            if color_index == 0:  # ByBlock
                return [255, 255, 255]  # Default to white
            elif color_index == 1:  # Red
                return [255, 0, 0]
            elif color_index == 2:  # Yellow
                return [255, 255, 0]
            elif color_index == 3:  # Green
                return [0, 255, 0]
            elif color_index == 4:  # Cyan
                return [0, 255, 255]
            elif color_index == 5:  # Blue
                return [0, 0, 255]
            elif color_index == 6:  # Magenta
                return [255, 0, 255]
            elif color_index == 7:  # Black/White
                return [0, 0, 0]
            else:
                return [128, 128, 128]  # Default gray

        except:
            return [0, 0, 0]  # Default black

    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract document metadata"""
        try:
            header = doc.header
            return {
                'acadver': header.get('$ACADVER', 'Unknown'),
                'dwgcodepage': header.get('$DWGCODEPAGE', 'Unknown'),
                'creation_date': str(header.get('$TDCREATE', '')),
                'modification_date': str(header.get('$TDUPDATE', '')),
                'title': header.get('$TITLE', ''),
                'subject': header.get('$SUBJECT', ''),
                'author': header.get('$AUTHOR', ''),
                'keywords': header.get('$KEYWORDS', ''),
                'comments': header.get('$COMMENTS', ''),
                'units': header.get('$INSUNITS', 0)
            }
        except:
            return {}

    def _get_units(self, doc) -> str:
        """Get drawing units"""
        try:
            units_code = doc.header.get('$INSUNITS', 0)
            units_map = {
                0: 'unitless',
                1: 'inches',
                2: 'feet',
                3: 'miles',
                4: 'millimeters',
                5: 'centimeters',
                6: 'meters',
                7: 'kilometers',
                8: 'microinches',
                9: 'mils',
                10: 'yards',
                11: 'angstroms',
                12: 'nanometers',
                13: 'microns',
                14: 'decimeters',
                15: 'decameters',
                16: 'hectometers',
                17: 'gigameters',
                18: 'astronomical units',
                19: 'light years',
                20: 'parsecs'
            }
            return units_map.get(units_code, 'unknown')
        except:
            return 'unknown'

    def _extract_layers(self, doc) -> Dict[str, Any]:
        """Extract layer information"""
        layers = {}
        try:
            # Use the layer table directly
            layer_table = doc.layers
            for layer in layer_table:
                layers[layer.dxf.name] = {
                    'name': layer.dxf.name,
                    'color': layer.dxf.color,
                    'linetype': layer.dxf.linetype,
                    'lineweight': getattr(layer.dxf, 'lineweight', 0),
                    'plot': layer.dxf.plot,
                    'on': layer.is_on(),
                    'frozen': layer.is_frozen(),
                    'locked': layer.is_locked()
                }
        except Exception as e:
            logger.warning(f"Could not extract layer information: {str(e)}")
            # Create default layer
            layers['0'] = {
                'name': '0',
                'color': 7,
                'linetype': 'CONTINUOUS',
                'lineweight': 0,
                'plot': True,
                'on': True,
                'frozen': False,
                'locked': False
            }

        return layers

    def _extract_blocks(self, doc) -> Dict[str, Any]:
        """Extract block definitions"""
        blocks = {}

        for block_name, block in doc.blocks.items():
            if not block_name.startswith('*'):  # Skip anonymous blocks
                block_entities = []
                for entity in block:
                    parsed_entity = self._parse_entity(entity)
                    if parsed_entity:
                        block_entities.append(parsed_entity)

                blocks[block_name] = {
                    'name': block_name,
                    'entities': block_entities,
                    'base_point': {
                        'x': block.block.dxf.base_point.x,
                        'y': block.block.dxf.base_point.y,
                        'z': block.block.dxf.base_point.z
                    }
                }

        return blocks

    def _calculate_bounds(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate bounding box of all entities"""
        if not entities:
            return {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for entity in entities:
            geom = entity.get('geometry', {})

            if geom.get('type') == 'line':
                points = [geom['start'], geom['end']]
            elif geom.get('type') in ['polyline', 'polygon']:
                points = geom.get('points', [])
            elif geom.get('type') == 'circle':
                center = geom['center']
                radius = geom['radius']
                points = [
                    {'x': center['x'] - radius, 'y': center['y'] - radius},
                    {'x': center['x'] + radius, 'y': center['y'] + radius}
                ]
            elif geom.get('type') == 'text':
                points = [geom['position']]
            else:
                continue

            for point in points:
                if isinstance(point, dict):
                    x, y = point.get('x', 0), point.get('y', 0)
                else:
                    x, y = point[0], point[1]

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        return {
            'min_x': min_x if min_x != float('inf') else 0,
            'min_y': min_y if min_y != float('inf') else 0,
            'max_x': max_x if max_x != float('-inf') else 0,
            'max_y': max_y if max_y != float('-inf') else 0
        }

    def parse_dxf_document(self, doc) -> Dict[str, Any]:
        """Parse an already loaded DXF document"""
        # Initialize result structure
        result = {
            'entities': [],
            'layers': {},
            'blocks': {},
            'metadata': self._extract_metadata(doc),
            'bounds': None,
            'units': self._get_units(doc)
        }

        # Extract entities from model space
        msp = doc.modelspace()

        # Process each entity
        for entity in msp:
            parsed_entity = self._parse_entity(entity)
            if parsed_entity:
                result['entities'].append(parsed_entity)

        # Extract layer information
        result['layers'] = self._extract_layers(doc)

        # Extract blocks
        result['blocks'] = self._extract_blocks(doc)

        # Calculate bounds
        result['bounds'] = self._calculate_bounds(result['entities'])

        return result