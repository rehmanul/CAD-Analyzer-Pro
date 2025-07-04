import logging
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
import requests
from datetime import datetime
import uuid
import numpy as np
from shapely.geometry import Point, Polygon, LineString, box
import pandas as pd
from pathlib import Path
import zipfile
import tempfile

logger = logging.getLogger(__name__)


class IFCConnector:
    """Connector for Industry Foundation Classes (IFC) BIM format"""
    
    def __init__(self):
        self.ifc_version = "IFC4"
        self.schema_url = "http://www.buildingsmart-tech.org/ifc/IFC4/final/html/schema/"
        
    def import_ifc(self, file_path: str) -> Dict[str, Any]:
        """Import IFC file and convert to internal format"""
        logger.info(f"Importing IFC file: {file_path}")
        
        try:
            # Parse IFC file
            ifc_data = self._parse_ifc_file(file_path)
            
            # Extract floor plan data
            floor_plan_data = {
                'entities': [],
                'metadata': self._extract_ifc_metadata(ifc_data),
                'spaces': self._extract_ifc_spaces(ifc_data),
                'walls': self._extract_ifc_walls(ifc_data),
                'doors': self._extract_ifc_doors(ifc_data),
                'windows': self._extract_ifc_windows(ifc_data),
                'furniture': self._extract_ifc_furniture(ifc_data),
                'building_info': self._extract_building_info(ifc_data)
            }
            
            # Convert to standard format
            return self._convert_ifc_to_standard(floor_plan_data)
            
        except Exception as e:
            logger.error(f"Error importing IFC file: {str(e)}")
            raise
    
    def export_ifc(self, floor_plan_data: Dict[str, Any], ilot_results: List[Dict[str, Any]],
                   output_path: str) -> str:
        """Export floor plan and îlots to IFC format"""
        logger.info("Exporting to IFC format")
        
        try:
            # Create IFC structure
            ifc_content = self._create_ifc_header()
            
            # Add project information
            project_guid = self._generate_guid()
            ifc_content += self._create_ifc_project(project_guid, floor_plan_data)
            
            # Add site and building
            site_guid = self._generate_guid()
            building_guid = self._generate_guid()
            ifc_content += self._create_ifc_site(site_guid, project_guid)
            ifc_content += self._create_ifc_building(building_guid, site_guid)
            
            # Add building storey
            storey_guid = self._generate_guid()
            ifc_content += self._create_ifc_storey(storey_guid, building_guid)
            
            # Add walls
            for wall in floor_plan_data.get('walls', []):
                wall_guid = self._generate_guid()
                ifc_content += self._create_ifc_wall(wall_guid, wall, storey_guid)
            
            # Add îlots as furniture
            for ilot in ilot_results:
                ilot_guid = self._generate_guid()
                ifc_content += self._create_ifc_furniture(ilot_guid, ilot, storey_guid)
            
            # Add footer
            ifc_content += "ENDSEC;\nEND-ISO-10303-21;"
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(ifc_content)
            
            logger.info(f"IFC file exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting IFC file: {str(e)}")
            raise
    
    def _parse_ifc_file(self, file_path: str) -> Dict[str, Any]:
        """Parse IFC file content"""
        ifc_data = {
            'header': {},
            'data': [],
            'entities': {}
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse STEP format
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('HEADER;'):
                current_section = 'header'
            elif line.startswith('DATA;'):
                current_section = 'data'
            elif line.startswith('ENDSEC;'):
                current_section = None
            elif current_section == 'data' and line.startswith('#'):
                # Parse entity
                entity_id, entity_data = self._parse_ifc_entity(line)
                if entity_id:
                    ifc_data['entities'][entity_id] = entity_data
        
        return ifc_data
    
    def _parse_ifc_entity(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """Parse single IFC entity"""
        try:
            # Extract entity ID and type
            parts = line.split('=', 1)
            if len(parts) != 2:
                return None, {}
            
            entity_id = parts[0].strip()
            entity_def = parts[1].strip()
            
            # Extract entity type
            if '(' not in entity_def:
                return None, {}
            
            entity_type = entity_def.split('(')[0]
            
            # Extract attributes
            attrs_str = entity_def[entity_def.index('(') + 1:entity_def.rindex(')')]
            attributes = self._parse_attributes(attrs_str)
            
            return entity_id, {
                'type': entity_type,
                'attributes': attributes
            }
            
        except Exception:
            return None, {}
    
    def _parse_attributes(self, attrs_str: str) -> List[Any]:
        """Parse IFC attributes"""
        # Simplified attribute parsing
        attributes = []
        current_attr = ''
        paren_count = 0
        
        for char in attrs_str:
            if char == ',' and paren_count == 0:
                attributes.append(self._parse_attribute_value(current_attr.strip()))
                current_attr = ''
            else:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                current_attr += char
        
        if current_attr:
            attributes.append(self._parse_attribute_value(current_attr.strip()))
        
        return attributes
    
    def _parse_attribute_value(self, value: str) -> Any:
        """Parse individual attribute value"""
        if value == '$':
            return None
        elif value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        elif value.startswith('#'):
            return value
        elif value.replace('.', '').replace('-', '').isdigit():
            return float(value) if '.' in value else int(value)
        else:
            return value
    
    def _extract_ifc_spaces(self, ifc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract space definitions from IFC"""
        spaces = []
        
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] == 'IFCSPACE':
                attrs = entity['attributes']
                space = {
                    'id': entity_id,
                    'name': attrs[2] if len(attrs) > 2 else 'Unknown',
                    'description': attrs[3] if len(attrs) > 3 else '',
                    'geometry': self._get_entity_geometry(entity_id, ifc_data)
                }
                spaces.append(space)
        
        return spaces
    
    def _extract_ifc_walls(self, ifc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract wall definitions from IFC"""
        walls = []
        
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] in ['IFCWALL', 'IFCWALLSTANDARDCASE']:
                attrs = entity['attributes']
                wall = {
                    'id': entity_id,
                    'name': attrs[2] if len(attrs) > 2 else 'Wall',
                    'geometry': self._get_entity_geometry(entity_id, ifc_data),
                    'thickness': self._get_wall_thickness(entity_id, ifc_data)
                }
                walls.append(wall)
        
        return walls
    
    def _get_entity_geometry(self, entity_id: str, ifc_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract geometry for an entity"""
        # This is a simplified implementation
        # In reality, you would traverse the IFC relationships to find geometry
        return None
    
    def _get_wall_thickness(self, wall_id: str, ifc_data: Dict[str, Any]) -> float:
        """Get wall thickness from IFC data"""
        # Default thickness
        return 0.2
    
    def _extract_ifc_doors(self, ifc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract door definitions from IFC"""
        doors = []
        
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] == 'IFCDOOR':
                attrs = entity['attributes']
                door = {
                    'id': entity_id,
                    'name': attrs[2] if len(attrs) > 2 else 'Door',
                    'width': 0.9,  # Default width
                    'height': 2.1,  # Default height
                    'geometry': self._get_entity_geometry(entity_id, ifc_data)
                }
                doors.append(door)
        
        return doors
    
    def _extract_ifc_windows(self, ifc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract window definitions from IFC"""
        windows = []
        
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] == 'IFCWINDOW':
                attrs = entity['attributes']
                window = {
                    'id': entity_id,
                    'name': attrs[2] if len(attrs) > 2 else 'Window',
                    'width': 1.2,  # Default width
                    'height': 1.5,  # Default height
                    'geometry': self._get_entity_geometry(entity_id, ifc_data)
                }
                windows.append(window)
        
        return windows
    
    def _extract_ifc_furniture(self, ifc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract furniture from IFC"""
        furniture = []
        
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] == 'IFCFURNISHINGELEMENT':
                attrs = entity['attributes']
                item = {
                    'id': entity_id,
                    'name': attrs[2] if len(attrs) > 2 else 'Furniture',
                    'type': attrs[4] if len(attrs) > 4 else 'generic',
                    'geometry': self._get_entity_geometry(entity_id, ifc_data)
                }
                furniture.append(item)
        
        return furniture
    
    def _extract_building_info(self, ifc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract building information from IFC"""
        building_info = {
            'name': 'Unknown Building',
            'address': '',
            'project': '',
            'floors': []
        }
        
        # Find building entity
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] == 'IFCBUILDING':
                attrs = entity['attributes']
                building_info['name'] = attrs[2] if len(attrs) > 2 else 'Unknown Building'
                break
        
        # Find building storeys
        for entity_id, entity in ifc_data['entities'].items():
            if entity['type'] == 'IFCBUILDINGSTOREY':
                attrs = entity['attributes']
                storey = {
                    'id': entity_id,
                    'name': attrs[2] if len(attrs) > 2 else 'Floor',
                    'elevation': attrs[9] if len(attrs) > 9 else 0
                }
                building_info['floors'].append(storey)
        
        return building_info
    
    def _extract_ifc_metadata(self, ifc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from IFC file"""
        metadata = {
            'ifc_version': self.ifc_version,
            'created_date': datetime.now().isoformat(),
            'units': 'meters',
            'coordinate_system': 'local'
        }
        
        # Extract from header if available
        # This is simplified - real implementation would parse header section
        
        return metadata
    
    def _convert_ifc_to_standard(self, ifc_floor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert IFC data to standard floor plan format"""
        entities = []
        
        # Convert walls
        for wall in ifc_floor_data.get('walls', []):
            # Create line entities for walls
            # This is simplified - real implementation would use actual geometry
            entity = {
                'type': 'line',
                'layer': 'walls',
                'color': [0, 0, 0],  # Black
                'geometry': {
                    'type': 'line',
                    'start': {'x': 0, 'y': 0, 'z': 0},
                    'end': {'x': 10, 'y': 0, 'z': 0},
                    'length': 10,
                    'angle': 0
                }
            }
            entities.append(entity)
        
        # Convert spaces to restricted areas or open spaces
        # This would involve complex geometry conversion
        
        return {
            'entities': entities,
            'metadata': ifc_floor_data['metadata'],
            'bounds': self._calculate_bounds(entities)
        }
    
    def _calculate_bounds(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate bounding box of entities"""
        if not entities:
            return {'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 100}
        
        # Simplified bounds calculation
        return {'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 100}
    
    def _create_ifc_header(self) -> str:
        """Create IFC file header"""
        return """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Floor Plan with Ilots'),'2;1');
FILE_NAME('floor_plan.ifc','%s','',(''),'',' ','');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
""" % datetime.now().isoformat()
    
    def _create_ifc_project(self, guid: str, floor_plan_data: Dict[str, Any]) -> str:
        """Create IFC project entity"""
        return f"#1= IFCPROJECT('{guid}',$,'Floor Plan Project',$,$,$,$,$,#2);\n"
    
    def _create_ifc_site(self, site_guid: str, project_guid: str) -> str:
        """Create IFC site entity"""
        return f"#10= IFCSITE('{site_guid}',$,'Site',$,$,$,$,$,.ELEMENT.,$,$,$,$,$);\n"
    
    def _create_ifc_building(self, building_guid: str, site_guid: str) -> str:
        """Create IFC building entity"""
        return f"#20= IFCBUILDING('{building_guid}',$,'Building',$,$,$,$,$,.ELEMENT.,$,$,$);\n"
    
    def _create_ifc_storey(self, storey_guid: str, building_guid: str) -> str:
        """Create IFC building storey entity"""
        return f"#30= IFCBUILDINGSTOREY('{storey_guid}',$,'Ground Floor',$,$,$,$,$,.ELEMENT.,0.);\n"
    
    def _create_ifc_wall(self, wall_guid: str, wall_data: Dict[str, Any], 
                        storey_guid: str) -> str:
        """Create IFC wall entity"""
        # Simplified wall creation
        return f"#100= IFCWALLSTANDARDCASE('{wall_guid}',$,'Wall',$,$,$,$,$,.BOTH.);\n"
    
    def _create_ifc_furniture(self, furniture_guid: str, ilot_data: Dict[str, Any],
                            storey_guid: str) -> str:
        """Create IFC furniture element for îlot"""
        name = f"Ilot_{ilot_data.get('id', '')}"
        return f"#200= IFCFURNISHINGELEMENT('{furniture_guid}',$,'{name}',$,$,$,$,$);\n"
    
    def _generate_guid(self) -> str:
        """Generate IFC compliant GUID"""
        return str(uuid.uuid4()).replace('-', '').upper()[:22]


class RevitConnector:
    """Connector for Autodesk Revit via API"""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """Get Revit project information"""
        try:
            response = requests.get(
                f"{self.api_endpoint}/projects/{project_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting Revit project info: {str(e)}")
            raise
    
    def import_floor_plan(self, project_id: str, level_name: str) -> Dict[str, Any]:
        """Import floor plan from Revit project"""
        try:
            # Get level information
            level_info = self._get_level_info(project_id, level_name)
            
            # Get elements on level
            elements = self._get_level_elements(project_id, level_info['id'])
            
            # Convert to standard format
            return self._convert_revit_to_standard(elements, level_info)
            
        except Exception as e:
            logger.error(f"Error importing from Revit: {str(e)}")
            raise
    
    def export_ilots(self, project_id: str, level_name: str, 
                    ilot_results: List[Dict[str, Any]]) -> bool:
        """Export îlots back to Revit"""
        try:
            level_info = self._get_level_info(project_id, level_name)
            
            # Create family instances for îlots
            for ilot in ilot_results:
                self._create_family_instance(project_id, level_info['id'], ilot)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Revit: {str(e)}")
            return False
    
    def _get_level_info(self, project_id: str, level_name: str) -> Dict[str, Any]:
        """Get level information from Revit"""
        response = requests.get(
            f"{self.api_endpoint}/projects/{project_id}/levels",
            headers=self.headers
        )
        response.raise_for_status()
        
        levels = response.json()
        for level in levels:
            if level['name'] == level_name:
                return level
        
        raise ValueError(f"Level '{level_name}' not found")
    
    def _get_level_elements(self, project_id: str, level_id: str) -> List[Dict[str, Any]]:
        """Get elements on a specific level"""
        response = requests.get(
            f"{self.api_endpoint}/projects/{project_id}/levels/{level_id}/elements",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def _convert_revit_to_standard(self, elements: List[Dict[str, Any]], 
                                  level_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Revit elements to standard format"""
        entities = []
        
        for element in elements:
            if element['category'] == 'Walls':
                entity = self._convert_revit_wall(element)
                entities.append(entity)
            elif element['category'] == 'Doors':
                entity = self._convert_revit_door(element)
                entities.append(entity)
            # Add more element types as needed
        
        return {
            'entities': entities,
            'metadata': {
                'source': 'revit',
                'project_id': level_info.get('project_id'),
                'level_name': level_info.get('name'),
                'elevation': level_info.get('elevation', 0)
            },
            'bounds': self._calculate_bounds_from_elements(entities)
        }
    
    def _convert_revit_wall(self, wall: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Revit wall to standard format"""
        # Extract wall curve/line
        curve = wall.get('location', {}).get('curve', {})
        
        return {
            'type': 'line',
            'layer': 'walls',
            'color': [0, 0, 0],
            'geometry': {
                'type': 'line',
                'start': curve.get('start', {'x': 0, 'y': 0, 'z': 0}),
                'end': curve.get('end', {'x': 0, 'y': 0, 'z': 0}),
                'length': curve.get('length', 0),
                'angle': 0
            },
            'properties': {
                'revit_id': wall.get('id'),
                'wall_type': wall.get('type_name'),
                'thickness': wall.get('width', 0.2)
            }
        }
    
    def _convert_revit_door(self, door: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Revit door to standard format"""
        location = door.get('location', {}).get('point', {})
        
        return {
            'type': 'point',
            'layer': 'doors',
            'color': [255, 0, 0],
            'geometry': {
                'type': 'point',
                'point': location,
                'width': door.get('width', 0.9),
                'height': door.get('height', 2.1)
            },
            'properties': {
                'revit_id': door.get('id'),
                'door_type': door.get('type_name')
            }
        }
    
    def _calculate_bounds_from_elements(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate bounds from Revit elements"""
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for entity in entities:
            geom = entity.get('geometry', {})
            
            if geom.get('type') == 'line':
                start = geom.get('start', {})
                end = geom.get('end', {})
                
                min_x = min(min_x, start.get('x', 0), end.get('x', 0))
                min_y = min(min_y, start.get('y', 0), end.get('y', 0))
                max_x = max(max_x, start.get('x', 0), end.get('x', 0))
                max_y = max(max_y, start.get('y', 0), end.get('y', 0))
        
        return {
            'min_x': min_x if min_x != float('inf') else 0,
            'min_y': min_y if min_y != float('inf') else 0,
            'max_x': max_x if max_x != float('-inf') else 100,
            'max_y': max_y if max_y != float('-inf') else 100
        }
    
    def _create_family_instance(self, project_id: str, level_id: str, 
                              ilot: Dict[str, Any]) -> str:
        """Create family instance in Revit for îlot"""
        family_type = self._get_ilot_family_type(ilot['size_category'])
        
        instance_data = {
            'family_type_id': family_type,
            'level_id': level_id,
            'location': {
                'point': ilot['position'],
                'rotation': ilot.get('rotation', 0)
            },
            'parameters': {
                'width': ilot['dimensions']['width'],
                'height': ilot['dimensions']['height'],
                'name': f"Ilot_{ilot['id']}"
            }
        }
        
        response = requests.post(
            f"{self.api_endpoint}/projects/{project_id}/instances",
            headers=self.headers,
            json=instance_data
        )
        response.raise_for_status()
        
        return response.json()['id']
    
    def _get_ilot_family_type(self, size_category: str) -> str:
        """Map îlot size category to Revit family type"""
        family_mapping = {
            'small': 'Workstation_Small',
            'medium': 'Workstation_Medium',
            'large': 'Workstation_Large'
        }
        return family_mapping.get(size_category, 'Generic_Furniture')


class ArchiCADConnector:
    """Connector for GRAPHISOFT ArchiCAD"""
    
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token
        self.headers = {
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json'
        }
    
    def import_floor_plan(self, project_path: str, story_name: str) -> Dict[str, Any]:
        """Import floor plan from ArchiCAD project"""
        try:
            # Connect to ArchiCAD API
            session = self._create_session(project_path)
            
            # Get story information
            story_info = self._get_story_info(session, story_name)
            
            # Get elements on story
            elements = self._get_story_elements(session, story_info['guid'])
            
            # Convert to standard format
            floor_plan = self._convert_archicad_to_standard(elements, story_info)
            
            # Close session
            self._close_session(session)
            
            return floor_plan
            
        except Exception as e:
            logger.error(f"Error importing from ArchiCAD: {str(e)}")
            raise
    
    def _create_session(self, project_path: str) -> str:
        """Create ArchiCAD API session"""
        response = requests.post(
            f"{self.api_url}/sessions",
            headers=self.headers,
            json={'project_path': project_path}
        )
        response.raise_for_status()
        return response.json()['session_id']
    
    def _close_session(self, session_id: str):
        """Close ArchiCAD API session"""
        requests.delete(
            f"{self.api_url}/sessions/{session_id}",
            headers=self.headers
        )
    
    def _get_story_info(self, session_id: str, story_name: str) -> Dict[str, Any]:
        """Get story information"""
        response = requests.get(
            f"{self.api_url}/sessions/{session_id}/stories",
            headers=self.headers
        )
        response.raise_for_status()
        
        stories = response.json()
        for story in stories:
            if story['name'] == story_name:
                return story
        
        raise ValueError(f"Story '{story_name}' not found")
    
    def _get_story_elements(self, session_id: str, story_guid: str) -> List[Dict[str, Any]]:
        """Get elements on a story"""
        response = requests.get(
            f"{self.api_url}/sessions/{session_id}/stories/{story_guid}/elements",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def _convert_archicad_to_standard(self, elements: List[Dict[str, Any]], 
                                    story_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ArchiCAD elements to standard format"""
        entities = []
        
        for element in elements:
            element_type = element.get('type')
            
            if element_type == 'Wall':
                entity = self._convert_archicad_wall(element)
                entities.append(entity)
            elif element_type == 'Door':
                entity = self._convert_archicad_door(element)
                entities.append(entity)
            elif element_type == 'Zone':
                entity = self._convert_archicad_zone(element)
                entities.append(entity)
        
        return {
            'entities': entities,
            'metadata': {
                'source': 'archicad',
                'story_name': story_info.get('name'),
                'story_level': story_info.get('level', 0)
            },
            'bounds': self._calculate_bounds_from_elements(entities)
        }
    
    def _convert_archicad_wall(self, wall: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ArchiCAD wall to standard format"""
        geometry = wall.get('geometry', {})
        
        return {
            'type': 'polyline',
            'layer': 'walls',
            'color': [0, 0, 0],
            'geometry': {
                'type': 'polyline',
                'points': geometry.get('reference_line', []),
                'closed': False,
                'thickness': wall.get('thickness', 0.2)
            },
            'properties': {
                'archicad_guid': wall.get('guid'),
                'wall_type': wall.get('type_name')
            }
        }
    
    def _convert_archicad_door(self, door: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ArchiCAD door to standard format"""
        return {
            'type': 'polygon',
            'layer': 'doors',
            'color': [255, 0, 0],
            'geometry': {
                'type': 'polygon',
                'points': door.get('geometry', {}).get('polygon', []),
                'width': door.get('width', 0.9)
            },
            'properties': {
                'archicad_guid': door.get('guid'),
                'door_type': door.get('type_name')
            }
        }
    
    def _convert_archicad_zone(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ArchiCAD zone to standard format"""
        zone_type = zone.get('zone_category', 'open')
        
        # Map zone categories to colors
        if 'restricted' in zone_type.lower():
            color = [0, 0, 255]  # Blue for restricted
            layer = 'restricted'
        else:
            color = [255, 255, 255]  # White for open
            layer = 'spaces'
        
        return {
            'type': 'polygon',
            'layer': layer,
            'color': color,
            'geometry': {
                'type': 'polygon',
                'points': zone.get('geometry', {}).get('polygon', []),
                'area': zone.get('area', 0)
            },
            'properties': {
                'archicad_guid': zone.get('guid'),
                'zone_name': zone.get('name'),
                'zone_category': zone_type
            }
        }


class BIMIntegrationManager:
    """Manager for BIM system integrations"""
    
    def __init__(self):
        self.connectors = {
            'ifc': IFCConnector(),
            'revit': None,  # Initialize with API credentials
            'archicad': None  # Initialize with API credentials
        }
        
        self.supported_formats = ['ifc', 'rvt', 'pln', 'dwg', 'dxf']
    
    def set_revit_connector(self, api_endpoint: str, api_key: str):
        """Configure Revit connector"""
        self.connectors['revit'] = RevitConnector(api_endpoint, api_key)
    
    def set_archicad_connector(self, api_url: str, api_token: str):
        """Configure ArchiCAD connector"""
        self.connectors['archicad'] = ArchiCADConnector(api_url, api_token)
    
    def import_from_bim(self, file_path: str, bim_system: str = 'auto',
                       **kwargs) -> Dict[str, Any]:
        """Import floor plan from BIM system"""
        logger.info(f"Importing from BIM: {file_path}")
        
        try:
            # Auto-detect BIM system if needed
            if bim_system == 'auto':
                bim_system = self._detect_bim_system(file_path)
            
            # Select appropriate connector
            connector = self.connectors.get(bim_system)
            if not connector:
                raise ValueError(f"Unsupported BIM system: {bim_system}")
            
            # Import using connector
            if bim_system == 'ifc':
                return connector.import_ifc(file_path)
            elif bim_system == 'revit':
                return connector.import_floor_plan(
                    kwargs.get('project_id'),
                    kwargs.get('level_name', 'Level 1')
                )
            elif bim_system == 'archicad':
                return connector.import_floor_plan(
                    file_path,
                    kwargs.get('story_name', 'Ground Floor')
                )
            else:
                raise ValueError(f"Unknown BIM system: {bim_system}")
            
        except Exception as e:
            logger.error(f"Error importing from BIM: {str(e)}")
            raise
    
    def export_to_bim(self, floor_plan_data: Dict[str, Any],
                     ilot_results: List[Dict[str, Any]],
                     output_path: str,
                     bim_system: str = 'ifc',
                     **kwargs) -> str:
        """Export floor plan and îlots to BIM format"""
        logger.info(f"Exporting to BIM format: {bim_system}")
        
        try:
            connector = self.connectors.get(bim_system)
            if not connector:
                raise ValueError(f"Unsupported BIM system: {bim_system}")
            
            if bim_system == 'ifc':
                return connector.export_ifc(floor_plan_data, ilot_results, output_path)
            elif bim_system == 'revit':
                success = connector.export_ilots(
                    kwargs.get('project_id'),
                    kwargs.get('level_name', 'Level 1'),
                    ilot_results
                )
                return output_path if success else None
            else:
                raise ValueError(f"Export not implemented for: {bim_system}")
            
        except Exception as e:
            logger.error(f"Error exporting to BIM: {str(e)}")
            raise
    
    def _detect_bim_system(self, file_path: str) -> str:
        """Auto-detect BIM system from file"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.ifc':
            return 'ifc'
        elif file_ext == '.rvt':
            return 'revit'
        elif file_ext == '.pln':
            return 'archicad'
        else:
            # Try to detect from file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)
                
                if 'ISO-10303-21' in content or 'IFC' in content:
                    return 'ifc'
            
            return 'unknown'
    
    def validate_bim_export(self, exported_file: str) -> Dict[str, Any]:
        """Validate exported BIM file"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Basic file validation
            if not Path(exported_file).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            file_size = Path(exported_file).stat().st_size
            validation_result['statistics']['file_size'] = file_size
            
            # Format-specific validation
            if exported_file.endswith('.ifc'):
                self._validate_ifc_file(exported_file, validation_result)
            
            return validation_result
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _validate_ifc_file(self, file_path: str, result: Dict[str, Any]):
        """Validate IFC file structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check basic structure
            if 'ISO-10303-21' not in content:
                result['errors'].append("Missing IFC header")
                result['valid'] = False
            
            if 'HEADER;' not in content:
                result['errors'].append("Missing HEADER section")
                result['valid'] = False
            
            if 'DATA;' not in content:
                result['errors'].append("Missing DATA section")
                result['valid'] = False
            
            # Count entities
            entity_count = content.count('#')
            result['statistics']['entity_count'] = entity_count
            
            if entity_count == 0:
                result['warnings'].append("No entities found in file")
            
        except Exception as e:
            result['errors'].append(f"File parsing error: {str(e)}")
            result['valid'] = False