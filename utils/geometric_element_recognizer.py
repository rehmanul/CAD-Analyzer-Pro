"""
Geometric Element Recognizer for Phase 1
Advanced recognition of walls, doors, windows, and architectural elements
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import unary_union, linemerge
import cv2
import logging
from dataclasses import dataclass
from enum import Enum
from utils.enhanced_cad_parser import CADElement, ElementType

class WallType(Enum):
    EXTERIOR = "exterior"
    INTERIOR = "interior"
    LOAD_BEARING = "load_bearing"
    PARTITION = "partition"

class OpeningType(Enum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"
    ARCHWAY = "archway"

@dataclass
class RecognizedWall:
    """Enhanced wall representation with recognition data"""
    geometry: LineString
    wall_type: WallType
    thickness: float
    length: float
    connected_walls: List[int]  # Indices of connected walls
    openings: List['RecognizedOpening']
    material: str = "concrete"
    is_load_bearing: bool = False

@dataclass
class RecognizedOpening:
    """Enhanced opening representation"""
    geometry: Any  # Point or LineString
    opening_type: OpeningType
    width: float
    height: float = 2100  # Default door/window height
    wall_index: Optional[int] = None
    swing_direction: Optional[str] = None  # For doors

class GeometricElementRecognizer:
    """
    Advanced geometric element recognition for CAD floor plans
    Recognizes walls, doors, windows with proper thickness, connectivity, and properties
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Recognition parameters
        self.wall_thickness_range = (80, 400)  # mm - typical wall thickness range
        self.door_width_range = (600, 1200)  # mm - standard door widths
        self.window_width_range = (400, 3000)  # mm - typical window widths
        self.min_wall_length = 200  # mm - minimum wall length to consider
        self.connection_tolerance = 50  # mm - tolerance for wall connections
        self.parallel_tolerance = 5  # degrees - tolerance for parallel wall detection
        
        # Opening detection parameters
        self.opening_detection_distance = 100  # mm - max distance from wall for opening
        self.door_arc_radius_range = (300, 800)  # mm - door swing arc radius range

    def recognize_elements(self, cad_elements: List[CADElement]) -> Dict[str, List]:
        """
        Main recognition method that processes CAD elements and returns recognized elements
        """
        try:
            # Separate elements by type for processing
            line_elements = [elem for elem in cad_elements 
                           if isinstance(elem.geometry, LineString)]
            arc_elements = [elem for elem in cad_elements 
                          if hasattr(elem.geometry, 'buffer') and 
                          elem.properties.get('is_arc', False)]
            
            # Recognize walls with proper thickness and connectivity
            recognized_walls = self._recognize_walls(line_elements)
            
            # Recognize doors and windows
            recognized_openings = self._recognize_openings(line_elements + arc_elements, recognized_walls)
            
            # Detect room boundaries from wall structure
            room_boundaries = self._detect_room_boundaries(recognized_walls)
            
            # Convert back to CADElement format
            wall_elements = self._convert_walls_to_cad_elements(recognized_walls)
            opening_elements = self._convert_openings_to_cad_elements(recognized_openings)
            room_elements = self._convert_rooms_to_cad_elements(room_boundaries)
            
            return {
                'walls': wall_elements,
                'doors': [elem for elem in opening_elements if elem.element_type == ElementType.DOOR],
                'windows': [elem for elem in opening_elements if elem.element_type == ElementType.WINDOW],
                'rooms': room_elements,
                'recognition_stats': self._generate_recognition_stats(recognized_walls, recognized_openings)
            }
            
        except Exception as e:
            self.logger.error(f"Error in element recognition: {str(e)}")
            return {'walls': [], 'doors': [], 'windows': [], 'rooms': [], 'recognition_stats': {}}

    def _recognize_walls(self, line_elements: List[CADElement]) -> List[RecognizedWall]:
        """Recognize walls with proper thickness detection and connectivity analysis"""
        recognized_walls = []
        
        # Filter lines that could be walls (minimum length)
        potential_walls = [elem for elem in line_elements 
                          if elem.geometry.length > self.min_wall_length]
        
        if not potential_walls:
            return recognized_walls
        
        # Group parallel lines that could form wall pairs
        wall_pairs = self._find_wall_pairs(potential_walls)
        
        # Process wall pairs to determine thickness
        for pair in wall_pairs:
            wall = self._process_wall_pair(pair)
            if wall:
                recognized_walls.append(wall)
        
        # Process single lines that might be wall centerlines
        used_elements = set()
        for pair in wall_pairs:
            used_elements.update(id(elem) for elem in pair)
        
        single_walls = [elem for elem in potential_walls if id(elem) not in used_elements]
        for elem in single_walls:
            wall = self._process_single_wall(elem)
            if wall:
                recognized_walls.append(wall)
        
        # Analyze wall connectivity
        self._analyze_wall_connectivity(recognized_walls)
        
        return recognized_walls

    def _find_wall_pairs(self, line_elements: List[CADElement]) -> List[List[CADElement]]:
        """Find pairs of parallel lines that could represent wall boundaries"""
        pairs = []
        processed = set()
        
        for i, elem1 in enumerate(line_elements):
            if id(elem1) in processed:
                continue
                
            for j, elem2 in enumerate(line_elements[i+1:], i+1):
                if id(elem2) in processed:
                    continue
                
                # Check if lines are parallel and at appropriate distance
                if self._are_parallel_walls(elem1.geometry, elem2.geometry):
                    distance = elem1.geometry.distance(elem2.geometry)
                    
                    if self.wall_thickness_range[0] <= distance <= self.wall_thickness_range[1]:
                        pairs.append([elem1, elem2])
                        processed.add(id(elem1))
                        processed.add(id(elem2))
                        break
        
        return pairs

    def _are_parallel_walls(self, line1: LineString, line2: LineString) -> bool:
        """Check if two lines are parallel within tolerance"""
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)
        
        if len(coords1) < 2 or len(coords2) < 2:
            return False
        
        # Calculate direction vectors
        v1 = np.array(coords1[-1]) - np.array(coords1[0])
        v2 = np.array(coords2[-1]) - np.array(coords2[0])
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate angle between vectors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(np.abs(dot_product))  # Use absolute value for parallel check
        angle_degrees = np.degrees(angle)
        
        return angle_degrees <= self.parallel_tolerance

    def _process_wall_pair(self, pair: List[CADElement]) -> Optional[RecognizedWall]:
        """Process a pair of parallel lines to create a wall"""
        if len(pair) != 2:
            return None
        
        line1, line2 = pair[0].geometry, pair[1].geometry
        
        # Calculate wall properties
        thickness = line1.distance(line2)
        length = max(line1.length, line2.length)
        
        # Create centerline as wall geometry
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)
        
        if len(coords1) >= 2 and len(coords2) >= 2:
            # Calculate centerline points
            center_start = ((coords1[0][0] + coords2[0][0]) / 2, 
                           (coords1[0][1] + coords2[0][1]) / 2)
            center_end = ((coords1[-1][0] + coords2[-1][0]) / 2, 
                         (coords1[-1][1] + coords2[-1][1]) / 2)
            
            centerline = LineString([center_start, center_end])
            
            # Determine wall type based on thickness
            wall_type = self._classify_wall_type(thickness, length)
            
            return RecognizedWall(
                geometry=centerline,
                wall_type=wall_type,
                thickness=thickness,
                length=length,
                connected_walls=[],
                openings=[],
                is_load_bearing=thickness > 200  # Thick walls are likely load-bearing
            )
        
        return None

    def _process_single_wall(self, element: CADElement) -> Optional[RecognizedWall]:
        """Process a single line as a wall centerline"""
        geometry = element.geometry
        length = geometry.length
        
        # Estimate thickness from layer or use default
        estimated_thickness = self._estimate_wall_thickness(element)
        
        # Determine wall type
        wall_type = self._classify_wall_type(estimated_thickness, length)
        
        return RecognizedWall(
            geometry=geometry,
            wall_type=wall_type,
            thickness=estimated_thickness,
            length=length,
            connected_walls=[],
            openings=[],
            is_load_bearing=estimated_thickness > 200
        )

    def _classify_wall_type(self, thickness: float, length: float) -> WallType:
        """Classify wall type based on thickness and length"""
        if thickness > 250:
            return WallType.LOAD_BEARING
        elif thickness > 150:
            return WallType.EXTERIOR
        elif length > 3000:  # Long walls are often exterior
            return WallType.EXTERIOR
        else:
            return WallType.INTERIOR

    def _estimate_wall_thickness(self, element: CADElement) -> float:
        """Estimate wall thickness from element properties"""
        # Check layer name for thickness hints
        layer = element.layer.lower()
        
        if 'exterior' in layer or 'external' in layer:
            return 250  # Typical exterior wall
        elif 'interior' in layer or 'partition' in layer:
            return 100  # Typical interior wall
        elif 'load' in layer or 'bearing' in layer:
            return 300  # Load-bearing wall
        else:
            return 150  # Default thickness

    def _analyze_wall_connectivity(self, walls: List[RecognizedWall]):
        """Analyze connectivity between walls"""
        for i, wall1 in enumerate(walls):
            wall1_coords = list(wall1.geometry.coords)
            if len(wall1_coords) < 2:
                continue
                
            wall1_start = Point(wall1_coords[0])
            wall1_end = Point(wall1_coords[-1])
            
            for j, wall2 in enumerate(walls):
                if i == j:
                    continue
                    
                wall2_coords = list(wall2.geometry.coords)
                if len(wall2_coords) < 2:
                    continue
                
                wall2_start = Point(wall2_coords[0])
                wall2_end = Point(wall2_coords[-1])
                
                # Check if walls are connected at endpoints
                connections = [
                    wall1_start.distance(wall2_start),
                    wall1_start.distance(wall2_end),
                    wall1_end.distance(wall2_start),
                    wall1_end.distance(wall2_end)
                ]
                
                if min(connections) <= self.connection_tolerance:
                    if j not in wall1.connected_walls:
                        wall1.connected_walls.append(j)

    def _recognize_openings(self, elements: List[CADElement], 
                          walls: List[RecognizedWall]) -> List[RecognizedOpening]:
        """Recognize doors and windows in relation to walls"""
        recognized_openings = []
        
        # Analyze different types of opening representations
        
        # 1. Arc elements (door swings)
        arc_elements = [elem for elem in elements 
                       if elem.properties.get('is_arc', False)]
        door_openings = self._recognize_door_arcs(arc_elements, walls)
        recognized_openings.extend(door_openings)
        
        # 2. Short line segments in walls (openings)
        line_elements = [elem for elem in elements 
                        if isinstance(elem.geometry, LineString)]
        opening_lines = self._recognize_opening_lines(line_elements, walls)
        recognized_openings.extend(opening_lines)
        
        # 3. Gaps in wall continuity
        gap_openings = self._detect_wall_gaps(walls)
        recognized_openings.extend(gap_openings)
        
        return recognized_openings

    def _recognize_door_arcs(self, arc_elements: List[CADElement], 
                           walls: List[RecognizedWall]) -> List[RecognizedOpening]:
        """Recognize doors from arc elements (door swings)"""
        door_openings = []
        
        for arc_elem in arc_elements:
            radius = arc_elem.properties.get('radius', 0)
            
            # Check if radius is in door range
            if self.door_arc_radius_range[0] <= radius <= self.door_arc_radius_range[1]:
                # Find the nearest wall
                nearest_wall_idx = self._find_nearest_wall(arc_elem.geometry, walls)
                
                if nearest_wall_idx is not None:
                    # Calculate door width (typically equals radius for 90-degree swing)
                    door_width = radius
                    
                    # Get center point of arc as door location
                    if hasattr(arc_elem.geometry, 'centroid'):
                        door_location = arc_elem.geometry.centroid
                    else:
                        door_location = arc_elem.geometry
                    
                    door_opening = RecognizedOpening(
                        geometry=door_location,
                        opening_type=OpeningType.DOOR,
                        width=door_width,
                        wall_index=nearest_wall_idx,
                        swing_direction=self._determine_swing_direction(arc_elem)
                    )
                    
                    door_openings.append(door_opening)
                    
                    # Add to wall's opening list
                    walls[nearest_wall_idx].openings.append(door_opening)
        
        return door_openings

    def _recognize_opening_lines(self, line_elements: List[CADElement], 
                               walls: List[RecognizedWall]) -> List[RecognizedOpening]:
        """Recognize openings from short line segments"""
        opening_lines = []
        
        for line_elem in line_elements:
            length = line_elem.geometry.length
            
            # Check if line length is in opening range
            if (self.door_width_range[0] <= length <= self.door_width_range[1] or
                self.window_width_range[0] <= length <= self.window_width_range[1]):
                
                # Find nearest wall
                nearest_wall_idx = self._find_nearest_wall(line_elem.geometry, walls)
                
                if nearest_wall_idx is not None:
                    wall = walls[nearest_wall_idx]
                    distance_to_wall = line_elem.geometry.distance(wall.geometry)
                    
                    if distance_to_wall <= self.opening_detection_distance:
                        # Classify as door or window based on length
                        if length <= self.door_width_range[1]:
                            opening_type = OpeningType.DOOR
                        else:
                            opening_type = OpeningType.WINDOW
                        
                        opening = RecognizedOpening(
                            geometry=line_elem.geometry,
                            opening_type=opening_type,
                            width=length,
                            wall_index=nearest_wall_idx
                        )
                        
                        opening_lines.append(opening)
                        wall.openings.append(opening)
        
        return opening_lines

    def _detect_wall_gaps(self, walls: List[RecognizedWall]) -> List[RecognizedOpening]:
        """Detect openings from gaps in wall continuity"""
        gap_openings = []
        
        # This is a simplified implementation
        # In a full implementation, you would analyze wall segments to find gaps
        
        for i, wall in enumerate(walls):
            # Look for significant gaps in connected walls
            if len(wall.connected_walls) < 2:  # Wall doesn't connect on both ends
                # Potentially has an opening
                coords = list(wall.geometry.coords)
                if len(coords) >= 2:
                    # Create a potential opening at one end
                    end_point = Point(coords[-1])
                    
                    gap_opening = RecognizedOpening(
                        geometry=end_point,
                        opening_type=OpeningType.OPENING,
                        width=800,  # Default opening width
                        wall_index=i
                    )
                    
                    gap_openings.append(gap_opening)
                    wall.openings.append(gap_opening)
        
        return gap_openings

    def _find_nearest_wall(self, geometry, walls: List[RecognizedWall]) -> Optional[int]:
        """Find the index of the nearest wall to the given geometry"""
        if not walls:
            return None
        
        min_distance = float('inf')
        nearest_wall_idx = None
        
        for i, wall in enumerate(walls):
            distance = geometry.distance(wall.geometry)
            if distance < min_distance:
                min_distance = distance
                nearest_wall_idx = i
        
        return nearest_wall_idx if min_distance <= self.opening_detection_distance else None

    def _determine_swing_direction(self, arc_element: CADElement) -> str:
        """Determine door swing direction from arc element"""
        # This is a simplified implementation
        # In practice, you would analyze the arc's start and end angles
        return "right"  # Default swing direction

    def _detect_room_boundaries(self, walls: List[RecognizedWall]) -> List[Polygon]:
        """Detect room boundaries from wall structure using geometric analysis"""
        room_boundaries = []
        
        if not walls:
            return room_boundaries
        
        try:
            # Create a simplified approach: use wall bounds to create rooms
            wall_geometries = [wall.geometry for wall in walls]
            
            if wall_geometries:
                # Get overall bounds
                all_walls = unary_union(wall_geometries)
                bounds = all_walls.bounds
                
                # Create a simple rectangular room
                room_polygon = Polygon([
                    (bounds[0] + 200, bounds[1] + 200),  # Add margin for wall thickness
                    (bounds[2] - 200, bounds[1] + 200),
                    (bounds[2] - 200, bounds[3] - 200),
                    (bounds[0] + 200, bounds[3] - 200)
                ])
                
                room_boundaries.append(room_polygon)
                
        except Exception as e:
            self.logger.warning(f"Error detecting room boundaries: {str(e)}")
        
        return room_boundaries

    def _convert_walls_to_cad_elements(self, walls: List[RecognizedWall]) -> List[CADElement]:
        """Convert recognized walls back to CADElement format"""
        cad_elements = []
        
        for i, wall in enumerate(walls):
            properties = {
                'wall_type': wall.wall_type.value,
                'thickness': wall.thickness,
                'length': wall.length,
                'connected_walls': wall.connected_walls,
                'opening_count': len(wall.openings),
                'is_load_bearing': wall.is_load_bearing,
                'material': wall.material,
                'wall_index': i
            }
            
            cad_element = CADElement(
                element_type=ElementType.WALL,
                geometry=wall.geometry,
                properties=properties,
                line_weight=3.0 if wall.is_load_bearing else 2.0
            )
            
            cad_elements.append(cad_element)
        
        return cad_elements

    def _convert_openings_to_cad_elements(self, openings: List[RecognizedOpening]) -> List[CADElement]:
        """Convert recognized openings back to CADElement format"""
        cad_elements = []
        
        for opening in openings:
            element_type = ElementType.DOOR if opening.opening_type in [OpeningType.DOOR, OpeningType.ARCHWAY] else ElementType.WINDOW
            
            properties = {
                'opening_type': opening.opening_type.value,
                'width': opening.width,
                'height': opening.height,
                'wall_index': opening.wall_index,
                'swing_direction': opening.swing_direction
            }
            
            cad_element = CADElement(
                element_type=element_type,
                geometry=opening.geometry,
                properties=properties,
                color="red" if element_type == ElementType.DOOR else "blue"
            )
            
            cad_elements.append(cad_element)
        
        return cad_elements

    def _convert_rooms_to_cad_elements(self, room_boundaries: List[Polygon]) -> List[CADElement]:
        """Convert room boundaries to CADElement format"""
        cad_elements = []
        
        for i, room_polygon in enumerate(room_boundaries):
            properties = {
                'room_id': f'room_{i+1}',
                'area': room_polygon.area,
                'perimeter': room_polygon.length,
                'room_type': 'general',
                'generated_from': 'wall_analysis'
            }
            
            cad_element = CADElement(
                element_type=ElementType.ROOM_BOUNDARY,
                geometry=room_polygon,
                properties=properties
            )
            
            cad_elements.append(cad_element)
        
        return cad_elements

    def _generate_recognition_stats(self, walls: List[RecognizedWall], 
                                  openings: List[RecognizedOpening]) -> Dict[str, Any]:
        """Generate statistics about the recognition process"""
        wall_types = {}
        opening_types = {}
        
        for wall in walls:
            wall_type = wall.wall_type.value
            wall_types[wall_type] = wall_types.get(wall_type, 0) + 1
        
        for opening in openings:
            opening_type = opening.opening_type.value
            opening_types[opening_type] = opening_types.get(opening_type, 0) + 1
        
        total_wall_length = sum(wall.length for wall in walls)
        avg_wall_thickness = np.mean([wall.thickness for wall in walls]) if walls else 0
        
        return {
            'total_walls': len(walls),
            'total_openings': len(openings),
            'wall_types': wall_types,
            'opening_types': opening_types,
            'total_wall_length': total_wall_length,
            'average_wall_thickness': avg_wall_thickness,
            'load_bearing_walls': sum(1 for wall in walls if wall.is_load_bearing),
            'connected_wall_pairs': sum(len(wall.connected_walls) for wall in walls) // 2
        }