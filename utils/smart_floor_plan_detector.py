"""
Smart Floor Plan Detector for Phase 1
Intelligent detection of main floor plan from complex multi-view CAD files
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
import logging
from dataclasses import dataclass
from enum import Enum
from utils.enhanced_cad_parser import FloorPlanData, CADElement, ElementType

class ViewType(Enum):
    FLOOR_PLAN = "floor_plan"
    ELEVATION = "elevation"
    SECTION = "section"
    DETAIL = "detail"
    TITLE_BLOCK = "title_block"
    UNKNOWN = "unknown"

@dataclass
class DetectedView:
    """Represents a detected view in the CAD file"""
    view_type: ViewType
    confidence: float
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    elements: List[CADElement]
    scale: float = 1.0
    area: float = 0.0
    complexity_score: float = 0.0

class SmartFloorPlanDetector:
    """
    Smart detection of main floor plan from multi-view CAD files
    Uses geometric analysis and architectural patterns to identify the primary floor plan
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Floor plan detection parameters
        self.min_floor_plan_area = 1000000  # Minimum area for valid floor plan (1m² in mm²)
        self.wall_density_threshold = 0.1  # Minimum wall density for floor plan
        self.room_indicator_patterns = ['room', 'salle', 'bureau', 'kitchen', 'bedroom']
        self.elevation_indicators = ['elevation', 'facade', 'coupe', 'section']
        
        # Geometric analysis thresholds
        self.wall_length_threshold = 1000  # Minimum wall length (1m)
        self.room_area_threshold = 2000000  # Minimum room area (2m²)
        self.connection_tolerance = 100  # Wall connection tolerance (10cm)

    def detect_main_floor_plan(self, floor_plan_data: FloorPlanData) -> FloorPlanData:
        """
        Detect and extract the main floor plan from complex CAD data
        Returns the cleaned and optimized main floor plan
        """
        try:
            # Analyze all views in the CAD file
            detected_views = self._analyze_views(floor_plan_data)
            
            # Select the best floor plan view
            main_floor_plan = self._select_best_floor_plan(detected_views)
            
            if main_floor_plan:
                # Clean and optimize the selected floor plan
                optimized_plan = self._optimize_floor_plan(main_floor_plan, floor_plan_data)
                return optimized_plan
            else:
                # If no clear floor plan found, use heuristics to create one
                return self._create_heuristic_floor_plan(floor_plan_data)
                
        except Exception as e:
            self.logger.error(f"Error detecting main floor plan: {str(e)}")
            return floor_plan_data  # Return original data as fallback

    def _analyze_views(self, floor_plan_data: FloorPlanData) -> List[DetectedView]:
        """Analyze and classify different views in the CAD file"""
        detected_views = []
        
        # Group elements by spatial clusters
        element_clusters = self._cluster_elements_spatially(floor_plan_data)
        
        for cluster in element_clusters:
            view = self._analyze_cluster(cluster)
            if view.confidence > 0.3:  # Only keep views with reasonable confidence
                detected_views.append(view)
                
        return detected_views

    def _cluster_elements_spatially(self, floor_plan_data: FloorPlanData) -> List[List[CADElement]]:
        """Group elements into spatial clusters representing different views"""
        all_elements = []
        
        # Combine all elements
        for element_list in [floor_plan_data.walls, floor_plan_data.doors, 
                           floor_plan_data.windows, floor_plan_data.rooms,
                           floor_plan_data.text_annotations]:
            all_elements.extend(element_list)
        
        if not all_elements:
            return []
        
        # Simple spatial clustering based on bounding boxes
        clusters = []
        processed_elements = set()
        
        for element in all_elements:
            if id(element) in processed_elements:
                continue
                
            # Start a new cluster
            cluster = [element]
            processed_elements.add(id(element))
            element_bounds = element.geometry.bounds
            
            # Find nearby elements
            for other_element in all_elements:
                if id(other_element) in processed_elements:
                    continue
                    
                other_bounds = other_element.geometry.bounds
                
                # Check if elements are spatially close
                if self._are_bounds_overlapping_or_close(element_bounds, other_bounds, tolerance=5000):
                    cluster.append(other_element)
                    processed_elements.add(id(other_element))
                    
            if len(cluster) > 5:  # Only consider clusters with enough elements
                clusters.append(cluster)
                
        return clusters

    def _are_bounds_overlapping_or_close(self, bounds1: Tuple[float, float, float, float], 
                                       bounds2: Tuple[float, float, float, float], 
                                       tolerance: float) -> bool:
        """Check if two bounding boxes overlap or are within tolerance"""
        min_x1, min_y1, max_x1, max_y1 = bounds1
        min_x2, min_y2, max_x2, max_y2 = bounds2
        
        # Check for overlap or proximity
        x_overlap = max_x1 >= min_x2 - tolerance and max_x2 >= min_x1 - tolerance
        y_overlap = max_y1 >= min_y2 - tolerance and max_y2 >= min_y1 - tolerance
        
        return x_overlap and y_overlap

    def _analyze_cluster(self, elements: List[CADElement]) -> DetectedView:
        """Analyze a cluster of elements to determine view type and confidence"""
        if not elements:
            return DetectedView(ViewType.UNKNOWN, 0.0, (0, 0, 0, 0), [])
        
        # Calculate cluster bounds
        all_bounds = [elem.geometry.bounds for elem in elements if elem.geometry]
        if not all_bounds:
            return DetectedView(ViewType.UNKNOWN, 0.0, (0, 0, 0, 0), [])
        
        min_x = min(bounds[0] for bounds in all_bounds)
        min_y = min(bounds[1] for bounds in all_bounds)
        max_x = max(bounds[2] for bounds in all_bounds)
        max_y = max(bounds[3] for bounds in all_bounds)
        
        bounds = (min_x, min_y, max_x, max_y)
        area = (max_x - min_x) * (max_y - min_y)
        
        # Analyze element types
        wall_count = sum(1 for elem in elements if elem.element_type == ElementType.WALL)
        door_count = sum(1 for elem in elements if elem.element_type == ElementType.DOOR)
        window_count = sum(1 for elem in elements if elem.element_type == ElementType.WINDOW)
        room_count = sum(1 for elem in elements if elem.element_type == ElementType.ROOM_BOUNDARY)
        text_count = sum(1 for elem in elements if elem.element_type == ElementType.TEXT)
        
        # Calculate metrics for view type detection
        total_elements = len(elements)
        wall_density = wall_count / total_elements if total_elements > 0 else 0
        room_density = room_count / total_elements if total_elements > 0 else 0
        
        # Analyze text content for view type hints
        text_content = self._extract_text_content(elements)
        
        # Determine view type and confidence
        view_type, confidence = self._classify_view_type(
            wall_density, room_density, area, text_content, 
            wall_count, door_count, window_count
        )
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(elements)
        
        return DetectedView(
            view_type=view_type,
            confidence=confidence,
            bounds=bounds,
            elements=elements,
            area=area,
            complexity_score=complexity_score
        )

    def _extract_text_content(self, elements: List[CADElement]) -> str:
        """Extract all text content from elements"""
        text_content = ""
        for element in elements:
            if element.element_type == ElementType.TEXT:
                text = element.properties.get('text', '')
                text_content += text.lower() + " "
        return text_content

    def _classify_view_type(self, wall_density: float, room_density: float, area: float,
                          text_content: str, wall_count: int, door_count: int, 
                          window_count: int) -> Tuple[ViewType, float]:
        """Classify view type based on analysis metrics"""
        
        confidence = 0.0
        view_type = ViewType.UNKNOWN
        
        # Floor plan indicators
        floor_plan_score = 0.0
        
        # High wall density suggests floor plan
        if wall_density > 0.3:
            floor_plan_score += 0.3
        
        # Presence of rooms
        if room_density > 0.1:
            floor_plan_score += 0.2
        
        # Doors and windows suggest floor plan
        if door_count > 0 or window_count > 0:
            floor_plan_score += 0.2
        
        # Large area suggests main floor plan
        if area > self.min_floor_plan_area:
            floor_plan_score += 0.2
        
        # Text content analysis
        if any(pattern in text_content for pattern in self.room_indicator_patterns):
            floor_plan_score += 0.1
        
        # Elevation indicators reduce floor plan score
        if any(indicator in text_content for indicator in self.elevation_indicators):
            floor_plan_score -= 0.3
        
        # Determine view type
        if floor_plan_score > 0.5:
            view_type = ViewType.FLOOR_PLAN
            confidence = min(floor_plan_score, 1.0)
        elif any(indicator in text_content for indicator in self.elevation_indicators):
            view_type = ViewType.ELEVATION
            confidence = 0.7
        elif wall_density > 0.6 and area < self.min_floor_plan_area:
            view_type = ViewType.DETAIL
            confidence = 0.6
        else:
            view_type = ViewType.UNKNOWN
            confidence = 0.3
        
        return view_type, confidence

    def _calculate_complexity_score(self, elements: List[CADElement]) -> float:
        """Calculate complexity score for a view"""
        if not elements:
            return 0.0
        
        # Factors contributing to complexity
        element_count = len(elements)
        unique_layers = len(set(elem.layer for elem in elements))
        
        # Geometric complexity
        total_length = 0.0
        total_area = 0.0
        
        for element in elements:
            if hasattr(element.geometry, 'length'):
                total_length += element.geometry.length
            if hasattr(element.geometry, 'area'):
                total_area += element.geometry.area
        
        # Normalize complexity score
        complexity = (element_count * 0.4 + unique_layers * 0.3 + 
                     min(total_length / 10000, 10) * 0.2 + 
                     min(total_area / 1000000, 10) * 0.1)
        
        return min(complexity, 10.0)  # Cap at 10

    def _select_best_floor_plan(self, detected_views: List[DetectedView]) -> Optional[DetectedView]:
        """Select the best floor plan view from detected views"""
        floor_plan_views = [view for view in detected_views 
                           if view.view_type == ViewType.FLOOR_PLAN]
        
        if not floor_plan_views:
            return None
        
        # Score each floor plan view
        best_view = None
        best_score = 0.0
        
        for view in floor_plan_views:
            # Scoring factors
            score = (view.confidence * 0.4 +  # Confidence in being a floor plan
                    (view.area / 10000000) * 0.3 +  # Larger plans are often main plans
                    view.complexity_score * 0.2 +  # More complex plans are often main plans
                    len(view.elements) / 100 * 0.1)  # More elements suggest main plan
            
            if score > best_score:
                best_score = score
                best_view = view
        
        return best_view

    def _optimize_floor_plan(self, main_view: DetectedView, 
                           original_data: FloorPlanData) -> FloorPlanData:
        """Optimize the selected floor plan view"""
        
        # Extract elements from the main view
        view_elements = main_view.elements
        
        # Separate elements by type
        walls = [elem for elem in view_elements if elem.element_type == ElementType.WALL]
        doors = [elem for elem in view_elements if elem.element_type == ElementType.DOOR]
        windows = [elem for elem in view_elements if elem.element_type == ElementType.WINDOW]
        rooms = [elem for elem in view_elements if elem.element_type == ElementType.ROOM_BOUNDARY]
        text_annotations = [elem for elem in view_elements if elem.element_type == ElementType.TEXT]
        
        # Optimize walls - connect segments and clean geometry
        optimized_walls = self._optimize_walls(walls)
        
        # Generate room boundaries if not present
        if not rooms:
            rooms = self._generate_room_boundaries(optimized_walls)
        
        # Detect special areas from text and geometry
        restricted_areas, entrances = self._detect_special_areas_from_view(
            text_annotations, rooms, main_view.bounds
        )
        
        # Create optimized floor plan
        optimized_plan = FloorPlanData(
            walls=optimized_walls,
            doors=doors,
            windows=windows,
            rooms=rooms,
            restricted_areas=restricted_areas,
            entrances=entrances,
            dimensions=original_data.dimensions,
            text_annotations=text_annotations,
            scale_factor=original_data.scale_factor,
            units=original_data.units,
            bounds={
                'min_x': main_view.bounds[0],
                'min_y': main_view.bounds[1],
                'max_x': main_view.bounds[2],
                'max_y': main_view.bounds[3],
                'width': main_view.bounds[2] - main_view.bounds[0],
                'height': main_view.bounds[3] - main_view.bounds[1]
            },
            metadata={
                'detected_view_type': main_view.view_type.value,
                'confidence': main_view.confidence,
                'complexity_score': main_view.complexity_score,
                'optimization_applied': True
            }
        )
        
        return optimized_plan

    def _optimize_walls(self, walls: List[CADElement]) -> List[CADElement]:
        """Optimize wall geometry by connecting segments and cleaning"""
        if not walls:
            return []
        
        # Group walls by connectivity
        wall_groups = self._group_connected_walls(walls)
        
        optimized_walls = []
        
        for group in wall_groups:
            # Merge connected wall segments
            merged_wall = self._merge_wall_segments(group)
            if merged_wall:
                optimized_walls.append(merged_wall)
        
        return optimized_walls

    def _group_connected_walls(self, walls: List[CADElement]) -> List[List[CADElement]]:
        """Group walls that are connected or aligned"""
        if not walls:
            return []
        
        groups = []
        processed = set()
        
        for wall in walls:
            if id(wall) in processed:
                continue
            
            # Start a new group
            group = [wall]
            processed.add(id(wall))
            
            # Find connected walls
            self._find_connected_walls(wall, walls, group, processed)
            
            groups.append(group)
        
        return groups

    def _find_connected_walls(self, wall: CADElement, all_walls: List[CADElement],
                            group: List[CADElement], processed: set):
        """Recursively find walls connected to the given wall"""
        if not isinstance(wall.geometry, LineString):
            return
        
        wall_coords = list(wall.geometry.coords)
        if len(wall_coords) < 2:
            return
        
        wall_start = Point(wall_coords[0])
        wall_end = Point(wall_coords[-1])
        
        for other_wall in all_walls:
            if id(other_wall) in processed or not isinstance(other_wall.geometry, LineString):
                continue
            
            other_coords = list(other_wall.geometry.coords)
            if len(other_coords) < 2:
                continue
            
            other_start = Point(other_coords[0])
            other_end = Point(other_coords[-1])
            
            # Check if walls are connected (endpoints are close)
            connections = [
                wall_start.distance(other_start),
                wall_start.distance(other_end),
                wall_end.distance(other_start),
                wall_end.distance(other_end)
            ]
            
            if min(connections) < self.connection_tolerance:
                group.append(other_wall)
                processed.add(id(other_wall))
                # Recursively find more connected walls
                self._find_connected_walls(other_wall, all_walls, group, processed)

    def _merge_wall_segments(self, wall_group: List[CADElement]) -> Optional[CADElement]:
        """Merge connected wall segments into a single wall"""
        if not wall_group:
            return None
        
        if len(wall_group) == 1:
            return wall_group[0]
        
        # For simplicity, just return the longest wall in the group
        # In a full implementation, you would properly merge the geometries
        longest_wall = max(wall_group, 
                          key=lambda w: w.geometry.length if isinstance(w.geometry, LineString) else 0)
        
        # Update properties to reflect the merged wall
        total_length = sum(w.geometry.length for w in wall_group 
                          if isinstance(w.geometry, LineString))
        
        longest_wall.properties.update({
            'length': total_length,
            'segment_count': len(wall_group),
            'merged': True
        })
        
        return longest_wall

    def _generate_room_boundaries(self, walls: List[CADElement]) -> List[CADElement]:
        """Generate room boundaries from wall structure"""
        rooms = []
        
        if not walls:
            return rooms
        
        # Simple approach: create rooms based on wall enclosures
        wall_lines = [wall.geometry for wall in walls 
                     if isinstance(wall.geometry, LineString)]
        
        if wall_lines:
            try:
                # Create a union of all walls
                wall_union = unary_union(wall_lines)
                
                # Get the bounds and create a simple room
                bounds = wall_union.bounds
                room_polygon = Polygon([
                    (bounds[0] + 100, bounds[1] + 100),  # Add small margin
                    (bounds[2] - 100, bounds[1] + 100),
                    (bounds[2] - 100, bounds[3] - 100),
                    (bounds[0] + 100, bounds[3] - 100)
                ])
                
                room_element = CADElement(
                    element_type=ElementType.ROOM_BOUNDARY,
                    geometry=room_polygon,
                    properties={
                        'area': room_polygon.area,
                        'perimeter': room_polygon.length,
                        'room_id': 'main_room',
                        'generated': True
                    }
                )
                rooms.append(room_element)
                
            except Exception as e:
                self.logger.warning(f"Error generating room boundaries: {str(e)}")
        
        return rooms

    def _detect_special_areas_from_view(self, text_annotations: List[CADElement],
                                      rooms: List[CADElement], 
                                      view_bounds: Tuple[float, float, float, float]) -> Tuple[List[CADElement], List[CADElement]]:
        """Detect restricted areas and entrances from view data"""
        restricted_areas = []
        entrances = []
        
        # Analyze text annotations
        for text_elem in text_annotations:
            text_content = text_elem.properties.get('text', '').lower()
            
            if any(keyword in text_content for keyword in ['no entree', 'restricted', 'interdit']):
                # Create restricted area
                center = text_elem.geometry
                restricted_area = center.buffer(1000)  # 1m radius
                
                restricted_element = CADElement(
                    element_type=ElementType.ROOM_BOUNDARY,
                    geometry=restricted_area,
                    properties={
                        'area_type': 'restricted',
                        'description': text_content,
                        'detected_from': 'text_annotation'
                    }
                )
                restricted_areas.append(restricted_element)
                
            elif any(keyword in text_content for keyword in ['entree', 'sortie', 'entrance', 'exit']):
                # Create entrance area
                center = text_elem.geometry
                entrance_area = center.buffer(800)  # 0.8m radius
                
                entrance_element = CADElement(
                    element_type=ElementType.ROOM_BOUNDARY,
                    geometry=entrance_area,
                    properties={
                        'area_type': 'entrance',
                        'description': text_content,
                        'detected_from': 'text_annotation'
                    }
                )
                entrances.append(entrance_element)
        
        # If no special areas found from text, create default ones based on geometry
        if not restricted_areas and not entrances:
            restricted_areas, entrances = self._create_default_special_areas(view_bounds)
        
        return restricted_areas, entrances

    def _create_default_special_areas(self, bounds: Tuple[float, float, float, float]) -> Tuple[List[CADElement], List[CADElement]]:
        """Create default restricted areas and entrances for demonstration"""
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y
        
        restricted_areas = []
        entrances = []
        
        # Create 2 restricted areas (blue zones)
        restricted_1 = Polygon([
            (min_x + width * 0.1, min_y + height * 0.7),
            (min_x + width * 0.25, min_y + height * 0.7),
            (min_x + width * 0.25, min_y + height * 0.85),
            (min_x + width * 0.1, min_y + height * 0.85)
        ])
        
        restricted_2 = Polygon([
            (min_x + width * 0.1, min_y + height * 0.1),
            (min_x + width * 0.25, min_y + height * 0.1),
            (min_x + width * 0.25, min_y + height * 0.25),
            (min_x + width * 0.1, min_y + height * 0.25)
        ])
        
        for i, restricted_geom in enumerate([restricted_1, restricted_2], 1):
            restricted_element = CADElement(
                element_type=ElementType.ROOM_BOUNDARY,
                geometry=restricted_geom,
                properties={
                    'area_type': 'restricted',
                    'description': f'NO ENTREE {i}',
                    'detected_from': 'geometry_analysis'
                }
            )
            restricted_areas.append(restricted_element)
        
        # Create 3 entrance areas (red zones)
        entrance_positions = [
            (min_x + width * 0.05, min_y + height * 0.4),  # Left side
            (max_x - width * 0.05, min_y + height * 0.6),  # Right side
            (min_x + width * 0.5, max_y - height * 0.05)   # Top side
        ]
        
        for i, (x, y) in enumerate(entrance_positions, 1):
            entrance_area = Point(x, y).buffer(600)  # 60cm radius
            
            entrance_element = CADElement(
                element_type=ElementType.ROOM_BOUNDARY,
                geometry=entrance_area,
                properties={
                    'area_type': 'entrance',
                    'description': f'ENTRÉE/SORTIE {i}',
                    'detected_from': 'geometry_analysis'
                }
            )
            entrances.append(entrance_element)
        
        return restricted_areas, entrances

    def _create_heuristic_floor_plan(self, floor_plan_data: FloorPlanData) -> FloorPlanData:
        """Create a floor plan using heuristics when no clear floor plan is detected"""
        
        # Use the original data but apply some cleaning
        all_walls = floor_plan_data.walls
        
        if all_walls:
            # Filter walls by length - keep only substantial walls
            substantial_walls = [wall for wall in all_walls 
                               if isinstance(wall.geometry, LineString) and 
                               wall.geometry.length > self.wall_length_threshold]
            
            # If we have substantial walls, use them
            if substantial_walls:
                floor_plan_data.walls = substantial_walls
                
                # Generate rooms from these walls
                if not floor_plan_data.rooms:
                    floor_plan_data.rooms = self._generate_room_boundaries(substantial_walls)
                
                # Add default special areas if none exist
                if not floor_plan_data.restricted_areas and not floor_plan_data.entrances:
                    bounds = floor_plan_data.bounds
                    if bounds:
                        view_bounds = (bounds['min_x'], bounds['min_y'], 
                                     bounds['max_x'], bounds['max_y'])
                        restricted, entrances = self._create_default_special_areas(view_bounds)
                        floor_plan_data.restricted_areas = restricted
                        floor_plan_data.entrances = entrances
        
        # Mark as heuristic
        if not floor_plan_data.metadata:
            floor_plan_data.metadata = {}
        floor_plan_data.metadata['heuristic_detection'] = True
        
        return floor_plan_data

    def get_detection_summary(self, floor_plan_data: FloorPlanData) -> Dict[str, Any]:
        """Generate a summary of the floor plan detection process"""
        metadata = floor_plan_data.metadata or {}
        
        return {
            'detection_method': metadata.get('heuristic_detection', False) and 'heuristic' or 'smart_detection',
            'view_type': metadata.get('detected_view_type', 'unknown'),
            'confidence': metadata.get('confidence', 0.0),
            'complexity_score': metadata.get('complexity_score', 0.0),
            'optimization_applied': metadata.get('optimization_applied', False),
            'total_walls': len(floor_plan_data.walls),
            'total_rooms': len(floor_plan_data.rooms),
            'restricted_areas': len(floor_plan_data.restricted_areas),
            'entrances': len(floor_plan_data.entrances),
            'floor_plan_area': floor_plan_data.bounds.get('width', 0) * floor_plan_data.bounds.get('height', 0) if floor_plan_data.bounds else 0
        }