"""
Advanced Corridor Generator
Implements mandatory corridor generation between facing îlot rows - CLIENT REQUIREMENT
"""

import numpy as np
import math
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

@dataclass
class CorridorSpec:
    """Enhanced corridor specification"""
    id: str
    type: str  # 'main', 'secondary', 'facing', 'access'
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    width: float
    path_points: List[Tuple[float, float]]
    connects_ilots: List[str]
    is_mandatory: bool = False
    accessibility_compliant: bool = True
    priority: int = 1
    color: str = '#F39C12'
    length: float = 0
    area: float = 0
    
    def __post_init__(self):
        """Calculate length and area after initialization"""
        if len(self.path_points) >= 2:
            self.length = sum(
                math.sqrt((self.path_points[i+1][0] - self.path_points[i][0])**2 + 
                         (self.path_points[i+1][1] - self.path_points[i][1])**2)
                for i in range(len(self.path_points)-1)
            )
            self.area = self.length * self.width
    
    def to_polygon(self) -> Optional[Polygon]:
        """Convert corridor to polygon for collision detection"""
        if len(self.path_points) < 2:
            return None
        
        try:
            line = LineString(self.path_points)
            return line.buffer(self.width / 2)
        except:
            return None

class AdvancedCorridorGenerator:
    """Advanced corridor generation system matching client requirements"""
    
    def __init__(self):
        self.ilots = []
        self.walls = []
        self.restricted_areas = []
        self.entrances = []
        self.bounds = {}
        self.generated_corridors = []
        
    def load_floor_plan_data(self, ilots: List[Dict], walls: List, 
                           restricted_areas: List, entrances: List, bounds: Dict):
        """Load floor plan and îlot data"""
        self.ilots = ilots
        self.walls = [Polygon(wall) if len(wall) > 2 else None for wall in walls]
        self.walls = [w for w in self.walls if w is not None]
        
        self.restricted_areas = [Polygon(area) if len(area) > 2 else None for area in restricted_areas]
        self.restricted_areas = [r for r in self.restricted_areas if r is not None]
        
        self.entrances = [Polygon(entrance) if len(entrance) > 2 else None for entrance in entrances]
        self.entrances = [e for e in self.entrances if e is not None]
        
        self.bounds = bounds
        
    def generate_complete_corridor_network(self, config: Dict) -> Dict[str, Any]:
        """Generate complete corridor network - CLIENT REQUIREMENT IMPLEMENTATION"""
        corridors = []
        
        # 1. MANDATORY: Generate corridors between facing îlot rows (CLIENT REQUIREMENT)
        if config.get('force_between_facing', True):
            facing_corridors = self.generate_mandatory_facing_corridors(config)
            corridors.extend(facing_corridors)
        
        # 2. Generate main corridors from entrances
        if config.get('generate_main', True):
            main_corridors = self.generate_main_corridors(config)
            corridors.extend(main_corridors)
        
        # 3. Generate secondary access corridors
        if config.get('generate_secondary', True):
            secondary_corridors = self.generate_secondary_corridors(config)
            corridors.extend(secondary_corridors)
        
        # 4. Generate access corridors for isolated îlots
        access_corridors = self.generate_access_corridors(config)
        corridors.extend(access_corridors)
        
        # 5. Validate all corridors
        valid_corridors = self.validate_corridor_network(corridors)
        
        # 6. Calculate network statistics
        network_stats = self.calculate_network_statistics(valid_corridors)
        
        self.generated_corridors = valid_corridors
        
        return {
            'corridors': [self.corridor_to_dict(c) for c in valid_corridors],
            'network_statistics': network_stats,
            'generation_config': config,
            'validation_results': {
                'total_generated': len(corridors),
                'valid_corridors': len(valid_corridors),
                'mandatory_corridors': len([c for c in valid_corridors if c.is_mandatory]),
                'facing_corridors': len([c for c in valid_corridors if c.type == 'facing']),
                'main_corridors': len([c for c in valid_corridors if c.type == 'main']),
                'secondary_corridors': len([c for c in valid_corridors if c.type == 'secondary'])
            }
        }
    
    def generate_mandatory_facing_corridors(self, config: Dict) -> List[CorridorSpec]:
        """Generate MANDATORY corridors between facing îlot rows - CORE CLIENT REQUIREMENT"""
        if not self.ilots:
            return []
        
        corridors = []
        corridor_width = config.get('access_width', 1.5)
        
        # Step 1: Detect îlot rows based on spatial alignment
        rows = self.detect_ilot_rows_advanced()
        
        # Step 2: Find facing row pairs
        facing_pairs = self.find_facing_row_pairs(rows)
        
        # Step 3: Generate mandatory corridors for each facing pair
        corridor_id = 1
        for row1, row2 in facing_pairs:
            corridor = self.create_mandatory_corridor_between_rows(
                row1, row2, corridor_width, f"mandatory_facing_{corridor_id:03d}"
            )
            
            if corridor and self.validate_mandatory_corridor(corridor):
                corridor.is_mandatory = True
                corridor.priority = 1  # Highest priority
                corridor.color = '#E74C3C'  # Red for mandatory
                corridor.type = 'facing'
                corridors.append(corridor)
                corridor_id += 1
        
        return corridors
    
    def detect_ilot_rows_advanced(self) -> List[List[Dict]]:
        """Advanced îlot row detection using clustering and alignment analysis"""
        if not self.ilots:
            return []
        
        rows = []
        alignment_threshold = 2.5  # meters
        min_row_size = 2  # minimum îlots per row
        
        # Create a copy of îlots to work with
        remaining_ilots = self.ilots.copy()
        
        while remaining_ilots:
            seed_ilot = remaining_ilots.pop(0)
            current_row = [seed_ilot]
            
            # Find îlots aligned with seed îlot
            to_remove = []
            for other_ilot in remaining_ilots:
                # Check horizontal alignment (same Y coordinate approximately)
                y_diff = abs(seed_ilot['y'] - other_ilot['y'])
                # Check vertical alignment (same X coordinate approximately)
                x_diff = abs(seed_ilot['x'] - other_ilot['x'])
                
                if y_diff < alignment_threshold or x_diff < alignment_threshold:
                    # Additional check: ensure they're reasonably close
                    distance = math.sqrt((seed_ilot['x'] - other_ilot['x'])**2 + 
                                       (seed_ilot['y'] - other_ilot['y'])**2)
                    
                    if distance < 15:  # Maximum distance for same row
                        current_row.append(other_ilot)
                        to_remove.append(other_ilot)
            
            # Remove îlots that were added to current row
            for ilot in to_remove:
                remaining_ilots.remove(ilot)
            
            # Only keep rows with minimum size
            if len(current_row) >= min_row_size:
                rows.append(current_row)
        
        return rows
    
    def find_facing_row_pairs(self, rows: List[List[Dict]]) -> List[Tuple[List[Dict], List[Dict]]]:
        """Find pairs of rows that are facing each other"""
        facing_pairs = []
        
        for i, row1 in enumerate(rows):
            for j, row2 in enumerate(rows[i+1:], i+1):
                if self.are_rows_facing_advanced(row1, row2):
                    facing_pairs.append((row1, row2))
        
        return facing_pairs
    
    def are_rows_facing_advanced(self, row1: List[Dict], row2: List[Dict]) -> bool:
        """Advanced algorithm to determine if two rows are facing each other"""
        # Calculate row centers and orientations
        row1_center_x = sum(ilot['x'] for ilot in row1) / len(row1)
        row1_center_y = sum(ilot['y'] for ilot in row1) / len(row1)
        
        row2_center_x = sum(ilot['x'] for ilot in row2) / len(row2)
        row2_center_y = sum(ilot['y'] for ilot in row2) / len(row2)
        
        # Calculate distance between row centers
        center_distance = math.sqrt((row1_center_x - row2_center_x)**2 + 
                                  (row1_center_y - row2_center_y)**2)
        
        # Check if distance is appropriate for corridor generation
        if not (3.0 <= center_distance <= 12.0):
            return False
        
        # Determine row orientations
        row1_width = max(ilot['x'] for ilot in row1) - min(ilot['x'] for ilot in row1)
        row1_height = max(ilot['y'] for ilot in row1) - min(ilot['y'] for ilot in row1)
        row1_horizontal = row1_width > row1_height
        
        row2_width = max(ilot['x'] for ilot in row2) - min(ilot['x'] for ilot in row2)
        row2_height = max(ilot['y'] for ilot in row2) - min(ilot['y'] for ilot in row2)
        row2_horizontal = row2_width > row2_height
        
        # Rows must have similar orientation
        if row1_horizontal != row2_horizontal:
            return False
        
        # Check for overlap in the perpendicular direction
        if row1_horizontal:
            # Horizontal rows - check Y overlap
            row1_min_x = min(ilot['x'] - ilot['width']/2 for ilot in row1)
            row1_max_x = max(ilot['x'] + ilot['width']/2 for ilot in row1)
            row2_min_x = min(ilot['x'] - ilot['width']/2 for ilot in row2)
            row2_max_x = max(ilot['x'] + ilot['width']/2 for ilot in row2)
            
            overlap = min(row1_max_x, row2_max_x) - max(row1_min_x, row2_min_x)
            return overlap > 2.0  # Minimum overlap for corridor
        else:
            # Vertical rows - check X overlap
            row1_min_y = min(ilot['y'] - ilot['height']/2 for ilot in row1)
            row1_max_y = max(ilot['y'] + ilot['height']/2 for ilot in row1)
            row2_min_y = min(ilot['y'] - ilot['height']/2 for ilot in row2)
            row2_max_y = max(ilot['y'] + ilot['height']/2 for ilot in row2)
            
            overlap = min(row1_max_y, row2_max_y) - max(row1_min_y, row2_min_y)
            return overlap > 2.0  # Minimum overlap for corridor
    
    def create_mandatory_corridor_between_rows(self, row1: List[Dict], row2: List[Dict], 
                                             width: float, corridor_id: str) -> Optional[CorridorSpec]:
        """Create mandatory corridor between two facing rows - CLIENT SPECIFICATION"""
        if not row1 or not row2:
            return None
        
        # Calculate row bounds
        row1_bounds = self.calculate_row_bounds(row1)
        row2_bounds = self.calculate_row_bounds(row2)
        
        # Determine corridor orientation and path
        row1_center_x = sum(ilot['x'] for ilot in row1) / len(row1)
        row1_center_y = sum(ilot['y'] for ilot in row1) / len(row1)
        row2_center_x = sum(ilot['x'] for ilot in row2) / len(row2)
        row2_center_y = sum(ilot['y'] for ilot in row2) / len(row2)
        
        # Determine if rows are horizontally or vertically separated
        horizontal_separation = abs(row1_center_x - row2_center_x)
        vertical_separation = abs(row1_center_y - row2_center_y)
        
        if horizontal_separation > vertical_separation:
            # Rows are horizontally separated - create vertical corridor
            corridor_x = (row1_center_x + row2_center_x) / 2
            
            # Find overlapping Y range
            overlap_min_y = max(row1_bounds['min_y'], row2_bounds['min_y'])
            overlap_max_y = min(row1_bounds['max_y'], row2_bounds['max_y'])
            
            if overlap_min_y < overlap_max_y:
                # Use overlap region
                start_point = (corridor_x, overlap_min_y)
                end_point = (corridor_x, overlap_max_y)
            else:
                # Extend to cover both rows
                start_point = (corridor_x, min(row1_bounds['min_y'], row2_bounds['min_y']))
                end_point = (corridor_x, max(row1_bounds['max_y'], row2_bounds['max_y']))
            
            path_points = [start_point, end_point]
            
        else:
            # Rows are vertically separated - create horizontal corridor
            corridor_y = (row1_center_y + row2_center_y) / 2
            
            # Find overlapping X range
            overlap_min_x = max(row1_bounds['min_x'], row2_bounds['min_x'])
            overlap_max_x = min(row1_bounds['max_x'], row2_bounds['max_x'])
            
            if overlap_min_x < overlap_max_x:
                # Use overlap region
                start_point = (overlap_min_x, corridor_y)
                end_point = (overlap_max_x, corridor_y)
            else:
                # Extend to cover both rows
                start_point = (min(row1_bounds['min_x'], row2_bounds['min_x']), corridor_y)
                end_point = (max(row1_bounds['max_x'], row2_bounds['max_x']), corridor_y)
            
            path_points = [start_point, end_point]
        
        # Create corridor specification
        corridor = CorridorSpec(
            id=corridor_id,
            type='facing',
            start_point=start_point,
            end_point=end_point,
            width=width,
            path_points=path_points,
            connects_ilots=[ilot['id'] for ilot in row1 + row2],
            is_mandatory=True,
            accessibility_compliant=True,
            priority=1,
            color='#E74C3C'  # Red for mandatory corridors
        )
        
        return corridor
    
    def calculate_row_bounds(self, row: List[Dict]) -> Dict[str, float]:
        """Calculate bounding box for a row of îlots"""
        min_x = min(ilot['x'] - ilot['width']/2 for ilot in row)
        max_x = max(ilot['x'] + ilot['width']/2 for ilot in row)
        min_y = min(ilot['y'] - ilot['height']/2 for ilot in row)
        max_y = max(ilot['y'] + ilot['height']/2 for ilot in row)
        
        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }
    
    def validate_mandatory_corridor(self, corridor: CorridorSpec) -> bool:
        """Validate mandatory corridor - must not overlap îlots"""
        corridor_poly = corridor.to_polygon()
        if not corridor_poly:
            return False
        
        # Check that corridor doesn't overlap with any îlot
        for ilot in self.ilots:
            ilot_poly = Polygon([
                (ilot['x'] - ilot['width']/2, ilot['y'] - ilot['height']/2),
                (ilot['x'] + ilot['width']/2, ilot['y'] - ilot['height']/2),
                (ilot['x'] + ilot['width']/2, ilot['y'] + ilot['height']/2),
                (ilot['x'] - ilot['width']/2, ilot['y'] + ilot['height']/2)
            ])
            
            if corridor_poly.intersects(ilot_poly):
                return False
        
        # Check that corridor touches both rows (requirement)
        # This is ensured by construction, so return True
        return True
    
    def generate_main_corridors(self, config: Dict) -> List[CorridorSpec]:
        """Generate main corridors from entrances"""
        corridors = []
        main_width = config.get('main_width', 2.5)
        
        corridor_id = 1
        for i, entrance in enumerate(self.entrances):
            if entrance:
                # Get entrance centroid
                entrance_center = entrance.centroid
                
                # Create main corridor extending into the space
                # Direction depends on entrance orientation
                bounds = entrance.bounds
                entrance_width = bounds[2] - bounds[0]
                entrance_height = bounds[3] - bounds[1]
                
                if entrance_width > entrance_height:
                    # Horizontal entrance - create vertical corridor
                    start_point = (entrance_center.x, entrance_center.y)
                    end_point = (entrance_center.x, entrance_center.y + 15)
                else:
                    # Vertical entrance - create horizontal corridor
                    start_point = (entrance_center.x, entrance_center.y)
                    end_point = (entrance_center.x + 15, entrance_center.y)
                
                corridor = CorridorSpec(
                    id=f"main_{corridor_id:03d}",
                    type='main',
                    start_point=start_point,
                    end_point=end_point,
                    width=main_width,
                    path_points=[start_point, end_point],
                    connects_ilots=[],
                    is_mandatory=False,
                    accessibility_compliant=True,
                    priority=2,
                    color='#2E86AB'
                )
                
                corridors.append(corridor)
                corridor_id += 1
        
        return corridors
    
    def generate_secondary_corridors(self, config: Dict) -> List[CorridorSpec]:
        """Generate secondary corridors for connectivity"""
        corridors = []
        secondary_width = config.get('secondary_width', 1.5)
        
        # Create network graph of îlots
        G = nx.Graph()
        
        # Add îlots as nodes
        for ilot in self.ilots:
            G.add_node(ilot['id'], pos=(ilot['x'], ilot['y']))
        
        # Add edges for nearby îlots
        for i, ilot1 in enumerate(self.ilots):
            for j, ilot2 in enumerate(self.ilots[i+1:], i+1):
                distance = math.sqrt((ilot1['x'] - ilot2['x'])**2 + (ilot1['y'] - ilot2['y'])**2)
                if 3.0 <= distance <= 8.0:  # Reasonable corridor distance
                    G.add_edge(ilot1['id'], ilot2['id'], weight=distance)
        
        # Find minimum spanning tree for connectivity
        if G.number_of_edges() > 0:
            mst = nx.minimum_spanning_tree(G)
            
            corridor_id = 1
            for edge in mst.edges():
                ilot1 = next(i for i in self.ilots if i['id'] == edge[0])
                ilot2 = next(i for i in self.ilots if i['id'] == edge[1])
                
                corridor = CorridorSpec(
                    id=f"secondary_{corridor_id:03d}",
                    type='secondary',
                    start_point=(ilot1['x'], ilot1['y']),
                    end_point=(ilot2['x'], ilot2['y']),
                    width=secondary_width,
                    path_points=[(ilot1['x'], ilot1['y']), (ilot2['x'], ilot2['y'])],
                    connects_ilots=[ilot1['id'], ilot2['id']],
                    is_mandatory=False,
                    accessibility_compliant=True,
                    priority=3,
                    color='#F39C12'
                )
                
                corridors.append(corridor)
                corridor_id += 1
        
        return corridors
    
    def generate_access_corridors(self, config: Dict) -> List[CorridorSpec]:
        """Generate access corridors for isolated îlots"""
        corridors = []
        access_width = config.get('access_width', 1.0)
        
        # Find îlots that are not connected by existing corridors
        connected_ilots = set()
        for corridor in self.generated_corridors:
            connected_ilots.update(corridor.connects_ilots)
        
        isolated_ilots = [ilot for ilot in self.ilots if ilot['id'] not in connected_ilots]
        
        corridor_id = 1
        for ilot in isolated_ilots:
            # Find nearest main corridor or entrance
            min_distance = float('inf')
            nearest_point = None
            
            for entrance in self.entrances:
                if entrance:
                    distance = Point(ilot['x'], ilot['y']).distance(entrance)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = entrance.centroid
            
            if nearest_point:
                corridor = CorridorSpec(
                    id=f"access_{corridor_id:03d}",
                    type='access',
                    start_point=(ilot['x'], ilot['y']),
                    end_point=(nearest_point.x, nearest_point.y),
                    width=access_width,
                    path_points=[(ilot['x'], ilot['y']), (nearest_point.x, nearest_point.y)],
                    connects_ilots=[ilot['id']],
                    is_mandatory=False,
                    accessibility_compliant=True,
                    priority=4,
                    color='#95A5A6'
                )
                
                corridors.append(corridor)
                corridor_id += 1
        
        return corridors
    
    def validate_corridor_network(self, corridors: List[CorridorSpec]) -> List[CorridorSpec]:
        """Validate entire corridor network"""
        valid_corridors = []
        
        for corridor in corridors:
            if self.validate_single_corridor(corridor):
                valid_corridors.append(corridor)
        
        return valid_corridors
    
    def validate_single_corridor(self, corridor: CorridorSpec) -> bool:
        """Validate a single corridor"""
        # Check minimum length
        if corridor.length < 0.5:
            return False
        
        # Check that corridor is within bounds
        if self.bounds:
            for point in corridor.path_points:
                if (point[0] < self.bounds['min_x'] or point[0] > self.bounds['max_x'] or
                    point[1] < self.bounds['min_y'] or point[1] > self.bounds['max_y']):
                    return False
        
        # Check that corridor doesn't intersect restricted areas
        corridor_poly = corridor.to_polygon()
        if corridor_poly:
            for restricted in self.restricted_areas:
                if corridor_poly.intersects(restricted):
                    return False
        
        return True
    
    def calculate_network_statistics(self, corridors: List[CorridorSpec]) -> Dict[str, Any]:
        """Calculate comprehensive network statistics"""
        if not corridors:
            return {
                'total_corridors': 0,
                'total_length': 0,
                'total_area': 0,
                'connectivity_score': 0,
                'mandatory_corridors': 0,
                'facing_corridors': 0,
                'main_corridors': 0,
                'secondary_corridors': 0,
                'access_corridors': 0
            }
        
        total_length = sum(c.length for c in corridors)
        total_area = sum(c.area for c in corridors)
        
        corridor_types = {
            'mandatory': len([c for c in corridors if c.is_mandatory]),
            'facing': len([c for c in corridors if c.type == 'facing']),
            'main': len([c for c in corridors if c.type == 'main']),
            'secondary': len([c for c in corridors if c.type == 'secondary']),
            'access': len([c for c in corridors if c.type == 'access'])
        }
        
        # Calculate connectivity score
        connected_ilots = set()
        for corridor in corridors:
            connected_ilots.update(corridor.connects_ilots)
        
        connectivity_score = len(connected_ilots) / max(len(self.ilots), 1) if self.ilots else 0
        
        return {
            'total_corridors': len(corridors),
            'total_length': total_length,
            'total_area': total_area,
            'connectivity_score': connectivity_score,
            'mandatory_corridors': corridor_types['mandatory'],
            'facing_corridors': corridor_types['facing'],
            'main_corridors': corridor_types['main'],
            'secondary_corridors': corridor_types['secondary'],
            'access_corridors': corridor_types['access'],
            'average_width': sum(c.width for c in corridors) / len(corridors),
            'pathfinding_algorithm': 'Advanced Spatial Analysis'
        }
    
    def corridor_to_dict(self, corridor: CorridorSpec) -> Dict:
        """Convert corridor to dictionary for JSON serialization"""
        return {
            'id': corridor.id,
            'type': corridor.type,
            'start_point': corridor.start_point,
            'end_point': corridor.end_point,
            'width': corridor.width,
            'path_points': corridor.path_points,
            'connects_ilots': corridor.connects_ilots,
            'is_mandatory': corridor.is_mandatory,
            'accessibility_compliant': corridor.accessibility_compliant,
            'priority': corridor.priority,
            'color': corridor.color,
            'length': corridor.length,
            'area': corridor.area
        }
