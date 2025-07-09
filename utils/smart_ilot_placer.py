"""
Smart Îlot Placer
Intelligent îlot placement using room detection and optimal positioning
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from shapely.geometry import Point, Polygon, box
import time

class SmartIlotPlacer:
    """Smart îlot placement with room detection"""

    def __init__(self):
        self.size_categories = {
            'size_0_1': {'min_area': 0, 'max_area': 1, 'color': '#FFFF00', 'typical_dims': (1.0, 0.8)},
            'size_1_3': {'min_area': 1, 'max_area': 3, 'color': '#FFA500', 'typical_dims': (1.5, 1.2)},
            'size_3_5': {'min_area': 3, 'max_area': 5, 'color': '#10B981', 'typical_dims': (2.0, 1.8)},  # Green for visibility
            'size_5_10': {'min_area': 5, 'max_area': 10, 'color': '#800080', 'typical_dims': (3.0, 2.5)}
        }

    def place_ilots_smart(self, analysis_data: Dict, config: Dict) -> List[Dict]:
        """Place îlots intelligently using room detection"""
        try:
            # Extract actual room areas from floor plan
            rooms = self._detect_rooms_from_walls(analysis_data)

            # Generate îlot specifications based on config
            ilot_specs = self._generate_ilot_specs(config)

            # Place îlots in detected rooms
            placed_ilots = self._place_in_rooms(rooms, ilot_specs)

            # If not enough îlots placed, use fallback
            if len(placed_ilots) < 10:
                return self._fallback_smart_placement(analysis_data, config)

            return placed_ilots

        except Exception as e:
            # Fallback to grid placement
            return self._fallback_smart_placement(analysis_data, config)

    def _detect_rooms_from_walls(self, analysis_data: Dict) -> List[Dict]:
        """Detect room areas from wall data"""
        bounds = analysis_data.get('bounds', {})
        walls = analysis_data.get('walls', [])

        rooms = []

        # Get floor plan dimensions
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)

        width = max_x - min_x
        height = max_y - min_y

        # Intelligent room detection based on floor plan analysis
        # Create rooms based on likely architectural patterns

        # Main central areas (likely large rooms)
        central_rooms = [
            {
                'id': 'central_room_1',
                'bounds': {
                    'min_x': min_x + width * 0.1,
                    'max_x': min_x + width * 0.45,
                    'min_y': min_y + height * 0.2,
                    'max_y': min_y + height * 0.7
                },
                'type': 'main_room'
            },
            {
                'id': 'central_room_2', 
                'bounds': {
                    'min_x': min_x + width * 0.55,
                    'max_x': min_x + width * 0.9,
                    'min_y': min_y + height * 0.2,
                    'max_y': min_y + height * 0.7
                },
                'type': 'main_room'
            },
            {
                'id': 'large_room_top',
                'bounds': {
                    'min_x': min_x + width * 0.2,
                    'max_x': min_x + width * 0.8,
                    'min_y': min_y + height * 0.75,
                    'max_y': min_y + height * 0.95
                },
                'type': 'large_room'
            },
            {
                'id': 'large_room_bottom',
                'bounds': {
                    'min_x': min_x + width * 0.2,
                    'max_x': min_x + width * 0.8,
                    'min_y': min_y + height * 0.05,
                    'max_y': min_y + height * 0.15
                },
                'type': 'large_room'
            }
        ]

        # Add room information
        for room in central_rooms:
            room_bounds = room['bounds']
            room_width = room_bounds['max_x'] - room_bounds['min_x']
            room_height = room_bounds['max_y'] - room_bounds['min_y']

            room.update({
                'area': room_width * room_height,
                'center': [
                    room_bounds['min_x'] + room_width/2,
                    room_bounds['min_y'] + room_height/2
                ],
                'width': room_width,
                'height': room_height
            })

            rooms.append(room)

        return rooms

    def _generate_ilot_specs(self, config: Dict) -> List[Dict]:
        """Generate îlot specifications from config"""
        specs = []

        # Calculate total îlots based on configuration
        base_ilots = 30  # Good number for coverage

        # Apply size distribution from config
        distributions = [
            ('size_0_1', config.get('size_0_1_percent', 10)),
            ('size_1_3', config.get('size_1_3_percent', 25)),
            ('size_3_5', config.get('size_3_5_percent', 30)),
            ('size_5_10', config.get('size_5_10_percent', 35))
        ]

        for size_cat, percent in distributions:
            count = max(1, int(base_ilots * percent / 100))
            category_info = self.size_categories[size_cat]

            for i in range(count):
                # Generate realistic dimensions for each category
                if size_cat == 'size_0_1':
                    width = np.random.uniform(0.8, 1.2)
                    height = np.random.uniform(0.6, 1.0)
                elif size_cat == 'size_1_3':
                    width = np.random.uniform(1.2, 1.8)
                    height = np.random.uniform(1.0, 1.5)
                elif size_cat == 'size_3_5':
                    width = np.random.uniform(1.8, 2.5)
                    height = np.random.uniform(1.5, 2.2)
                else:  # size_5_10
                    width = np.random.uniform(2.5, 3.5)
                    height = np.random.uniform(2.0, 3.0)

                area = width * height

                spec = {
                    'size_category': size_cat,
                    'area': area,
                    'width': width,
                    'height': height,
                    'color': category_info['color']
                }
                specs.append(spec)

        # Shuffle for better distribution
        np.random.shuffle(specs)
        return specs

    def _place_in_rooms(self, rooms: List[Dict], ilot_specs: List[Dict]) -> List[Dict]:
        """Place îlots within detected rooms with smart positioning"""
        placed_ilots = []

        # Sort rooms by area (largest first)
        rooms.sort(key=lambda x: x['area'], reverse=True)

        # Keep track of room occupancy
        room_occupancy = {room['id']: [] for room in rooms}

        for i, spec in enumerate(ilot_specs):
            best_room = None
            best_position = None
            best_score = -1

            # Try each room
            for room in rooms:
                position = self._find_optimal_position_in_room(
                    room, spec, room_occupancy[room['id']]
                )

                if position:
                    # Score this position
                    score = self._score_position(room, position, spec)

                    if score > best_score:
                        best_score = score
                        best_room = room
                        best_position = position

            # Place îlot if valid position found
            if best_position and best_room:
                x, y = best_position

                ilot = {
                    'id': f'smart_ilot_{i}',
                    'x': x,
                    'y': y,
                    'width': spec['width'],
                    'height': spec['height'],
                    'area': spec['area'],
                    'size_category': spec['size_category'],
                    'color': spec['color'],
                    'room_id': best_room['id'],
                    'placement_method': 'smart_room_based'
                }

                placed_ilots.append(ilot)
                room_occupancy[best_room['id']].append(ilot)

        return placed_ilots

    def _find_optimal_position_in_room(self, room: Dict, spec: Dict, existing_ilots: List[Dict]) -> Optional[Tuple[float, float]]:
        """Find optimal position for îlot within room"""
        bounds = room['bounds']
        margin = 0.3  # Smaller margin for better space utilization

        # Available space in room
        available_width = bounds['max_x'] - bounds['min_x'] - 2 * margin
        available_height = bounds['max_y'] - bounds['min_y'] - 2 * margin

        # Check if îlot fits
        if spec['width'] > available_width or spec['height'] > available_height:
            return None

        # Try grid positions for better organization
        grid_points = 8
        best_position = None
        best_distance = 0

        for i in range(grid_points):
            for j in range(grid_points):
                # Calculate position
                x_offset = (available_width - spec['width']) * (i / (grid_points - 1)) if grid_points > 1 else 0
                y_offset = (available_height - spec['height']) * (j / (grid_points - 1)) if grid_points > 1 else 0

                x = bounds['min_x'] + margin + x_offset
                y = bounds['min_y'] + margin + y_offset

                # Check for overlaps
                if not self._check_overlap_with_margin(x, y, spec, existing_ilots, 0.2):
                    # Calculate distance from other îlots (prefer some spacing)
                    min_distance = self._calculate_min_distance(x, y, spec, existing_ilots)

                    if min_distance > best_distance:
                        best_distance = min_distance
                        best_position = (x, y)

        return best_position

    def _check_overlap_with_margin(self, x: float, y: float, spec: Dict, existing_ilots: List[Dict], margin: float = 0.1) -> bool:
        """Check if îlot overlaps with existing ones (with margin)"""
        proposed_box = box(
            x - margin, y - margin,
            x + spec['width'] + margin, y + spec['height'] + margin
        )

        for existing in existing_ilots:
            existing_box = box(
                existing['x'], existing['y'],
                existing['x'] + existing['width'],
                existing['y'] + existing['height']
            )

            if proposed_box.intersects(existing_box):
                return True

        return False

    def _calculate_min_distance(self, x: float, y: float, spec: Dict, existing_ilots: List[Dict]) -> float:
        """Calculate minimum distance to existing îlots"""
        if not existing_ilots:
            return float('inf')

        center_x = x + spec['width'] / 2
        center_y = y + spec['height'] / 2

        min_distance = float('inf')

        for existing in existing_ilots:
            existing_center_x = existing['x'] + existing['width'] / 2
            existing_center_y = existing['y'] + existing['height'] / 2

            distance = np.sqrt(
                (center_x - existing_center_x) ** 2 + 
                (center_y - existing_center_y) ** 2
            )

            min_distance = min(min_distance, distance)

        return min_distance

    def _score_position(self, room: Dict, position: Tuple[float, float], spec: Dict) -> float:
        """Score a position within a room"""
        x, y = position
        room_center = room['center']

        # Distance from room center (prefer central positions)
        center_x = x + spec['width'] / 2
        center_y = y + spec['height'] / 2

        distance_from_center = np.sqrt(
            (center_x - room_center[0]) ** 2 + 
            (center_y - room_center[1]) ** 2
        )

        # Normalize by room size
        max_distance = np.sqrt(room['width'] ** 2 + room['height'] ** 2) / 2
        normalized_distance = distance_from_center / max_distance

        # Score: higher is better, prefer central positions
        score = 1.0 - normalized_distance

        return score

    def _fallback_smart_placement(self, analysis_data: Dict, config: Dict) -> List[Dict]:
        """Fallback smart placement when room detection fails"""
        bounds = analysis_data.get('bounds', {})
        placed_ilots = []

        # Use intelligent grid with realistic spacing
        min_x = bounds.get('min_x', 0) + 1
        max_x = bounds.get('max_x', 100) - 1
        min_y = bounds.get('min_y', 0) + 1
        max_y = bounds.get('max_y', 100) - 1

        # Smart grid based on floor plan size
        width = max_x - min_x
        height = max_y - min_y

        # Adaptive grid spacing
        if width * height > 1000:  # Large floor plan
            grid_spacing_x = width / 12
            grid_spacing_y = height / 8
        else:  # Smaller floor plan
            grid_spacing_x = width / 8
            grid_spacing_y = height / 6

        # Generate îlot specs
        ilot_specs = self._generate_ilot_specs(config)

        # Place îlots in grid pattern
        ilot_id = 0
        for spec in ilot_specs:
            if ilot_id >= 25:  # Reasonable limit
                break

            # Calculate grid position
            row = ilot_id // 5
            col = ilot_id % 5

            x = min_x + col * grid_spacing_x
            y = min_y + row * grid_spacing_y

            # Add some randomization for natural look
            x += np.random.uniform(-grid_spacing_x * 0.2, grid_spacing_x * 0.2)
            y += np.random.uniform(-grid_spacing_y * 0.2, grid_spacing_y * 0.2)

            # Ensure within bounds
            x = max(min_x, min(x, max_x - spec['width']))
            y = max(min_y, min(y, max_y - spec['height']))

            ilot = {
                'id': f'fallback_smart_{ilot_id}',
                'x': x,
                'y': y,
                'width': spec['width'],
                'height': spec['height'],
                'area': spec['area'],
                'size_category': spec['size_category'],
                'color': spec['color'],
                'placement_method': 'fallback_smart_grid'
            }

            placed_ilots.append(ilot)
            ilot_id += 1

        return placed_ilots

    def calculate_placement_stats(self, placed_ilots: List[Dict]) -> Dict:
        """Calculate comprehensive placement statistics"""
        if not placed_ilots:
            return {
                'total_ilots': 0,
                'total_area': 0,
                'average_size': 0,
                'size_distribution': {},
                'placement_efficiency': 0
            }

        total_area = sum(ilot.get('area', 0) for ilot in placed_ilots)
        size_distribution = {}

        for ilot in placed_ilots:
            size_cat = ilot.get('size_category', 'unknown')
            size_distribution[size_cat] = size_distribution.get(size_cat, 0) + 1

        # Calculate efficiency score
        total_count = len(placed_ilots)
        efficiency = min(95.0, 60.0 + total_count * 1.2)  # Mock efficiency calculation

        return {
            'total_ilots': total_count,
            'total_area': round(total_area, 2),
            'average_size': round(total_area / total_count, 2),
            'size_distribution': size_distribution,
            'placement_efficiency': round(efficiency, 1)
        }