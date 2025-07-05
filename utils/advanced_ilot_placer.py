"""
Advanced Îlot Placement System
Intelligently places îlots based on zone detection and constraints
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from scipy.spatial import distance
from scipy.optimize import differential_evolution
import random

class AdvancedIlotPlacer:
    """Advanced îlot placement with AI-driven optimization"""
    
    def __init__(self):
        self.ilot_categories = {
            'small': {'min': 0, 'max': 1, 'color': '#FFE5B4'},
            'medium': {'min': 1, 'max': 3, 'color': '#FFD700'},
            'large': {'min': 3, 'max': 5, 'color': '#FFA500'},
            'extra_large': {'min': 5, 'max': 10, 'color': '#FF8C00'}
        }
        
        self.placement_rules = {
            'min_wall_clearance': 0.3,  # 30cm from walls
            'min_entrance_clearance': 2.0,  # 2m from entrances
            'min_restricted_clearance': 1.0,  # 1m from restricted areas
            'min_ilot_spacing': 0.8,  # 80cm between îlots
            'corridor_clearance': 1.2,  # 1.2m for corridors
            'emergency_route_clearance': 1.5  # 1.5m for emergency routes
        }
    
    def place_ilots_intelligently(self, zones: Dict[str, List[Dict]], 
                                 ilot_config: Dict[str, float],
                                 constraints: Dict[str, Any]) -> List[Dict]:
        """
        Place îlots intelligently based on detected zones
        """
        # Calculate total îlots needed
        total_ilots = self._calculate_total_ilots(zones, constraints)
        
        # Generate îlot specifications based on config
        ilot_specs = self._generate_ilot_specs(ilot_config, total_ilots)
        
        # Identify placeable zones
        placeable_zones = self._identify_placeable_zones(zones)
        
        # Place îlots using advanced algorithms
        placed_ilots = self._place_ilots_advanced(ilot_specs, placeable_zones, zones)
        
        # Optimize placement
        optimized_ilots = self._optimize_placement(placed_ilots, zones)
        
        # Generate facing corridors
        corridors = self._generate_facing_corridors(optimized_ilots, zones)
        
        # Return ilots list with additional data
        for ilot in optimized_ilots:
            ilot['corridors'] = corridors
            ilot['metrics'] = self._calculate_placement_metrics(optimized_ilots, zones)
        
        return optimized_ilots
    
    def _calculate_total_ilots(self, zones: Dict, constraints: Dict) -> int:
        """Calculate optimal number of îlots based on available space"""
        total_area = 0
        
        # Calculate placeable area
        for room in zones.get('rooms', []):
            if room['subtype'] not in ['corridor', 'restricted']:
                total_area += room['area']
        
        for open_space in zones.get('open_spaces', []):
            total_area += open_space['area']
        
        # Average îlot size including spacing
        avg_ilot_size = 4.0  # sqm including circulation space
        
        # Calculate based on density factor
        density_factor = constraints.get('density_factor', 0.6)
        
        return int((total_area * density_factor) / avg_ilot_size)
    
    def _generate_ilot_specs(self, config: Dict, total_count: int) -> List[Dict]:
        """Generate îlot specifications based on configuration"""
        specs = []
        
        for category, percentage in config.items():
            count = int(total_count * (percentage / 100))
            size_range = self.ilot_categories.get(category, self.ilot_categories['medium'])
            
            for i in range(count):
                # Generate size within range
                size = random.uniform(size_range['min'], size_range['max'])
                
                # Generate dimensions
                aspect_ratio = random.uniform(0.6, 1.4)
                width = np.sqrt(size * aspect_ratio)
                height = size / width
                
                spec = {
                    'id': f'ilot_{len(specs)}',
                    'category': category,
                    'area': size,
                    'width': width,
                    'height': height,
                    'color': size_range['color'],
                    'rotation': 0,  # Will be optimized later
                    'type': self._determine_ilot_type(category)
                }
                specs.append(spec)
        
        return specs
    
    def _identify_placeable_zones(self, zones: Dict) -> List[Dict]:
        """Identify zones suitable for îlot placement"""
        placeable = []
        
        # Rooms suitable for îlots
        for room in zones.get('rooms', []):
            if room['subtype'] in ['office', 'open_office', 'meeting_room', 'general_room']:
                placeable_zone = {
                    'zone': room,
                    'type': 'room',
                    'priority': 1,
                    'capacity': self._estimate_zone_capacity(room)
                }
                placeable.append(placeable_zone)
        
        # Open spaces
        for space in zones.get('open_spaces', []):
            placeable_zone = {
                'zone': space,
                'type': 'open_space',
                'priority': 2,
                'capacity': space.get('capacity', 10)
            }
            placeable.append(placeable_zone)
        
        return sorted(placeable, key=lambda x: x['priority'])
    
    def _place_ilots_advanced(self, ilot_specs: List[Dict], 
                            placeable_zones: List[Dict],
                            all_zones: Dict) -> List[Dict]:
        """Advanced îlot placement using intelligent algorithms"""
        placed_ilots = []
        
        # Sort îlots by size (larger first for better packing)
        sorted_specs = sorted(ilot_specs, key=lambda x: x['area'], reverse=True)
        
        for zone_info in placeable_zones:
            zone = zone_info['zone']
            zone_geom = zone['geometry']
            
            # Get placement strategy based on zone type
            strategy = self._get_placement_strategy(zone)
            
            # Place îlots in this zone
            for spec in sorted_specs[:]:
                if len(placed_ilots) >= len(ilot_specs):
                    break
                
                # Find optimal position
                position = self._find_optimal_position(spec, zone_geom, placed_ilots, 
                                                     all_zones, strategy)
                
                if position:
                    ilot = {
                        **spec,
                        'position': position,
                        'zone_id': zone['id'],
                        'geometry': self._create_ilot_geometry(spec, position)
                    }
                    
                    # Check all constraints
                    if self._validate_placement(ilot, placed_ilots, all_zones):
                        placed_ilots.append(ilot)
                        sorted_specs.remove(spec)
        
        return placed_ilots
    
    def _get_placement_strategy(self, zone: Dict) -> str:
        """Determine placement strategy based on zone characteristics"""
        if zone['subtype'] == 'open_office':
            return 'grid'
        elif zone['subtype'] == 'meeting_room':
            return 'perimeter'
        elif zone.get('type') == 'open_space':
            return 'cluster'
        else:
            return 'optimized'
    
    def _find_optimal_position(self, spec: Dict, zone_geom: Polygon,
                              placed_ilots: List[Dict], all_zones: Dict,
                              strategy: str) -> Optional[Dict]:
        """Find optimal position for îlot based on strategy"""
        if strategy == 'grid':
            return self._find_grid_position(spec, zone_geom, placed_ilots, all_zones)
        elif strategy == 'perimeter':
            return self._find_perimeter_position(spec, zone_geom, placed_ilots, all_zones)
        elif strategy == 'cluster':
            return self._find_cluster_position(spec, zone_geom, placed_ilots, all_zones)
        else:
            return self._find_optimized_position(spec, zone_geom, placed_ilots, all_zones)
    
    def _find_grid_position(self, spec: Dict, zone_geom: Polygon,
                           placed_ilots: List[Dict], all_zones: Dict) -> Optional[Dict]:
        """Find position using grid placement"""
        bounds = zone_geom.bounds
        
        # Calculate grid parameters
        grid_spacing = spec['width'] + self.placement_rules['min_ilot_spacing']
        
        # Generate grid points
        x_points = np.arange(bounds[0] + spec['width']/2, 
                           bounds[2] - spec['width']/2, 
                           grid_spacing)
        y_points = np.arange(bounds[1] + spec['height']/2,
                           bounds[3] - spec['height']/2,
                           grid_spacing)
        
        # Find first valid position
        for x in x_points:
            for y in y_points:
                position = {'x': x, 'y': y}
                test_geom = self._create_ilot_geometry(spec, position)
                
                if zone_geom.contains(test_geom):
                    # Check collision with other îlots
                    valid = True
                    for other in placed_ilots:
                        if test_geom.distance(other['geometry']) < self.placement_rules['min_ilot_spacing']:
                            valid = False
                            break
                    
                    if valid:
                        return position
        
        return None
    
    def _find_perimeter_position(self, spec: Dict, zone_geom: Polygon,
                                placed_ilots: List[Dict], all_zones: Dict) -> Optional[Dict]:
        """Find position along zone perimeter"""
        boundary = zone_geom.boundary
        
        # Sample points along boundary
        num_samples = int(boundary.length / 0.5)  # Sample every 0.5m
        
        for i in range(num_samples):
            point = boundary.interpolate(i * 0.5)
            
            # Try different orientations
            for angle in [0, 90, 180, 270]:
                position = {
                    'x': point.x,
                    'y': point.y,
                    'rotation': angle
                }
                
                test_geom = self._create_ilot_geometry(spec, position)
                
                if zone_geom.contains(test_geom) and \
                   self._check_clearances(test_geom, placed_ilots, all_zones):
                    return position
        
        return None
    
    def _find_cluster_position(self, spec: Dict, zone_geom: Polygon,
                             placed_ilots: List[Dict], all_zones: Dict) -> Optional[Dict]:
        """Find position using clustering approach"""
        if not placed_ilots:
            # First îlot - place at centroid
            centroid = zone_geom.centroid
            return {'x': centroid.x, 'y': centroid.y}
        
        # Find position near existing îlots
        for ilot in placed_ilots:
            # Try positions around this îlot
            base_x, base_y = ilot['position']['x'], ilot['position']['y']
            
            for angle in range(0, 360, 45):
                distance = ilot['width'] + spec['width'] + self.placement_rules['min_ilot_spacing']
                x = base_x + distance * np.cos(np.radians(angle))
                y = base_y + distance * np.sin(np.radians(angle))
                
                position = {'x': x, 'y': y}
                test_geom = self._create_ilot_geometry(spec, position)
                
                if zone_geom.contains(test_geom) and \
                   self._check_clearances(test_geom, placed_ilots, all_zones):
                    return position
        
        return None
    
    def _find_optimized_position(self, spec: Dict, zone_geom: Polygon,
                               placed_ilots: List[Dict], all_zones: Dict) -> Optional[Dict]:
        """Find position using optimization algorithm"""
        bounds = zone_geom.bounds
        
        def objective(x):
            position = {'x': x[0], 'y': x[1], 'rotation': x[2]}
            test_geom = self._create_ilot_geometry(spec, position)
            
            # Check basic constraints
            if not zone_geom.contains(test_geom):
                return 1000000
            
            # Calculate quality score
            score = 0
            
            # Distance from walls
            wall_dist = test_geom.distance(zone_geom.boundary)
            score += max(0, self.placement_rules['min_wall_clearance'] - wall_dist) * 100
            
            # Distance from other îlots
            for other in placed_ilots:
                dist = test_geom.distance(other['geometry'])
                score += max(0, self.placement_rules['min_ilot_spacing'] - dist) * 200
            
            # Prefer positions that maintain alignment
            if placed_ilots:
                alignment_score = self._calculate_alignment_score(position, placed_ilots)
                score -= alignment_score * 10
            
            return score
        
        # Optimization bounds
        bounds = [
            (bounds[0] + spec['width']/2, bounds[2] - spec['width']/2),
            (bounds[1] + spec['height']/2, bounds[3] - spec['height']/2),
            (0, 360)
        ]
        
        # Run optimization
        result = differential_evolution(objective, bounds, maxiter=50)
        
        if result.fun < 1000:
            return {'x': result.x[0], 'y': result.x[1], 'rotation': result.x[2]}
        
        return None
    
    def _create_ilot_geometry(self, spec: Dict, position: Dict) -> Polygon:
        """Create îlot geometry at given position"""
        x, y = position['x'], position['y']
        w, h = spec['width'] / 2, spec['height'] / 2
        rotation = position.get('rotation', 0)
        
        # Create rectangle
        corners = [
            (-w, -h), (w, -h), (w, h), (-w, h)
        ]
        
        # Rotate if needed
        if rotation != 0:
            angle_rad = np.radians(rotation)
            rotated_corners = []
            for cx, cy in corners:
                rx = cx * np.cos(angle_rad) - cy * np.sin(angle_rad)
                ry = cx * np.sin(angle_rad) + cy * np.cos(angle_rad)
                rotated_corners.append((rx, ry))
            corners = rotated_corners
        
        # Translate to position
        polygon_points = [(x + cx, y + cy) for cx, cy in corners]
        
        return Polygon(polygon_points)
    
    def _validate_placement(self, ilot: Dict, placed_ilots: List[Dict],
                          all_zones: Dict) -> bool:
        """Validate îlot placement against all constraints"""
        geom = ilot['geometry']
        
        # Check entrance clearance
        for entrance in all_zones.get('entrances', []):
            if geom.distance(entrance['geometry']) < self.placement_rules['min_entrance_clearance']:
                return False
        
        # Check restricted area clearance
        for restricted in all_zones.get('restricted', []):
            if geom.distance(restricted['geometry']) < self.placement_rules['min_restricted_clearance']:
                return False
        
        # Check corridor clearance
        for corridor in all_zones.get('corridors', []):
            if geom.intersects(corridor['geometry']):
                return False
        
        # Check emergency route clearance
        # This would be more complex in practice
        
        return True
    
    def _check_clearances(self, geom: Polygon, placed_ilots: List[Dict],
                         all_zones: Dict) -> bool:
        """Check all clearance requirements"""
        # Check îlot spacing
        for other in placed_ilots:
            if geom.distance(other['geometry']) < self.placement_rules['min_ilot_spacing']:
                return False
        
        return self._validate_placement({'geometry': geom}, placed_ilots, all_zones)
    
    def _calculate_alignment_score(self, position: Dict, placed_ilots: List[Dict]) -> float:
        """Calculate how well aligned this position is with existing îlots"""
        if not placed_ilots:
            return 0
        
        x, y = position['x'], position['y']
        score = 0
        
        # Check horizontal alignment
        for ilot in placed_ilots:
            if abs(ilot['position']['y'] - y) < 0.1:  # Nearly aligned horizontally
                score += 1
        
        # Check vertical alignment
        for ilot in placed_ilots:
            if abs(ilot['position']['x'] - x) < 0.1:  # Nearly aligned vertically
                score += 1
        
        return score
    
    def _optimize_placement(self, ilots: List[Dict], zones: Dict) -> List[Dict]:
        """Post-process optimization of placement"""
        # Align îlots where possible
        self._align_ilots(ilots)
        
        # Optimize rotations
        self._optimize_rotations(ilots, zones)
        
        # Balance distribution
        self._balance_distribution(ilots, zones)
        
        return ilots
    
    def _align_ilots(self, ilots: List[Dict]):
        """Align îlots to create cleaner layouts"""
        # Group by approximate rows
        tolerance = 0.5
        rows = []
        
        for ilot in ilots:
            y = ilot['position']['y']
            placed = False
            
            for row in rows:
                if abs(row['y'] - y) < tolerance:
                    row['ilots'].append(ilot)
                    placed = True
                    break
            
            if not placed:
                rows.append({'y': y, 'ilots': [ilot]})
        
        # Align within rows
        for row in rows:
            if len(row['ilots']) > 1:
                # Calculate average y position
                avg_y = np.mean([ilot['position']['y'] for ilot in row['ilots']])
                
                # Update positions
                for ilot in row['ilots']:
                    ilot['position']['y'] = avg_y
                    ilot['geometry'] = self._create_ilot_geometry(ilot, ilot['position'])
    
    def _optimize_rotations(self, ilots: List[Dict], zones: Dict):
        """Optimize îlot rotations for better space utilization"""
        for ilot in ilots:
            # Find the zone containing this îlot
            zone_geom = None
            for room in zones.get('rooms', []):
                if room['geometry'].contains(ilot['geometry']):
                    zone_geom = room['geometry']
                    break
            
            if zone_geom:
                # Try different rotations
                best_rotation = ilot['position'].get('rotation', 0)
                best_score = self._calculate_rotation_score(ilot, zone_geom, ilots)
                
                for angle in [0, 90, 180, 270]:
                    ilot['position']['rotation'] = angle
                    ilot['geometry'] = self._create_ilot_geometry(ilot, ilot['position'])
                    
                    score = self._calculate_rotation_score(ilot, zone_geom, ilots)
                    if score > best_score:
                        best_score = score
                        best_rotation = angle
                
                ilot['position']['rotation'] = best_rotation
                ilot['geometry'] = self._create_ilot_geometry(ilot, ilot['position'])
    
    def _calculate_rotation_score(self, ilot: Dict, zone_geom: Polygon,
                                all_ilots: List[Dict]) -> float:
        """Calculate quality score for rotation"""
        score = 100
        
        # Penalize if too close to boundary
        dist_to_boundary = ilot['geometry'].distance(zone_geom.boundary)
        if dist_to_boundary < self.placement_rules['min_wall_clearance']:
            score -= 50
        
        # Reward alignment with other îlots
        for other in all_ilots:
            if other['id'] != ilot['id']:
                if abs(ilot['position'].get('rotation', 0) - 
                      other['position'].get('rotation', 0)) < 10:
                    score += 5
        
        return score
    
    def _balance_distribution(self, ilots: List[Dict], zones: Dict):
        """Balance îlot distribution across zones"""
        # Group îlots by zone
        zone_ilots = {}
        for ilot in ilots:
            zone_id = ilot.get('zone_id', 'unknown')
            if zone_id not in zone_ilots:
                zone_ilots[zone_id] = []
            zone_ilots[zone_id].append(ilot)
        
        # Calculate density for each zone
        zone_densities = {}
        for zone_id, zone_ilots_list in zone_ilots.items():
            # Find zone area
            zone_area = 0
            for room in zones.get('rooms', []):
                if room['id'] == zone_id:
                    zone_area = room['area']
                    break
            
            if zone_area > 0:
                density = len(zone_ilots_list) / zone_area
                zone_densities[zone_id] = density
        
        # Rebalance if needed (future enhancement)
    
    def _generate_facing_corridors(self, ilots: List[Dict], zones: Dict) -> List[Dict]:
        """Generate corridors between facing îlots"""
        corridors = []
        
        # Group îlots by proximity
        ilot_groups = self._group_facing_ilots(ilots)
        
        for group in ilot_groups:
            if len(group) >= 2:
                # Check if îlots face each other
                corridor = self._create_corridor_between_ilots(group)
                if corridor:
                    corridors.append(corridor)
        
        return corridors
    
    def _group_facing_ilots(self, ilots: List[Dict]) -> List[List[Dict]]:
        """Group îlots that face each other"""
        groups = []
        processed = set()
        
        for i, ilot1 in enumerate(ilots):
            if ilot1['id'] in processed:
                continue
            
            group = [ilot1]
            processed.add(ilot1['id'])
            
            for j, ilot2 in enumerate(ilots[i+1:], i+1):
                if ilot2['id'] in processed:
                    continue
                
                # Check if they face each other
                if self._are_facing(ilot1, ilot2):
                    group.append(ilot2)
                    processed.add(ilot2['id'])
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _are_facing(self, ilot1: Dict, ilot2: Dict) -> bool:
        """Check if two îlots face each other"""
        # Get centroids
        c1 = ilot1['geometry'].centroid
        c2 = ilot2['geometry'].centroid
        
        # Calculate distance
        dist = c1.distance(c2)
        
        # Check if within reasonable distance
        max_facing_distance = 5.0  # 5 meters
        if dist > max_facing_distance:
            return False
        
        # Check alignment
        dx = abs(c1.x - c2.x)
        dy = abs(c1.y - c2.y)
        
        # Either horizontally or vertically aligned
        if dx < 1.0 or dy < 1.0:
            return True
        
        return False
    
    def _create_corridor_between_ilots(self, ilot_group: List[Dict]) -> Optional[Dict]:
        """Create corridor between facing îlots"""
        if len(ilot_group) < 2:
            return None
        
        # Find the space between îlots
        geoms = [ilot['geometry'] for ilot in ilot_group]
        union = unary_union(geoms)
        
        # Get bounding box of the group
        bounds = union.bounds
        
        # Determine corridor orientation
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        if width > height:
            # Horizontal corridor
            corridor_width = self.placement_rules['corridor_clearance']
            
            # Find y positions of îlots
            y_positions = [ilot['position']['y'] for ilot in ilot_group]
            y_min = min(y_positions) + ilot_group[0]['height'] / 2
            y_max = max(y_positions) - ilot_group[0]['height'] / 2
            
            if y_max - y_min > corridor_width:
                corridor_geom = box(bounds[0], y_min, bounds[2], y_min + corridor_width)
                
                return {
                    'id': f'corridor_{len(self._corridor_id_counter)}',
                    'type': 'facing_corridor',
                    'geometry': corridor_geom,
                    'width': corridor_width,
                    'connects': [ilot['id'] for ilot in ilot_group]
                }
        else:
            # Vertical corridor
            corridor_width = self.placement_rules['corridor_clearance']
            
            # Find x positions of îlots
            x_positions = [ilot['position']['x'] for ilot in ilot_group]
            x_min = min(x_positions) + ilot_group[0]['width'] / 2
            x_max = max(x_positions) - ilot_group[0]['width'] / 2
            
            if x_max - x_min > corridor_width:
                corridor_geom = box(x_min, bounds[1], x_min + corridor_width, bounds[3])
                
                return {
                    'id': f'corridor_{len(self._corridor_id_counter)}',
                    'type': 'facing_corridor',
                    'geometry': corridor_geom,
                    'width': corridor_width,
                    'connects': [ilot['id'] for ilot in ilot_group]
                }
        
        return None
    
    def _calculate_placement_metrics(self, ilots: List[Dict], zones: Dict) -> Dict:
        """Calculate metrics for placement quality"""
        metrics = {
            'total_ilots': len(ilots),
            'coverage_ratio': 0,
            'utilization_score': 0,
            'alignment_score': 0,
            'distribution_score': 0,
            'accessibility_score': 0
        }
        
        # Calculate coverage ratio
        total_ilot_area = sum(ilot['area'] for ilot in ilots)
        total_room_area = sum(room['area'] for room in zones.get('rooms', [])
                             if room['subtype'] not in ['corridor', 'restricted'])
        
        if total_room_area > 0:
            metrics['coverage_ratio'] = (total_ilot_area / total_room_area) * 100
        
        # Calculate utilization score
        metrics['utilization_score'] = min(metrics['coverage_ratio'] / 60 * 100, 100)
        
        # Calculate alignment score
        aligned_count = 0
        for i, ilot1 in enumerate(ilots):
            for ilot2 in ilots[i+1:]:
                if abs(ilot1['position']['x'] - ilot2['position']['x']) < 0.1 or \
                   abs(ilot1['position']['y'] - ilot2['position']['y']) < 0.1:
                    aligned_count += 1
        
        if len(ilots) > 1:
            metrics['alignment_score'] = (aligned_count / (len(ilots) * (len(ilots) - 1) / 2)) * 100
        
        # Calculate distribution score
        zone_counts = {}
        for ilot in ilots:
            zone_id = ilot.get('zone_id', 'unknown')
            zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
        
        if zone_counts:
            variance = np.var(list(zone_counts.values()))
            metrics['distribution_score'] = max(0, 100 - variance * 10)
        
        # Calculate accessibility score
        accessible_ilots = 0
        for ilot in ilots:
            # Check if îlot has clear path to corridor
            has_access = False
            for corridor in zones.get('corridors', []):
                if ilot['geometry'].distance(corridor['geometry']) < 2.0:
                    has_access = True
                    break
            
            if has_access:
                accessible_ilots += 1
        
        if ilots:
            metrics['accessibility_score'] = (accessible_ilots / len(ilots)) * 100
        
        return metrics
    
    def _determine_ilot_type(self, category: str) -> str:
        """Determine îlot type based on category"""
        if category == 'small':
            return 'single_desk'
        elif category == 'medium':
            return 'double_desk'
        elif category == 'large':
            return 'workstation_group'
        else:
            return 'meeting_area'
    
    def _estimate_zone_capacity(self, room: Dict) -> int:
        """Estimate how many îlots can fit in a zone"""
        # Consider usable area (80% of total)
        usable_area = room['area'] * 0.8
        
        # Average space per îlot including circulation
        space_per_ilot = 5.0  # sqm
        
        return int(usable_area / space_per_ilot)
    
    # Initialize corridor ID counter
    _corridor_id_counter = 0