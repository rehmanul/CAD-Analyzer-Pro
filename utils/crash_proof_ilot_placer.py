"""
Crash-Proof Îlot Placer - Prevents crashes with large files
Robust placement system with error handling and memory management
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from shapely.geometry import Polygon, Point, box
import gc
import traceback
from dataclasses import dataclass
import time
import streamlit as st

logger = logging.getLogger(__name__)

@dataclass
class SafeIlot:
    """Safe îlot representation with error handling"""
    id: str
    x: float
    y: float
    width: float
    height: float
    size_category: str
    area: float = 0
    
    def __post_init__(self):
        if self.area == 0:
            self.area = self.width * self.height

class CrashProofIlotPlacer:
    """Crash-proof îlot placement with error handling"""
    
    def __init__(self):
        self.max_ilots = 50  # Reduced for memory efficiency
        self.grid_size = 8.0  # Larger grid for performance
        self.placement_timeout = 15  # Reduced timeout
        
        # Client-specified size distribution
        self.size_distribution = {
            'size_0_1': 0.10,   # 10%
            'size_1_3': 0.25,   # 25%
            'size_3_5': 0.30,   # 30%
            'size_5_10': 0.35   # 35%
        }
        
        self.size_ranges = {
            'size_0_1': (2.0, 3.0),
            'size_1_3': (3.0, 5.0),
            'size_3_5': (5.0, 8.0),
            'size_5_10': (8.0, 12.0)
        }
    
    def place_ilots_safely(self, analysis_data: Dict) -> List[SafeIlot]:
        """Place îlots with comprehensive error handling"""
        try:
            start_time = time.time()
            
            # Extract bounds safely
            bounds = self._extract_bounds_safely(analysis_data)
            if not bounds:
                return self._generate_fallback_ilots()
            
            # Calculate total area
            total_area = (bounds['max_x'] - bounds['min_x']) * (bounds['max_y'] - bounds['min_y'])
            
            # Determine number of îlots based on area
            num_ilots = min(int(total_area / 50), self.max_ilots)  # Limit îlots
            
            if num_ilots < 5:
                num_ilots = 10  # Minimum îlots for demo
            
            # Generate îlots with timeout protection
            ilots = []
            for category, percentage in self.size_distribution.items():
                category_count = max(1, int(num_ilots * percentage))
                
                for i in range(category_count):
                    # Check timeout
                    if time.time() - start_time > self.placement_timeout:
                        logger.warning("Placement timeout reached, using generated îlots")
                        break
                    
                    ilot = self._create_safe_ilot(bounds, category, f"{category}_{i}")
                    if ilot:
                        ilots.append(ilot)
            
            logger.info(f"Generated {len(ilots)} îlots safely")
            return ilots
            
        except Exception as e:
            logger.error(f"Îlot placement failed: {str(e)}")
            st.error(f"Placement error: {str(e)}")
            return self._generate_fallback_ilots()
    
    def _extract_bounds_safely(self, analysis_data: Dict) -> Optional[Dict]:
        """Extract bounds with error handling"""
        try:
            if 'bounds' in analysis_data:
                bounds = analysis_data['bounds']
                if all(key in bounds for key in ['min_x', 'max_x', 'min_y', 'max_y']):
                    return bounds
            
            # Try to extract from entities
            if 'entities' in analysis_data:
                entities = analysis_data['entities']
                if entities and len(entities) > 0:
                    return self._calculate_bounds_from_entities(entities)
            
            # Fallback bounds
            return {
                'min_x': 0, 'max_x': 100,
                'min_y': 0, 'max_y': 80
            }
            
        except Exception as e:
            logger.error(f"Bounds extraction failed: {str(e)}")
            return None
    
    def _calculate_bounds_from_entities(self, entities: List) -> Dict:
        """Calculate bounds from entities"""
        try:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for entity in entities[:100]:  # Limit to prevent crashes
                if isinstance(entity, dict):
                    points = entity.get('points', [])
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            x, y = point[0], point[1]
                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)
            
            # Ensure valid bounds
            if min_x == float('inf'):
                return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}
            
            return {
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y
            }
            
        except Exception as e:
            logger.error(f"Bounds calculation failed: {str(e)}")
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}
    
    def _create_safe_ilot(self, bounds: Dict, category: str, id: str) -> Optional[SafeIlot]:
        """Create îlot with error handling"""
        try:
            # Get size range for category
            size_range = self.size_ranges.get(category, (3.0, 5.0))
            
            # Generate safe dimensions
            width = np.random.uniform(size_range[0], size_range[1])
            height = np.random.uniform(size_range[0], size_range[1])
            
            # Safe placement within bounds
            margin = max(width, height) / 2 + 1
            safe_min_x = bounds['min_x'] + margin
            safe_max_x = bounds['max_x'] - margin
            safe_min_y = bounds['min_y'] + margin
            safe_max_y = bounds['max_y'] - margin
            
            # Check if placement is possible
            if safe_min_x >= safe_max_x or safe_min_y >= safe_max_y:
                # Fallback to center placement
                x = (bounds['min_x'] + bounds['max_x']) / 2
                y = (bounds['min_y'] + bounds['max_y']) / 2
                width = height = 2.0
            else:
                x = np.random.uniform(safe_min_x, safe_max_x)
                y = np.random.uniform(safe_min_y, safe_max_y)
            
            return SafeIlot(
                id=id,
                x=x,
                y=y,
                width=width,
                height=height,
                size_category=category,
                area=width * height
            )
            
        except Exception as e:
            logger.error(f"Îlot creation failed: {str(e)}")
            return None
    
    def _generate_fallback_ilots(self) -> List[SafeIlot]:
        """Generate fallback îlots for demo"""
        try:
            fallback_ilots = []
            
            # Generate a simple grid of îlots
            for i in range(20):
                x = 20 + (i % 5) * 15
                y = 20 + (i // 5) * 15
                
                # Distribute by size
                if i < 2:
                    category = 'size_0_1'
                    size = 2.5
                elif i < 7:
                    category = 'size_1_3'
                    size = 4.0
                elif i < 13:
                    category = 'size_3_5'
                    size = 6.0
                else:
                    category = 'size_5_10'
                    size = 9.0
                
                ilot = SafeIlot(
                    id=f"fallback_{i}",
                    x=x,
                    y=y,
                    width=size,
                    height=size,
                    size_category=category,
                    area=size * size
                )
                fallback_ilots.append(ilot)
            
            logger.info("Generated fallback îlots")
            return fallback_ilots
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {str(e)}")
            return []
    
    def get_placement_metrics(self, ilots: List[SafeIlot]) -> Dict:
        """Get placement metrics safely"""
        try:
            if not ilots:
                return {
                    'total_ilots': 0,
                    'size_distribution': {},
                    'total_area': 0,
                    'average_size': 0
                }
            
            # Count by size category
            size_counts = {}
            total_area = 0
            
            for ilot in ilots:
                category = ilot.size_category
                size_counts[category] = size_counts.get(category, 0) + 1
                total_area += ilot.area
            
            # Calculate distribution percentages
            total_count = len(ilots)
            size_distribution = {}
            for category, count in size_counts.items():
                size_distribution[category] = (count / total_count) * 100
            
            return {
                'total_ilots': total_count,
                'size_distribution': size_distribution,
                'total_area': total_area,
                'average_size': total_area / total_count if total_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            return {
                'total_ilots': 0,
                'size_distribution': {},
                'total_area': 0,
                'average_size': 0
            }

# Global instance
crash_proof_placer = CrashProofIlotPlacer()