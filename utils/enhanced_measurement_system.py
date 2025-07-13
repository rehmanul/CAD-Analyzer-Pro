"""
Enhanced Measurement System - Precise Area Calculations
Creates exact measurement displays matching reference Image 3
with precise area calculations and professional labeling
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional
import math

class EnhancedMeasurementSystem:
    """Professional measurement system for CAD floor plans"""
    
    def __init__(self):
        self.measurement_colors = {
            'text': '#8B5CF6',           # Purple measurement text
            'lines': '#6366F1',          # Blue measurement lines
            'background': 'rgba(255,255,255,0.9)',  # White background for text
            'border': '#8B5CF6'          # Purple border for text boxes
        }
        
        self.measurement_styles = {
            'font_size': 10,
            'line_width': 1.0,
            'text_border_width': 1,
            'leader_line_length': 15
        }
    
    def add_precise_measurements(self, fig: go.Figure, ilots: List[Dict], analysis_data: Dict):
        """Add precise measurements exactly like reference Image 3"""
        print(f"Adding precise measurements for {len(ilots)} îlots")
        
        for i, ilot in enumerate(ilots):
            try:
                # Calculate precise area
                area = self._calculate_precise_area(ilot)
                
                # Get position
                x = ilot.get('x', ilot.get('center_x', 0))
                y = ilot.get('y', ilot.get('center_y', 0))
                
                # Add measurement annotation
                self._add_measurement_annotation(fig, x, y, area, i + 1)
                
            except Exception as e:
                print(f"Error adding measurement for îlot {i}: {e}")
                continue
    
    def add_corridor_measurements(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridor length measurements"""
        for i, corridor in enumerate(corridors):
            try:
                path = corridor.get('path', [])
                if len(path) >= 2:
                    # Calculate corridor length
                    length = self._calculate_path_length(path)
                    
                    # Get midpoint for label
                    mid_idx = len(path) // 2
                    mid_x = path[mid_idx][0]
                    mid_y = path[mid_idx][1]
                    
                    # Add length annotation
                    fig.add_annotation(
                        x=mid_x,
                        y=mid_y,
                        text=f"{length:.1f}m",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor=self.measurement_colors['lines'],
                        font=dict(
                            size=self.measurement_styles['font_size'] - 1,
                            color=self.measurement_colors['text']
                        ),
                        bgcolor=self.measurement_colors['background'],
                        bordercolor=self.measurement_colors['border'],
                        borderwidth=1
                    )
                    
            except Exception as e:
                continue
    
    def add_room_dimensions(self, fig: go.Figure, analysis_data: Dict):
        """Add room dimension measurements"""
        bounds = analysis_data.get('bounds', {})
        
        # Calculate overall dimensions
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 80) - bounds.get('min_y', 0)
        
        # Add dimension annotations
        center_x = (bounds.get('max_x', 100) + bounds.get('min_x', 0)) / 2
        center_y = (bounds.get('max_y', 80) + bounds.get('min_y', 0)) / 2
        
        # Width annotation (bottom)
        fig.add_annotation(
            x=center_x,
            y=bounds.get('min_y', 0) - height * 0.05,
            text=f"Largeur: {width:.1f}m",
            showarrow=False,
            font=dict(
                size=self.measurement_styles['font_size'],
                color=self.measurement_colors['text']
            ),
            bgcolor=self.measurement_colors['background'],
            bordercolor=self.measurement_colors['border'],
            borderwidth=1
        )
        
        # Height annotation (left side)
        fig.add_annotation(
            x=bounds.get('min_x', 0) - width * 0.05,
            y=center_y,
            text=f"Hauteur: {height:.1f}m",
            showarrow=False,
            textangle=90,
            font=dict(
                size=self.measurement_styles['font_size'],
                color=self.measurement_colors['text']
            ),
            bgcolor=self.measurement_colors['background'],
            bordercolor=self.measurement_colors['border'],
            borderwidth=1
        )
    
    def _calculate_precise_area(self, ilot: Dict) -> float:
        """Calculate precise area for an îlot"""
        try:
            # Get dimensions
            width = ilot.get('width', 2.0)
            height = ilot.get('height', 1.5)
            
            # Handle different area calculation methods
            if 'area' in ilot:
                return float(ilot['area'])
            elif 'polygon' in ilot:
                # Calculate polygon area using shoelace formula
                return self._calculate_polygon_area(ilot['polygon'])
            else:
                # Calculate rectangular area
                return width * height
                
        except Exception as e:
            print(f"Error calculating area: {e}")
            return 2.0  # Default fallback area
    
    def _calculate_polygon_area(self, polygon_points: List[List[float]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        try:
            if len(polygon_points) < 3:
                return 0.0
            
            n = len(polygon_points)
            area = 0.0
            
            for i in range(n):
                j = (i + 1) % n
                area += polygon_points[i][0] * polygon_points[j][1]
                area -= polygon_points[j][0] * polygon_points[i][1]
            
            return abs(area) / 2.0
            
        except Exception as e:
            print(f"Error calculating polygon area: {e}")
            return 0.0
    
    def _calculate_path_length(self, path: List[List[float]]) -> float:
        """Calculate total length of a path"""
        try:
            total_length = 0.0
            
            for i in range(len(path) - 1):
                x1, y1 = path[i][0], path[i][1]
                x2, y2 = path[i + 1][0], path[i + 1][1]
                
                segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_length += segment_length
            
            return total_length
            
        except Exception as e:
            print(f"Error calculating path length: {e}")
            return 0.0
    
    def _add_measurement_annotation(self, fig: go.Figure, x: float, y: float, area: float, ilot_num: int):
        """Add measurement annotation exactly like reference Image 3"""
        try:
            # Format area text
            area_text = f"{area:.1f}m²"
            
            # Add annotation with exact styling from reference
            fig.add_annotation(
                x=x,
                y=y,
                text=area_text,
                showarrow=False,
                font=dict(
                    size=self.measurement_styles['font_size'],
                    color=self.measurement_colors['text'],
                    family="Arial"
                ),
                bgcolor=self.measurement_colors['background'],
                bordercolor=self.measurement_colors['border'],
                borderwidth=self.measurement_styles['text_border_width'],
                # Position the text slightly above center for better visibility
                yshift=3
            )
            
        except Exception as e:
            print(f"Error adding measurement annotation: {e}")
    
    def create_measurement_summary(self, ilots: List[Dict], corridors: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive measurement summary"""
        summary = {
            'total_ilots': len(ilots),
            'total_corridors': len(corridors),
            'total_area': 0.0,
            'total_corridor_length': 0.0,
            'ilot_areas': [],
            'corridor_lengths': [],
            'statistics': {}
        }
        
        # Calculate îlot areas
        for ilot in ilots:
            area = self._calculate_precise_area(ilot)
            summary['ilot_areas'].append(area)
            summary['total_area'] += area
        
        # Calculate corridor lengths
        for corridor in corridors:
            path = corridor.get('path', [])
            length = self._calculate_path_length(path)
            summary['corridor_lengths'].append(length)
            summary['total_corridor_length'] += length
        
        # Calculate statistics
        if summary['ilot_areas']:
            summary['statistics']['average_ilot_area'] = np.mean(summary['ilot_areas'])
            summary['statistics']['min_ilot_area'] = np.min(summary['ilot_areas'])
            summary['statistics']['max_ilot_area'] = np.max(summary['ilot_areas'])
        
        if summary['corridor_lengths']:
            summary['statistics']['average_corridor_length'] = np.mean(summary['corridor_lengths'])
            summary['statistics']['total_corridor_length'] = summary['total_corridor_length']
        
        return summary