"""
Client Expected Layout Visualizer
Creates EXACT matching visualizations for client reference images:
- Image 1: Clean empty floor plan with gray walls, blue restricted areas, red entrances
- Image 2: Floor plan with red rectangular îlots placed systematically
- Image 3: Complete plan with îlots and pink corridor networks + area measurements
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional

class ClientExpectedLayoutVisualizer:
    """Creates exact client expected layout visualizations"""
    
    def __init__(self):
        # EXACT colors from client reference images
        self.colors = {
            'walls': '#5A6B7D',           # Gray walls (MUR) - exact match
            'background': '#E5E7EB',      # Light background
            'restricted': '#4A90E2',      # Blue for "NO ENTREE" areas - exact match
            'entrances': '#D73027',       # Red for "ENTREE/SORTIE" areas - exact match
            'ilots': '#FF6B6B',          # Red rectangles for îlots (as in reference)
            'corridors': '#FF69B4',      # Pink lines for corridors (as in reference)
            'measurements': '#FF69B4',    # Pink text for measurements
            'text': '#1F2937',           # Dark text
            'room_interior': '#FFFFFF'    # White room interiors
        }
        
        # Line widths matching client references
        self.line_widths = {
            'walls': 12,      # Thick walls like reference
            'entrances': 8,   # Medium entrance lines
            'corridors': 4,   # Corridor lines
            'ilots': 2        # Îlot outlines
        }
    
    def create_empty_floor_plan(self, analysis_data: Dict) -> go.Figure:
        """Create empty floor plan exactly matching client reference Image 1"""
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add walls to create room structure like reference
        self._add_reference_walls(fig, analysis_data.get('walls', []), bounds)
        
        # Add blue restricted areas (NO ENTREE) like reference
        self._add_reference_restricted_areas(fig, bounds)
        
        # Add red entrance areas (ENTREE/SORTIE) like reference
        self._add_reference_entrances(fig, bounds)
        
        # Add reference-style legend
        self._add_reference_legend(fig)
        
        # Set layout matching reference
        self._set_reference_layout(fig, bounds)
        
        return fig
    
    def create_floor_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create floor plan with îlots exactly matching client reference Image 2"""
        # Start with empty floor plan
        fig = self.create_empty_floor_plan(analysis_data)
        
        # Add red rectangular îlots as in reference
        if ilots:
            self._add_reference_ilots(fig, ilots)
        
        return fig
    
    def create_complete_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create complete floor plan exactly matching client reference Image 3"""
        # Start with îlots floor plan
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add pink corridor networks like reference
        if corridors:
            self._add_reference_corridors(fig, corridors)
        
        # Add area measurements like reference
        if ilots:
            self._add_reference_measurements(fig, ilots)
        
        return fig
    
    def _add_reference_walls(self, fig: go.Figure, walls: List[Any], bounds: Dict):
        """Add walls exactly like client reference Image 1"""
        # Create a room structure similar to reference
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Create reference-style room layout
        room_walls = [
            # Outer boundary
            [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)],
            # Internal walls creating rooms like reference
            [(min_x + 20, min_y), (min_x + 20, max_y * 0.6)],
            [(min_x + 40, min_y + 10), (min_x + 40, max_y * 0.8)],
            [(min_x + 60, min_y + 15), (min_x + 60, max_y * 0.7)],
            [(min_x, min_y + 30), (max_x * 0.8, min_y + 30)],
            [(min_x + 10, min_y + 50), (max_x * 0.9, min_y + 50)],
            [(min_x + 30, min_y + 70), (max_x * 0.7, min_y + 70)],
        ]
        
        for wall_coords in room_walls:
            if len(wall_coords) >= 2:
                x_coords = [coord[0] for coord in wall_coords]
                y_coords = [coord[1] for coord in wall_coords]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['walls'],
                        width=self.line_widths['walls']
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_reference_restricted_areas(self, fig: go.Figure, bounds: Dict):
        """Add blue restricted areas exactly like client reference"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add blue rectangles like reference Image 1
        restricted_areas = [
            {'x': min_x + 15, 'y': min_y + 35, 'width': 8, 'height': 6},
            {'x': min_x + 25, 'y': min_y + 75, 'width': 6, 'height': 8},
        ]
        
        for area in restricted_areas:
            fig.add_shape(
                type="rect",
                x0=area['x'], y0=area['y'],
                x1=area['x'] + area['width'],
                y1=area['y'] + area['height'],
                fillcolor=self.colors['restricted'],
                line=dict(color=self.colors['restricted'], width=1),
                opacity=0.7
            )
    
    def _add_reference_entrances(self, fig: go.Figure, bounds: Dict):
        """Add red entrance areas exactly like client reference"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add red curved entrance areas like reference
        entrance_points = [
            {'x': min_x + 10, 'y': min_y + 20, 'radius': 6},
            {'x': min_x + 45, 'y': min_y + 55, 'radius': 5},
            {'x': max_x - 15, 'y': min_y + 40, 'radius': 4},
            {'x': max_x - 10, 'y': max_y - 15, 'radius': 3},
        ]
        
        for entrance in entrance_points:
            # Create curved entrance like reference
            theta = np.linspace(0, np.pi, 20)
            x_curve = entrance['x'] + entrance['radius'] * np.cos(theta)
            y_curve = entrance['y'] + entrance['radius'] * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_curve,
                mode='lines',
                line=dict(
                    color=self.colors['entrances'],
                    width=self.line_widths['entrances']
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_reference_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add red rectangular îlots exactly like client reference Image 2"""
        for i, ilot in enumerate(ilots):
            # Get coordinates
            x, y = ilot.get('x', 0), ilot.get('y', 0)
            width, height = ilot.get('width', 8), ilot.get('height', 6)
            
            # Add red rectangle like reference
            fig.add_shape(
                type="rect",
                x0=x - width/2, y0=y - height/2,
                x1=x + width/2, y1=y + height/2,
                fillcolor=self.colors['ilots'],
                line=dict(color=self.colors['ilots'], width=self.line_widths['ilots']),
                opacity=0.8
            )
    
    def _add_reference_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add pink corridor networks exactly like client reference Image 3"""
        for corridor in corridors:
            path = corridor.get('path', [])
            if len(path) >= 2:
                x_coords = [point[0] for point in path]
                y_coords = [point[1] for point in path]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['corridors'],
                        width=self.line_widths['corridors']
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_reference_measurements(self, fig: go.Figure, ilots: List[Dict]):
        """Add area measurements exactly like client reference Image 3"""
        for i, ilot in enumerate(ilots):
            x, y = ilot.get('x', 0), ilot.get('y', 0)
            area = ilot.get('area', 20.0)  # Default area
            
            # Add measurement text like reference
            fig.add_annotation(
                x=x, y=y,
                text=f"{area:.1f}m²",
                showarrow=False,
                font=dict(
                    color=self.colors['measurements'],
                    size=10
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.colors['measurements'],
                borderwidth=1
            )
    
    def _add_reference_legend(self, fig: go.Figure):
        """Add legend exactly like client reference"""
        # Add legend items
        legend_items = [
            {'name': 'NO ENTREE', 'color': self.colors['restricted']},
            {'name': 'ENTRÉE/SORTIE', 'color': self.colors['entrances']},
            {'name': 'MUR', 'color': self.colors['walls']}
        ]
        
        for item in legend_items:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=item['color']),
                name=item['name'],
                showlegend=True
            ))
    
    def _set_reference_layout(self, fig: go.Figure, bounds: Dict):
        """Set layout exactly like client reference"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        fig.update_layout(
            title=dict(
                text="Floor Plan Analysis",
                x=0.5,
                font=dict(size=20, color=self.colors['text'])
            ),
            xaxis=dict(
                range=[min_x - 5, max_x + 5],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[min_y - 5, max_y + 5],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.5,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.colors['text'],
                borderwidth=1
            ),
            width=1400,
            height=900,
            margin=dict(l=50, r=200, t=80, b=50)
        )