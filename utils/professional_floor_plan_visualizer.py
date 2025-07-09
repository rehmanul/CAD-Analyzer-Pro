"""
Professional Floor Plan Visualizer
Creates modern, sophisticated floor plan visualizations matching reference images
with clean styling, proper room labeling, and architectural presentation
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
import colorsys

class ProfessionalFloorPlanVisualizer:
    """Professional floor plan visualizer matching reference image styles"""
    
    def __init__(self):
        # Modern architectural color palette
        self.colors = {
            'walls': '#2d3748',           # Dark charcoal for walls
            'floor': '#f8f9fa',           # Light gray floor
            'restricted': '#fbb6ce',      # Soft pink for restricted areas
            'entrances': '#9ae6b4',       # Soft green for entrances
            'room_small': '#fed7d7',      # Light beige/pink for small rooms
            'room_medium': '#fefcbf',     # Light yellow for medium rooms
            'room_large': '#c6f6d5',      # Light green for large rooms
            'room_xlarge': '#e9d8fd',     # Light purple for extra large
            'corridors': '#bee3f8',       # Light blue for corridors
            'furniture': '#8b5cf6',       # Purple for furniture
            'text_dark': '#1a202c',       # Dark text
            'text_medium': '#4a5568',     # Medium text
            'text_light': '#718096',      # Light text
            'accent': '#4299e1',          # Blue accent
            'background': '#ffffff',      # Pure white
            'grid': '#e2e8f0'            # Very light grid
        }
        
        # Typography settings
        self.fonts = {
            'title': {'family': 'Inter, Arial, sans-serif', 'size': 24, 'color': self.colors['text_dark']},
            'subtitle': {'family': 'Inter, Arial, sans-serif', 'size': 16, 'color': self.colors['text_medium']},
            'room_label': {'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': self.colors['text_dark']},
            'measurement': {'family': 'Inter, Arial, sans-serif', 'size': 10, 'color': self.colors['text_medium']},
            'legend': {'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': self.colors['text_dark']}
        }
    
    def create_professional_floor_plan(self, analysis_data: Dict, ilots: List[Dict] = None, 
                                     corridors: List[Dict] = None, show_3d: bool = False) -> go.Figure:
        """Create professional floor plan visualization"""
        
        if show_3d:
            return self._create_3d_floor_plan(analysis_data, ilots, corridors)
        else:
            return self._create_2d_floor_plan(analysis_data, ilots, corridors)
    
    def _create_2d_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create modern 2D floor plan with architectural styling"""
        
        fig = go.Figure()
        
        # Set up professional layout
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add floor background
        self._add_floor_background(fig, bounds)
        
        # Add walls with proper thickness and styling
        self._add_professional_walls(fig, analysis_data.get('walls', []))
        
        # Add rooms/zones with modern colors and labels
        self._add_professional_rooms(fig, ilots or [])
        
        # Add entrances with clear marking
        self._add_professional_entrances(fig, analysis_data.get('entrances', []))
        
        # Add corridors with proper styling
        if corridors:
            self._add_professional_corridors(fig, corridors)
        
        # Add measurements and dimensions
        self._add_professional_measurements(fig, bounds, ilots or [])
        
        # Set professional layout and styling
        self._apply_professional_layout(fig, bounds)
        
        return fig
    
    def _create_3d_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create 3D isometric floor plan view"""
        
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Create 3D room volumes
        if ilots:
            for i, room in enumerate(ilots):
                self._add_3d_room(fig, room, i)
        
        # Add 3D walls
        self._add_3d_walls(fig, analysis_data.get('walls', []))
        
        # Set 3D layout
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    projection=dict(type='orthographic')
                ),
                aspectmode='cube',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor=self.colors['background']
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            title={
                'text': '3D Floor Plan Visualization',
                'x': 0.5,
                'font': self.fonts['title']
            },
            showlegend=True,
            width=1200,
            height=800
        )
        
        return fig
    
    def _add_floor_background(self, fig: go.Figure, bounds: Dict):
        """Add floor background with subtle texture"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add floor rectangle
        fig.add_shape(
            type="rect",
            x0=min_x, y0=min_y, x1=max_x, y1=max_y,
            fillcolor=self.colors['floor'],
            line=dict(color=self.colors['walls'], width=2),
            layer="below"
        )
    
    def _add_professional_walls(self, fig: go.Figure, walls: List):
        """Add walls with professional thickness and styling"""
        for wall in walls:
            if len(wall) >= 2:
                x_coords = []
                y_coords = []
                
                for point in wall:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                
                if len(x_coords) >= 2:
                    # Close the wall if it's a polygon
                    if len(x_coords) > 2:
                        x_coords.append(x_coords[0])
                        y_coords.append(y_coords[0])
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(
                            color=self.colors['walls'],
                            width=4
                        ),
                        name='Walls',
                        showlegend=True,
                        hoverinfo='name'
                    ))
    
    def _add_professional_rooms(self, fig: go.Figure, rooms: List[Dict]):
        """Add rooms with modern colors and professional labels"""
        room_types = {}
        
        for room in rooms:
            size_cat = room.get('size_category', 'medium')
            
            # Group by size category for legend
            if size_cat not in room_types:
                room_types[size_cat] = []
            room_types[size_cat].append(room)
        
        # Add rooms by category with proper color coding
        for size_cat, room_list in room_types.items():
            # Map size categories to proper colors
            color_map = {
                'small': '#fed7d7',      # Light pink for small (0-1 m²)
                'medium': '#fefcbf',     # Light yellow for medium (1-3 m²)
                'large': '#c6f6d5',      # Light green for large (3-5 m²)
                'xlarge': '#e9d8fd'      # Light purple for xlarge (5-10 m²)
            }
            color = color_map.get(size_cat, self.colors['room_medium'])
            
            for i, room in enumerate(room_list):
                x = room.get('x', 0)
                y = room.get('y', 0)
                width = room.get('width', 3)
                height = room.get('height', 2)
                area = room.get('area', width * height)
                
                # Add room rectangle with proper color
                fig.add_shape(
                    type="rect",
                    x0=x, y0=y, x1=x+width, y1=y+height,
                    fillcolor=color,
                    line=dict(color=self.colors['walls'], width=2),
                    opacity=0.9
                )
                
                # Add room label with area
                fig.add_annotation(
                    x=x + width/2,
                    y=y + height/2,
                    text=f"{area:.1f} m²",
                    showarrow=False,
                    font=dict(
                        family='Inter, Arial, sans-serif', 
                        size=12, 
                        color='#1a202c'
                    ),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=self.colors['text_dark'],
                    borderwidth=1,
                    borderpad=4
                )
            
            # Add legend entry with proper color
            if room_list:
                size_range_map = {
                    'small': '0-1 m²',
                    'medium': '1-3 m²', 
                    'large': '3-5 m²',
                    'xlarge': '5-10 m²'
                }
                size_range = size_range_map.get(size_cat, size_cat)
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=15,
                        symbol='square'
                    ),
                    name=f'{size_range} ({len(room_list)})',
                    showlegend=True
                ))
    
    def _add_professional_entrances(self, fig: go.Figure, entrances: List):
        """Add entrances with clear architectural symbols"""
        for i, entrance in enumerate(entrances):
            if len(entrance) >= 2:
                x_coords = [point[0] for point in entrance]
                y_coords = [point[1] for point in entrance]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    line=dict(
                        color=self.colors['entrances'],
                        width=6
                    ),
                    marker=dict(
                        color=self.colors['entrances'],
                        size=10,
                        symbol='diamond'
                    ),
                    name='Entrances' if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo='name'
                ))
    
    def _add_professional_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors with modern architectural styling"""
        for i, corridor in enumerate(corridors):
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
                        width=3,
                        dash='dot'
                    ),
                    name='Corridors' if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo='name'
                ))
    
    def _add_professional_measurements(self, fig: go.Figure, bounds: Dict, rooms: List[Dict]):
        """Add professional measurements and dimensions"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Add overall dimensions
        fig.add_annotation(
            x=(min_x + max_x) / 2,
            y=min_y - height * 0.08,
            text=f"{width:.1f}m",
            showarrow=False,
            font=self.fonts['measurement'],
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=self.colors['accent'],
            borderwidth=1
        )
        
        fig.add_annotation(
            x=min_x - width * 0.08,
            y=(min_y + max_y) / 2,
            text=f"{height:.1f}m",
            showarrow=False,
            textangle=-90,
            font=self.fonts['measurement'],
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=self.colors['accent'],
            borderwidth=1
        )
        
        # Add room measurements
        total_area = sum(room.get('area', 0) for room in rooms)
        if total_area > 0:
            fig.add_annotation(
                x=max_x,
                y=max_y + height * 0.05,
                text=f"Total Area: {total_area:.1f} m²",
                showarrow=False,
                font=self.fonts['subtitle'],
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=self.colors['accent'],
                borderwidth=2,
                xanchor='right'
            )
    
    def _add_3d_room(self, fig: go.Figure, room: Dict, index: int):
        """Add 3D room volume"""
        x = room.get('x', 0)
        y = room.get('y', 0)
        width = room.get('width', 3)
        height = room.get('height', 2)
        z_height = 3.0  # Standard room height
        
        # Create 3D box for room
        vertices = [
            [x, y, 0], [x+width, y, 0], [x+width, y+height, 0], [x, y+height, 0],  # bottom
            [x, y, z_height], [x+width, y, z_height], [x+width, y+height, z_height], [x, y+height, z_height]  # top
        ]
        
        size_cat = room.get('size_category', 'medium')
        
        # Use the same color mapping as 2D
        color_map = {
            'small': '#fed7d7',      # Light pink for small (0-1 m²)
            'medium': '#fefcbf',     # Light yellow for medium (1-3 m²)
            'large': '#c6f6d5',      # Light green for large (3-5 m²)
            'xlarge': '#e9d8fd'      # Light purple for xlarge (5-10 m²)
        }
        color = color_map.get(size_cat, '#fefcbf')
        
        size_range_map = {
            'small': '0-1 m²',
            'medium': '1-3 m²', 
            'large': '3-5 m²',
            'xlarge': '5-10 m²'
        }
        size_range = size_range_map.get(size_cat, size_cat)
        
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in vertices],
            y=[v[1] for v in vertices],
            z=[v[2] for v in vertices],
            color=color,
            opacity=0.8,
            name=f'{size_range}',
            showlegend=True
        ))
    
    def _add_3d_walls(self, fig: go.Figure, walls: List):
        """Add 3D walls"""
        for wall in walls:
            if len(wall) >= 2:
                for i in range(len(wall) - 1):
                    x1, y1 = wall[i][0], wall[i][1]
                    x2, y2 = wall[i+1][0], wall[i+1][1]
                    
                    # Create wall as 3D surface
                    fig.add_trace(go.Mesh3d(
                        x=[x1, x2, x2, x1],
                        y=[y1, y2, y2, y1],
                        z=[0, 0, 3, 3],
                        color=self.colors['walls'],
                        opacity=0.9,
                        name='Walls',
                        showlegend=False
                    ))
    
    def _apply_professional_layout(self, fig: go.Figure, bounds: Dict):
        """Apply professional layout and styling"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Calculate padding
        width = max_x - min_x
        height = max_y - min_y
        padding = max(width, height) * 0.15
        
        fig.update_layout(
            title={
                'text': "Professional Floor Plan",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': self.fonts['title']
            },
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            width=1200,
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=self.colors['grid'],
                borderwidth=1,
                font=self.fonts['legend']
            ),
            xaxis=dict(
                range=[min_x - padding, max_x + padding],
                showgrid=True,
                gridcolor=self.colors['grid'],
                gridwidth=1,
                zeroline=False,
                showticklabels=True,
                title="Width (m)",
                title_font=self.fonts['measurement'],
                tickfont=self.fonts['measurement'],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[min_y - padding, max_y + padding],
                showgrid=True,
                gridcolor=self.colors['grid'],
                gridwidth=1,
                zeroline=False,
                showticklabels=True,
                title="Height (m)",
                title_font=self.fonts['measurement'],
                tickfont=self.fonts['measurement']
            )
        )