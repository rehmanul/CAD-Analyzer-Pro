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
        """Create realistic 3D architectural visualization like reference image"""
        
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add realistic floor foundation
        self._add_3d_foundation(fig, bounds)
        
        # Create detailed 3D room volumes with realistic textures
        if ilots:
            for i, room in enumerate(ilots):
                self._add_realistic_3d_room(fig, room, i)
        
        # Add realistic 3D walls with thickness and texture
        self._add_realistic_3d_walls(fig, analysis_data.get('walls', []))
        
        # Add furniture and interior details
        if ilots:
            self._add_3d_furniture(fig, ilots)
        
        # Add corridor pathways
        if corridors:
            self._add_3d_corridors(fig, corridors)
        
        # Set professional 3D layout matching reference image quality
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=2.2, y=2.2, z=1.8),  # Better elevated angle
                    projection=dict(type='perspective')
                ),
                aspectmode='data',
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    showbackground=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    showbackground=False,
                    visible=False
                ),
                zaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    showbackground=False,
                    visible=False
                ),
                bgcolor='rgba(248,250,252,0.2)'  # Very light blue-gray background
            ),
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            title={
                'text': '3D Architectural Visualization',
                'x': 0.5,
                'font': dict(family='Inter, Arial, sans-serif', size=22, color='#1a202c', weight='bold')
            },
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=12, family='Inter, Arial, sans-serif')
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
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
    
    def _add_realistic_3d_room(self, fig: go.Figure, room: Dict, index: int):
        """Add realistic 3D room with detailed interior styling"""
        x = room.get('x', 0)
        y = room.get('y', 0)
        width = room.get('width', 3)
        height = room.get('height', 2)
        z_height = 3.2  # Realistic room height
        
        size_cat = room.get('size_category', 'medium')
        
        # Realistic color scheme based on room type
        room_colors = {
            'small': '#f8f6f0',      # Warm cream for small rooms
            'medium': '#fff8e7',     # Light warm white for medium
            'large': '#f0f8f0',      # Very light green for large
            'xlarge': '#f5f0ff'      # Very light lavender for xlarge
        }
        
        # Floor colors for variety
        floor_colors = {
            'small': '#d4af37',      # Light wood tone
            'medium': '#deb887',     # Burlywood
            'large': '#c8b99c',      # Light taupe
            'xlarge': '#d3d3d3'      # Light gray
        }
        
        room_color = room_colors.get(size_cat, '#fff8e7')
        floor_color = floor_colors.get(size_cat, '#deb887')
        
        # Floor surface
        fig.add_trace(go.Mesh3d(
            x=[x, x+width, x+width, x, x],
            y=[y, y, y+height, y+height, y],
            z=[0, 0, 0, 0, 0],
            color=floor_color,
            opacity=0.9,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Ceiling surface
        fig.add_trace(go.Mesh3d(
            x=[x, x+width, x+width, x, x],
            y=[y, y, y+height, y+height, y],
            z=[z_height, z_height, z_height, z_height, z_height],
            color='white',
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Room walls (interior surfaces)
        wall_color = room_color
        
        # Front wall
        fig.add_trace(go.Mesh3d(
            x=[x, x+width, x+width, x],
            y=[y, y, y, y],
            z=[0, 0, z_height, z_height],
            color=wall_color,
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Back wall  
        fig.add_trace(go.Mesh3d(
            x=[x, x+width, x+width, x],
            y=[y+height, y+height, y+height, y+height],
            z=[0, 0, z_height, z_height],
            color=wall_color,
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Left wall
        fig.add_trace(go.Mesh3d(
            x=[x, x, x, x],
            y=[y, y+height, y+height, y],
            z=[0, 0, z_height, z_height],
            color=wall_color,
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Right wall
        fig.add_trace(go.Mesh3d(
            x=[x+width, x+width, x+width, x+width],
            y=[y, y+height, y+height, y],
            z=[0, 0, z_height, z_height],
            color=wall_color,
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add room label in 3D space
        fig.add_trace(go.Scatter3d(
            x=[x + width/2],
            y=[y + height/2],
            z=[z_height + 0.3],
            mode='text',
            text=[f"{room.get('area', width*height):.1f}m²"],
            textfont=dict(size=12, color='#2d3748'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Legend entry for room type
        size_range_map = {
            'small': '0-1 m²',
            'medium': '1-3 m²', 
            'large': '3-5 m²',
            'xlarge': '5-10 m²'
        }
        
        if index < 4:  # Only show legend for first few rooms
            size_range = size_range_map.get(size_cat, size_cat)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(color=room_color, size=8),
                name=f'{size_range}',
                showlegend=True
            ))
    
    def _add_3d_foundation(self, fig: go.Figure, bounds: Dict):
        """Add realistic foundation/ground plane"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Extend foundation beyond building
        padding = 10
        
        fig.add_trace(go.Mesh3d(
            x=[min_x-padding, max_x+padding, max_x+padding, min_x-padding],
            y=[min_y-padding, min_y-padding, max_y+padding, max_y+padding],
            z=[-0.2, -0.2, -0.2, -0.2],
            color='#e8e8e8',
            opacity=0.4,
            showlegend=False,
            hoverinfo='skip'
        ))
        
    def _add_realistic_3d_walls(self, fig: go.Figure, walls: List):
        """Add realistic 3D walls with proper thickness"""
        wall_height = 3.2
        wall_thickness = 0.2
        
        for wall in walls:
            if len(wall) >= 2:
                for i in range(len(wall) - 1):
                    x1, y1 = wall[i][0], wall[i][1]
                    x2, y2 = wall[i+1][0], wall[i+1][1]
                    
                    # Calculate wall normal for thickness
                    dx = x2 - x1
                    dy = y2 - y1
                    length = (dx**2 + dy**2)**0.5
                    if length > 0:
                        nx = -dy / length * wall_thickness
                        ny = dx / length * wall_thickness
                        
                        # Create wall volume
                        wall_vertices_x = [x1, x2, x2+nx, x1+nx, x1, x2, x2+nx, x1+nx]
                        wall_vertices_y = [y1, y2, y2+ny, y1+ny, y1, y2, y2+ny, y1+ny]
                        wall_vertices_z = [0, 0, 0, 0, wall_height, wall_height, wall_height, wall_height]
                        
                        fig.add_trace(go.Mesh3d(
                            x=wall_vertices_x,
                            y=wall_vertices_y,
                            z=wall_vertices_z,
                            color='#d4d4aa',
                            opacity=0.9,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
    
    def _add_3d_furniture(self, fig: go.Figure, rooms: List[Dict]):
        """Add detailed furniture elements to create realistic interiors"""
        furniture_colors = {
            'desk': '#8b4513',        # Saddle brown
            'chair': '#654321',       # Dark brown
            'bed': '#daa520',         # Goldenrod
            'cabinet': '#a0522d',     # Sienna
            'decoration': '#228b22'   # Forest green for plants
        }
        
        for i, room in enumerate(rooms):
            x = room.get('x', 0)
            y = room.get('y', 0)
            width = room.get('width', 3)
            height = room.get('height', 2)
            size_cat = room.get('size_category', 'medium')
            
            # Add furniture based on room size and type
            if width > 1.5 and height > 1.5:
                # Office/workspace furniture for hotel rooms
                if size_cat in ['medium', 'large', 'xlarge']:
                    # Desk
                    desk_width = min(1.2, width * 0.4)
                    desk_height = 0.75
                    desk_depth = min(0.6, height * 0.3)
                    desk_x = x + width * 0.1
                    desk_y = y + height * 0.1
                    
                    self._add_furniture_piece(fig, desk_x, desk_y, desk_width, desk_depth, desk_height, furniture_colors['desk'])
                    
                    # Chair
                    chair_size = 0.5
                    chair_height = 0.9
                    chair_x = desk_x + desk_width + 0.2
                    chair_y = desk_y + desk_depth/2 - chair_size/2
                    
                    self._add_furniture_piece(fig, chair_x, chair_y, chair_size, chair_size, chair_height, furniture_colors['chair'])
                
                # Bed for larger rooms
                if size_cat in ['large', 'xlarge'] and width > 2.5:
                    bed_width = min(1.4, width * 0.5)
                    bed_length = min(2.0, height * 0.6)
                    bed_height = 0.6
                    bed_x = x + width - bed_width - 0.2
                    bed_y = y + height - bed_length - 0.2
                    
                    self._add_furniture_piece(fig, bed_x, bed_y, bed_width, bed_length, bed_height, furniture_colors['bed'])
                
                # Small decoration/plant
                if width > 2 and height > 2:
                    plant_size = 0.3
                    plant_height = 0.4
                    plant_x = x + width * 0.8
                    plant_y = y + height * 0.2
                    
                    self._add_furniture_piece(fig, plant_x, plant_y, plant_size, plant_size, plant_height, furniture_colors['decoration'])
    
    def _add_furniture_piece(self, fig: go.Figure, x: float, y: float, width: float, depth: float, height: float, color: str):
        """Add a single furniture piece as a 3D box"""
        vertices_x = [x, x+width, x+width, x, x, x+width, x+width, x]
        vertices_y = [y, y, y+depth, y+depth, y, y, y+depth, y+depth]
        vertices_z = [0, 0, 0, 0, height, height, height, height]
        
        fig.add_trace(go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            color=color,
            opacity=0.85,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def _add_3d_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add 3D corridor pathways"""
        for corridor in corridors:
            path = corridor.get('path', [])
            if len(path) >= 2:
                for i in range(len(path) - 1):
                    x1, y1 = path[i][0], path[i][1]
                    x2, y2 = path[i+1][0], path[i+1][1]
                    
                    # Simple corridor line at floor level
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2],
                        y=[y1, y2],
                        z=[0.1, 0.1],
                        mode='lines',
                        line=dict(color='#4299e1', width=8),
                        showlegend=False,
                        hoverinfo='skip'
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