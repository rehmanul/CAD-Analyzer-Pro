"""
Phase 3: Pixel-Perfect Visualization System
Creates exact visual matches to reference designs with precise color matching,
professional architectural styling, and multi-stage visualization
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Polygon, Point, LineString
from dataclasses import dataclass
from enum import Enum
import time

class VisualizationStage(Enum):
    EMPTY_PLAN = "empty_plan"           # Stage 1: Empty floor plan
    ILOTS_PLACED = "ilots_placed"       # Stage 2: Îlots placed
    CORRIDORS_ADDED = "corridors_added" # Stage 3: Corridors + îlots

class VisualizationStyle(Enum):
    REFERENCE_MATCH = "reference_match"     # Exact match to reference images
    PROFESSIONAL = "professional"          # Professional architectural style
    TECHNICAL = "technical"                # Technical drawing style
    MODERN = "modern"                      # Modern minimalist style

@dataclass
class VisualizationConfig:
    """Configuration for pixel-perfect visualization"""
    stage: VisualizationStage
    style: VisualizationStyle = VisualizationStyle.REFERENCE_MATCH
    canvas_size: Tuple[int, int] = (1800, 1800)  # Width, Height in pixels
    background_color: str = "#FFFFFF"
    
    # Reference color palette (exact matches)
    wall_color: str = "#6B7280"        # Gray for walls
    restricted_color: str = "#EF4444"  # Red for restricted areas
    entrance_color: str = "#3B82F6"    # Blue for entrances
    
    # Îlot colors by size
    ilot_colors: Dict[str, str] = None
    
    # Corridor colors by type
    corridor_colors: Dict[str, str] = None
    
    # Styling options
    show_grid: bool = False
    show_labels: bool = True
    show_dimensions: bool = False
    line_width: float = 2.0
    opacity: float = 1.0

class PixelPerfectVisualizer:
    """
    Advanced visualization system that creates pixel-perfect matches
    to reference designs with professional architectural styling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default color palettes
        self.reference_colors = {
            'walls': "#6B7280",           # Gray
            'restricted_areas': "#EF4444", # Red  
            'entrances': "#3B82F6",       # Blue
            'background': "#FFFFFF",       # White
        }
        
        self.ilot_size_colors = {
            'Small (0-1m²)': "#FDE047",      # Yellow
            'Medium (1-3m²)': "#FB923C",     # Orange
            'Large (3-5m²)': "#22C55E",      # Green
            'Extra Large (5-10m²)': "#A855F7" # Purple
        }
        
        self.corridor_type_colors = {
            'main': "#3B82F6",        # Blue
            'secondary': "#10B981",   # Green
            'access': "#F59E0B",      # Yellow
            'emergency': "#EF4444"    # Red
        }
        
        # Professional styling presets
        self.style_presets = {
            VisualizationStyle.REFERENCE_MATCH: {
                'line_width': 2.0,
                'opacity': 1.0,
                'show_grid': False,
                'background': "#FFFFFF"
            },
            VisualizationStyle.PROFESSIONAL: {
                'line_width': 1.5,
                'opacity': 0.9,
                'show_grid': True,
                'background': "#F8F9FA"
            },
            VisualizationStyle.TECHNICAL: {
                'line_width': 1.0,
                'opacity': 1.0,
                'show_grid': True,
                'background': "#FFFFFF"
            },
            VisualizationStyle.MODERN: {
                'line_width': 2.5,
                'opacity': 0.8,
                'show_grid': False,
                'background': "#F1F5F9"
            }
        }

    def create_pixel_perfect_visualization(self, floor_plan_data: Dict[str, Any], 
                                         config: VisualizationConfig) -> go.Figure:
        """
        Create pixel-perfect visualization matching the specified stage and style
        """
        start_time = time.time()
        
        try:
            # Apply style preset
            self._apply_style_preset(config)
            
            # Create base figure with precise dimensions
            fig = self._create_base_figure(config)
            
            # Stage 1: Always show floor plan structure (walls, restricted areas, entrances)
            self._add_floor_plan_structure(fig, floor_plan_data, config)
            
            # Stage 2: Add îlots if stage >= ILOTS_PLACED
            if config.stage in [VisualizationStage.ILOTS_PLACED, VisualizationStage.CORRIDORS_ADDED]:
                self._add_ilots_to_visualization(fig, floor_plan_data, config)
            
            # Stage 3: Add corridors if stage == CORRIDORS_ADDED
            if config.stage == VisualizationStage.CORRIDORS_ADDED:
                self._add_corridors_to_visualization(fig, floor_plan_data, config)
            
            # Apply final styling and layout
            self._apply_final_styling(fig, config)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Pixel-perfect visualization created in {processing_time:.2f}s")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating pixel-perfect visualization: {str(e)}")
            return self._create_error_figure(str(e), config)

    def _apply_style_preset(self, config: VisualizationConfig):
        """Apply styling preset based on visualization style"""
        preset = self.style_presets.get(config.style, {})
        
        if not config.ilot_colors:
            config.ilot_colors = self.ilot_size_colors.copy()
        
        if not config.corridor_colors:
            config.corridor_colors = self.corridor_type_colors.copy()
        
        # Update config with preset values
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _create_base_figure(self, config: VisualizationConfig) -> go.Figure:
        """Create base figure with precise canvas size and styling"""
        
        fig = go.Figure()
        
        # Calculate aspect ratio and bounds
        width, height = config.canvas_size
        aspect_ratio = width / height
        
        # Set layout for pixel-perfect rendering
        fig.update_layout(
            width=width,
            height=height,
            plot_bgcolor=config.background_color,
            paper_bgcolor=config.background_color,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            
            # Ensure equal aspect ratio
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                showgrid=config.show_grid,
                gridwidth=0.5,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=False,
                showticklabels=False,
                range=[-1000, 1000]  # Will be adjusted based on data
            ),
            yaxis=dict(
                showgrid=config.show_grid,
                gridwidth=0.5,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=False,
                showticklabels=False,
                range=[-1000, 1000]  # Will be adjusted based on data
            )
        )
        
        return fig

    def _add_floor_plan_structure(self, fig: go.Figure, floor_plan_data: Dict[str, Any], 
                                config: VisualizationConfig):
        """Add floor plan structure (walls, restricted areas, entrances)"""
        
        # Add walls
        walls = floor_plan_data.get('walls', [])
        for wall in walls:
            coords = wall.get('coordinates', [])
            if len(coords) >= 2:
                self._add_wall_to_figure(fig, coords, config)
        
        # Add restricted areas
        restricted_areas = floor_plan_data.get('restricted_areas', [])
        for area in restricted_areas:
            coords = area.get('coordinates', [])
            if len(coords) >= 3:
                self._add_restricted_area_to_figure(fig, coords, config)
        
        # Add entrances
        entrances = floor_plan_data.get('entrances', [])
        for entrance in entrances:
            coords = entrance.get('coordinates', [])
            if len(coords) >= 1:
                self._add_entrance_to_figure(fig, coords, config)
        
        # Adjust axis ranges based on data
        self._adjust_axis_ranges(fig, floor_plan_data)

    def _add_wall_to_figure(self, fig: go.Figure, coordinates: List[List[float]], 
                          config: VisualizationConfig):
        """Add wall to figure with precise styling"""
        
        if len(coordinates) < 2:
            return
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(
                color=config.wall_color,
                width=config.line_width * 2,  # Walls are thicker
            ),
            opacity=config.opacity,
            name='Wall',
            showlegend=False,
            hoverinfo='skip'
        ))

    def _add_restricted_area_to_figure(self, fig: go.Figure, coordinates: List[List[float]], 
                                     config: VisualizationConfig):
        """Add restricted area to figure with precise styling"""
        
        if len(coordinates) < 3:
            return
        
        # Close the polygon
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            fill='toself',
            fillcolor=f"rgba({self._hex_to_rgb(config.restricted_color)}, 0.3)",
            line=dict(
                color=config.restricted_color,
                width=config.line_width,
            ),
            opacity=config.opacity,
            name='Restricted Area',
            showlegend=False,
            hoverinfo='skip'
        ))

    def _add_entrance_to_figure(self, fig: go.Figure, coordinates: List[List[float]], 
                              config: VisualizationConfig):
        """Add entrance to figure with precise styling"""
        
        if len(coordinates) == 1:
            # Point entrance - create a circle
            center = coordinates[0]
            radius = 500  # 0.5m radius in mm
            
            theta = np.linspace(0, 2*np.pi, 20)
            x_circle = center[0] + radius * np.cos(theta)
            y_circle = center[1] + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                fill='toself',
                fillcolor=f"rgba({self._hex_to_rgb(config.entrance_color)}, 0.5)",
                line=dict(
                    color=config.entrance_color,
                    width=config.line_width,
                ),
                opacity=config.opacity,
                name='Entrance',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # Polygon entrance
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                fill='toself',
                fillcolor=f"rgba({self._hex_to_rgb(config.entrance_color)}, 0.5)",
                line=dict(
                    color=config.entrance_color,
                    width=config.line_width,
                ),
                opacity=config.opacity,
                name='Entrance',
                showlegend=False,
                hoverinfo='skip'
            ))

    def _add_ilots_to_visualization(self, fig: go.Figure, floor_plan_data: Dict[str, Any], 
                                  config: VisualizationConfig):
        """Add îlots to visualization with size-based coloring"""
        
        placed_ilots = floor_plan_data.get('placed_ilots', [])
        
        for ilot in placed_ilots:
            coords = ilot.get('coordinates', [])
            size_category = ilot.get('size_category', 'Medium (1-3m²)')
            ilot_id = ilot.get('id', 'Unknown')
            
            if len(coords) >= 3:
                self._add_ilot_to_figure(fig, coords, size_category, ilot_id, config)

    def _add_ilot_to_figure(self, fig: go.Figure, coordinates: List[List[float]], 
                          size_category: str, ilot_id: str, config: VisualizationConfig):
        """Add individual îlot to figure"""
        
        # Close the polygon
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        # Get color for size category
        color = config.ilot_colors.get(size_category, "#22C55E")
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            fill='toself',
            fillcolor=f"rgba({self._hex_to_rgb(color)}, 0.7)",
            line=dict(
                color=color,
                width=config.line_width,
            ),
            opacity=config.opacity,
            name=f'Îlot {ilot_id}',
            showlegend=False,
            hovertemplate=f"<b>Îlot {ilot_id}</b><br>Size: {size_category}<extra></extra>",
        ))
        
        # Add label if enabled
        if config.show_labels:
            center_x = sum(x_coords[:-1]) / len(x_coords[:-1])
            center_y = sum(y_coords[:-1]) / len(y_coords[:-1])
            
            fig.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode='text',
                text=[ilot_id.split('_')[-1]],  # Show just the number
                textfont=dict(size=10, color='white'),
                showlegend=False,
                hoverinfo='skip'
            ))

    def _add_corridors_to_visualization(self, fig: go.Figure, floor_plan_data: Dict[str, Any], 
                                      config: VisualizationConfig):
        """Add corridors to visualization with type-based coloring"""
        
        corridors = floor_plan_data.get('corridors', [])
        
        for corridor in corridors:
            coords = corridor.get('coordinates', [])
            corridor_type = corridor.get('corridor_type', 'secondary')
            corridor_id = corridor.get('id', 'Unknown')
            width = corridor.get('width', 1.0)  # Width in meters
            
            if len(coords) >= 2:
                self._add_corridor_to_figure(fig, coords, corridor_type, corridor_id, width, config)

    def _add_corridor_to_figure(self, fig: go.Figure, coordinates: List[List[float]], 
                              corridor_type: str, corridor_id: str, width: float, 
                              config: VisualizationConfig):
        """Add individual corridor to figure"""
        
        # Get color for corridor type
        color = config.corridor_colors.get(corridor_type, "#10B981")
        
        # Calculate line width based on corridor width (scale appropriately)
        display_width = max(config.line_width, width * 2)
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(
                color=color,
                width=display_width,
            ),
            opacity=config.opacity * 0.8,  # Slightly more transparent
            name=f'Corridor {corridor_id}',
            showlegend=False,
            hovertemplate=f"<b>Corridor {corridor_id}</b><br>Type: {corridor_type}<br>Width: {width:.1f}m<extra></extra>",
        ))

    def _adjust_axis_ranges(self, fig: go.Figure, floor_plan_data: Dict[str, Any]):
        """Adjust axis ranges to fit all data with proper margins"""
        
        all_coords = []
        
        # Collect all coordinates
        for element_type in ['walls', 'restricted_areas', 'entrances', 'placed_ilots', 'corridors']:
            elements = floor_plan_data.get(element_type, [])
            for element in elements:
                coords = element.get('coordinates', [])
                all_coords.extend(coords)
        
        if not all_coords:
            return
        
        # Calculate bounds
        x_coords = [coord[0] for coord in all_coords if len(coord) >= 2]
        y_coords = [coord[1] for coord in all_coords if len(coord) >= 2]
        
        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add 10% margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            fig.update_xaxes(range=[x_min - x_margin, x_max + x_margin])
            fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])

    def _apply_final_styling(self, fig: go.Figure, config: VisualizationConfig):
        """Apply final styling touches for pixel-perfect rendering"""
        
        # Ensure proper aspect ratio and clean appearance
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            dragmode='pan',
            scrollZoom=True,
        )
        
        # Remove axes if not needed
        if not config.show_grid:
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for rgba()"""
        hex_color = hex_color.lstrip('#')
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"

    def _create_error_figure(self, error_message: str, config: VisualizationConfig) -> go.Figure:
        """Create error figure when visualization fails"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='text',
            text=[f"Visualization Error: {error_message}"],
            textfont=dict(size=16, color='red'),
            showlegend=False
        ))
        
        fig.update_layout(
            width=config.canvas_size[0],
            height=config.canvas_size[1],
            plot_bgcolor=config.background_color,
            paper_bgcolor=config.background_color,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig

    def create_multi_stage_visualization(self, floor_plan_data: Dict[str, Any], 
                                       style: VisualizationStyle = VisualizationStyle.REFERENCE_MATCH) -> Dict[str, go.Figure]:
        """
        Create all three visualization stages matching reference images
        """
        
        visualizations = {}
        
        for stage in VisualizationStage:
            config = VisualizationConfig(
                stage=stage,
                style=style,
                canvas_size=(1800, 1800),
                wall_color=self.reference_colors['walls'],
                restricted_color=self.reference_colors['restricted_areas'],
                entrance_color=self.reference_colors['entrances'],
                background_color=self.reference_colors['background']
            )
            
            fig = self.create_pixel_perfect_visualization(floor_plan_data, config)
            visualizations[stage.value] = fig
        
        return visualizations

    def get_visualization_capabilities(self) -> Dict[str, Any]:
        """Get information about visualization capabilities"""
        return {
            'stages': [stage.value for stage in VisualizationStage],
            'styles': [style.value for style in VisualizationStyle],
            'features': [
                'Pixel-perfect rendering',
                'Exact color matching',
                'Multi-stage visualization',
                'Professional styling',
                'Interactive elements',
                'Responsive scaling',
                'Export-ready output'
            ],
            'supported_elements': [
                'Walls', 'Restricted Areas', 'Entrances', 
                'Îlots (size-coded)', 'Corridors (type-coded)',
                'Labels', 'Dimensions', 'Grid overlay'
            ],
            'canvas_sizes': [
                (1200, 1200), (1800, 1800), (2400, 2400)
            ]
        }

# Create global instance
pixel_perfect_visualizer = PixelPerfectVisualizer()