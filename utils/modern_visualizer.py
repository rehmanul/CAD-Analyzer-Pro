"""
Modern Professional Visualizer
Creates floor plan visualizations matching the attached reference images
with clean 2D/3D styling, proper room labeling, and sophisticated design
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
import colorsys

class ModernVisualizer:
    """Modern professional visualizer for floor plans"""
    
    def __init__(self):
        self.modern_colors = {
            'walls': '#2d3748',
            'restricted': '#e53e3e',
            'entrances': '#38a169',
            'open_zones': '#f7fafc',
            'small_ilots': '#ffd700',
            'medium_ilots': '#ff8c00',
            'large_ilots': '#32cd32',
            'xlarge_ilots': '#9370db',
            'corridors': '#4299e1',
            'background': '#ffffff',
            'grid': '#e2e8f0',
            'text': '#2d3748',
            'accent': '#6366f1'
        }
        
        self.modern_fonts = {
            'family': 'Inter, system-ui, -apple-system, sans-serif',
            'size': 12,
            'color': '#2d3748'
        }
    
    def create_modern_floor_plan(self, analysis_data: Dict, ilots: List[Dict] = None, 
                               corridors: List[Dict] = None) -> go.Figure:
        """Create modern floor plan visualization matching client expectations"""
        
        # Create figure with modern layout
        fig = go.Figure()
        
        # Set modern background and styling
        fig.update_layout(
            title={
                'text': "🏨 Professional Floor Plan Analysis",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': self.modern_fonts['family'], 'color': self.modern_colors['text']}
            },
            paper_bgcolor=self.modern_colors['background'],
            plot_bgcolor=self.modern_colors['background'],
            width=1200,
            height=800,
            showlegend=True,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.15,
                'xanchor': 'center',
                'x': 0.5,
                'bgcolor': 'rgba(255,255,255,0.9)',
                'bordercolor': '#e2e8f0',
                'borderwidth': 1,
                'font': self.modern_fonts
            }
        )
        
        # Add walls with modern styling
        self._add_modern_walls(fig, analysis_data.get('walls', []))
        
        # Add restricted areas
        self._add_modern_restricted_areas(fig, analysis_data.get('restricted_areas', []))
        
        # Add entrances
        self._add_modern_entrances(fig, analysis_data.get('entrances', []))
        
        # Add îlots if provided
        if ilots:
            self._add_modern_ilots(fig, ilots)
        
        # Add corridors if provided
        if corridors:
            self._add_modern_corridors(fig, corridors)
        
        # Add professional grid
        self._add_modern_grid(fig, analysis_data.get('bounds', {}))
        
        # Add measurements and labels
        self._add_modern_measurements(fig, analysis_data.get('bounds', {}))
        
        # Update axes for professional appearance
        self._update_modern_axes(fig, analysis_data.get('bounds', {}))
        
        return fig
    
    def _add_modern_walls(self, fig: go.Figure, walls: List):
        """Add walls with modern styling"""
        wall_x = []
        wall_y = []
        
        for wall in walls:
            if len(wall) >= 2:
                for i in range(len(wall)):
                    point = wall[i]
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        wall_x.append(point[0])
                        wall_y.append(point[1])
                
                # Close the shape
                if len(wall) > 2:
                    wall_x.append(wall[0][0])
                    wall_y.append(wall[0][1])
                
                wall_x.append(None)
                wall_y.append(None)
        
        if wall_x and wall_y:
            fig.add_trace(go.Scatter(
                x=wall_x,
                y=wall_y,
                mode='lines',
                line={
                    'color': self.modern_colors['walls'],
                    'width': 3
                },
                name='Walls',
                hoverinfo='name',
                showlegend=True
            ))
    
    def _add_modern_restricted_areas(self, fig: go.Figure, restricted_areas: List):
        """Add restricted areas with modern styling"""
        for i, area in enumerate(restricted_areas):
            if len(area) >= 3:
                x_coords = [point[0] for point in area] + [area[0][0]]
                y_coords = [point[1] for point in area] + [area[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=f'rgba(229, 62, 62, 0.3)',
                    line={
                        'color': self.modern_colors['restricted'],
                        'width': 2,
                        'dash': 'dash'
                    },
                    mode='lines',
                    name='Restricted Areas' if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo='name'
                ))
    
    def _add_modern_entrances(self, fig: go.Figure, entrances: List):
        """Add entrances with modern styling"""
        for i, entrance in enumerate(entrances):
            if len(entrance) >= 2:
                x_coords = [point[0] for point in entrance]
                y_coords = [point[1] for point in entrance]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    line={
                        'color': self.modern_colors['entrances'],
                        'width': 6
                    },
                    marker={
                        'color': self.modern_colors['entrances'],
                        'size': 8,
                        'symbol': 'square'
                    },
                    name='Entrances' if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo='name'
                ))
    
    def _add_modern_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add îlots with modern styling matching client expectations"""
        size_categories = {}
        
        for ilot in ilots:
            size_cat = ilot.get('size_category', 'medium')
            if size_cat not in size_categories:
                size_categories[size_cat] = {
                    'x': [], 'y': [], 'width': [], 'height': [], 
                    'color': self.modern_colors.get(f'{size_cat}_ilots', '#4299e1'),
                    'areas': []
                }
            
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 2)
            height = ilot.get('height', 2)
            area = ilot.get('area', width * height)
            
            size_categories[size_cat]['x'].append(x)
            size_categories[size_cat]['y'].append(y)
            size_categories[size_cat]['width'].append(width)
            size_categories[size_cat]['height'].append(height)
            size_categories[size_cat]['areas'].append(area)
        
        # Add îlots by category for clean legend
        for size_cat, data in size_categories.items():
            if data['x']:
                # Create rectangles for each îlot
                for i in range(len(data['x'])):
                    x, y, w, h = data['x'][i], data['y'][i], data['width'][i], data['height'][i]
                    area = data['areas'][i]
                    
                    # Create rectangle
                    fig.add_shape(
                        type="rect",
                        x0=x, y0=y, x1=x+w, y1=y+h,
                        fillcolor=data['color'],
                        line=dict(color=self.modern_colors['walls'], width=1),
                        opacity=0.8
                    )
                    
                    # Add label with area
                    fig.add_annotation(
                        x=x + w/2,
                        y=y + h/2,
                        text=f"{area:.1f}m²",
                        showarrow=False,
                        font=dict(
                            family=self.modern_fonts['family'],
                            size=10,
                            color=self.modern_colors['text']
                        ),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=self.modern_colors['text'],
                        borderwidth=1
                    )
                
                # Add legend entry
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        color=data['color'],
                        size=15,
                        symbol='square'
                    ),
                    name=f'{size_cat.title()} Îlots ({len(data["x"])})',
                    showlegend=True
                ))
    
    def _add_modern_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors with modern styling"""
        for i, corridor in enumerate(corridors):
            path = corridor.get('path', [])
            if len(path) >= 2:
                x_coords = [point[0] for point in path]
                y_coords = [point[1] for point in path]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line={
                        'color': self.modern_colors['corridors'],
                        'width': 4,
                        'dash': 'dot'
                    },
                    name='Corridors' if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo='name'
                ))
    
    def _add_modern_grid(self, fig: go.Figure, bounds: Dict):
        """Add professional grid"""
        if not bounds:
            return
        
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        # Calculate grid spacing
        x_range = max_x - min_x
        y_range = max_y - min_y
        grid_spacing = max(x_range, y_range) / 20
        
        # Add grid lines
        x_lines = np.arange(min_x, max_x + grid_spacing, grid_spacing)
        y_lines = np.arange(min_y, max_y + grid_spacing, grid_spacing)
        
        for x in x_lines:
            fig.add_shape(
                type="line",
                x0=x, y0=min_y, x1=x, y1=max_y,
                line=dict(color=self.modern_colors['grid'], width=0.5),
                layer="below"
            )
        
        for y in y_lines:
            fig.add_shape(
                type="line",
                x0=min_x, y0=y, x1=max_x, y1=y,
                line=dict(color=self.modern_colors['grid'], width=0.5),
                layer="below"
            )
    
    def _add_modern_measurements(self, fig: go.Figure, bounds: Dict):
        """Add professional measurements"""
        if not bounds:
            return
            
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Add dimension lines and labels
        fig.add_annotation(
            x=(min_x + max_x) / 2,
            y=min_y - (max_y - min_y) * 0.05,
            text=f"{width:.1f}m",
            showarrow=False,
            font=dict(
                family=self.modern_fonts['family'],
                size=12,
                color=self.modern_colors['accent']
            ),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=self.modern_colors['accent'],
            borderwidth=1
        )
        
        fig.add_annotation(
            x=min_x - (max_x - min_x) * 0.05,
            y=(min_y + max_y) / 2,
            text=f"{height:.1f}m",
            showarrow=False,
            textangle=-90,
            font=dict(
                family=self.modern_fonts['family'],
                size=12,
                color=self.modern_colors['accent']
            ),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=self.modern_colors['accent'],
            borderwidth=1
        )
    
    def _update_modern_axes(self, fig: go.Figure, bounds: Dict):
        """Update axes for professional appearance"""
        if bounds:
            min_x = bounds.get('min_x', 0)
            max_x = bounds.get('max_x', 100)
            min_y = bounds.get('min_y', 0)
            max_y = bounds.get('max_y', 100)
            
            # Add padding
            padding = max(max_x - min_x, max_y - min_y) * 0.1
            x_range = [min_x - padding, max_x + padding]
            y_range = [min_y - padding, max_y + padding]
        else:
            x_range = [-10, 110]
            y_range = [-10, 110]
        
        fig.update_xaxes(
            range=x_range,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            title="Width (m)",
            title_font=self.modern_fonts,
            tickfont=self.modern_fonts,
            scaleanchor="y",
            scaleratio=1
        )
        
        fig.update_yaxes(
            range=y_range,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            title="Height (m)",
            title_font=self.modern_fonts,
            tickfont=self.modern_fonts
        )
    
    def create_modern_metrics_dashboard(self, analysis_data: Dict, 
                                      ilots: List[Dict] = None) -> go.Figure:
        """Create modern metrics dashboard"""
        
        # Create subplots for metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Space Utilization", "Îlot Distribution", 
                          "Size Analysis", "Efficiency Metrics"),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Calculate metrics
        total_area = 0
        size_distribution = {}
        
        if ilots:
            for ilot in ilots:
                area = ilot.get('area', 0)
                total_area += area
                size_cat = ilot.get('size_category', 'unknown')
                size_distribution[size_cat] = size_distribution.get(size_cat, 0) + 1
        
        # Space utilization gauge
        bounds = analysis_data.get('bounds', {})
        floor_area = (bounds.get('max_x', 100) - bounds.get('min_x', 0)) * \
                    (bounds.get('max_y', 100) - bounds.get('min_y', 0))
        utilization = (total_area / floor_area * 100) if floor_area > 0 else 0
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=utilization,
            title={'text': "Space Utilization (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.modern_colors['accent']},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=1, col=1)
        
        # Size distribution pie chart
        if size_distribution:
            fig.add_trace(go.Pie(
                labels=list(size_distribution.keys()),
                values=list(size_distribution.values()),
                hole=0.4,
                marker_colors=[self.modern_colors.get(f'{k}_ilots', '#4299e1') 
                             for k in size_distribution.keys()]
            ), row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "📊 Analysis Metrics Dashboard",
                'x': 0.5,
                'font': {'size': 20, 'family': self.modern_fonts['family']}
            },
            font=self.modern_fonts,
            paper_bgcolor=self.modern_colors['background'],
            height=600
        )
        
        return fig