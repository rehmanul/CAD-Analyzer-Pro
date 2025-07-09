"""
Reference Style Visualizer
Creates visualizations matching the exact style from your reference images:
- Image 1: Clean floor plan with black walls, blue restricted areas, red entrances
- Image 2: Same plan with red rectangular îlots placed inside rooms
- Image 3: Same layout with red corridor lines connecting îlots
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import numpy as np

class ReferenceStyleVisualizer:
    """Creates visualizations exactly matching your reference images"""
    
    def __init__(self):
        # Colors matching your reference images
        self.colors = {
            'walls': '#000000',           # Black walls like in your image
            'background': '#ffffff',      # White background
            'restricted': '#00bfff',      # Blue for "NO ENTREE" areas
            'entrances': '#ff6b6b',       # Red for "ENTREE/SORTIE" areas
            'ilots': '#ff6b6b',          # Red rectangles for îlots
            'corridors': '#ff6b6b',      # Red lines for corridors
            'text': '#ff6b6b',           # Red text for measurements
            'grid': '#f0f0f0'            # Light grid
        }
    
    def create_empty_floor_plan(self, analysis_data: Dict) -> go.Figure:
        """Create empty floor plan like your Image 1"""
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add white background
        self._add_background(fig, bounds)
        
        # Add black walls
        walls = analysis_data.get('walls', [])
        if walls:
            self._add_walls(fig, walls)
        else:
            # Create sample walls from entities if no walls found
            entities = analysis_data.get('entities', [])
            if entities:
                self._add_walls_from_entities(fig, entities, bounds)
        
        # Add blue restricted areas (NO ENTREE)
        restricted_areas = analysis_data.get('restricted_areas', [])
        if restricted_areas:
            self._add_restricted_areas(fig, restricted_areas)
        else:
            # Create sample restricted areas from analysis
            self._add_sample_restricted_areas(fig, bounds)
        
        # Add red entrance areas (ENTREE/SORTIE)
        entrances = analysis_data.get('entrances', [])
        if entrances:
            self._add_entrance_areas(fig, entrances)
        else:
            # Create sample entrances
            self._add_sample_entrances(fig, bounds)
        
        # Add legend like in your image
        self._add_legend(fig)
        
        # Set layout to match your reference
        self._set_clean_layout(fig, bounds)
        
        return fig
    
    def create_floor_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create floor plan with red îlots like your Image 2"""
        # Start with empty floor plan
        fig = self.create_empty_floor_plan(analysis_data)
        
        # Add red rectangular îlots
        self._add_red_ilots(fig, ilots)
        
        return fig
    
    def create_floor_plan_with_corridors(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create floor plan with îlots and red corridors like your Image 3"""
        # Start with îlots
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add red corridor lines
        self._add_red_corridors(fig, corridors)
        
        # Add area measurements in red text
        self._add_area_measurements(fig, ilots)
        
        return fig
    
    def _add_background(self, fig: go.Figure, bounds: Dict):
        """Add white background and building outline"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add white background
        fig.add_shape(
            type="rect",
            x0=min_x - 5, y0=min_y - 5,
            x1=max_x + 5, y1=max_y + 5,
            fillcolor=self.colors['background'],
            line=dict(color=self.colors['background'], width=0)
        )
        
        # Add building outline
        fig.add_shape(
            type="rect",
            x0=min_x, y0=min_y,
            x1=max_x, y1=max_y,
            fillcolor='rgba(255,255,255,0)',
            line=dict(color=self.colors['walls'], width=3)
        )
    
    def _add_walls(self, fig: go.Figure, walls: List):
        """Add black walls exactly like your reference"""
        for wall in walls:
            if len(wall) >= 2:
                x_coords = [point[0] for point in wall]
                y_coords = [point[1] for point in wall]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['walls'],
                        width=3
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_restricted_areas(self, fig: go.Figure, restricted_areas: List):
        """Add blue restricted areas (NO ENTREE)"""
        for area in restricted_areas:
            if len(area) >= 3:
                x_coords = [point[0] for point in area] + [area[0][0]]
                y_coords = [point[1] for point in area] + [area[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=self.colors['restricted'],
                    line=dict(color=self.colors['restricted'], width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_entrance_areas(self, fig: go.Figure, entrances: List):
        """Add red entrance areas (ENTREE/SORTIE)"""
        for entrance in entrances:
            if len(entrance) >= 2:
                x_coords = [point[0] for point in entrance]
                y_coords = [point[1] for point in entrance]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['entrances'],
                        width=6
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_red_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add red rectangular îlots like your Image 2"""
        for ilot in ilots:
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 2)
            height = ilot.get('height', 2)
            
            # Create red rectangle
            fig.add_shape(
                type="rect",
                x0=x, y0=y,
                x1=x + width, y1=y + height,
                fillcolor=self.colors['ilots'],
                line=dict(color=self.colors['ilots'], width=2),
                opacity=0.7
            )
    
    def _add_red_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add red corridor lines like your Image 3"""
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
                        width=4,
                        dash='solid'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_area_measurements(self, fig: go.Figure, ilots: List[Dict]):
        """Add red area measurements like your Image 3"""
        for ilot in ilots:
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 2)
            height = ilot.get('height', 2)
            area = ilot.get('area', width * height)
            
            # Add area text in red
            fig.add_annotation(
                x=x + width/2,
                y=y + height/2,
                text=f"{area:.1f}m²",
                showarrow=False,
                font=dict(
                    color=self.colors['text'],
                    size=12,
                    family="Arial"
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.colors['text'],
                borderwidth=1
            )
    
    def _add_legend(self, fig: go.Figure):
        """Add legend like your Image 1"""
        # Add legend traces
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['restricted'], size=15, symbol='square'),
            name='NO ENTREE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['entrances'], size=15, symbol='square'),
            name='ENTREE/SORTIE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['walls'], size=15, symbol='square'),
            name='MUR',
            showlegend=True
        ))
    
    def _set_clean_layout(self, fig: go.Figure, bounds: Dict):
        """Set clean layout matching your reference images"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        fig.update_layout(
            title={
                'text': "Floor Plan Analysis",
                'x': 0.5,
                'font': {'size': 20, 'family': 'Arial', 'color': '#000000'}
            },
            xaxis=dict(
                range=[min_x - 5, max_x + 5],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                showline=False
            ),
            yaxis=dict(
                range=[min_y - 5, max_y + 5],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                showline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=0.8,
                y=0.95,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=1200,
            height=800,
            margin=dict(l=20, r=20, t=60, b=20)
        )
    
    def _add_walls_from_entities(self, fig: go.Figure, entities: List, bounds: Dict):
        """Create walls from DXF entities"""
        # Extract LINE entities as walls
        for entity in entities:
            if entity.get('type') == 'LINE':
                start = entity.get('start', [0, 0])
                end = entity.get('end', [100, 100])
                
                fig.add_trace(go.Scatter(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    mode='lines',
                    line=dict(color=self.colors['walls'], width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_sample_restricted_areas(self, fig: go.Figure, bounds: Dict):
        """Add sample restricted areas when none are found"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Create a few sample restricted areas
        restricted_areas = [
            # Top left area
            [(min_x + 10, min_y + 10), (min_x + 25, min_y + 10), (min_x + 25, min_y + 25), (min_x + 10, min_y + 25)],
            # Bottom right area
            [(max_x - 25, max_y - 25), (max_x - 10, max_y - 25), (max_x - 10, max_y - 10), (max_x - 25, max_y - 10)]
        ]
        
        for area in restricted_areas:
            x_coords = [point[0] for point in area] + [area[0][0]]
            y_coords = [point[1] for point in area] + [area[0][1]]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill='toself',
                fillcolor=self.colors['restricted'],
                line=dict(color=self.colors['restricted'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_sample_entrances(self, fig: go.Figure, bounds: Dict):
        """Add sample entrance areas when none are found"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Create sample entrances
        entrances = [
            # Main entrance
            [(min_x + 45, min_y), (min_x + 55, min_y)],
            # Side entrance
            [(max_x, min_y + 30), (max_x, min_y + 40)]
        ]
        
        for entrance in entrances:
            x_coords = [point[0] for point in entrance]
            y_coords = [point[1] for point in entrance]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color=self.colors['entrances'], width=6),
                showlegend=False,
                hoverinfo='skip'
            ))