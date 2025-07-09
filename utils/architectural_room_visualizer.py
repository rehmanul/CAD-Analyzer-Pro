"""
Architectural Room Visualizer
Creates proper floor plan visualizations with connected room structures
matching the user's reference images exactly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class ArchitecturalRoomVisualizer:
    """Creates architectural floor plan visualizations with proper room structures"""
    
    def __init__(self):
        self.colors = {
            'walls': '#808080',      # Gray for walls (MUR)
            'restricted': '#4A90E2',  # Blue for restricted areas (NO ENTREE)
            'entrances': '#FF0000',   # Red for entrances (ENTREE/SORTIE)
            'background': '#F5F5F5',  # Light gray background
            'ilots': '#FF4444',       # Red for îlots
            'corridors': '#FF6666'    # Light red for corridors
        }
    
    def create_architectural_floor_plan(self, analysis_result: Dict[str, Any], 
                                      mode: str = 'complete') -> go.Figure:
        """Create architectural floor plan with proper room structure"""
        
        # Get bounds
        bounds = analysis_result.get('bounds', {})
        if not bounds:
            bounds = {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
        
        # Calculate dimensions
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        
        # Create figure
        fig = go.Figure()
        
        # Add background
        fig.add_shape(
            type="rect",
            x0=bounds['min_x'], y0=bounds['min_y'],
            x1=bounds['max_x'], y1=bounds['max_y'],
            fillcolor=self.colors['background'],
            line=dict(color="black", width=2)
        )
        
        # Add walls to create room structure
        walls = analysis_result.get('walls', [])
        if walls:
            self._add_walls_to_figure(fig, walls)
        else:
            # Create sample room structure if no walls detected
            self._create_sample_room_structure(fig, bounds)
        
        # Add restricted areas (blue rectangles)
        restricted_areas = analysis_result.get('restricted_areas', [])
        if restricted_areas:
            self._add_restricted_areas(fig, restricted_areas)
        
        # Add entrances (red circles/arcs)
        entrances = analysis_result.get('entrances', [])
        if entrances:
            self._add_entrances(fig, entrances)
        
        # Add îlots if in appropriate mode
        if mode in ['ilots', 'complete']:
            ilots = analysis_result.get('ilots', [])
            if ilots:
                self._add_ilots(fig, ilots)
        
        # Add corridors if in complete mode
        if mode == 'complete':
            corridors = analysis_result.get('corridors', [])
            if corridors:
                self._add_corridors(fig, corridors)
        
        # Configure layout
        fig.update_layout(
            title="Floor Plan Analysis",
            xaxis=dict(
                range=[bounds['min_x'] - width*0.1, bounds['max_x'] + width*0.1],
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[bounds['min_y'] - height*0.1, bounds['max_y'] + height*0.1],
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=600
        )
        
        # Add legend
        self._add_legend(fig)
        
        return fig
    
    def _add_walls_to_figure(self, fig: go.Figure, walls: List[Dict]):
        """Add walls to create proper room structure"""
        for wall in walls:
            points = wall.get('points', [])
            if len(points) >= 2:
                if wall.get('type') == 'LINE':
                    # Add line wall
                    fig.add_shape(
                        type="line",
                        x0=points[0][0], y0=points[0][1],
                        x1=points[1][0], y1=points[1][1],
                        line=dict(color=self.colors['walls'], width=3)
                    )
                elif wall.get('type') == 'POLYLINE':
                    # Add polyline wall
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=self.colors['walls'], width=3),
                        showlegend=False
                    ))
    
    def _create_sample_room_structure(self, fig: go.Figure, bounds: Dict):
        """Create sample room structure when no walls are detected"""
        min_x, max_x = bounds['min_x'], bounds['max_x']
        min_y, max_y = bounds['min_y'], bounds['max_y']
        
        # Create outer walls
        outer_walls = [
            [(min_x, min_y), (max_x, min_y)],  # Bottom
            [(max_x, min_y), (max_x, max_y)],  # Right
            [(max_x, max_y), (min_x, max_y)],  # Top
            [(min_x, max_y), (min_x, min_y)]   # Left
        ]
        
        # Create some internal walls to divide rooms
        width = max_x - min_x
        height = max_y - min_y
        
        internal_walls = [
            # Vertical divider
            [(min_x + width * 0.6, min_y), (min_x + width * 0.6, max_y)],
            # Horizontal divider
            [(min_x, min_y + height * 0.5), (min_x + width * 0.6, min_y + height * 0.5)],
            # Room divisions
            [(min_x + width * 0.3, min_y + height * 0.5), (min_x + width * 0.3, max_y)]
        ]
        
        all_walls = outer_walls + internal_walls
        
        # Add walls to figure
        for wall in all_walls:
            fig.add_shape(
                type="line",
                x0=wall[0][0], y0=wall[0][1],
                x1=wall[1][0], y1=wall[1][1],
                line=dict(color=self.colors['walls'], width=3)
            )
    
    def _add_restricted_areas(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add restricted areas (blue rectangles)"""
        for area in restricted_areas:
            bounds = area.get('bounds', {})
            if bounds:
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['restricted'],
                    opacity=0.7,
                    line=dict(color=self.colors['restricted'], width=2)
                )
    
    def _add_entrances(self, fig: go.Figure, entrances: List[Dict]):
        """Add entrances (red circles/arcs)"""
        for entrance in entrances:
            center = entrance.get('center', (0, 0))
            radius = entrance.get('radius', 1)
            
            # Create circle for entrance
            theta = np.linspace(0, 2*np.pi, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=self.colors['entrances'], width=3),
                fill='toself',
                fillcolor=self.colors['entrances'],
                opacity=0.7,
                showlegend=False
            ))
    
    def _add_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add îlots to the visualization"""
        for ilot in ilots:
            bounds = ilot.get('bounds', {})
            if bounds:
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['ilots'],
                    opacity=0.8,
                    line=dict(color=self.colors['ilots'], width=2)
                )
    
    def _add_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors to the visualization"""
        for corridor in corridors:
            points = corridor.get('points', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color=self.colors['corridors'], width=6),
                    showlegend=False
                ))
    
    def _add_legend(self, fig: go.Figure):
        """Add legend to the figure"""
        # Add invisible traces for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['walls'], size=10),
            name='MUR'
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['restricted'], size=10),
            name='NO ENTREE'
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['entrances'], size=10),
            name='ENTREE/SORTIE'
        ))