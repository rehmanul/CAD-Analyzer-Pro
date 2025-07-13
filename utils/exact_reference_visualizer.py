"""
Exact Reference Visualizer
Creates visualizations that EXACTLY match your reference images:
- Professional architectural floor plans with proper room boundaries
- Correct color coding: Gray walls (MUR), Blue restricted (NO ENTREE), Red entrances (ENTREE/SORTIE)
- Proper room structure and layout matching your expectations
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional

class ExactReferenceVisualizer:
    """Creates exact match visualizations for your reference images"""
    
    def __init__(self):
        # Exact colors from your reference images
        self.colors = {
            'walls': '#6b7280',           # Gray walls (MUR)
            'background': '#f3f4f6',      # Light gray background
            'restricted': '#3b82f6',      # Blue for "NO ENTREE" areas
            'entrances': '#ef4444',       # Red for "ENTREE/SORTIE" areas
            'ilots': '#22c55e',          # Green rectangles for îlots (like your reference)
            'corridors': '#ef4444',      # Red lines for corridors
            'text': '#374151',           # Dark text
            'room_interior': '#ffffff'    # White room interiors
        }
        
        # Line widths matching architectural standards
        self.line_widths = {
            'walls': 4,
            'entrances': 3,
            'corridors': 2,
            'ilots': 1
        }
    
    def create_architectural_floor_plan(self, analysis_data: Dict, mode: str = 'base') -> go.Figure:
        """Create architectural floor plan matching your reference images exactly"""
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add background
        self._add_architectural_background(fig, bounds)
        
        # Add walls to form room boundaries (like your reference)
        self._add_room_structure_walls(fig, analysis_data)
        
        # Add restricted areas (blue zones)
        self._add_restricted_zones(fig, analysis_data.get('restricted_areas', []))
        
        # Add entrance areas (red curved areas)
        self._add_entrance_zones(fig, analysis_data.get('entrances', []))
        
        # Add îlots if in appropriate mode
        if mode == 'with_ilots':
            self._add_green_ilots(fig, analysis_data.get('ilots', []))
        elif mode == 'detailed':
            self._add_green_ilots(fig, analysis_data.get('ilots', []))
            self._add_corridor_lines(fig, analysis_data.get('corridors', []))
        
        # Add professional legend
        self._add_professional_legend(fig)
        
        # Set architectural layout
        self._set_professional_layout(fig, bounds)
        
        return fig
    
    def _add_architectural_background(self, fig: go.Figure, bounds: Dict):
        """Add professional architectural background"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add background
        fig.add_shape(
            type="rect",
            x0=min_x, y0=min_y,
            x1=max_x, y1=max_y,
            fillcolor=self.colors['background'],
            line=dict(width=0),
            layer="below"
        )
    
    def _add_room_structure_walls(self, fig: go.Figure, analysis_data: Dict):
        """Add walls that form room boundaries like your reference image"""
        walls = analysis_data.get('walls', [])
        entities = analysis_data.get('entities', [])
        
        walls_added = 0
        
        # Process wall data
        if walls:
            for wall in walls:
                if isinstance(wall, list) and len(wall) >= 2:
                    # Create continuous wall lines
                    x_coords = [point[0] for point in wall]
                    y_coords = [point[1] for point in wall]
                    
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
                    walls_added += 1
        
        # Process entities as walls if no walls found
        if entities and walls_added == 0:
            for entity in entities:
                if entity.get('type') in ['LINE', 'POLYLINE', 'LWPOLYLINE']:
                    if 'points' in entity and len(entity['points']) >= 2:
                        points = entity['points']
                        x_coords = [point[0] for point in points]
                        y_coords = [point[1] for point in points]
                        
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
                        walls_added += 1
        
        print(f"DEBUG: Added {walls_added} wall elements to create room structure")
    
    def _add_restricted_zones(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add blue restricted areas (NO ENTREE) matching your reference"""
        for area in restricted_areas:
            if 'bounds' in area:
                bounds = area['bounds']
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['restricted'],
                    opacity=0.7,
                    line=dict(color=self.colors['restricted'], width=1),
                    layer="below"
                )
    
    def _add_entrance_zones(self, fig: go.Figure, entrances: List[Dict]):
        """Add red entrance zones (ENTREE/SORTIE) with curves like your reference"""
        for entrance in entrances:
            if 'center' in entrance and 'radius' in entrance:
                center = entrance.get('center', [0, 0])
                radius = entrance.get('radius', 1.0)
                
                # Create curved entrance marking
                theta = np.linspace(0, np.pi, 50)  # Half circle for entrance
                center_x, center_y = center
                x_curve = center_x + radius * np.cos(theta)
                center_x, center_y = center
                y_curve = center_y + radius * np.sin(theta)
                
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
            elif 'bounds' in entrance:
                # Rectangular entrance area
                bounds = entrance['bounds']
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['entrances'],
                    opacity=0.7,
                    line=dict(color=self.colors['entrances'], width=2),
                    layer="below"
                )
    
    def _add_green_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add green rectangular îlots like your reference Image 2"""
        for ilot in ilots:
            if 'bounds' in ilot:
                bounds = ilot['bounds']
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['ilots'],
                    opacity=0.8,
                    line=dict(color=self.colors['ilots'], width=self.line_widths['ilots']),
                    layer="above"
                )
    
    def _add_corridor_lines(self, fig: go.Figure, corridors: List[Dict]):
        """Add red corridor lines like your reference Image 3"""
        for corridor in corridors:
            if 'path' in corridor and len(corridor['path']) >= 2:
                path = corridor['path']
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
    
    def _add_professional_legend(self, fig: go.Figure):
        """Add legend exactly matching your reference image"""
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=self.colors['restricted'], symbol='square'),
            name='NO ENTREE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=self.colors['entrances'], symbol='square'),
            name='ENTREE/SORTIE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=self.colors['walls'], symbol='square'),
            name='MUR',
            showlegend=True
        ))
    
    def _set_professional_layout(self, fig: go.Figure, bounds: Dict):
        """Set professional architectural layout matching your reference"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Calculate appropriate padding
        width = max_x - min_x if max_x > min_x else 100
        height = max_y - min_y if max_y > min_y else 100
        padding = max(width * 0.05, height * 0.05, 10)
        
        fig.update_layout(
            title=dict(
                text="Floor Plan Analysis",
                x=0.5,
                font=dict(size=18, color=self.colors['text'], family="Arial, sans-serif")
            ),
            xaxis=dict(
                range=[min_x - padding, max_x + padding],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[min_y - padding, max_y + padding],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='white',
                bordercolor=self.colors['walls'],
                borderwidth=1,
                font=dict(size=12, color=self.colors['text'])
            ),
            margin=dict(l=50, r=150, t=80, b=50),
            height=600
        )