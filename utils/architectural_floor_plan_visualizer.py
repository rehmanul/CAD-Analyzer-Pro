"""
Architectural Floor Plan Visualizer
Creates exact match visualizations for your reference images:
- Clean architectural floor plans with proper color coding
- Gray walls (MUR), blue restricted areas (NO ENTREE), red entrances (ENTREE/SORTIE)
- Professional architectural drawing standards
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional

class ArchitecturalFloorPlanVisualizer:
    """Creates architectural floor plans matching your reference images exactly"""
    
    def __init__(self):
        # Exact colors from your reference images
        self.colors = {
            'walls': '#4a5568',           # Gray walls (MUR)
            'background': '#f7fafc',      # Light gray background
            'restricted': '#3182ce',      # Blue for "NO ENTREE" areas
            'entrances': '#e53e3e',       # Red for "ENTREE/SORTIE" areas
            'ilots': '#e53e3e',          # Red rectangles for îlots
            'corridors': '#e53e3e',      # Red lines for corridors
            'text': '#2d3748',           # Dark text
            'legend_bg': '#ffffff'        # White legend background
        }
        
        # Line widths for architectural drawing
        self.line_widths = {
            'walls': 3,
            'entrances': 2,
            'corridors': 2,
            'ilots': 2
        }
    
    def create_empty_floor_plan(self, analysis_data: Dict) -> go.Figure:
        """Create empty floor plan like your reference image - Image 1"""
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add background
        self._add_background(fig, bounds)
        
        # Add walls as gray lines
        self._add_architectural_walls(fig, analysis_data)
        
        # Add blue restricted areas (NO ENTREE)
        self._add_restricted_areas(fig, analysis_data.get('restricted_areas', []))
        
        # Add red entrance areas (ENTREE/SORTIE)
        self._add_entrance_areas(fig, analysis_data.get('entrances', []))
        
        # Add legend
        self._add_architectural_legend(fig)
        
        # Set professional layout
        self._set_architectural_layout(fig, bounds, "Floor Plan Analysis")
        
        return fig
    
    def create_floor_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create floor plan with red îlots like your reference image - Image 2"""
        fig = self.create_empty_floor_plan(analysis_data)
        
        # Add red rectangular îlots
        self._add_architectural_ilots(fig, ilots)
        
        return fig
    
    def create_complete_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create complete floor plan with îlots and corridors like your reference image - Image 3"""
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add red corridor lines
        self._add_architectural_corridors(fig, corridors)
        
        return fig
    
    def _add_background(self, fig: go.Figure, bounds: Dict):
        """Add light gray background"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add subtle background
        fig.add_shape(
            type="rect",
            x0=min_x, y0=min_y,
            x1=max_x, y1=max_y,
            fillcolor=self.colors['background'],
            line=dict(width=0),
            layer="below"
        )
    
    def _add_architectural_walls(self, fig: go.Figure, analysis_data: Dict):
        """Add gray walls with proper architectural styling"""
        walls = analysis_data.get('walls', [])
        entities = analysis_data.get('entities', [])
        
        walls_added = 0
        
        # Try walls first
        if walls:
            for wall in walls:
                if isinstance(wall, list) and len(wall) >= 2:
                    # Wall is a list of points
                    x_coords = [point[0] for point in wall if len(point) >= 2]
                    y_coords = [point[1] for point in wall if len(point) >= 2]
                    
                    if len(x_coords) >= 2:
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
                elif isinstance(wall, dict):
                    self._add_wall_line(fig, wall)
                    walls_added += 1
        
        # If no walls found, try entities (DXF entities)
        if entities and walls_added < 5:  # Add more entities if few walls found
            entity_count = 0
            for entity in entities:
                if entity_count >= 200:  # Limit to prevent overload
                    break
                    
                entity_type = entity.get('type', '').upper()
                if entity_type in ['LINE', 'POLYLINE', 'LWPOLYLINE', 'SPLINE']:
                    try:
                        self._add_entity_as_wall(fig, entity)
                        walls_added += 1
                        entity_count += 1
                    except:
                        continue
        
        print(f"DEBUG: Added {walls_added} wall elements to architectural visualization")
    
    def _add_wall_line(self, fig: go.Figure, wall: Dict):
        """Add individual wall line"""
        if 'start' in wall and 'end' in wall:
            fig.add_trace(go.Scatter(
                x=[wall['start'][0], wall['end'][0]],
                y=[wall['start'][1], wall['end'][1]],
                mode='lines',
                line=dict(
                    color=self.colors['walls'],
                    width=self.line_widths['walls']
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_entity_as_wall(self, fig: go.Figure, entity: Dict):
        """Add entity as wall line"""
        try:
            # Handle different entity formats
            if 'points' in entity and len(entity['points']) >= 2:
                points = entity['points']
                x_coords = []
                y_coords = []
                
                for point in points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                
                if len(x_coords) >= 2:
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
                    
            elif 'start' in entity and 'end' in entity:
                # Single line entity
                start = entity['start']
                end = entity['end']
                
                if (isinstance(start, (list, tuple)) and len(start) >= 2 and 
                    isinstance(end, (list, tuple)) and len(end) >= 2):
                    
                    fig.add_trace(go.Scatter(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        mode='lines',
                        line=dict(
                            color=self.colors['walls'],
                            width=self.line_widths['walls']
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
            elif 'coordinates' in entity:
                # Handle coordinate-based entities
                coords = entity['coordinates']
                if isinstance(coords, list) and len(coords) >= 2:
                    x_coords = []
                    y_coords = []
                    
                    for coord in coords:
                        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            x_coords.append(coord[0])
                            y_coords.append(coord[1])
                    
                    if len(x_coords) >= 2:
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
        except Exception as e:
            # Skip problematic entities silently
            pass
    
    def _add_restricted_areas(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add blue restricted areas (NO ENTREE)"""
        for area in restricted_areas:
            if 'bounds' in area:
                bounds = area['bounds']
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['restricted'],
                    opacity=0.6,
                    line=dict(color=self.colors['restricted'], width=2),
                    layer="below"
                )
    
    def _add_entrance_areas(self, fig: go.Figure, entrances: List[Dict]):
        """Add red entrance areas (ENTREE/SORTIE)"""
        for entrance in entrances:
            if 'bounds' in entrance:
                bounds = entrance['bounds']
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.colors['entrances'],
                    opacity=0.6,
                    line=dict(color=self.colors['entrances'], width=2),
                    layer="below"
                )
            elif 'center' in entrance and 'radius' in entrance:
                # Circular entrance
                center = entrance['center']
                radius = entrance['radius']
                
                # Create circle points
                theta = np.linspace(0, 2 * np.pi, 100)
                x_circle = center[0] + radius * np.cos(theta)
                y_circle = center[1] + radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    fill='toself',
                    fillcolor=self.colors['entrances'],
                    opacity=0.6,
                    line=dict(color=self.colors['entrances'], width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_architectural_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add red rectangular îlots"""
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
    
    def _add_architectural_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add red corridor lines"""
        for corridor in corridors:
            if 'start' in corridor and 'end' in corridor:
                fig.add_trace(go.Scatter(
                    x=[corridor['start'][0], corridor['end'][0]],
                    y=[corridor['start'][1], corridor['end'][1]],
                    mode='lines',
                    line=dict(
                        color=self.colors['corridors'],
                        width=self.line_widths['corridors']
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_architectural_legend(self, fig: go.Figure):
        """Add legend matching your reference image"""
        # Add legend items (invisible traces for legend)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['restricted']),
            name='NO ENTREE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['entrances']),
            name='ENTREE/SORTIE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['walls']),
            name='MUR',
            showlegend=True
        ))
    
    def _set_architectural_layout(self, fig: go.Figure, bounds: Dict, title: str):
        """Set professional architectural layout"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add padding
        padding = max(abs(max_x - min_x), abs(max_y - min_y)) * 0.05
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color=self.colors['text'], family="Arial, sans-serif")
            ),
            xaxis=dict(
                range=[min_x - padding, max_x + padding],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[min_y - padding, max_y + padding],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor=self.colors['legend_bg'],
                bordercolor=self.colors['walls'],
                borderwidth=1,
                font=dict(size=12, color=self.colors['text'])
            ),
            margin=dict(l=50, r=150, t=80, b=50),
            height=600
        )