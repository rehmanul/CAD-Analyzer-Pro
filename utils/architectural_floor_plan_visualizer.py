
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
            'walls': '#6B7280',           # Gray walls (MUR)
            'background': '#F3F4F6',      # Light gray background
            'restricted': '#3B82F6',      # Blue for "NO ENTREE" areas
            'entrances': '#EF4444',       # Red for "ENTREE/SORTIE" areas
            'ilots': '#10B981',          # Green rectangles for îlots
            'corridors': '#F59E0B',      # Orange lines for corridors
            'text': '#1F2937',           # Dark text
            'legend_bg': '#FFFFFF'        # White legend background
        }
        
        # Line widths for architectural drawing
        self.line_widths = {
            'walls': 4,
            'entrances': 3,
            'corridors': 3,
            'ilots': 2
        }
    
    def create_empty_floor_plan(self, analysis_data: Dict) -> go.Figure:
        """Create empty floor plan like your reference image - Image 1"""
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add walls as gray lines - this is the main fix
        self._add_walls_from_analysis(fig, analysis_data)
        
        # Add blue restricted areas if they exist
        restricted_areas = analysis_data.get('restricted_areas', [])
        if restricted_areas:
            self._add_restricted_areas(fig, restricted_areas)
        
        # Add red entrances if they exist
        entrances = analysis_data.get('entrances', [])
        if entrances:
            self._add_entrances(fig, entrances)
        
        # Set clean layout matching your reference
        self._set_clean_layout(fig, bounds)
        
        # Add legend
        self._add_legend(fig)
        
        return fig
    
    def create_floor_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create floor plan with green rectangular îlots"""
        fig = self.create_empty_floor_plan(analysis_data)
        
        # Add green rectangular îlots
        if ilots:
            self._add_ilots(fig, ilots)
        
        # Update legend
        self._add_legend_with_ilots(fig)
        
        return fig
    
    def create_complete_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create complete floor plan with îlots and corridors"""
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add corridors
        if corridors:
            self._add_corridors(fig, corridors)
        
        # Update legend
        self._add_legend_with_corridors(fig)
        
        return fig
    
    def _add_walls_from_analysis(self, fig: go.Figure, analysis_data: Dict):
        """Add walls from analysis data - the key fix"""
        # First try to get walls from the 'walls' key
        walls = analysis_data.get('walls', [])
        
        # If no walls, try to get from 'entities' (raw DXF data)
        if not walls:
            entities = analysis_data.get('entities', [])
            if entities:
                print(f"DEBUG: Using {len(entities)} entities as walls")
                walls = entities[:1000]  # Limit to first 1000 for performance
        
        if not walls:
            print("DEBUG: No wall data found")
            return
        
        print(f"DEBUG: Processing {len(walls)} walls for visualization")
        
        # Process walls and add to figure
        wall_count = 0
        for wall in walls:
            try:
                # Handle different wall data formats
                if isinstance(wall, dict):
                    # Wall is a dictionary with points
                    points = wall.get('points', [])
                    if points and len(points) >= 2:
                        x_coords = [p[0] for p in points if len(p) >= 2]
                        y_coords = [p[1] for p in points if len(p) >= 2]
                        
                        if len(x_coords) >= 2:
                            fig.add_trace(go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode='lines',
                                line=dict(color=self.colors['walls'], width=self.line_widths['walls']),
                                name='Walls' if wall_count == 0 else None,
                                showlegend=(wall_count == 0),
                                hoverinfo='skip'
                            ))
                            wall_count += 1
                
                elif isinstance(wall, (list, tuple)):
                    # Wall is a list of points
                    if len(wall) >= 2:
                        # Check if wall contains coordinate pairs
                        if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in wall):
                            x_coords = [p[0] for p in wall]
                            y_coords = [p[1] for p in wall]
                            
                            fig.add_trace(go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode='lines',
                                line=dict(color=self.colors['walls'], width=self.line_widths['walls']),
                                name='Walls' if wall_count == 0 else None,
                                showlegend=(wall_count == 0),
                                hoverinfo='skip'
                            ))
                            wall_count += 1
                        
                        # Handle case where wall is just two points [x1, y1, x2, y2]
                        elif len(wall) == 4 and all(isinstance(p, (int, float)) for p in wall):
                            x_coords = [wall[0], wall[2]]
                            y_coords = [wall[1], wall[3]]
                            
                            fig.add_trace(go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode='lines',
                                line=dict(color=self.colors['walls'], width=self.line_widths['walls']),
                                name='Walls' if wall_count == 0 else None,
                                showlegend=(wall_count == 0),
                                hoverinfo='skip'
                            ))
                            wall_count += 1
                            
            except Exception as e:
                print(f"DEBUG: Error processing wall: {e}")
                continue
        
        print(f"DEBUG: Added {wall_count} wall elements to architectural visualization")
    
    def _add_restricted_areas(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add blue restricted areas (NO ENTREE)"""
        for i, area in enumerate(restricted_areas):
            if isinstance(area, dict):
                # Rectangle format
                x = area.get('x', 0)
                y = area.get('y', 0)
                width = area.get('width', 10)
                height = area.get('height', 10)
                
                fig.add_shape(
                    type="rect",
                    x0=x, y0=y, x1=x+width, y1=y+height,
                    fillcolor=self.colors['restricted'],
                    line=dict(color=self.colors['restricted'], width=2),
                    opacity=0.7
                )
            
            elif isinstance(area, (list, tuple)) and len(area) >= 2:
                # Polygon format
                x_coords = [p[0] for p in area if len(p) >= 2]
                y_coords = [p[1] for p in area if len(p) >= 2]
                
                if len(x_coords) >= 3:
                    fig.add_trace(go.Scatter(
                        x=x_coords + [x_coords[0]],  # Close the polygon
                        y=y_coords + [y_coords[0]],
                        fill='toself',
                        fillcolor=self.colors['restricted'],
                        line=dict(color=self.colors['restricted'], width=2),
                        mode='lines',
                        name='NO ENTREE' if i == 0 else None,
                        showlegend=(i == 0),
                        opacity=0.7
                    ))
    
    def _add_entrances(self, fig: go.Figure, entrances: List[Dict]):
        """Add red entrances (ENTREE/SORTIE)"""
        for i, entrance in enumerate(entrances):
            if isinstance(entrance, dict):
                # Point format
                x = entrance.get('x', 0)
                y = entrance.get('y', 0)
                
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(
                        color=self.colors['entrances'],
                        size=15,
                        symbol='circle'
                    ),
                    name='ENTREE/SORTIE' if i == 0 else None,
                    showlegend=(i == 0)
                ))
            
            elif isinstance(entrance, (list, tuple)) and len(entrance) >= 2:
                # Line/arc format
                x_coords = [p[0] for p in entrance if len(p) >= 2]
                y_coords = [p[1] for p in entrance if len(p) >= 2]
                
                if len(x_coords) >= 2:
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=self.colors['entrances'], width=self.line_widths['entrances']),
                        name='ENTREE/SORTIE' if i == 0 else None,
                        showlegend=(i == 0)
                    ))
    
    def _add_ilots(self, fig: go.Figure, ilots: List[Dict]):
        """Add green rectangular îlots"""
        for i, ilot in enumerate(ilots):
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 2)
            height = ilot.get('height', 2)
            
            fig.add_shape(
                type="rect",
                x0=x, y0=y, x1=x+width, y1=y+height,
                fillcolor=self.colors['ilots'],
                line=dict(color=self.colors['ilots'], width=self.line_widths['ilots']),
                opacity=0.8
            )
            
            # Add area label
            area = ilot.get('area', width * height)
            fig.add_annotation(
                x=x + width/2,
                y=y + height/2,
                text=f"{area:.1f}m²",
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor=self.colors['ilots'],
                bordercolor='white',
                borderwidth=1
            )
    
    def _add_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add orange corridor lines"""
        for i, corridor in enumerate(corridors):
            path = corridor.get('path', [])
            if len(path) >= 2:
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['corridors'],
                        width=self.line_widths['corridors'],
                        dash='dash'
                    ),
                    name='Corridors' if i == 0 else None,
                    showlegend=(i == 0)
                ))
    
    def _set_clean_layout(self, fig: go.Figure, bounds: Dict):
        """Set layout matching your reference image"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Calculate padding
        width = max_x - min_x
        height = max_y - min_y
        padding = max(width, height) * 0.05  # 5% padding
        
        fig.update_layout(
            title=None,
            paper_bgcolor='white',
            plot_bgcolor=self.colors['background'],
            width=1200,
            height=800,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor=self.colors['legend_bg'],
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            xaxis=dict(
                range=[min_x - padding, max_x + padding],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[min_y - padding, max_y + padding],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            margin=dict(l=0, r=150, t=0, b=0)
        )
    
    def _add_legend(self, fig: go.Figure):
        """Add legend matching your reference image"""
        # Add legend entries with colored squares
        legend_items = [
            ('NO ENTREE', self.colors['restricted']),
            ('ENTREE/SORTIE', self.colors['entrances']),
            ('MUR', self.colors['walls'])
        ]
        
        for name, color in legend_items:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    color=color,
                    size=15,
                    symbol='square'
                ),
                name=name,
                showlegend=True
            ))
    
    def _add_legend_with_ilots(self, fig: go.Figure):
        """Add legend including îlots"""
        self._add_legend(fig)
        
        # Add îlots to legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                color=self.colors['ilots'],
                size=15,
                symbol='square'
            ),
            name='Îlots',
            showlegend=True
        ))
    
    def _add_legend_with_corridors(self, fig: go.Figure):
        """Add legend including corridors"""
        self._add_legend_with_ilots(fig)
        
        # Add corridors to legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(
                color=self.colors['corridors'],
                width=3,
                dash='dash'
            ),
            name='Corridors',
            showlegend=True
        ))
