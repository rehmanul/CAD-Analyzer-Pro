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
        """Create empty floor plan like your Image 1 - ONLY REAL DATA"""
        fig = go.Figure()
        
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add white background
        self._add_background(fig, bounds)
        
        # Add black walls - ONLY if real data exists
        walls = analysis_data.get('walls', [])
        entities = analysis_data.get('entities', [])
        
        print(f"DEBUG: Found {len(walls)} walls and {len(entities)} entities")
        
        if walls:
            self._add_walls(fig, walls)
        elif entities:
            # Try to extract walls from entities - ONLY real data
            self._add_walls_from_entities(fig, entities, bounds)
        else:
            print("DEBUG: No wall data found in analysis_data")
        
        # Add blue restricted areas (NO ENTREE) - ONLY if real data exists
        restricted_areas = analysis_data.get('restricted_areas', [])
        if restricted_areas:
            self._add_restricted_areas(fig, restricted_areas)
        
        # Add red entrance areas (ENTREE/SORTIE) - ONLY if real data exists
        entrances = analysis_data.get('entrances', [])
        if entrances:
            self._add_entrance_areas(fig, entrances)
        
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
    
    def create_complete_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create floor plan with îlots and red corridors like your Image 3"""
        # Start with îlots
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add red corridor lines
        self._add_red_corridors(fig, corridors)
        
        # Add area measurements in red text
        self._add_area_measurements(fig, ilots)
        
        return fig
    
    def _add_background(self, fig: go.Figure, bounds: Dict):
        """Add white background - no mock outlines"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        # Add white background only
        fig.add_shape(
            type="rect",
            x0=min_x - 5, y0=min_y - 5,
            x1=max_x + 5, y1=max_y + 5,
            fillcolor=self.colors['background'],
            line=dict(color=self.colors['background'], width=0)
        )
    
    def _add_walls(self, fig: go.Figure, walls: List):
        """Add black walls exactly like your reference"""
        print(f"DEBUG: Adding {len(walls)} walls")
        walls_added = 0
        
        for wall in walls:
            if len(wall) >= 2:
                x_coords = [point[0] for point in wall]
                y_coords = [point[1] for point in wall]
                
                print(f"DEBUG: Wall coordinates: {x_coords}, {y_coords}")
                
                # Add wall trace with guaranteed visibility
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['walls'],
                        width=4  # Increased width for visibility
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    name='wall'  # Add name for debugging
                ))
                walls_added += 1
        
        print(f"DEBUG: Successfully added {walls_added} wall traces to figure")
    
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
    
    def _add_walls_from_entities(self, fig: go.Figure, entities: List, bounds: Dict):
        """Create walls from DXF entities"""
        print(f"DEBUG: Processing {len(entities)} entities for walls")
        
        # Extract LINE entities as walls
        for entity in entities:
            if entity.get('type') == 'LINE':
                start = entity.get('start', [0, 0])
                end = entity.get('end', [100, 100])
                
                print(f"DEBUG: Adding wall from {start} to {end}")
                
                fig.add_trace(go.Scatter(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    mode='lines',
                    line=dict(color=self.colors['walls'], width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            elif entity.get('type') == 'POLYLINE':
                # Handle polylines as walls
                points = entity.get('points', [])
                if len(points) >= 2:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=self.colors['walls'], width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            elif entity.get('type') == 'LWPOLYLINE':
                # Handle lightweight polylines
                points = entity.get('points', [])
                if len(points) >= 2:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=self.colors['walls'], width=3),
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
                font=dict(color=self.colors['text'], size=10),
                showarrow=False,
                bgcolor='white',
                bordercolor=self.colors['text'],
                borderwidth=1
            )
    
    def _add_legend(self, fig: go.Figure):
        """Add legend like your reference image"""
        # Add legend entries as invisible traces
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['restricted']),
            name='NO ENTRÉE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['entrances']),
            name='ENTRÉE/SORTIE',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['walls']),
            name='MUR',
            showlegend=True
        ))
    
    def _set_clean_layout(self, fig: go.Figure, bounds: Dict):
        """Set clean layout matching your reference"""
        min_x, max_x = bounds.get('min_x', 0), bounds.get('max_x', 100)
        min_y, max_y = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        print(f"DEBUG: Setting layout with bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
        
        # Calculate proper padding
        width = max_x - min_x if max_x > min_x else 1000
        height = max_y - min_y if max_y > min_y else 1000
        padding = max(width * 0.05, height * 0.05, 100)  # Minimum 100 unit padding
        
        print(f"DEBUG: Using padding: {padding:.1f}")
        
        fig.update_layout(
            title="Floor Plan Analysis",
            title_font_size=20,
            title_x=0.5,
            xaxis=dict(
                showgrid=True,  # Enable grid temporarily for debugging
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=True,  # Show tick labels for debugging
                range=[min_x - padding, max_x + padding]
            ),
            yaxis=dict(
                showgrid=True,  # Enable grid temporarily for debugging
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=True,  # Show tick labels for debugging
                range=[min_y - padding, max_y + padding],
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            height=700,
            margin=dict(l=50, r=50, t=80, b=50)
        )
    

    
