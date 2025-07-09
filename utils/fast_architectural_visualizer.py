"""
Fast Architectural Visualizer
Optimized for rendering large numbers of walls without simplification
Uses WebGL and batch rendering for speed
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Tuple
import time

class FastArchitecturalVisualizer:
    """Fast visualization for large architectural data"""
    
    def __init__(self):
        self.colors = {
            'background': '#FFFFFF',
            'walls': '#2C2C2C',  # Dark gray walls
            'restricted': '#4285F4',  # Blue restricted areas
            'entrances': '#EA4335',  # Red entrances
            'ilots': '#EA4335',  # Red îlots
            'corridors': '#EA4335'  # Red corridors
        }
    
    def create_fast_floor_plan(self, analysis_data: Dict) -> go.Figure:
        """Create fast floor plan visualization using WebGL"""
        start_time = time.time()
        
        # Initialize figure with WebGL renderer
        fig = go.Figure()
        
        # Configure for WebGL performance
        fig.update_layout(
            template='plotly_white',
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode='pan'
        )
        
        # Extract data
        walls = analysis_data.get('walls', [])
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Add walls using batch rendering
        self._add_walls_fast(fig, walls)
        
        # Add restricted areas
        restricted_areas = analysis_data.get('restricted_areas', [])
        self._add_restricted_areas_fast(fig, restricted_areas)
        
        # Add entrances
        entrances = analysis_data.get('entrances', [])
        self._add_entrances_fast(fig, entrances)
        
        # Set axis ranges
        self._set_axis_ranges(fig, bounds)
        
        # Add legend
        self._add_legend(fig)
        
        render_time = time.time() - start_time
        print(f"Fast visualization rendered in {render_time:.2f}s")
        
        return fig
    
    def _add_walls_fast(self, fig: go.Figure, walls: List):
        """Add walls using batch rendering for speed"""
        if not walls:
            return
        
        # Batch all wall segments into single trace
        all_x = []
        all_y = []
        
        for wall in walls:
            points = wall.get('points', [])
            if len(points) >= 2:
                # Add wall segment
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    
                    all_x.extend([x1, x2, None])  # None creates line break
                    all_y.extend([y1, y2, None])
        
        if all_x and all_y:
            # Add single trace for all walls
            fig.add_trace(go.Scattergl(  # Use WebGL for performance
                x=all_x,
                y=all_y,
                mode='lines',
                line=dict(
                    color=self.colors['walls'],
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip',
                name='walls'
            ))
            
            print(f"Added {len(walls)} walls in single WebGL trace")
    
    def _add_restricted_areas_fast(self, fig: go.Figure, restricted_areas: List):
        """Add restricted areas efficiently"""
        for area in restricted_areas:
            bounds = area.get('bounds', {})
            if bounds:
                min_x = bounds.get('min_x', 0)
                max_x = bounds.get('max_x', 10)
                min_y = bounds.get('min_y', 0)
                max_y = bounds.get('max_y', 10)
                
                # Add as rectangle shape (faster than scatter)
                fig.add_shape(
                    type="rect",
                    x0=min_x, y0=min_y,
                    x1=max_x, y1=max_y,
                    fillcolor=self.colors['restricted'],
                    line=dict(color=self.colors['restricted'], width=1),
                    opacity=0.6
                )
    
    def _add_entrances_fast(self, fig: go.Figure, entrances: List):
        """Add entrances efficiently"""
        for entrance in entrances:
            center = entrance.get('center', (0, 0))
            radius = entrance.get('radius', 2)
            
            # Add as circle shape (faster than scatter)
            fig.add_shape(
                type="circle",
                x0=center[0] - radius, y0=center[1] - radius,
                x1=center[0] + radius, y1=center[1] + radius,
                fillcolor=self.colors['entrances'],
                line=dict(color=self.colors['entrances'], width=2),
                opacity=0.8
            )
    
    def _set_axis_ranges(self, fig: go.Figure, bounds: Dict):
        """Set appropriate axis ranges"""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        # Add small padding
        padding_x = (max_x - min_x) * 0.05
        padding_y = (max_y - min_y) * 0.05
        
        fig.update_xaxes(
            range=[min_x - padding_x, max_x + padding_x],
            constrain='domain'
        )
        fig.update_yaxes(
            range=[min_y - padding_y, max_y + padding_y],
            constrain='domain'
        )
    
    def _add_legend(self, fig: go.Figure):
        """Add legend matching client's reference image"""
        # Add legend as annotations positioned in top-right
        legend_x = 0.98
        legend_y = 0.98
        
        # Legend title
        fig.add_annotation(
            x=legend_x, y=legend_y,
            text="<b>LÉGENDE</b>",
            showarrow=False,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        # Legend items
        legend_items = [
            {"color": "#4285F4", "text": "NO ENTREE", "y_offset": -0.05},
            {"color": "#EA4335", "text": "ENTRÉE/SORTIE", "y_offset": -0.08},
            {"color": "#2C2C2C", "text": "MUR", "y_offset": -0.11}
        ]
        
        for item in legend_items:
            # Color box
            fig.add_annotation(
                x=legend_x - 0.08, y=legend_y + item["y_offset"],
                text="■",
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                font=dict(size=16, color=item["color"]),
                bgcolor="white"
            )
            
            # Text label
            fig.add_annotation(
                x=legend_x - 0.02, y=legend_y + item["y_offset"],
                text=item["text"],
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="left", yanchor="middle",
                font=dict(size=11, color="black"),
                bgcolor="white"
            )
    
    def create_floor_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create floor plan with red rectangular îlots like your reference image"""
        fig = self.create_fast_floor_plan(analysis_data)
        
        # Add red rectangular îlots exactly like your printed floor plan
        for ilot in ilots:
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)  
            width = ilot.get('width', 2)
            height = ilot.get('height', 2)
            
            # Create red rectangle with thin border
            fig.add_shape(
                type="rect",
                x0=x, y0=y,
                x1=x + width, y1=y + height,
                fillcolor=self.colors['ilots'],  # Red color
                line=dict(color=self.colors['ilots'], width=1),
                opacity=0.8
            )
        
        # Update legend to include îlots
        self._add_legend_with_ilots(fig)
        
        return fig
    
    def _add_legend_with_ilots(self, fig: go.Figure):
        """Add legend with îlots included"""
        # Clear existing legend annotations
        fig.layout.annotations = []
        
        # Add legend as annotations positioned in top-right
        legend_x = 0.98
        legend_y = 0.98
        
        # Legend title
        fig.add_annotation(
            x=legend_x, y=legend_y,
            text="<b>LÉGENDE</b>",
            showarrow=False,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        # Legend items including îlots
        legend_items = [
            {"color": "#4285F4", "text": "NO ENTREE", "y_offset": -0.05},
            {"color": "#EA4335", "text": "ENTRÉE/SORTIE", "y_offset": -0.08},
            {"color": "#EA4335", "text": "ÎLOTS", "y_offset": -0.11},
            {"color": "#2C2C2C", "text": "MUR", "y_offset": -0.14}
        ]
        
        for item in legend_items:
            # Color box
            fig.add_annotation(
                x=legend_x - 0.08, y=legend_y + item["y_offset"],
                text="■",
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                font=dict(size=16, color=item["color"]),
                bgcolor="white"
            )
            
            # Text label
            fig.add_annotation(
                x=legend_x - 0.02, y=legend_y + item["y_offset"],
                text=item["text"],
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="left", yanchor="middle",
                font=dict(size=11, color="black"),
                bgcolor="white"
            )
    
    def create_complete_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create complete floor plan with corridors"""
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add corridors as lines
        for corridor in corridors:
            points = corridor.get('points', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                fig.add_trace(go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['corridors'],
                        width=3
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        return fig