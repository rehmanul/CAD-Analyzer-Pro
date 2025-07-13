"""
Empty Plan Visualizer
Creates clean architectural floor plan matching the reference image
Shows simplified room structure, not raw DXF data
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Tuple
import time

class EmptyPlanVisualizer:
    """Creates clean empty floor plan like reference image"""
    
    def __init__(self):
        self.colors = {
            'background': '#F5F5F5',  # Light gray background
            'walls': '#6B7280',       # Gray walls (MUR)
            'restricted': '#3B82F6',  # Blue restricted areas (NO ENTREE)
            'entrances': '#EF4444',   # Red entrances (ENTRÉE/SORTIE)
            'ilots': '#10B981',       # Green îlots like reference image
            'corridors': '#EF4444'    # Red corridors
        }
    
    def create_empty_plan(self, analysis_data: Dict) -> go.Figure:
        """Create clean empty floor plan matching reference image"""
        fig = go.Figure()
        
        # Configure layout
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
            plot_bgcolor=self.colors['background'],
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        # Get bounds
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        
        # Create simplified room structure from DXF data
        rooms = self._create_room_structure(analysis_data, bounds)
        
        # Add room walls
        self._add_room_walls(fig, rooms)
        
        # Add restricted areas
        self._add_restricted_areas(fig, analysis_data.get('restricted_areas', []))
        
        # Add entrances
        self._add_entrances(fig, analysis_data.get('entrances', []))
        
        # Set axis ranges
        self._set_axis_ranges(fig, bounds)
        
        # Add legend
        self._add_legend(fig)
        
        return fig
    
    def _create_room_structure(self, analysis_data: Dict, bounds: Dict) -> List[Dict]:
        """Use actual DXF wall data to create authentic floor plan"""
        walls = analysis_data.get('walls', [])
        if not walls:
            # Fallback to simple layout if no wall data
            return self._create_fallback_layout(bounds)
        
        # Use actual wall data for authentic visualization
        return walls
    
    def _add_room_walls(self, fig: go.Figure, walls: List[Dict]):
        """Add authentic walls from DXF data"""
        # Batch all wall segments for performance
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
            fig.add_trace(go.Scatter(
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
    
    def _create_fallback_layout(self, bounds: Dict) -> List[Dict]:
        """Create fallback simple layout if no wall data available"""
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        
        # Simple rectangular room
        return [{'points': [
            [bounds['min_x'], bounds['min_y']],
            [bounds['max_x'], bounds['min_y']],
            [bounds['max_x'], bounds['max_y']],
            [bounds['min_x'], bounds['max_y']],
            [bounds['min_x'], bounds['min_y']]
        ]}]
    
    def _add_restricted_areas(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add blue restricted areas"""
        for area in restricted_areas:
            bounds = area.get('bounds', {})
            if bounds:
                min_x = bounds.get('min_x', 0)
                max_x = bounds.get('max_x', 10)
                min_y = bounds.get('min_y', 0)
                max_y = bounds.get('max_y', 10)
                
                fig.add_shape(
                    type="rect",
                    x0=min_x, y0=min_y,
                    x1=max_x, y1=max_y,
                    fillcolor=self.colors['restricted'],
                    line=dict(color=self.colors['restricted'], width=2),
                    opacity=0.8
                )
    
    def _add_entrances(self, fig: go.Figure, entrances: List[Dict]):
        """Add red curved entrances"""
        for entrance in entrances:
            center = entrance.get('center', (0, 0))
            radius = entrance.get('radius', 2)
            
            # Create curved entrance arc
            theta = np.linspace(0, np.pi, 20)
            center_x, center_y = center
            x_arc = center_x + radius * np.cos(theta)
            center_x, center_y = center
            y_arc = center_y + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_arc,
                y=y_arc,
                mode='lines',
                line=dict(
                    color=self.colors['entrances'],
                    width=4
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _set_axis_ranges(self, fig: go.Figure, bounds: Dict):
        """Set axis ranges"""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        
        fig.update_xaxes(range=[min_x - padding_x, max_x + padding_x])
        fig.update_yaxes(range=[min_y - padding_y, max_y + padding_y])
    
    def _add_legend(self, fig: go.Figure):
        """Add legend matching reference image"""
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
            {"color": "#3B82F6", "text": "NO ENTREE", "y_offset": -0.05},
            {"color": "#EF4444", "text": "ENTRÉE/SORTIE", "y_offset": -0.08},
            {"color": "#6B7280", "text": "MUR", "y_offset": -0.11}
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
    
    def create_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create plan with green rectangular îlots matching reference image"""
        fig = self.create_empty_plan(analysis_data)
        
        # Add green rectangular îlots
        for ilot in ilots:
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 2)
            height = ilot.get('height', 2)
            
            fig.add_shape(
                type="rect",
                x0=x, y0=y,
                x1=x + width, y1=y + height,
                fillcolor=self.colors['ilots'],
                line=dict(color=self.colors['ilots'], width=1),
                opacity=0.8
            )
        
        # Update legend to include îlots
        self._add_legend_with_ilots(fig)
        
        return fig
    
    def _add_legend_with_ilots(self, fig: go.Figure):
        """Add legend with îlots"""
        fig.layout.annotations = []
        
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
        
        # Legend items with îlots
        legend_items = [
            {"color": "#6B7280", "text": "MUR", "y_offset": -0.05},
            {"color": "#3B82F6", "text": "NO ENTREE", "y_offset": -0.08},
            {"color": "#EF4444", "text": "ENTRÉE/SORTIE", "y_offset": -0.11},
            {"color": "#10B981", "text": "ÎLOTS", "y_offset": -0.14}
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