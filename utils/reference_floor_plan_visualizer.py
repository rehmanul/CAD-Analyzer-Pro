"""
Reference Floor Plan Visualizer
Creates exact match visualization to user's reference images
Clean, simple floor plan with gray walls, blue restricted areas, red entrances
"""

import plotly.graph_objects as go
from typing import Dict, List, Any
import numpy as np

class ReferenceFloorPlanVisualizer:
    """Creates visualization exactly matching user's reference images"""
    
    def __init__(self):
        self.wall_color = "#6B7280"  # Gray walls
        self.restricted_color = "#3B82F6"  # Blue restricted areas
        self.entrance_color = "#EF4444"  # Red entrances
        self.ilot_color = "#10B981"  # Green îlots
        self.corridor_color = "#F59E0B"  # Orange corridors
        self.background_color = "#F3F4F6"  # Light gray background
        
    def create_empty_floor_plan(self, analysis_data: Dict) -> go.Figure:
        """Create empty floor plan matching reference image 1"""
        fig = go.Figure()
        
        # Set background
        fig.update_layout(
            plot_bgcolor=self.background_color,
            paper_bgcolor="white",
            title="Floor Plan Visualization",
            title_x=0.5,
            showlegend=False,
            width=1200,
            height=800
        )
        
        # Add walls as gray lines
        walls = analysis_data.get('walls', [])
        if walls:
            self._add_walls_clean(fig, walls)
        
        # Add restricted areas as blue rectangles
        restricted_areas = analysis_data.get('restricted_areas', [])
        if restricted_areas:
            self._add_restricted_areas_clean(fig, restricted_areas)
        
        # Add entrances as red circles
        entrances = analysis_data.get('entrances', [])
        if entrances:
            self._add_entrances_clean(fig, entrances)
        
        # Set proper axis ranges
        bounds = analysis_data.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        self._set_clean_axis_ranges(fig, bounds)
        
        # Add clean legend
        self._add_clean_legend(fig)
        
        return fig
    
    def create_floor_plan_with_ilots(self, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create floor plan with green rectangular îlots - matching reference image 2"""
        fig = self.create_empty_floor_plan(analysis_data)
        
        # Add green rectangular îlots
        if ilots:
            self._add_ilots_clean(fig, ilots)
        
        # Update legend to include îlots
        self._add_legend_with_ilots(fig)
        
        return fig
    
    def create_complete_floor_plan(self, analysis_data: Dict, ilots: List[Dict], corridors: List[Dict]) -> go.Figure:
        """Create complete floor plan with îlots and corridors - matching reference image 3"""
        fig = self.create_floor_plan_with_ilots(analysis_data, ilots)
        
        # Add corridors
        if corridors:
            self._add_corridors_clean(fig, corridors)
        
        # Update legend to include corridors
        self._add_legend_with_corridors(fig)
        
        return fig
    
    def _add_walls_clean(self, fig: go.Figure, walls: List[Dict]):
        """Add walls as clean gray lines"""
        x_coords = []
        y_coords = []
        
        for wall in walls:
            points = wall.get('points', [])
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    x_coords.extend([points[i][0], points[i+1][0], None])
                    y_coords.extend([points[i][1], points[i+1][1], None])
        
        if x_coords:
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color=self.wall_color, width=2),
                name='Walls',
                hoverinfo='skip'
            ))
    
    def _add_restricted_areas_clean(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add restricted areas as clean blue rectangles"""
        for area in restricted_areas:
            bounds = area.get('bounds', {})
            if bounds:
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.restricted_color,
                    opacity=0.7,
                    line=dict(color=self.restricted_color, width=1)
                )
    
    def _add_entrances_clean(self, fig: go.Figure, entrances: List[Dict]):
        """Add entrances as clean red circles"""
        for entrance in entrances:
            center = entrance.get('center', (0, 0))
            radius = entrance.get('radius', 2)
            
            fig.add_shape(
                type="circle",
                center_x, center_y = center
                x0=center_x - radius, y0=center_y - radius,
                center_x, center_y = center
                x1=center_x + radius, y1=center_y + radius,
                fillcolor=self.entrance_color,
                opacity=0.8,
                line=dict(color=self.entrance_color, width=2)
            )
    
    def _add_ilots_clean(self, fig: go.Figure, ilots: List[Dict]):
        """Add îlots as clean green rectangles"""
        for ilot in ilots:
            bounds = ilot.get('bounds', {})
            if bounds:
                fig.add_shape(
                    type="rect",
                    x0=bounds['min_x'], y0=bounds['min_y'],
                    x1=bounds['max_x'], y1=bounds['max_y'],
                    fillcolor=self.ilot_color,
                    opacity=0.8,
                    line=dict(color=self.ilot_color, width=1)
                )
    
    def _add_corridors_clean(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors as clean orange lines"""
        for corridor in corridors:
            points = corridor.get('path', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color=self.corridor_color, width=3),
                    name='Corridor',
                    hoverinfo='skip'
                ))
    
    def _set_clean_axis_ranges(self, fig: go.Figure, bounds: Dict):
        """Set clean axis ranges with proper padding"""
        padding_x = (bounds['max_x'] - bounds['min_x']) * 0.1
        padding_y = (bounds['max_y'] - bounds['min_y']) * 0.1
        
        fig.update_xaxes(
            range=[bounds['min_x'] - padding_x, bounds['max_x'] + padding_x],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        )
        fig.update_yaxes(
            range=[bounds['min_y'] - padding_y, bounds['max_y'] + padding_y],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        )
    
    def _add_clean_legend(self, fig: go.Figure):
        """Add clean legend matching reference image"""
        fig.add_annotation(
            x=0.95, y=0.95,
            text="<b>LÉGENDE</b>",
            showarrow=False,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        legend_items = [
            {"color": self.restricted_color, "text": "NO ENTRÉE", "y": 0.88},
            {"color": self.entrance_color, "text": "ENTRÉE/SORTIE", "y": 0.84},
            {"color": self.wall_color, "text": "MUR", "y": 0.80}
        ]
        
        for item in legend_items:
            # Color square
            fig.add_shape(
                type="rect",
                x0=0.82, y0=item["y"] - 0.01,
                x1=0.85, y1=item["y"] + 0.01,
                fillcolor=item["color"],
                line=dict(color=item["color"], width=1),
                xref="paper", yref="paper"
            )
            
            # Text
            fig.add_annotation(
                x=0.87, y=item["y"],
                text=item["text"],
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="left", yanchor="middle",
                font=dict(size=11, color="black")
            )
    
    def _add_legend_with_ilots(self, fig: go.Figure):
        """Add legend including îlots"""
        fig.layout.annotations = []
        fig.layout.shapes = [s for s in fig.layout.shapes if not (s.get('x0', 0) > 0.8 and s.get('fillcolor') == 'white')]
        
        fig.add_annotation(
            x=0.95, y=0.95,
            text="<b>LÉGENDE</b>",
            showarrow=False,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        legend_items = [
            {"color": self.restricted_color, "text": "NO ENTRÉE", "y": 0.88},
            {"color": self.entrance_color, "text": "ENTRÉE/SORTIE", "y": 0.84},
            {"color": self.wall_color, "text": "MUR", "y": 0.80},
            {"color": self.ilot_color, "text": "ÎLOTS", "y": 0.76}
        ]
        
        for item in legend_items:
            # Color square
            fig.add_shape(
                type="rect",
                x0=0.82, y0=item["y"] - 0.01,
                x1=0.85, y1=item["y"] + 0.01,
                fillcolor=item["color"],
                line=dict(color=item["color"], width=1),
                xref="paper", yref="paper"
            )
            
            # Text
            fig.add_annotation(
                x=0.87, y=item["y"],
                text=item["text"],
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="left", yanchor="middle",
                font=dict(size=11, color="black")
            )
    
    def _add_legend_with_corridors(self, fig: go.Figure):
        """Add legend including corridors"""
        fig.layout.annotations = []
        fig.layout.shapes = [s for s in fig.layout.shapes if not (s.get('x0', 0) > 0.8 and s.get('fillcolor') == 'white')]
        
        fig.add_annotation(
            x=0.95, y=0.95,
            text="<b>LÉGENDE</b>",
            showarrow=False,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        legend_items = [
            {"color": self.restricted_color, "text": "NO ENTRÉE", "y": 0.88},
            {"color": self.entrance_color, "text": "ENTRÉE/SORTIE", "y": 0.84},
            {"color": self.wall_color, "text": "MUR", "y": 0.80},
            {"color": self.ilot_color, "text": "ÎLOTS", "y": 0.76},
            {"color": self.corridor_color, "text": "CORRIDORS", "y": 0.72}
        ]
        
        for item in legend_items:
            # Color square
            fig.add_shape(
                type="rect",
                x0=0.82, y0=item["y"] - 0.01,
                x1=0.85, y1=item["y"] + 0.01,
                fillcolor=item["color"],
                line=dict(color=item["color"], width=1),
                xref="paper", yref="paper"
            )
            
            # Text
            fig.add_annotation(
                x=0.87, y=item["y"],
                text=item["text"],
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="left", yanchor="middle",
                font=dict(size=11, color="black")
            )