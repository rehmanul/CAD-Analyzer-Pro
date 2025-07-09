"""
Simple Floor Plan Creator
Creates clean, connected floor plan visualizations from DXF wall data
"""

import plotly.graph_objects as go
from typing import Dict, List, Any
import numpy as np

class SimpleFloorPlanCreator:
    """Creates simplified but accurate floor plan visualizations"""
    
    def __init__(self):
        self.colors = {
            'walls': '#000000',  # Black
            'restricted': '#3B82F6',  # Blue
            'entrances': '#EF4444',  # Red
            'ilots': '#EF4444',  # Red
            'corridors': '#EF4444',  # Red
            'background': '#FFFFFF',  # White
            'text': '#EF4444'  # Red
        }
    
    def create_floor_plan_from_walls(self, analysis_data: Dict) -> go.Figure:
        """Create a clean floor plan from wall data"""
        print("DEBUG: Creating simplified floor plan visualization")
        
        # Initialize figure
        fig = go.Figure()
        
        # Get wall data
        walls = analysis_data.get('walls', [])
        bounds = analysis_data.get('bounds', {})
        
        print(f"DEBUG: Processing {len(walls)} walls for simplified visualization")
        
        if not walls:
            print("DEBUG: No walls found, creating empty plot")
            return self._create_empty_plot()
        
        # Process walls to create connected floor plan
        connected_walls = self._process_walls_for_connection(walls)
        
        # Add walls to figure
        self._add_connected_walls(fig, connected_walls)
        
        # Set up clean layout
        self._set_simple_layout(fig, bounds)
        
        # Add legend
        self._add_simple_legend(fig)
        
        return fig
    
    def _process_walls_for_connection(self, walls: List) -> List:
        """Process walls to create better connections"""
        print("DEBUG: Processing walls for better connections")
        
        # Group walls by proximity to create building outline
        processed_walls = []
        
        for wall in walls:
            if len(wall) >= 2:
                # Create line segments
                for i in range(len(wall) - 1):
                    start_point = wall[i]
                    end_point = wall[i + 1]
                    
                    if self._is_valid_wall_segment(start_point, end_point):
                        processed_walls.append([start_point, end_point])
        
        print(f"DEBUG: Created {len(processed_walls)} wall segments")
        return processed_walls
    
    def _is_valid_wall_segment(self, start_point, end_point) -> bool:
        """Check if wall segment is valid"""
        try:
            # Check if points are different
            if start_point[0] == end_point[0] and start_point[1] == end_point[1]:
                return False
            
            # Check if segment is not too small (filter out tiny segments)
            distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            return distance > 10  # Minimum wall length
            
        except (IndexError, TypeError):
            return False
    
    def _add_connected_walls(self, fig: go.Figure, walls: List):
        """Add walls to create connected floor plan"""
        print(f"DEBUG: Adding {len(walls)} connected wall segments")
        
        for wall in walls:
            if len(wall) >= 2:
                x_coords = [wall[0][0], wall[1][0]]
                y_coords = [wall[0][1], wall[1][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.colors['walls'],
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _set_simple_layout(self, fig: go.Figure, bounds: Dict):
        """Set simple, clean layout"""
        min_x = bounds.get('min_x', 0)
        max_x = bounds.get('max_x', 100)
        min_y = bounds.get('min_y', 0)
        max_y = bounds.get('max_y', 100)
        
        # Calculate padding
        width = max_x - min_x
        height = max_y - min_y
        padding = max(width * 0.1, height * 0.1, 1000)
        
        print(f"DEBUG: Setting layout - X: [{min_x:.1f}, {max_x:.1f}], Y: [{min_y:.1f}, {max_y:.1f}], Padding: {padding:.1f}")
        
        fig.update_layout(
            title="Floor Plan Analysis",
            title_x=0.5,
            xaxis=dict(
                range=[min_x - padding, max_x + padding],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                range=[min_y - padding, max_y + padding],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )
    
    def _add_simple_legend(self, fig: go.Figure):
        """Add simple legend"""
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=self.colors['walls'], size=10),
            name='MUR',
            showlegend=True
        ))
    
    def _create_empty_plot(self) -> go.Figure:
        """Create empty plot when no data"""
        fig = go.Figure()
        fig.update_layout(
            title="No Floor Plan Data Available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        return fig