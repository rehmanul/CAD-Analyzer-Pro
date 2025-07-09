"""
Client Expected Visualizer - Exact Match to Expected Output
Creates visualizations that precisely match the client's expected output images
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json
import math

class ClientExpectedVisualizer:
    """Visualizer that creates exact matches to client expected output"""
    
    def __init__(self):
        # Color scheme matching client expected output EXACTLY
        self.colors = {
            'walls': '#000000',           # Black walls (MUR)
            'restricted_no_entry': '#0080FF',  # Blue (NO ENTREE)
            'entrances': '#FF0000',       # Red (ENTREE/SORTIE)
            'ilot_measurements': '#FF0000',  # Red measurements text
            'room_labels': '#000000',     # Black room labels
            'background': '#FFFFFF',      # White background
            'grid': '#E5E5E5',           # Light gray grid
            'dimensions': '#FF0000',      # Red dimension lines
            'area_text': '#FF0000'        # Red area text
        }
        
        # Size categories for îlots matching client requirements
        self.size_categories = {
            'small': {'min': 0, 'max': 1, 'color': '#FFFF00'},     # Yellow
            'medium': {'min': 1, 'max': 3, 'color': '#FFA500'},    # Orange  
            'large': {'min': 3, 'max': 5, 'color': '#008000'},     # Green
            'xlarge': {'min': 5, 'max': 10, 'color': '#800080'}    # Purple
        }
    
    def create_client_expected_visualization(self, analysis_data: Dict, ilots: List[Dict], 
                                           corridors: List[Dict], show_measurements: bool = True) -> go.Figure:
        """
        Create visualization that EXACTLY matches client expected output
        Based on attached images: walls (black), restricted areas (blue), entrances (red)
        With precise measurements and room labels
        """
        
        fig = go.Figure()
        
        # Set background to white
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add walls exactly as in client expected output - black lines
        self._add_client_expected_walls(fig, analysis_data.get('walls', []))
        
        # Add restricted areas (NO ENTREE) - blue zones
        self._add_client_expected_restricted_areas(fig, analysis_data.get('restricted_areas', []))
        
        # Add entrances (ENTREE/SORTIE) - red zones  
        self._add_client_expected_entrances(fig, analysis_data.get('entrances', []))
        
        # Add îlots with exact color coding and measurements
        self._add_client_expected_ilots(fig, ilots, show_measurements)
        
        # Add corridors matching client requirements
        self._add_client_expected_corridors(fig, corridors)
        
        # Add room measurements and labels exactly as in expected output
        if show_measurements:
            self._add_client_expected_measurements(fig, analysis_data, ilots)
        
        # Configure layout to match client expected output exactly
        self._configure_client_expected_layout(fig, analysis_data.get('bounds', {}))
        
        return fig
    
    def _add_client_expected_walls(self, fig: go.Figure, walls: List):
        """Add walls exactly as shown in client expected output - black lines"""
        for wall in walls:
            # Handle both dict format and simple coordinate array format
            if isinstance(wall, dict) and 'type' in wall:
                if wall['type'] == 'line':
                    coords = wall['coordinates']
                elif wall['type'] == 'polygon':
                    coords = wall['coordinates']
                else:
                    coords = wall.get('coordinates', [])
            else:
                # Handle simple coordinate array format from DXF processing
                coords = wall
            
            if len(coords) >= 2:
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                
                # Close polygon if more than 2 points
                if len(coords) > 2:
                    x_coords.append(coords[0][0])
                    y_coords.append(coords[0][1])
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color=self.colors['walls'], width=2),
                    showlegend=False,
                    hoverinfo='skip'
                    ))
            
            elif wall['type'] == 'polyline':
                coords = wall['coordinates']
                if len(coords) >= 2:
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=self.colors['walls'], width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    def _add_client_expected_restricted_areas(self, fig: go.Figure, restricted_areas: List):
        """Add restricted areas exactly as in client expected output - blue zones marked 'NO ENTREE'"""
        for i, area in enumerate(restricted_areas):
            # Handle both dict format and simple coordinate array format
            if isinstance(area, dict) and 'type' in area:
                if area['type'] == 'polygon':
                    coords = area['coordinates']
                elif area['type'] == 'circle':
                    center = area['center']
                    radius = area['radius']
                    # Convert circle to polygon
                    angles = np.linspace(0, 2*np.pi, 50)
                    coords = [[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles]
                else:
                    coords = area.get('coordinates', [])
            else:
                # Handle simple coordinate array format
                coords = area
            
            if len(coords) >= 3:
                x_coords = [coord[0] for coord in coords] + [coords[0][0]]
                y_coords = [coord[1] for coord in coords] + [coords[0][1]]
                
                # Add filled blue area
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=self.colors['restricted_no_entry'],
                    line=dict(color=self.colors['restricted_no_entry'], width=1),
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add "NO ENTREE" label
                center_x = sum(x_coords[:-1]) / len(x_coords[:-1])
                center_y = sum(y_coords[:-1]) / len(y_coords[:-1])
                
                fig.add_annotation(
                    x=center_x,
                    y=center_y,
                    text="NO ENTREE",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial"),
                    bgcolor=self.colors['restricted_no_entry'],
                    bordercolor="white",
                    borderwidth=1
                )
                if len(coords) >= 3:
                    x_coords = [coord[0] for coord in coords] + [coords[0][0]]
                    y_coords = [coord[1] for coord in coords] + [coords[0][1]]
                    
                    # Add filled blue area
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill='toself',
                        fillcolor=self.colors['restricted_no_entry'],
                        line=dict(color=self.colors['restricted_no_entry'], width=1),
                        opacity=0.7,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Add "NO ENTREE" label
                    center_x = sum(x_coords[:-1]) / len(x_coords[:-1])
                    center_y = sum(y_coords[:-1]) / len(y_coords[:-1])
                    
                    fig.add_annotation(
                        x=center_x,
                        y=center_y,
                        text="NO ENTREE",
                        showarrow=False,
                        font=dict(color="white", size=10, family="Arial"),
                        bgcolor=self.colors['restricted_no_entry'],
                        bordercolor="white",
                        borderwidth=1
                    )
            
            elif area['type'] == 'circle':
                center = area['center']
                radius = area['radius']
                
                # Create circle coordinates
                angles = np.linspace(0, 2*np.pi, 50)
                x_coords = [center[0] + radius * np.cos(angle) for angle in angles]
                y_coords = [center[1] + radius * np.sin(angle) for angle in angles]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=self.colors['restricted_no_entry'],
                    line=dict(color=self.colors['restricted_no_entry'], width=1),
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add "NO ENTREE" label
                fig.add_annotation(
                    x=center[0],
                    y=center[1],
                    text="NO ENTREE",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial"),
                    bgcolor=self.colors['restricted_no_entry'],
                    bordercolor="white",
                    borderwidth=1
                )
    
    def _add_client_expected_entrances(self, fig: go.Figure, entrances: List):
        """Add entrances exactly as in client expected output - red zones marked 'ENTREE/SORTIE'"""
        for entrance in entrances:
            # Handle both dict format and simple coordinate array format
            if isinstance(entrance, dict) and 'type' in entrance:
                if entrance['type'] == 'polygon':
                    coords = entrance['coordinates']
                else:
                    coords = entrance.get('coordinates', [])
            else:
                # Handle simple coordinate array format
                coords = entrance
            
            if len(coords) >= 3:
                coords = entrance['coordinates']
                if len(coords) >= 3:
                    x_coords = [coord[0] for coord in coords] + [coords[0][0]]
                    y_coords = [coord[1] for coord in coords] + [coords[0][1]]
                    
                    # Add filled red area
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill='toself',
                        fillcolor=self.colors['entrances'],
                        line=dict(color=self.colors['entrances'], width=1),
                        opacity=0.7,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Add "ENTREE/SORTIE" label
                    center_x = sum(x_coords[:-1]) / len(x_coords[:-1])
                    center_y = sum(y_coords[:-1]) / len(y_coords[:-1])
                    
                    fig.add_annotation(
                        x=center_x,
                        y=center_y,
                        text="ENTREE/SORTIE",
                        showarrow=False,
                        font=dict(color="white", size=8, family="Arial"),
                        bgcolor=self.colors['entrances'],
                        bordercolor="white",
                        borderwidth=1
                    )
            
            elif entrance['type'] == 'rectangle':
                coords = entrance['coordinates']
                if len(coords) >= 4:
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill='toself',
                        fillcolor=self.colors['entrances'],
                        line=dict(color=self.colors['entrances'], width=1),
                        opacity=0.7,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Add label
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    
                    fig.add_annotation(
                        x=center_x,
                        y=center_y,
                        text="ENTREE/SORTIE",
                        showarrow=False,
                        font=dict(color="white", size=8, family="Arial"),
                        bgcolor=self.colors['entrances'],
                        bordercolor="white",
                        borderwidth=1
                    )
    
    def _add_client_expected_ilots(self, fig: go.Figure, ilots: List[Dict], show_measurements: bool):
        """Add îlots with exact color coding and measurements as per client expected output"""
        for ilot in ilots:
            # Get îlot color based on size category
            area = ilot.get('area', 0)
            color = self._get_ilot_color_by_area(area)
            
            # Add îlot rectangle with flexible data handling
            position = ilot.get('position', [ilot.get('x', 0), ilot.get('y', 0)])
            x, y = position
            width = ilot.get('width', 3.0)
            height = ilot.get('height', 2.0)
            
            fig.add_shape(
                type="rect",
                x0=x, y0=y,
                x1=x + width, y1=y + height,
                fillcolor=color,
                line=dict(color='black', width=1),
                opacity=0.8
            )
            
            # Add area measurement in red text (matching client expected output)
            if show_measurements:
                center_x = x + width / 2
                center_y = y + height / 2
                
                fig.add_annotation(
                    x=center_x,
                    y=center_y,
                    text=f"{area:.1f}m²",
                    showarrow=False,
                    font=dict(color=self.colors['area_text'], size=10, family="Arial Black"),
                    bgcolor="white",
                    bordercolor=self.colors['area_text'],
                    borderwidth=1
                )
    
    def _add_client_expected_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors matching client requirements - blue lines between îlots"""
        for corridor in corridors:
            if corridor['type'] == 'mandatory':
                # Mandatory corridors between facing îlots - blue lines
                path = corridor['path']
                if len(path) >= 2:
                    x_coords = [point[0] for point in path]
                    y_coords = [point[1] for point in path]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color='#0000FF', width=3, dash='solid'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            elif corridor['type'] == 'access':
                # Access corridors - lighter blue
                path = corridor['path']
                if len(path) >= 2:
                    x_coords = [point[0] for point in path]
                    y_coords = [point[1] for point in path]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color='#00FFFF', width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    def _add_client_expected_measurements(self, fig: go.Figure, analysis_data: Dict, ilots: List[Dict]):
        """Add precise measurements and room labels exactly as in client expected output"""
        bounds = analysis_data.get('bounds', {})
        
        # Add dimension lines and measurements matching client expected output
        if bounds:
            # Overall dimensions
            total_width = bounds['max_x'] - bounds['min_x']
            total_height = bounds['max_y'] - bounds['min_y']
            
            # Add horizontal dimension line at top
            fig.add_shape(
                type="line",
                x0=bounds['min_x'], y0=bounds['max_y'] + 2,
                x1=bounds['max_x'], y1=bounds['max_y'] + 2,
                line=dict(color=self.colors['dimensions'], width=1)
            )
            
            # Add measurement text
            fig.add_annotation(
                x=(bounds['min_x'] + bounds['max_x']) / 2,
                y=bounds['max_y'] + 4,
                text=f"{total_width:.1f}m",
                showarrow=False,
                font=dict(color=self.colors['dimensions'], size=10, family="Arial"),
                bgcolor="white"
            )
            
            # Add vertical dimension line at right
            fig.add_shape(
                type="line",
                x0=bounds['max_x'] + 2, y0=bounds['min_y'],
                x1=bounds['max_x'] + 2, y1=bounds['max_y'],
                line=dict(color=self.colors['dimensions'], width=1)
            )
            
            # Add measurement text
            fig.add_annotation(
                x=bounds['max_x'] + 4,
                y=(bounds['min_y'] + bounds['max_y']) / 2,
                text=f"{total_height:.1f}m",
                showarrow=False,
                font=dict(color=self.colors['dimensions'], size=10, family="Arial"),
                bgcolor="white",
                textangle=90
            )
        
        # Add individual îlot measurements
        for ilot in ilots:
            x, y = ilot['position']
            width = ilot.get('width', 3.0)
            height = ilot.get('height', 2.0)
            
            # Add width measurement
            fig.add_annotation(
                x=x + width / 2,
                y=y - 1,
                text=f"{width:.1f}m",
                showarrow=False,
                font=dict(color=self.colors['dimensions'], size=8, family="Arial"),
                bgcolor="white"
            )
            
            # Add height measurement
            fig.add_annotation(
                x=x - 1,
                y=y + height / 2,
                text=f"{height:.1f}m",
                showarrow=False,
                font=dict(color=self.colors['dimensions'], size=8, family="Arial"),
                bgcolor="white",
                textangle=90
            )
    
    def _get_ilot_color_by_area(self, area: float) -> str:
        """Get îlot color based on area (matching client size categories)"""
        for category, info in self.size_categories.items():
            if info['min'] <= area <= info['max']:
                return info['color']
        
        # Default to largest category if area exceeds all ranges
        return self.size_categories['xlarge']['color']
    
    def _configure_client_expected_layout(self, fig: go.Figure, bounds: Dict):
        """Configure layout to match client expected output exactly"""
        if not bounds:
            bounds = {'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 100}
        
        # Add padding for measurements
        padding = 10
        
        # Calculate legend position outside of visual area
        legend_margin = 120  # Extra margin for legend
        
        fig.update_layout(
            title={
                'text': 'Plan d\'étage avec îlots et couloirs',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial', 'color': 'black'}
            },
            xaxis=dict(
                range=[bounds['min_x'] - padding, bounds['max_x'] + padding],
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor=self.colors['grid'],
                gridwidth=1,
                showticklabels=True,
                title="Distance (m)"
            ),
            yaxis=dict(
                range=[bounds['min_y'] - padding, bounds['max_y'] + padding],
                showgrid=True,
                gridcolor=self.colors['grid'],
                gridwidth=1,
                showticklabels=True,
                title="Distance (m)"
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=50, r=legend_margin, t=80, b=50)  # Extra right margin for legend
        )
        
        # Add legend completely outside the visual area using relative positioning
        fig.add_annotation(
            xref="paper", yref="paper",  # Use paper coordinates (0-1) instead of data coordinates
            x=1.02, y=0.98,  # Position at 102% of plot width, 98% of plot height
            text="<b>LÉGENDE</b><br>" +
                 "▬ MUR<br>" +
                 "🔵 NO ENTREE<br>" +
                 "🔴 ENTREE/SORTIE<br>" +
                 "🟡 Îlot 0-1m²<br>" +
                 "🟠 Îlot 1-3m²<br>" +
                 "🟢 Îlot 3-5m²<br>" +
                 "🟣 Îlot 5-10m²",
            showarrow=False,
            font=dict(size=10, family="Arial"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            align="left",
            xanchor="left",
            yanchor="top"
        )
    
    def create_measurements_overlay(self, fig: go.Figure, analysis_data: Dict, ilots: List[Dict]) -> go.Figure:
        """Create measurements overlay exactly matching client expected output"""
        # This creates the red measurement annotations seen in expected output view 3
        
        for ilot in ilots:
            x, y = ilot['position']
            width = ilot.get('width', 3.0)
            height = ilot.get('height', 2.0)
            area = ilot.get('area', width * height)
            
            # Add precise area measurement in red
            fig.add_annotation(
                x=x + width / 2,
                y=y + height / 2,
                text=f"{area:.2f}m²",
                showarrow=False,
                font=dict(color=self.colors['area_text'], size=12, family="Arial Black"),
                bgcolor="white",
                bordercolor=self.colors['area_text'],
                borderwidth=2
            )
            
            # Add dimension lines
            # Width dimension
            fig.add_shape(
                type="line",
                x0=x, y0=y - 0.5,
                x1=x + width, y1=y - 0.5,
                line=dict(color=self.colors['dimensions'], width=1)
            )
            
            fig.add_annotation(
                x=x + width / 2,
                y=y - 0.5,
                text=f"{width:.1f}m",
                showarrow=False,
                font=dict(color=self.colors['dimensions'], size=8, family="Arial"),
                bgcolor="white"
            )
            
            # Height dimension
            fig.add_shape(
                type="line",
                x0=x - 0.5, y0=y,
                x1=x - 0.5, y1=y + height,
                line=dict(color=self.colors['dimensions'], width=1)
            )
            
            fig.add_annotation(
                x=x - 0.5,
                y=y + height / 2,
                text=f"{height:.1f}m",
                showarrow=False,
                font=dict(color=self.colors['dimensions'], size=8, family="Arial"),
                bgcolor="white",
                textangle=90
            )
        
        return fig