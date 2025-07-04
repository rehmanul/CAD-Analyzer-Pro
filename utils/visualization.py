import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloorPlanVisualizer:
    """Professional floor plan visualization with interactive controls"""
    
    def __init__(self):
        self.color_scheme = {
            'walls': '#2C3E50',
            'restricted_areas': '#E74C3C',
            'entrances': '#E67E22',
            'open_spaces': '#ECF0F1',
            'ilots': {
                'small': '#3498DB',
                'medium': '#2ECC71',
                'large': '#9B59B6'
            },
            'corridors': {
                'main': '#34495E',
                'secondary': '#7F8C8D',
                'access': '#95A5A6'
            },
            'background': '#FFFFFF'
        }
        
        self.layer_visibility = {
            'walls': True,
            'restricted_areas': True,
            'entrances': True,
            'open_spaces': True,
            'ilots': True,
            'corridors': True,
            'text': False,
            'grid': False
        }
    
    def create_interactive_view(self, floor_plan_data: Dict[str, Any],
                               analysis_results: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create interactive floor plan visualization
        
        Args:
            floor_plan_data: Original floor plan data
            analysis_results: Analysis results (optional)
            
        Returns:
            Plotly figure with interactive floor plan
        """
        logger.info("Creating interactive floor plan visualization")
        
        try:
            # Initialize figure
            fig = go.Figure()
            
            # Add base floor plan elements
            self._add_floor_plan_elements(fig, floor_plan_data)
            
            # Add analysis results if available
            if analysis_results:
                self._add_analysis_elements(fig, analysis_results)
            
            # Configure layout
            self._configure_layout(fig, floor_plan_data)
            
            # Add interactive controls
            self._add_interactive_controls(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise
    
    def _add_floor_plan_elements(self, fig: go.Figure, floor_plan_data: Dict[str, Any]):
        """Add basic floor plan elements to figure"""
        entities = floor_plan_data.get('entities', [])
        
        for entity in entities:
            geometry = entity.get('geometry', {})
            layer = entity.get('layer', 'default')
            
            if geometry.get('type') == 'line':
                self._add_line_element(fig, entity, geometry, layer)
            elif geometry.get('type') in ['polyline', 'polygon']:
                self._add_polyline_element(fig, entity, geometry, layer)
            elif geometry.get('type') == 'circle':
                self._add_circle_element(fig, entity, geometry, layer)
            elif geometry.get('type') == 'text':
                self._add_text_element(fig, entity, geometry, layer)
    
    def _add_line_element(self, fig: go.Figure, entity: Dict[str, Any],
                         geometry: Dict[str, Any], layer: str):
        """Add line element to figure"""
        start = geometry.get('start', {})
        end = geometry.get('end', {})
        
        color = self._get_entity_color(entity)
        line_width = self._get_line_width(entity)
        
        fig.add_trace(go.Scatter(
            x=[start.get('x', 0), end.get('x', 0)],
            y=[start.get('y', 0), end.get('y', 0)],
            mode='lines',
            line=dict(color=color, width=line_width),
            name=f'Line ({layer})',
            legendgroup='lines',
            showlegend=False,
            hovertemplate=f'<b>Line</b><br>Layer: {layer}<br>Length: {geometry.get("length", 0):.2f}m<extra></extra>'
        ))
    
    def _add_polyline_element(self, fig: go.Figure, entity: Dict[str, Any],
                             geometry: Dict[str, Any], layer: str):
        """Add polyline/polygon element to figure"""
        points = geometry.get('points', [])
        
        if not points:
            return
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        color = self._get_entity_color(entity)
        line_width = self._get_line_width(entity)
        
        if geometry.get('type') == 'polygon' and geometry.get('is_closed', False):
            # Add fill for polygons
            fig.add_trace(go.Scatter(
                x=x_coords + [x_coords[0]],  # Close polygon
                y=y_coords + [y_coords[0]],
                mode='lines',
                fill='tonexty' if len(fig.data) > 0 else 'tozeroy',
                fillcolor=f'rgba({color[4:-1]}, 0.3)',  # Semi-transparent fill
                line=dict(color=color, width=line_width),
                name=f'Polygon ({layer})',
                legendgroup='polygons',
                showlegend=False,
                hovertemplate=f'<b>Polygon</b><br>Layer: {layer}<br>Area: {geometry.get("area", 0):.2f}m²<extra></extra>'
            ))
        else:
            # Polyline
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color=color, width=line_width),
                name=f'Polyline ({layer})',
                legendgroup='polylines',
                showlegend=False,
                hovertemplate=f'<b>Polyline</b><br>Layer: {layer}<br>Length: {geometry.get("length", 0):.2f}m<extra></extra>'
            ))
    
    def _add_circle_element(self, fig: go.Figure, entity: Dict[str, Any],
                           geometry: Dict[str, Any], layer: str):
        """Add circle element to figure"""
        center = geometry.get('center', {})
        radius = geometry.get('radius', 0)
        
        if radius <= 0:
            return
        
        # Generate circle points
        theta = np.linspace(0, 2*np.pi, 50)
        x_circle = center.get('x', 0) + radius * np.cos(theta)
        y_circle = center.get('y', 0) + radius * np.sin(theta)
        
        color = self._get_entity_color(entity)
        line_width = self._get_line_width(entity)
        
        fig.add_trace(go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color=color, width=line_width),
            name=f'Circle ({layer})',
            legendgroup='circles',
            showlegend=False,
            hovertemplate=f'<b>Circle</b><br>Layer: {layer}<br>Radius: {radius:.2f}m<br>Area: {geometry.get("area", 0):.2f}m²<extra></extra>'
        ))
    
    def _add_text_element(self, fig: go.Figure, entity: Dict[str, Any],
                         geometry: Dict[str, Any], layer: str):
        """Add text element to figure"""
        position = geometry.get('position', {})
        text = geometry.get('text', '')
        
        if not text:
            return
        
        fig.add_trace(go.Scatter(
            x=[position.get('x', 0)],
            y=[position.get('y', 0)],
            mode='text',
            text=[text],
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            name=f'Text ({layer})',
            legendgroup='text',
            showlegend=False,
            hovertemplate=f'<b>Text</b><br>Layer: {layer}<br>Content: {text}<extra></extra>'
        ))
    
    def _add_analysis_elements(self, fig: go.Figure, analysis_results: Dict[str, Any]):
        """Add analysis results to visualization"""
        # Add walls
        if 'walls' in analysis_results:
            self._add_walls(fig, analysis_results['walls'])
        
        # Add restricted areas
        if 'restricted_areas' in analysis_results:
            self._add_restricted_areas(fig, analysis_results['restricted_areas'])
        
        # Add entrances
        if 'entrances' in analysis_results:
            self._add_entrances(fig, analysis_results['entrances'])
        
        # Add open spaces
        if 'open_spaces' in analysis_results:
            self._add_open_spaces(fig, analysis_results['open_spaces'])
        
        # Add îlots
        if 'ilots' in analysis_results:
            self._add_ilots(fig, analysis_results['ilots'])
        
        # Add corridors
        if 'corridors' in analysis_results:
            self._add_corridors(fig, analysis_results['corridors'])
    
    def _add_walls(self, fig: go.Figure, walls: List[Dict[str, Any]]):
        """Add wall elements to visualization"""
        for wall in walls:
            start = wall.get('start', {})
            end = wall.get('end', {})
            thickness = wall.get('thickness', 0.1)
            
            # Create wall as thick line
            fig.add_trace(go.Scatter(
                x=[start.get('x', 0), end.get('x', 0)],
                y=[start.get('y', 0), end.get('y', 0)],
                mode='lines',
                line=dict(color=self.color_scheme['walls'], width=max(3, thickness * 10)),
                name='Walls',
                legendgroup='walls',
                showlegend=len([t for t in fig.data if t.legendgroup == 'walls']) == 0,
                hovertemplate=f'<b>Wall</b><br>Length: {wall.get("length", 0):.2f}m<br>Thickness: {thickness:.2f}m<extra></extra>'
            ))
    
    def _add_restricted_areas(self, fig: go.Figure, restricted_areas: List[Dict[str, Any]]):
        """Add restricted area elements to visualization"""
        for area in restricted_areas:
            if area.get('type') == 'restricted_polygon':
                points = area.get('points', [])
                if points:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords + [x_coords[0]],
                        y=y_coords + [y_coords[0]],
                        mode='lines',
                        fill='toself',
                        fillcolor=f'rgba(231, 76, 60, 0.3)',
                        line=dict(color=self.color_scheme['restricted_areas'], width=2),
                        name='Restricted Areas',
                        legendgroup='restricted',
                        showlegend=len([t for t in fig.data if t.legendgroup == 'restricted']) == 0,
                        hovertemplate=f'<b>Restricted Area</b><br>Area: {area.get("area", 0):.2f}m²<extra></extra>'
                    ))
            
            elif area.get('type') == 'restricted_circle':
                center = area.get('center', {})
                radius = area.get('radius', 0)
                
                theta = np.linspace(0, 2*np.pi, 50)
                x_circle = center.get('x', 0) + radius * np.cos(theta)
                y_circle = center.get('y', 0) + radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode='lines',
                    fill='toself',
                    fillcolor=f'rgba(231, 76, 60, 0.3)',
                    line=dict(color=self.color_scheme['restricted_areas'], width=2),
                    name='Restricted Areas',
                    legendgroup='restricted',
                    showlegend=len([t for t in fig.data if t.legendgroup == 'restricted']) == 0,
                    hovertemplate=f'<b>Restricted Area</b><br>Radius: {radius:.2f}m<br>Area: {area.get("area", 0):.2f}m²<extra></extra>'
                ))
    
    def _add_entrances(self, fig: go.Figure, entrances: List[Dict[str, Any]]):
        """Add entrance elements to visualization"""
        for entrance in entrances:
            if entrance.get('type') == 'entrance_line':
                start = entrance.get('start', {})
                end = entrance.get('end', {})
                
                fig.add_trace(go.Scatter(
                    x=[start.get('x', 0), end.get('x', 0)],
                    y=[start.get('y', 0), end.get('y', 0)],
                    mode='lines',
                    line=dict(color=self.color_scheme['entrances'], width=4),
                    name='Entrances',
                    legendgroup='entrances',
                    showlegend=len([t for t in fig.data if t.legendgroup == 'entrances']) == 0,
                    hovertemplate=f'<b>Entrance</b><br>Width: {entrance.get("width", 0):.2f}m<extra></extra>'
                ))
            
            elif entrance.get('type') == 'entrance_area':
                center = entrance.get('center', {})
                
                fig.add_trace(go.Scatter(
                    x=[center.get('x', 0)],
                    y=[center.get('y', 0)],
                    mode='markers',
                    marker=dict(
                        color=self.color_scheme['entrances'],
                        size=15,
                        symbol='square'
                    ),
                    name='Entrances',
                    legendgroup='entrances',
                    showlegend=len([t for t in fig.data if t.legendgroup == 'entrances']) == 0,
                    hovertemplate=f'<b>Entrance Area</b><br>Area: {entrance.get("area", 0):.2f}m²<extra></extra>'
                ))
    
    def _add_open_spaces(self, fig: go.Figure, open_spaces: List[Dict[str, Any]]):
        """Add open space elements to visualization"""
        for space in open_spaces:
            if space.get('shapely_geom'):
                geom = space['shapely_geom']
                
                if hasattr(geom, 'exterior'):
                    # Polygon
                    coords = list(geom.exterior.coords)
                    x_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        fill='toself',
                        fillcolor=f'rgba(236, 240, 241, 0.5)',
                        line=dict(color=self.color_scheme['open_spaces'], width=1, dash='dash'),
                        name='Open Spaces',
                        legendgroup='open_spaces',
                        showlegend=len([t for t in fig.data if t.legendgroup == 'open_spaces']) == 0,
                        hovertemplate=f'<b>Open Space</b><br>Area: {space.get("area", 0):.2f}m²<br>Usable: {space.get("usable_area", 0):.2f}m²<extra></extra>'
                    ))
    
    def _add_ilots(self, fig: go.Figure, ilots: List[Dict[str, Any]]):
        """Add îlot elements to visualization"""
        for ilot in ilots:
            geometry = ilot.get('geometry')
            if not geometry:
                continue
            
            size_category = ilot.get('size_category', 'medium')
            color = self.color_scheme['ilots'].get(size_category, self.color_scheme['ilots']['medium'])
            
            # Extract coordinates from geometry
            if hasattr(geometry, 'exterior'):
                coords = list(geometry.exterior.coords)
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    fill='toself',
                    fillcolor=f'rgba{self._hex_to_rgba(color, 0.7)}',
                    line=dict(color=color, width=2),
                    name=f'Îlots ({size_category})',
                    legendgroup=f'ilots_{size_category}',
                    showlegend=len([t for t in fig.data if t.legendgroup == f'ilots_{size_category}']) == 0,
                    hovertemplate=f'<b>Îlot</b><br>ID: {ilot.get("id", "")}<br>Size: {size_category}<br>Area: {ilot.get("area", 0):.2f}m²<extra></extra>'
                ))
    
    def _add_corridors(self, fig: go.Figure, corridors: List[Dict[str, Any]]):
        """Add corridor elements to visualization"""
        for corridor in corridors:
            coordinates = corridor.get('coordinates', [])
            if not coordinates:
                continue
            
            corridor_type = corridor.get('type', 'secondary')
            color = self.color_scheme['corridors'].get(corridor_type, self.color_scheme['corridors']['secondary'])
            width = corridor.get('width', 1.5)
            
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color=color, width=max(2, width * 2)),
                name=f'Corridors ({corridor_type})',
                legendgroup=f'corridors_{corridor_type}',
                showlegend=len([t for t in fig.data if t.legendgroup == f'corridors_{corridor_type}']) == 0,
                hovertemplate=f'<b>Corridor</b><br>Type: {corridor_type}<br>Length: {corridor.get("length", 0):.2f}m<br>Width: {width:.2f}m<extra></extra>'
            ))
    
    def _get_entity_color(self, entity: Dict[str, Any]) -> str:
        """Get color for entity based on its properties"""
        entity_color = entity.get('color', [0, 0, 0])
        
        # Convert RGB to hex
        if isinstance(entity_color, list) and len(entity_color) >= 3:
            r, g, b = entity_color[:3]
            return f'rgb({r}, {g}, {b})'
        
        return 'rgb(0, 0, 0)'
    
    def _get_line_width(self, entity: Dict[str, Any]) -> float:
        """Get line width for entity"""
        lineweight = entity.get('lineweight', 1)
        
        # Convert lineweight to visual width
        if lineweight <= 0:
            return 1
        elif lineweight < 5:
            return 1
        elif lineweight < 10:
            return 2
        else:
            return 3
    
    def _hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        """Convert hex color to RGBA"""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        return f'({r}, {g}, {b}, {alpha})'
    
    def _configure_layout(self, fig: go.Figure, floor_plan_data: Dict[str, Any]):
        """Configure figure layout"""
        bounds = floor_plan_data.get('bounds', {})
        
        # Calculate margins
        margin_percent = 0.1
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 100) - bounds.get('min_y', 0)
        
        margin_x = width * margin_percent
        margin_y = height * margin_percent
        
        fig.update_layout(
            title={
                'text': 'Professional Floor Plan Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis=dict(
                range=[bounds.get('min_x', 0) - margin_x, bounds.get('max_x', 100) + margin_x],
                scaleanchor='y',
                scaleratio=1,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                title='X (meters)'
            ),
            yaxis=dict(
                range=[bounds.get('min_y', 0) - margin_y, bounds.get('max_y', 100) + margin_y],
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                title='Y (meters)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1000,
            height=700,
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02
            ),
            margin=dict(l=50, r=150, t=50, b=50)
        )
    
    def _add_interactive_controls(self, fig: go.Figure):
        """Add interactive controls to figure"""
        # Add zoom and pan controls
        fig.update_layout(
            dragmode='pan',
            selectdirection='diagonal'
        )
        
        # Add custom buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    direction='left',
                    buttons=list([
                        dict(
                            args=[{'visible': [True] * len(fig.data)}],
                            label='Show All',
                            method='restyle'
                        ),
                        dict(
                            args=[{'visible': [False] * len(fig.data)}],
                            label='Hide All',
                            method='restyle'
                        )
                    ]),
                    pad={'r': 10, 't': 10},
                    showactive=True,
                    x=0.01,
                    xanchor='left',
                    y=1.02,
                    yanchor='top'
                )
            ]
        )
    
    def create_analysis_dashboard(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create analysis dashboard with metrics"""
        spatial_metrics = analysis_results.get('spatial_metrics', {})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Area Distribution', 'Element Counts', 'Efficiency Metrics', 'Accessibility Score'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'indicator'}]]
        )
        
        # Area distribution pie chart
        area_ratios = spatial_metrics.get('area_ratios', {})
        fig.add_trace(
            go.Pie(
                labels=list(area_ratios.keys()),
                values=list(area_ratios.values()),
                name='Area Distribution'
            ),
            row=1, col=1
        )
        
        # Element counts bar chart
        counts = spatial_metrics.get('counts', {})
        fig.add_trace(
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                name='Element Counts'
            ),
            row=1, col=2
        )
        
        # Efficiency metrics
        efficiency_data = {
            'Space Utilization': spatial_metrics.get('space_utilization', 0),
            'Circulation Efficiency': spatial_metrics.get('circulation_efficiency', 0),
            'Accessibility Score': spatial_metrics.get('accessibility_score', 0)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(efficiency_data.keys()),
                y=list(efficiency_data.values()),
                name='Efficiency Metrics'
            ),
            row=2, col=1
        )
        
        # Accessibility indicator
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=spatial_metrics.get('accessibility_score', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Accessibility Score'},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': 'darkblue'},
                       'steps': [
                           {'range': [0, 50], 'color': 'lightgray'},
                           {'range': [50, 80], 'color': 'gray'}],
                       'threshold': {'line': {'color': 'red', 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Floor Plan Analysis Dashboard',
            showlegend=False,
            height=600
        )
        
        return fig
