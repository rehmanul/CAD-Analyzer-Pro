
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FloorPlanVisualizer:
    """Advanced visualization for floor plan analysis results"""
    
    def __init__(self):
        self.color_scheme = {
            'walls': 'black',
            'restricted_areas': 'red',
            'entrances': 'green',
            'open_spaces': 'lightblue',
            'ilots': {
                'small': 'orange',
                'medium': 'blue',
                'large': 'purple'
            },
            'corridors': 'yellow'
        }
        
        self.default_layout = {
            'width': 1200,
            'height': 800,
            'margin': dict(l=20, r=20, t=40, b=20),
            'showlegend': True,
            'hovermode': 'closest'
        }
    
    def create_interactive_view(self, floor_plan_data: Dict[str, Any], 
                              analysis_results: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Create interactive floor plan visualization"""
        logger.info("Creating interactive floor plan view")
        
        try:
            fig = go.Figure()
            
            # Add floor plan entities
            self._add_floor_plan_entities(fig, floor_plan_data)
            
            # Add analysis results if available
            if analysis_results:
                self._add_analysis_overlays(fig, analysis_results)
            
            # Configure layout
            self._configure_layout(fig, floor_plan_data)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive view: {str(e)}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
    
    def _add_floor_plan_entities(self, fig: go.Figure, floor_plan_data: Dict[str, Any]):
        """Add basic floor plan entities to the figure"""
        entities = floor_plan_data.get('entities', [])
        
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            geometry = entity.get('geometry', {})
            
            if entity_type == 'line':
                self._add_line_entity(fig, entity, geometry)
            elif entity_type == 'polyline':
                self._add_polyline_entity(fig, entity, geometry)
            elif entity_type == 'circle':
                self._add_circle_entity(fig, entity, geometry)
            elif entity_type == 'rectangle':
                self._add_rectangle_entity(fig, entity, geometry)
            elif entity_type == 'polygon':
                self._add_polygon_entity(fig, entity, geometry)
    
    def _add_line_entity(self, fig: go.Figure, entity: Dict[str, Any], geometry: Dict[str, Any]):
        """Add a line entity to the figure"""
        start = geometry.get('start', {})
        end = geometry.get('end', {})
        
        if start and end:
            fig.add_trace(go.Scatter(
                x=[start.get('x', 0), end.get('x', 0)],
                y=[start.get('y', 0), end.get('y', 0)],
                mode='lines',
                line=dict(
                    color=self._get_entity_color(entity),
                    width=2
                ),
                name=f"Line ({entity.get('layer', 'default')})",
                showlegend=False,
                hovertemplate=f"Layer: {entity.get('layer', 'default')}<br>" +
                             f"Length: {geometry.get('length', 0):.2f}m<extra></extra>"
            ))
    
    def _add_polyline_entity(self, fig: go.Figure, entity: Dict[str, Any], geometry: Dict[str, Any]):
        """Add a polyline entity to the figure"""
        points = geometry.get('points', [])
        
        if points:
            x_coords = [p.get('x', 0) for p in points]
            y_coords = [p.get('y', 0) for p in points]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    color=self._get_entity_color(entity),
                    width=2
                ),
                name=f"Polyline ({entity.get('layer', 'default')})",
                showlegend=False,
                hovertemplate=f"Layer: {entity.get('layer', 'default')}<br>" +
                             f"Points: {len(points)}<extra></extra>"
            ))
    
    def _add_circle_entity(self, fig: go.Figure, entity: Dict[str, Any], geometry: Dict[str, Any]):
        """Add a circle entity to the figure"""
        center = geometry.get('center', {})
        radius = geometry.get('radius', 1)
        
        if center:
            # Create circle using parametric equations
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center.get('x', 0) + radius * np.cos(theta)
            y_circle = center.get('y', 0) + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                line=dict(
                    color=self._get_entity_color(entity),
                    width=2
                ),
                fill='toself',
                fillcolor=f"rgba{tuple(list(self._hex_to_rgb(self._get_entity_color(entity))) + [0.3])}",
                name=f"Circle ({entity.get('layer', 'default')})",
                showlegend=False,
                hovertemplate=f"Layer: {entity.get('layer', 'default')}<br>" +
                             f"Radius: {radius:.2f}m<extra></extra>"
            ))
    
    def _add_rectangle_entity(self, fig: go.Figure, entity: Dict[str, Any], geometry: Dict[str, Any]):
        """Add a rectangle entity to the figure"""
        min_point = geometry.get('min_point', {})
        max_point = geometry.get('max_point', {})
        
        if min_point and max_point:
            x_coords = [
                min_point.get('x', 0), max_point.get('x', 0),
                max_point.get('x', 0), min_point.get('x', 0),
                min_point.get('x', 0)
            ]
            y_coords = [
                min_point.get('y', 0), min_point.get('y', 0),
                max_point.get('y', 0), max_point.get('y', 0),
                min_point.get('y', 0)
            ]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    color=self._get_entity_color(entity),
                    width=2
                ),
                fill='toself',
                fillcolor=f"rgba{tuple(list(self._hex_to_rgb(self._get_entity_color(entity))) + [0.3])}",
                name=f"Rectangle ({entity.get('layer', 'default')})",
                showlegend=False,
                hovertemplate=f"Layer: {entity.get('layer', 'default')}<br>" +
                             f"Area: {geometry.get('area', 0):.2f}m²<extra></extra>"
            ))
    
    def _add_polygon_entity(self, fig: go.Figure, entity: Dict[str, Any], geometry: Dict[str, Any]):
        """Add a polygon entity to the figure"""
        points = geometry.get('points', [])
        
        if points:
            x_coords = [p.get('x', 0) for p in points] + [points[0].get('x', 0)]
            y_coords = [p.get('y', 0) for p in points] + [points[0].get('y', 0)]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    color=self._get_entity_color(entity),
                    width=2
                ),
                fill='toself',
                fillcolor=f"rgba{tuple(list(self._hex_to_rgb(self._get_entity_color(entity))) + [0.3])}",
                name=f"Polygon ({entity.get('layer', 'default')})",
                showlegend=False,
                hovertemplate=f"Layer: {entity.get('layer', 'default')}<br>" +
                             f"Points: {len(points)}<extra></extra>"
            ))
    
    def _add_analysis_overlays(self, fig: go.Figure, analysis_results: Dict[str, Any]):
        """Add analysis result overlays"""
        # Add walls
        if 'walls' in analysis_results:
            self._add_walls_overlay(fig, analysis_results['walls'])
        
        # Add restricted areas
        if 'restricted_areas' in analysis_results:
            self._add_restricted_areas_overlay(fig, analysis_results['restricted_areas'])
        
        # Add entrances
        if 'entrances' in analysis_results:
            self._add_entrances_overlay(fig, analysis_results['entrances'])
        
        # Add îlots
        if 'ilots' in analysis_results:
            self._add_ilots_overlay(fig, analysis_results['ilots'])
        
        # Add corridors
        if 'corridors' in analysis_results:
            self._add_corridors_overlay(fig, analysis_results['corridors'])
    
    def _add_walls_overlay(self, fig: go.Figure, walls: List[Dict[str, Any]]):
        """Add walls overlay"""
        for i, wall in enumerate(walls):
            start = wall.get('start', {})
            end = wall.get('end', {})
            
            if start and end:
                fig.add_trace(go.Scatter(
                    x=[start.get('x', 0), end.get('x', 0)],
                    y=[start.get('y', 0), end.get('y', 0)],
                    mode='lines',
                    line=dict(
                        color=self.color_scheme['walls'],
                        width=4
                    ),
                    name='Walls' if i == 0 else None,
                    legendgroup='walls',
                    showlegend=i == 0,
                    hovertemplate=f"Wall {i}<br>" +
                                 f"Length: {wall.get('length', 0):.2f}m<br>" +
                                 f"Thickness: {wall.get('thickness', 0):.2f}m<extra></extra>"
                ))
    
    def _add_restricted_areas_overlay(self, fig: go.Figure, restricted_areas: List[Dict[str, Any]]):
        """Add restricted areas overlay"""
        for i, area in enumerate(restricted_areas):
            bounds = area.get('bounds', {})
            
            if bounds:
                x_coords = [
                    bounds.get('min_x', 0), bounds.get('max_x', 0),
                    bounds.get('max_x', 0), bounds.get('min_x', 0),
                    bounds.get('min_x', 0)
                ]
                y_coords = [
                    bounds.get('min_y', 0), bounds.get('min_y', 0),
                    bounds.get('max_y', 0), bounds.get('max_y', 0),
                    bounds.get('min_y', 0)
                ]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.color_scheme['restricted_areas'],
                        width=2
                    ),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    name='Restricted Areas' if i == 0 else None,
                    legendgroup='restricted',
                    showlegend=i == 0,
                    hovertemplate=f"Restricted Area {i}<br>" +
                                 f"Area: {area.get('area', 0):.2f}m²<extra></extra>"
                ))
    
    def _add_entrances_overlay(self, fig: go.Figure, entrances: List[Dict[str, Any]]):
        """Add entrances overlay"""
        for i, entrance in enumerate(entrances):
            position = entrance.get('position', {})
            
            if position:
                fig.add_trace(go.Scatter(
                    x=[position.get('x', 0)],
                    y=[position.get('y', 0)],
                    mode='markers',
                    marker=dict(
                        color=self.color_scheme['entrances'],
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Entrances' if i == 0 else None,
                    legendgroup='entrances',
                    showlegend=i == 0,
                    hovertemplate=f"Entrance {i}<br>" +
                                 f"Type: {entrance.get('type', 'Unknown')}<br>" +
                                 f"Width: {entrance.get('width', 0):.2f}m<extra></extra>"
                ))
    
    def _add_ilots_overlay(self, fig: go.Figure, ilots: List[Dict[str, Any]]):
        """Add îlots overlay"""
        size_categories = ['small', 'medium', 'large']
        
        for category in size_categories:
            category_ilots = [i for i in ilots if i.get('size_category') == category]
            
            if category_ilots:
                x_coords = []
                y_coords = []
                hover_texts = []
                
                for i, ilot in enumerate(category_ilots):
                    position = ilot.get('position', {})
                    if position:
                        x_coords.append(position.get('x', 0))
                        y_coords.append(position.get('y', 0))
                        hover_texts.append(
                            f"Îlot {category.title()} {i}<br>" +
                            f"Area: {ilot.get('area', 0):.2f}m²<br>" +
                            f"Rotation: {ilot.get('rotation', 0):.1f}°"
                        )
                
                if x_coords:
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            color=self.color_scheme['ilots'][category],
                            size=8 + (size_categories.index(category) * 4),
                            symbol='square',
                            line=dict(width=1, color='black')
                        ),
                        name=f'Îlots ({category.title()})',
                        legendgroup=f'ilots_{category}',
                        showlegend=True,
                        hovertemplate='%{text}<extra></extra>',
                        text=hover_texts
                    ))
    
    def _add_corridors_overlay(self, fig: go.Figure, corridors: List[Dict[str, Any]]):
        """Add corridors overlay"""
        for i, corridor in enumerate(corridors):
            path = corridor.get('path', [])
            
            if path:
                x_coords = [p.get('x', 0) for p in path]
                y_coords = [p.get('y', 0) for p in path]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=self.color_scheme['corridors'],
                        width=6,
                        dash='dash'
                    ),
                    name='Corridors' if i == 0 else None,
                    legendgroup='corridors',
                    showlegend=i == 0,
                    hovertemplate=f"Corridor {i}<br>" +
                                 f"Type: {corridor.get('type', 'Unknown')}<br>" +
                                 f"Length: {corridor.get('length', 0):.2f}m<br>" +
                                 f"Width: {corridor.get('width', 0):.2f}m<extra></extra>"
                ))
    
    def _configure_layout(self, fig: go.Figure, floor_plan_data: Dict[str, Any]):
        """Configure figure layout"""
        bounds = floor_plan_data.get('bounds', {})
        
        # Calculate aspect ratio
        width = bounds.get('max_x', 100) - bounds.get('min_x', 0)
        height = bounds.get('max_y', 100) - bounds.get('min_y', 0)
        
        fig.update_layout(
            title=dict(
                text="Interactive Floor Plan Analysis",
                font=dict(size=20)
            ),
            xaxis=dict(
                title="X Coordinate (m)",
                scaleanchor="y",
                scaleratio=1,
                range=[bounds.get('min_x', 0) - 5, bounds.get('max_x', 100) + 5]
            ),
            yaxis=dict(
                title="Y Coordinate (m)",
                range=[bounds.get('min_y', 0) - 5, bounds.get('max_y', 100) + 5]
            ),
            **self.default_layout
        )
    
    def _get_entity_color(self, entity: Dict[str, Any]) -> str:
        """Get color for an entity based on its layer or type"""
        layer = entity.get('layer', '').lower()
        
        # Map common layer names to colors
        if 'wall' in layer:
            return 'black'
        elif 'door' in layer or 'entrance' in layer:
            return 'green'
        elif 'window' in layer:
            return 'blue'
        elif 'furniture' in layer:
            return 'brown'
        else:
            # Use entity color if available
            color = entity.get('color', [0, 0, 0])
            if isinstance(color, list) and len(color) >= 3:
                return f"rgb({color[0]}, {color[1]}, {color[2]})"
            return 'gray'
    
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (128, 128, 128)  # Default gray
    
    def create_heatmap_visualization(self, analysis_results: Dict[str, Any], 
                                   metric: str = 'accessibility') -> go.Figure:
        """Create heatmap visualization for specific metrics"""
        logger.info(f"Creating heatmap visualization for {metric}")
        
        fig = go.Figure()
        
        # Generate sample heatmap data
        bounds = analysis_results.get('metadata', {}).get('analysis_bounds', {})
        x_min, x_max = bounds.get('min_x', 0), bounds.get('max_x', 100)
        y_min, y_max = bounds.get('min_y', 0), bounds.get('max_y', 100)
        
        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)
        
        # Generate heatmap based on metric
        if metric == 'accessibility':
            Z = self._calculate_accessibility_heatmap(X, Y, analysis_results)
        elif metric == 'density':
            Z = self._calculate_density_heatmap(X, Y, analysis_results)
        elif metric == 'flow':
            Z = self._calculate_flow_heatmap(X, Y, analysis_results)
        else:
            Z = np.random.rand(50, 50) * 100
        
        fig.add_trace(go.Heatmap(
            x=x,
            y=y,
            z=Z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f"{metric.title()} Score")
        ))
        
        fig.update_layout(
            title=f"{metric.title()} Heatmap",
            xaxis_title="X Coordinate (m)",
            yaxis_title="Y Coordinate (m)",
            **self.default_layout
        )
        
        return fig
    
    def _calculate_accessibility_heatmap(self, X: np.ndarray, Y: np.ndarray, 
                                       analysis_results: Dict[str, Any]) -> np.ndarray:
        """Calculate accessibility heatmap"""
        # Simple accessibility calculation based on distance to entrances
        entrances = analysis_results.get('entrances', [])
        
        if not entrances:
            return np.ones_like(X) * 50
        
        Z = np.zeros_like(X)
        
        for entrance in entrances:
            pos = entrance.get('position', {})
            ex, ey = pos.get('x', 0), pos.get('y', 0)
            
            # Calculate distance from each point to entrance
            dist = np.sqrt((X - ex)**2 + (Y - ey)**2)
            # Convert to accessibility score (closer = higher score)
            accessibility = 100 * np.exp(-dist / 20)
            Z = np.maximum(Z, accessibility)
        
        return Z
    
    def _calculate_density_heatmap(self, X: np.ndarray, Y: np.ndarray, 
                                 analysis_results: Dict[str, Any]) -> np.ndarray:
        """Calculate density heatmap"""
        # Simple density calculation based on îlot placement
        ilots = analysis_results.get('ilots', [])
        
        Z = np.zeros_like(X)
        
        for ilot in ilots:
            pos = ilot.get('position', {})
            ix, iy = pos.get('x', 0), pos.get('y', 0)
            
            # Add density contribution from each îlot
            dist = np.sqrt((X - ix)**2 + (Y - iy)**2)
            density = 100 * np.exp(-dist / 10)
            Z += density
        
        return np.clip(Z, 0, 100)
    
    def _calculate_flow_heatmap(self, X: np.ndarray, Y: np.ndarray, 
                              analysis_results: Dict[str, Any]) -> np.ndarray:
        """Calculate flow heatmap"""
        # Simple flow calculation based on corridor network
        corridors = analysis_results.get('corridors', [])
        
        Z = np.zeros_like(X)
        
        for corridor in corridors:
            path = corridor.get('path', [])
            
            for point in path:
                px, py = point.get('x', 0), point.get('y', 0)
                
                # Add flow contribution from each corridor point
                dist = np.sqrt((X - px)**2 + (Y - py)**2)
                flow = 100 * np.exp(-dist / 5)
                Z += flow
        
        return np.clip(Z, 0, 100)
