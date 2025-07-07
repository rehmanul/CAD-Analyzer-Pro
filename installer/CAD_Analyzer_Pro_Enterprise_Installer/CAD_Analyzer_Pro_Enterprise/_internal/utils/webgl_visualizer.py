import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from shapely.geometry import Point, Polygon, LineString
import colorsys

logger = logging.getLogger(__name__)

class WebGLVisualizer:
    """Advanced 3D visualization engine using WebGL for complex building analysis"""
    
    def __init__(self):
        self.floor_height = 3.0  # Standard floor height in meters
        self.wall_height = 2.7  # Wall height
        self.ilot_heights = {
            'small': 0.9,
            'medium': 1.2,
            'large': 1.5
        }
        
        self.color_scheme = {
            'walls': 'rgb(60, 60, 60)',
            'restricted': 'rgba(100, 149, 237, 0.6)',  # Cornflower blue
            'entrances': 'rgba(255, 69, 69, 0.8)',     # Red
            'ilots_small': 'rgb(144, 238, 144)',       # Light green
            'ilots_medium': 'rgb(100, 200, 100)',      # Medium green
            'ilots_large': 'rgb(60, 150, 60)',         # Dark green
            'corridors': 'rgba(255, 215, 0, 0.4)',     # Gold
            'floor': 'rgb(245, 245, 245)',
            'grid': 'rgba(200, 200, 200, 0.2)'
        }
        
        self.material_properties = {
            'metallic': 0.3,
            'roughness': 0.7,
            'ambient_occlusion': 0.8
        }
    
    def create_3d_visualization(self, analysis_results: Dict[str, Any],
                               ilot_results: List[Dict[str, Any]],
                               corridor_results: List[Dict[str, Any]],
                               multi_floor: bool = False,
                               num_floors: int = 1) -> go.Figure:
        """
        Create comprehensive 3D visualization with WebGL rendering
        
        Args:
            analysis_results: Zone analysis results
            ilot_results: Placed îlots
            corridor_results: Generated corridors
            multi_floor: Enable multi-floor visualization
            num_floors: Number of floors to visualize
            
        Returns:
            Plotly figure with 3D visualization
        """
        logger.info("Creating 3D WebGL visualization")
        
        try:
            # Create figure with WebGL renderer
            fig = go.Figure()
            
            # Add floor(s)
            for floor_idx in range(num_floors):
                z_offset = floor_idx * self.floor_height
                
                # Add floor plane
                self._add_floor_plane(fig, analysis_results, z_offset)
                
                # Add walls with realistic height
                self._add_3d_walls(fig, analysis_results.get('walls', []), 
                                  z_offset, floor_idx)
                
                # Add restricted areas as elevated zones
                self._add_3d_restricted_areas(fig, 
                                            analysis_results.get('restricted_areas', []),
                                            z_offset, floor_idx)
                
                # Add entrances as highlighted zones
                self._add_3d_entrances(fig, analysis_results.get('entrances', []),
                                     z_offset, floor_idx)
                
                # Add îlots as 3D objects
                self._add_3d_ilots(fig, ilot_results, z_offset, floor_idx)
                
                # Add corridors as textured paths
                self._add_3d_corridors(fig, corridor_results, z_offset, floor_idx)
                
                # Add grid reference
                if floor_idx == 0:
                    self._add_grid_reference(fig, analysis_results)
            
            # Configure 3D scene with WebGL optimizations
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="X (meters)",
                        backgroundcolor="white",
                        gridcolor="lightgray",
                        showbackground=True,
                        zerolinecolor="lightgray"
                    ),
                    yaxis=dict(
                        title="Y (meters)",
                        backgroundcolor="white",
                        gridcolor="lightgray",
                        showbackground=True,
                        zerolinecolor="lightgray"
                    ),
                    zaxis=dict(
                        title="Z (meters)",
                        backgroundcolor="white",
                        gridcolor="lightgray",
                        showbackground=True,
                        zerolinecolor="lightgray",
                        range=[0, num_floors * self.floor_height + 1]
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.3),
                    bgcolor='rgb(250, 250, 250)'
                ),
                title=dict(
                    text="3D Floor Plan Analysis - WebGL Rendering",
                    font=dict(size=20, family="Arial, sans-serif")
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="black",
                    borderwidth=1
                ),
                width=1400,
                height=900,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Add lighting effects
            self._configure_lighting(fig)
            
            # Enable WebGL renderer
            fig.update_traces(lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.5,
                roughness=0.7,
                fresnel=0.2
            ))
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {str(e)}")
            raise
    
    def _add_floor_plane(self, fig: go.Figure, analysis_results: Dict[str, Any],
                        z_offset: float):
        """Add floor plane with texture"""
        bounds = analysis_results.get('metadata', {}).get('analysis_bounds', {})
        
        if not bounds:
            return
        
        # Create floor mesh
        x_range = np.linspace(bounds.get('min_x', 0), bounds.get('max_x', 100), 50)
        y_range = np.linspace(bounds.get('min_y', 0), bounds.get('max_y', 100), 50)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = np.full_like(x_mesh, z_offset)
        
        # Add floor surface
        fig.add_trace(go.Surface(
            x=x_mesh,
            y=y_mesh,
            z=z_mesh,
            colorscale=[[0, self.color_scheme['floor']], 
                       [1, self.color_scheme['floor']]],
            showscale=False,
            opacity=1.0,
            name=f"Floor Level {int(z_offset/self.floor_height)}",
            hovertemplate="Floor<br>X: %{x:.1f}m<br>Y: %{y:.1f}m<extra></extra>"
        ))
    
    def _add_3d_walls(self, fig: go.Figure, walls: List[Dict[str, Any]],
                     z_offset: float, floor_idx: int):
        """Add 3D walls with realistic height and thickness"""
        for wall in walls:
            start = wall.get('start', {})
            end = wall.get('end', {})
            thickness = wall.get('thickness', 0.2)
            
            # Calculate wall direction
            dx = end.get('x', 0) - start.get('x', 0)
            dy = end.get('y', 0) - start.get('y', 0)
            length = np.sqrt(dx**2 + dy**2)
            
            if length == 0:
                continue
            
            # Perpendicular direction for thickness
            perp_dx = -dy / length * thickness / 2
            perp_dy = dx / length * thickness / 2
            
            # Create wall vertices
            vertices = np.array([
                [start['x'] - perp_dx, start['y'] - perp_dy, z_offset],
                [start['x'] + perp_dx, start['y'] + perp_dy, z_offset],
                [end['x'] + perp_dx, end['y'] + perp_dy, z_offset],
                [end['x'] - perp_dx, end['y'] - perp_dy, z_offset],
                [start['x'] - perp_dx, start['y'] - perp_dy, z_offset + self.wall_height],
                [start['x'] + perp_dx, start['y'] + perp_dy, z_offset + self.wall_height],
                [end['x'] + perp_dx, end['y'] + perp_dy, z_offset + self.wall_height],
                [end['x'] - perp_dx, end['y'] - perp_dy, z_offset + self.wall_height]
            ])
            
            # Define faces
            faces = [
                [0, 1, 5, 4],  # Front
                [2, 3, 7, 6],  # Back
                [0, 3, 7, 4],  # Left
                [1, 2, 6, 5],  # Right
                [4, 5, 6, 7],  # Top
                [0, 1, 2, 3]   # Bottom
            ]
            
            # Add each face
            for face in faces:
                face_vertices = vertices[face]
                fig.add_trace(go.Mesh3d(
                    x=face_vertices[:, 0],
                    y=face_vertices[:, 1],
                    z=face_vertices[:, 2],
                    i=[0, 0, 0, 0],
                    j=[1, 1, 2, 2],
                    k=[2, 3, 3, 0],
                    color=self.color_scheme['walls'],
                    opacity=1.0,
                    name='Wall' if floor_idx == 0 else None,
                    showlegend=floor_idx == 0 and wall == walls[0],
                    hovertemplate="Wall<br>Height: %.1fm<extra></extra>" % self.wall_height
                ))
    
    def _add_3d_restricted_areas(self, fig: go.Figure, 
                                restricted_areas: List[Dict[str, Any]],
                                z_offset: float, floor_idx: int):
        """Add 3D restricted areas with transparency"""
        for area in restricted_areas:
            if area['type'] == 'restricted_polygon':
                points = area.get('points', [])
                if len(points) < 3:
                    continue
                
                # Create elevated platform
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                # Add base
                fig.add_trace(go.Scatter3d(
                    x=x_coords + [x_coords[0]],
                    y=y_coords + [y_coords[0]],
                    z=[z_offset + 0.1] * (len(x_coords) + 1),
                    mode='lines',
                    line=dict(color=self.color_scheme['restricted'], width=4),
                    fill='toself',
                    surfaceaxis=2,
                    surfacecolor=self.color_scheme['restricted'],
                    name='Restricted Area' if floor_idx == 0 else None,
                    showlegend=floor_idx == 0 and area == restricted_areas[0],
                    hovertemplate="Restricted Area<br>Type: %s<extra></extra>" % area.get('type')
                ))
                
                # Add elevated surface
                height = 0.5  # Restricted area elevation
                for i in range(len(points)):
                    next_i = (i + 1) % len(points)
                    
                    # Vertical wall
                    fig.add_trace(go.Mesh3d(
                        x=[points[i][0], points[next_i][0], points[next_i][0], points[i][0]],
                        y=[points[i][1], points[next_i][1], points[next_i][1], points[i][1]],
                        z=[z_offset + 0.1, z_offset + 0.1, z_offset + height, z_offset + height],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color=self.color_scheme['restricted'],
                        opacity=0.6,
                        showlegend=False
                    ))
    
    def _add_3d_entrances(self, fig: go.Figure, entrances: List[Dict[str, Any]],
                         z_offset: float, floor_idx: int):
        """Add 3D entrance markers"""
        for entrance in entrances:
            if entrance['type'] == 'entrance_line':
                start = entrance.get('start', {})
                end = entrance.get('end', {})
                
                # Create entrance arch
                x_coords = [start['x'], end['x']]
                y_coords = [start['y'], end['y']]
                
                # Add entrance marker
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=[z_offset + self.wall_height * 0.8] * 2,
                    mode='lines+markers',
                    line=dict(color=self.color_scheme['entrances'], width=10),
                    marker=dict(size=8, color=self.color_scheme['entrances']),
                    name='Entrance/Exit' if floor_idx == 0 else None,
                    showlegend=floor_idx == 0 and entrance == entrances[0],
                    hovertemplate="Entrance<br>Width: %.1fm<extra></extra>" % entrance.get('width', 1.0)
                ))
                
                # Add door frame visualization
                self._add_door_frame(fig, start, end, z_offset)
    
    def _add_3d_ilots(self, fig: go.Figure, ilots: List[Dict[str, Any]],
                     z_offset: float, floor_idx: int):
        """Add 3D îlots with varying heights based on size"""
        ilot_groups = {'small': [], 'medium': [], 'large': []}
        
        for ilot in ilots:
            size_category = ilot.get('size_category', 'medium')
            position = ilot.get('position', {})
            dimensions = ilot.get('dimensions', {})
            
            width = dimensions.get('width', 2.0)
            height = dimensions.get('height', 2.0)
            ilot_height = self.ilot_heights.get(size_category, 1.2)
            
            # Create îlot box
            x_min = position.get('x', 0) - width / 2
            x_max = position.get('x', 0) + width / 2
            y_min = position.get('y', 0) - height / 2
            y_max = position.get('y', 0) + height / 2
            
            # Create vertices for the box
            vertices = np.array([
                [x_min, y_min, z_offset],
                [x_max, y_min, z_offset],
                [x_max, y_max, z_offset],
                [x_min, y_max, z_offset],
                [x_min, y_min, z_offset + ilot_height],
                [x_max, y_min, z_offset + ilot_height],
                [x_max, y_max, z_offset + ilot_height],
                [x_min, y_max, z_offset + ilot_height]
            ])
            
            # Add îlot as mesh
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=[0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3],
                j=[1, 2, 3, 7, 5, 6, 7, 3, 4, 5, 6, 7],
                k=[2, 3, 7, 4, 6, 7, 3, 0, 5, 6, 7, 4],
                color=self.color_scheme[f'ilots_{size_category}'],
                opacity=0.9,
                name=f'Îlot {size_category.capitalize()}' if floor_idx == 0 and ilot == next((i for i in ilots if i.get('size_category') == size_category), None) else None,
                showlegend=floor_idx == 0 and ilot == next((i for i in ilots if i.get('size_category') == size_category), None),
                hovertemplate=f"Îlot {ilot.get('id', '')}<br>Size: {size_category}<br>Area: {dimensions.get('area', 0):.1f}m²<extra></extra>"
            ))
            
            # Add top surface detail
            self._add_ilot_top_detail(fig, vertices[4:], size_category)
    
    def _add_3d_corridors(self, fig: go.Figure, corridors: List[Dict[str, Any]],
                         z_offset: float, floor_idx: int):
        """Add 3D corridors with visual indicators"""
        for corridor in corridors:
            path = corridor.get('path', [])
            width = corridor.get('width', 1.5)
            corridor_type = corridor.get('type', 'secondary')
            
            if len(path) < 2:
                continue
            
            # Create corridor surface
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                
                # Calculate perpendicular direction
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx**2 + dy**2)
                
                if length == 0:
                    continue
                
                perp_dx = -dy / length * width / 2
                perp_dy = dx / length * width / 2
                
                # Create corridor segment
                x_coords = [
                    start[0] - perp_dx,
                    start[0] + perp_dx,
                    end[0] + perp_dx,
                    end[0] - perp_dx
                ]
                y_coords = [
                    start[1] - perp_dy,
                    start[1] + perp_dy,
                    end[1] + perp_dy,
                    end[1] - perp_dy
                ]
                
                # Add corridor surface
                fig.add_trace(go.Scatter3d(
                    x=x_coords + [x_coords[0]],
                    y=y_coords + [y_coords[0]],
                    z=[z_offset + 0.05] * 5,
                    mode='lines',
                    line=dict(color=self.color_scheme['corridors'], width=2),
                    fill='toself',
                    surfaceaxis=2,
                    surfacecolor=self.color_scheme['corridors'],
                    name=f'Corridor ({corridor_type})' if floor_idx == 0 and corridor == corridors[0] else None,
                    showlegend=floor_idx == 0 and corridor == corridors[0],
                    hovertemplate=f"Corridor<br>Type: {corridor_type}<br>Width: {width:.1f}m<extra></extra>"
                ))
                
                # Add directional arrows
                if i % 3 == 0:  # Add arrows every 3 segments
                    self._add_corridor_arrow(fig, start, end, z_offset + 0.06)
    
    def _add_door_frame(self, fig: go.Figure, start: Dict[str, float],
                       end: Dict[str, float], z_offset: float):
        """Add door frame visualization"""
        # Create door frame outline
        frame_height = self.wall_height * 0.9
        
        # Left post
        fig.add_trace(go.Scatter3d(
            x=[start['x'], start['x']],
            y=[start['y'], start['y']],
            z=[z_offset, z_offset + frame_height],
            mode='lines',
            line=dict(color=self.color_scheme['entrances'], width=6),
            showlegend=False
        ))
        
        # Right post
        fig.add_trace(go.Scatter3d(
            x=[end['x'], end['x']],
            y=[end['y'], end['y']],
            z=[z_offset, z_offset + frame_height],
            mode='lines',
            line=dict(color=self.color_scheme['entrances'], width=6),
            showlegend=False
        ))
        
        # Top beam
        fig.add_trace(go.Scatter3d(
            x=[start['x'], end['x']],
            y=[start['y'], end['y']],
            z=[z_offset + frame_height, z_offset + frame_height],
            mode='lines',
            line=dict(color=self.color_scheme['entrances'], width=6),
            showlegend=False
        ))
    
    def _add_ilot_top_detail(self, fig: go.Figure, top_vertices: np.ndarray,
                            size_category: str):
        """Add detail to îlot top surface"""
        # Add workspace indicator based on size
        center_x = np.mean(top_vertices[:, 0])
        center_y = np.mean(top_vertices[:, 1])
        center_z = np.mean(top_vertices[:, 2])
        
        # Add workspace marker
        marker_size = {'small': 15, 'medium': 20, 'large': 25}
        fig.add_trace(go.Scatter3d(
            x=[center_x],
            y=[center_y],
            z=[center_z + 0.1],
            mode='markers',
            marker=dict(
                size=marker_size.get(size_category, 20),
                color='white',
                symbol='square',
                line=dict(color='gray', width=2)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def _add_corridor_arrow(self, fig: go.Figure, start: List[float],
                          end: List[float], z: float):
        """Add directional arrow in corridor"""
        # Calculate arrow position and direction
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Create arrow points
        arrow_length = 0.5
        arrow_width = 0.2
        
        fig.add_trace(go.Cone(
            x=[mid_x],
            y=[mid_y],
            z=[z],
            u=[dx * arrow_length],
            v=[dy * arrow_length],
            w=[0],
            sizemode='absolute',
            sizeref=arrow_width,
            showscale=False,
            colorscale=[[0, 'white'], [1, 'white']],
            opacity=0.8,
            hoverinfo='skip'
        ))
    
    def _add_grid_reference(self, fig: go.Figure, analysis_results: Dict[str, Any]):
        """Add reference grid for measurements"""
        bounds = analysis_results.get('metadata', {}).get('analysis_bounds', {})
        
        if not bounds:
            return
        
        # Create grid lines
        grid_spacing = 5.0  # 5 meter grid
        
        x_lines = np.arange(bounds.get('min_x', 0), bounds.get('max_x', 100), grid_spacing)
        y_lines = np.arange(bounds.get('min_y', 0), bounds.get('max_y', 100), grid_spacing)
        
        # Add vertical grid lines
        for x in x_lines:
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[bounds.get('min_y', 0), bounds.get('max_y', 100)],
                z=[0.01, 0.01],
                mode='lines',
                line=dict(color=self.color_scheme['grid'], width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add horizontal grid lines
        for y in y_lines:
            fig.add_trace(go.Scatter3d(
                x=[bounds.get('min_x', 0), bounds.get('max_x', 100)],
                y=[y, y],
                z=[0.01, 0.01],
                mode='lines',
                line=dict(color=self.color_scheme['grid'], width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _configure_lighting(self, fig: go.Figure):
        """Configure advanced lighting for the scene"""
        # This would typically involve WebGL shader configuration
        # Plotly handles this internally with the lighting parameter
        pass
    
    def create_heatmap_visualization(self, analysis_results: Dict[str, Any],
                                   metric: str = 'accessibility') -> go.Figure:
        """Create 3D heatmap visualization for various metrics"""
        logger.info(f"Creating 3D heatmap for {metric}")
        
        bounds = analysis_results.get('metadata', {}).get('analysis_bounds', {})
        
        # Generate heatmap data
        resolution = 50
        x_range = np.linspace(bounds.get('min_x', 0), bounds.get('max_x', 100), resolution)
        y_range = np.linspace(bounds.get('min_y', 0), bounds.get('max_y', 100), resolution)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        
        # Calculate metric values
        z_mesh = self._calculate_heatmap_values(x_mesh, y_mesh, analysis_results, metric)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=x_mesh,
            y=y_mesh,
            z=z_mesh,
            colorscale='Viridis',
            name=f'{metric.capitalize()} Heatmap',
            hovertemplate=f"{metric.capitalize()}: %{{z:.2f}}<br>X: %{{x:.1f}}m<br>Y: %{{y:.1f}}m<extra></extra>"
        )])
        
        fig.update_layout(
            title=f"3D {metric.capitalize()} Analysis",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title=metric.capitalize(),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1200,
            height=800
        )
        
        return fig
    
    def _calculate_heatmap_values(self, x_mesh: np.ndarray, y_mesh: np.ndarray,
                                 analysis_results: Dict[str, Any], metric: str) -> np.ndarray:
        """Calculate heatmap values for given metric"""
        z_mesh = np.zeros_like(x_mesh)
        
        if metric == 'accessibility':
            # Calculate distance to entrances
            entrances = analysis_results.get('entrances', [])
            for entrance in entrances:
                pos = entrance.get('position', {})
                if pos:
                    distances = np.sqrt((x_mesh - pos.get('x', 0))**2 + 
                                      (y_mesh - pos.get('y', 0))**2)
                    z_mesh += 100 / (1 + distances)
        
        elif metric == 'density':
            # Calculate îlot density
            # This would use actual îlot positions
            z_mesh = np.random.rand(*x_mesh.shape) * 50 + 25
        
        elif metric == 'flow':
            # Calculate traffic flow potential
            # This would use corridor network analysis
            z_mesh = np.sin(x_mesh/10) * np.cos(y_mesh/10) * 30 + 50
        
        return z_mesh
    
    def export_webgl_scene(self, fig: go.Figure, filename: str = "floor_plan_3d.html"):
        """Export 3D scene as standalone WebGL HTML file"""
        logger.info(f"Exporting WebGL scene to {filename}")
        
        # Add WebGL-specific optimizations
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'floor_plan_3d',
                'height': 1920,
                'width': 1080,
                'scale': 2
            },
            'displaylogo': False,
            'modeBarButtonsToAdd': ['toggleSpikelines']
        }
        
        # Export to HTML with WebGL renderer
        fig.write_html(filename, config=config, include_plotlyjs='cdn')
        
        return filename