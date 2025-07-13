"""
Advanced 3D Real-Time CAD Floor Plan Renderer
Professional 3D rendering system with real-time interaction and high-fidelity visuals
Based on the implementation guide for advanced 3D CAD visualization
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import base64
import math
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

class Advanced3DRenderer:
    """
    Professional 3D renderer for CAD floor plans with advanced features:
    - Real-time 3D visualization with Three.js integration
    - Interactive object manipulation
    - Advanced lighting and materials
    - Professional architectural rendering
    """
    
    def __init__(self):
        self.wall_height = 2.8  # Standard wall height in meters
        self.door_height = 2.1  # Standard door height
        self.window_height = 1.2  # Standard window height
        self.floor_thickness = 0.2  # Floor thickness
        
        # Material properties
        self.materials = {
            'wall': {'color': '#CCCCCC', 'opacity': 0.9, 'metallic': 0.1, 'roughness': 0.8},
            'floor': {'color': '#D4C4A8', 'opacity': 1.0, 'metallic': 0.0, 'roughness': 0.6},
            'door': {'color': '#8B4513', 'opacity': 1.0, 'metallic': 0.0, 'roughness': 0.4},
            'window': {'color': '#87CEEB', 'opacity': 0.3, 'metallic': 0.9, 'roughness': 0.1},
            'furniture': {'color': '#8B4513', 'opacity': 1.0, 'metallic': 0.0, 'roughness': 0.5}
        }
        
        # Lighting configuration
        self.lighting = {
            'ambient': {'intensity': 0.4, 'color': '#FFFFFF'},
            'directional': {'intensity': 0.8, 'color': '#FFFFFF', 'position': [10, 10, 10]},
            'point_lights': [
                {'intensity': 0.6, 'color': '#FFFFFF', 'position': [5, 5, 3]},
                {'intensity': 0.4, 'color': '#FFFFFF', 'position': [-5, -5, 3]}
            ]
        }
    
    def create_advanced_3d_visualization(self, analysis_data: Dict, ilots: List[Dict], 
                                       corridors: List[Dict], show_wireframe: bool = False,
                                       enable_shadows: bool = True) -> go.Figure:
        """Create advanced 3D visualization with professional rendering"""
        
        fig = go.Figure()
        
        # Add floor
        self._add_3d_floor(fig, analysis_data)
        
        # Add walls with proper 3D geometry
        self._add_3d_walls(fig, analysis_data)
        
        # Add doors and windows
        self._add_3d_openings(fig, analysis_data)
        
        # Add furniture (îlots) with 3D models
        self._add_3d_furniture(fig, ilots)
        
        # Add corridors as 3D paths
        self._add_3d_corridors(fig, corridors)
        
        # Configure advanced 3D scene
        self._configure_3d_scene(fig, analysis_data, enable_shadows)
        
        # Add interactive controls
        self._add_interactive_controls(fig)
        
        return fig
    
    def _add_3d_floor(self, fig: go.Figure, analysis_data: Dict):
        """Add 3D floor with proper geometry and materials"""
        bounds = analysis_data.get('bounds', {})
        
        # Create floor mesh
        x_min, x_max = bounds.get('x_min', 0), bounds.get('x_max', 100)
        y_min, y_max = bounds.get('y_min', 0), bounds.get('y_max', 100)
        
        # Floor vertices
        x_floor = [x_min, x_max, x_max, x_min, x_min]
        y_floor = [y_min, y_min, y_max, y_max, y_min]
        z_floor = [0, 0, 0, 0, 0]
        
        # Add floor surface
        fig.add_trace(go.Mesh3d(
            x=x_floor,
            y=y_floor,
            z=z_floor,
            color=self.materials['floor']['color'],
            opacity=self.materials['floor']['opacity'],
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.2,
                roughness=self.materials['floor']['roughness']
            ),
            name='Floor',
            showlegend=False
        ))
    
    def _add_3d_walls(self, fig: go.Figure, analysis_data: Dict):
        """Add 3D walls with proper extrusion and materials"""
        walls = analysis_data.get('walls', [])
        
        for i, wall in enumerate(walls):
            coords = self._extract_wall_coordinates(wall)
            if coords and len(coords) >= 2:
                self._create_3d_wall_mesh(fig, coords, f'Wall_{i}')
    
    def _create_3d_wall_mesh(self, fig: go.Figure, coords: List[List[float]], name: str):
        """Create 3D wall mesh from 2D coordinates"""
        if len(coords) < 2:
            return
        
        # Create wall vertices by extruding 2D coordinates
        vertices_x, vertices_y, vertices_z = [], [], []
        faces_i, faces_j, faces_k = [], [], []
        
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            
            # Wall thickness (assume 0.2m)
            thickness = 0.1
            
            # Calculate perpendicular vector for thickness
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                norm_x = -dy / length * thickness
                norm_y = dx / length * thickness
                
                # Create wall vertices (8 vertices per wall segment)
                base_idx = len(vertices_x)
                
                # Bottom vertices
                vertices_x.extend([x1 + norm_x, x1 - norm_x, x2 - norm_x, x2 + norm_x])
                vertices_y.extend([y1 + norm_y, y1 - norm_y, y2 - norm_y, y2 + norm_y])
                vertices_z.extend([0, 0, 0, 0])
                
                # Top vertices
                vertices_x.extend([x1 + norm_x, x1 - norm_x, x2 - norm_x, x2 + norm_x])
                vertices_y.extend([y1 + norm_y, y1 - norm_y, y2 - norm_y, y2 + norm_y])
                vertices_z.extend([self.wall_height, self.wall_height, self.wall_height, self.wall_height])
                
                # Create faces (triangles)
                # Bottom face
                faces_i.extend([base_idx, base_idx + 1, base_idx + 2])
                faces_j.extend([base_idx + 1, base_idx + 2, base_idx + 3])
                faces_k.extend([base_idx + 2, base_idx + 3, base_idx])
                
                # Top face
                faces_i.extend([base_idx + 4, base_idx + 6, base_idx + 5])
                faces_j.extend([base_idx + 5, base_idx + 7, base_idx + 6])
                faces_k.extend([base_idx + 6, base_idx + 4, base_idx + 7])
                
                # Side faces
                for j in range(4):
                    next_j = (j + 1) % 4
                    # Connect bottom to top
                    faces_i.extend([base_idx + j, base_idx + next_j, base_idx + j + 4])
                    faces_j.extend([base_idx + next_j, base_idx + next_j + 4, base_idx + j + 4])
                    faces_k.extend([base_idx + j + 4, base_idx + j, base_idx + next_j + 4])
        
        if vertices_x:
            fig.add_trace(go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=faces_i,
                j=faces_j,
                k=faces_k,
                color=self.materials['wall']['color'],
                opacity=self.materials['wall']['opacity'],
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.8,
                    specular=0.4,
                    roughness=self.materials['wall']['roughness']
                ),
                name=name,
                showlegend=False
            ))
    
    def _add_3d_openings(self, fig: go.Figure, analysis_data: Dict):
        """Add doors and windows as 3D objects"""
        doors = analysis_data.get('doors', [])
        windows = analysis_data.get('windows', [])
        
        # Add doors
        for i, door in enumerate(doors):
            self._create_3d_door(fig, door, f'Door_{i}')
        
        # Add windows
        for i, window in enumerate(windows):
            self._create_3d_window(fig, window, f'Window_{i}')
    
    def _create_3d_door(self, fig: go.Figure, door: Dict, name: str):
        """Create 3D door object"""
        # Extract door position and dimensions
        x = door.get('x', 0)
        y = door.get('y', 0)
        width = door.get('width', 0.8)
        thickness = 0.05
        
        # Create door frame
        door_x = [x - width/2, x + width/2, x + width/2, x - width/2, x - width/2]
        door_y = [y - thickness/2, y - thickness/2, y + thickness/2, y + thickness/2, y - thickness/2]
        door_z_bottom = [0, 0, 0, 0, 0]
        door_z_top = [self.door_height, self.door_height, self.door_height, self.door_height, self.door_height]
        
        # Add door as mesh
        fig.add_trace(go.Mesh3d(
            x=door_x + door_x,
            y=door_y + door_y,
            z=door_z_bottom + door_z_top,
            color=self.materials['door']['color'],
            opacity=self.materials['door']['opacity'],
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.3,
                roughness=self.materials['door']['roughness']
            ),
            name=name,
            showlegend=False
        ))
    
    def _create_3d_window(self, fig: go.Figure, window: Dict, name: str):
        """Create 3D window object"""
        # Extract window position and dimensions
        x = window.get('x', 0)
        y = window.get('y', 0)
        width = window.get('width', 1.2)
        height = window.get('height', self.window_height)
        thickness = 0.02
        
        # Create window frame
        window_x = [x - width/2, x + width/2, x + width/2, x - width/2, x - width/2]
        window_y = [y - thickness/2, y - thickness/2, y + thickness/2, y + thickness/2, y - thickness/2]
        window_z_bottom = [0.8, 0.8, 0.8, 0.8, 0.8]  # Windows start higher
        window_z_top = [0.8 + height, 0.8 + height, 0.8 + height, 0.8 + height, 0.8 + height]
        
        # Add window as mesh
        fig.add_trace(go.Mesh3d(
            x=window_x + window_x,
            y=window_y + window_y,
            z=window_z_bottom + window_z_top,
            color=self.materials['window']['color'],
            opacity=self.materials['window']['opacity'],
            lighting=dict(
                ambient=0.6,
                diffuse=0.4,
                specular=0.9,
                roughness=self.materials['window']['roughness']
            ),
            name=name,
            showlegend=False
        ))
    
    def _add_3d_furniture(self, fig: go.Figure, ilots: List[Dict]):
        """Add furniture as 3D objects with proper geometry"""
        for i, ilot in enumerate(ilots):
            self._create_3d_furniture_piece(fig, ilot, f'Furniture_{i}')
    
    def _create_3d_furniture_piece(self, fig: go.Figure, ilot: Dict, name: str):
        """Create 3D furniture piece"""
        x = ilot.get('x', 0)
        y = ilot.get('y', 0)
        width = ilot.get('width', 1.0)
        height = ilot.get('height', 0.6)
        furniture_height = 0.75  # Standard furniture height
        
        # Create furniture box
        furniture_x = [x - width/2, x + width/2, x + width/2, x - width/2, 
                      x - width/2, x + width/2, x + width/2, x - width/2]
        furniture_y = [y - height/2, y - height/2, y + height/2, y + height/2,
                      y - height/2, y - height/2, y + height/2, y + height/2]
        furniture_z = [0, 0, 0, 0, furniture_height, furniture_height, furniture_height, furniture_height]
        
        # Define faces for the furniture box
        faces_i = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
        faces_j = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]
        faces_k = [2, 3, 0, 1, 6, 7, 4, 5, 1, 2, 3, 0]
        
        # Add furniture as mesh
        fig.add_trace(go.Mesh3d(
            x=furniture_x,
            y=furniture_y,
            z=furniture_z,
            i=faces_i,
            j=faces_j,
            k=faces_k,
            color=self.materials['furniture']['color'],
            opacity=self.materials['furniture']['opacity'],
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.3,
                roughness=self.materials['furniture']['roughness']
            ),
            name=name,
            showlegend=True
        ))
    
    def _add_3d_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors as 3D paths with proper geometry"""
        for i, corridor in enumerate(corridors):
            self._create_3d_corridor_path(fig, corridor, f'Corridor_{i}')
    
    def _create_3d_corridor_path(self, fig: go.Figure, corridor: Dict, name: str):
        """Create 3D corridor path"""
        start_x = corridor.get('start_x', 0)
        start_y = corridor.get('start_y', 0)
        end_x = corridor.get('end_x', 0)
        end_y = corridor.get('end_y', 0)
        width = corridor.get('width', 1.2)
        
        # Create corridor path
        fig.add_trace(go.Scatter3d(
            x=[start_x, end_x],
            y=[start_y, end_y],
            z=[0.1, 0.1],  # Slightly above floor
            mode='lines',
            line=dict(
                color='#FF4444',
                width=8
            ),
            name=name,
            showlegend=False
        ))
    
    def _configure_3d_scene(self, fig: go.Figure, analysis_data: Dict, enable_shadows: bool):
        """Configure 3D scene with professional lighting and camera"""
        bounds = analysis_data.get('bounds', {})
        
        # Calculate scene center and size
        x_center = (bounds.get('x_min', 0) + bounds.get('x_max', 100)) / 2
        y_center = (bounds.get('y_min', 0) + bounds.get('y_max', 100)) / 2
        z_center = self.wall_height / 2
        
        # Configure 3D scene
        fig.update_layout(
            title={
                'text': 'Advanced 3D Floor Plan Visualization',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis=dict(
                    title='X (meters)',
                    showgrid=True,
                    gridcolor='#E0E0E0',
                    backgroundcolor='#F8F9FA'
                ),
                yaxis=dict(
                    title='Y (meters)',
                    showgrid=True,
                    gridcolor='#E0E0E0',
                    backgroundcolor='#F8F9FA'
                ),
                zaxis=dict(
                    title='Z (meters)',
                    showgrid=True,
                    gridcolor='#E0E0E0',
                    backgroundcolor='#F8F9FA'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data',
                bgcolor='#F8F9FA'
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#CCCCCC',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=800
        )
    
    def _add_interactive_controls(self, fig: go.Figure):
        """Add interactive controls for 3D manipulation"""
        # Add annotations for user instructions
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref='paper',
            yref='paper',
            text='🖱️ Drag to rotate • 🔍 Scroll to zoom • 📱 Right-click to pan',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font=dict(size=12)
        )
    
    def _extract_wall_coordinates(self, wall: Any) -> Optional[List[List[float]]]:
        """Extract wall coordinates from various wall data formats"""
        if isinstance(wall, dict):
            if 'points' in wall:
                return wall['points']
            elif 'coordinates' in wall:
                return wall.get('coordinates', [])
            elif 'start' in wall and 'end' in wall:
                return [wall.get('start', [0, 0]), wall.get('end', [0, 0])]
        elif isinstance(wall, (list, tuple)):
            if len(wall) >= 2:
                return wall
        return None
    
    def create_interactive_3d_scene(self, analysis_data: Dict, ilots: List[Dict], 
                                  corridors: List[Dict]) -> str:
        """Create interactive 3D scene with Three.js integration"""
        
        # Generate Three.js scene configuration
        scene_config = {
            'scene': {
                'background': '#F8F9FA',
                'fog': {'color': '#F8F9FA', 'near': 1, 'far': 100}
            },
            'camera': {
                'type': 'PerspectiveCamera',
                'fov': 60,
                'position': [15, 15, 10],
                'target': [0, 0, 0]
            },
            'lighting': self.lighting,
            'objects': self._generate_3d_objects(analysis_data, ilots, corridors),
            'controls': {
                'type': 'OrbitControls',
                'enableDamping': True,
                'dampingFactor': 0.05,
                'enableZoom': True,
                'enablePan': True,
                'enableRotate': True
            }
        }
        
        # Return JSON configuration for Three.js
        return json.dumps(scene_config, indent=2)
    
    def _generate_3d_objects(self, analysis_data: Dict, ilots: List[Dict], 
                           corridors: List[Dict]) -> List[Dict]:
        """Generate 3D objects configuration for Three.js"""
        objects = []
        
        # Add floor
        bounds = analysis_data.get('bounds', {})
        floor_config = {
            'type': 'PlaneGeometry',
            'width': bounds.get('x_max', 100) - bounds.get('x_min', 0),
            'height': bounds.get('y_max', 100) - bounds.get('y_min', 0),
            'material': self.materials['floor'],
            'position': [0, 0, 0],
            'rotation': [-Math.pi/2, 0, 0]
        }
        objects.append(floor_config)
        
        # Add walls
        for wall in analysis_data.get('walls', []):
            wall_config = self._create_wall_config(wall)
            if wall_config:
                objects.append(wall_config)
        
        # Add furniture
        for ilot in ilots:
            furniture_config = self._create_furniture_config(ilot)
            objects.append(furniture_config)
        
        return objects
    
    def _create_wall_config(self, wall: Dict) -> Optional[Dict]:
        """Create wall configuration for Three.js"""
        coords = self._extract_wall_coordinates(wall)
        if not coords or len(coords) < 2:
            return None
        
        return {
            'type': 'BoxGeometry',
            'width': 0.2,  # Wall thickness
            'height': self.wall_height,
            'depth': 1.0,  # Will be calculated from coordinates
            'material': self.materials['wall'],
            'coordinates': coords
        }
    
    def _create_furniture_config(self, ilot: Dict) -> Dict:
        """Create furniture configuration for Three.js"""
        return {
            'type': 'BoxGeometry',
            'width': ilot.get('width', 1.0),
            'height': 0.75,  # Furniture height
            'depth': ilot.get('height', 0.6),
            'material': self.materials['furniture'],
            'position': [ilot.get('x', 0), ilot.get('y', 0), 0.375]  # Half height
        }