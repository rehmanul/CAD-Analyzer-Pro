"""
WebGL 3D Renderer for Real-Time CAD Visualization
Professional WebGL-based 3D rendering with Three.js integration
"""

import streamlit as st
import json
import math
from typing import Dict, List, Any, Optional
import base64

class WebGL3DRenderer:
    """WebGL-based 3D renderer with Three.js integration"""
    
    def __init__(self):
        self.three_js_cdn = "https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js"
        self.controls_cdn = "https://cdn.jsdelivr.net/npm/three@0.155.0/examples/jsm/controls/OrbitControls.js"
        
    def render_3d_scene(self, analysis_data: Dict, ilots: List[Dict], 
                       corridors: List[Dict], container_id: str = "3d-scene"):
        """Render interactive 3D scene using WebGL and Three.js"""
        
        # Generate scene configuration
        scene_config = self._generate_scene_config(analysis_data, ilots, corridors)
        
        # Create HTML container with Three.js
        html_content = self._create_threejs_html(scene_config, container_id)
        
        # Render in Streamlit
        st.components.v1.html(html_content, height=600)
    
    def _generate_scene_config(self, analysis_data: Dict, ilots: List[Dict], 
                             corridors: List[Dict]) -> Dict:
        """Generate complete scene configuration"""
        
        bounds = analysis_data.get('bounds', {})
        
        return {
            'scene': {
                'background': '#F8F9FA',
                'fog': {'color': '#F8F9FA', 'near': 10, 'far': 200}
            },
            'camera': {
                'type': 'PerspectiveCamera',
                'fov': 60,
                'position': [20, 20, 15],
                'target': [0, 0, 0]
            },
            'lighting': self._generate_lighting_config(),
            'objects': self._generate_objects_config(analysis_data, ilots, corridors),
            'controls': {
                'type': 'OrbitControls',
                'enableDamping': True,
                'dampingFactor': 0.05,
                'enableZoom': True,
                'enablePan': True,
                'enableRotate': True,
                'maxPolarAngle': 1.4  # Limit rotation
            },
            'bounds': bounds
        }
    
    def _generate_lighting_config(self) -> Dict:
        """Generate professional lighting configuration"""
        return {
            'ambient': {
                'color': '#404040',
                'intensity': 0.4
            },
            'directional': {
                'color': '#FFFFFF',
                'intensity': 0.8,
                'position': [10, 10, 10],
                'castShadow': True,
                'shadowMapSize': 2048
            },
            'point_lights': [
                {
                    'color': '#FFFFFF',
                    'intensity': 0.6,
                    'position': [5, 5, 5],
                    'distance': 50,
                    'decay': 2
                },
                {
                    'color': '#FFFFFF',
                    'intensity': 0.4,
                    'position': [-5, -5, 5],
                    'distance': 50,
                    'decay': 2
                }
            ]
        }
    
    def _generate_objects_config(self, analysis_data: Dict, ilots: List[Dict], 
                               corridors: List[Dict]) -> List[Dict]:
        """Generate 3D objects configuration"""
        objects = []
        
        # Add floor
        bounds = analysis_data.get('bounds', {})
        floor_width = bounds.get('x_max', 100) - bounds.get('x_min', 0)
        floor_height = bounds.get('y_max', 100) - bounds.get('y_min', 0)
        
        objects.append({
            'type': 'floor',
            'geometry': 'PlaneGeometry',
            'width': floor_width,
            'height': floor_height,
            'material': {
                'type': 'MeshStandardMaterial',
                'color': '#D4C4A8',
                'roughness': 0.8,
                'metalness': 0.0
            },
            'position': [0, 0, 0],
            'rotation': [-1.5708, 0, 0]  # -90 degrees in radians
        })
        
        # Add walls
        walls = analysis_data.get('walls', [])
        for i, wall in enumerate(walls):
            wall_objects = self._create_wall_objects(wall, i)
            objects.extend(wall_objects)
        
        # Add furniture (îlots)
        for i, ilot in enumerate(ilots):
            furniture_obj = self._create_furniture_object(ilot, i)
            objects.append(furniture_obj)
        
        # Add corridors
        for i, corridor in enumerate(corridors):
            corridor_obj = self._create_corridor_object(corridor, i)
            objects.append(corridor_obj)
        
        return objects
    
    def _create_wall_objects(self, wall: Dict, wall_id: int) -> List[Dict]:
        """Create wall objects configuration"""
        coords = self._extract_wall_coordinates(wall)
        if not coords or len(coords) < 2:
            return []
        
        wall_objects = []
        wall_height = 2.8
        wall_thickness = 0.2
        
        for i in range(len(coords) - 1):
            start_point = coords[i]
            end_point = coords[i + 1]
            
            # Calculate wall segment properties
            length = ((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)**0.5
            center_x = (start_point[0] + end_point[0]) / 2
            center_y = (start_point[1] + end_point[1]) / 2
            
            # Calculate rotation
            angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
            
            wall_objects.append({
                'type': 'wall',
                'geometry': 'BoxGeometry',
                'width': length,
                'height': wall_height,
                'depth': wall_thickness,
                'material': {
                    'type': 'MeshStandardMaterial',
                    'color': '#CCCCCC',
                    'roughness': 0.8,
                    'metalness': 0.1
                },
                'position': [center_x, center_y, wall_height / 2],
                'rotation': [0, 0, angle]
            })
        
        return wall_objects
    
    def _create_furniture_object(self, ilot: Dict, furniture_id: int) -> Dict:
        """Create furniture object configuration"""
        return {
            'type': 'furniture',
            'geometry': 'BoxGeometry',
            'width': ilot.get('width', 1.0),
            'height': 0.75,  # Furniture height
            'depth': ilot.get('height', 0.6),
            'material': {
                'type': 'MeshStandardMaterial',
                'color': '#8B4513',
                'roughness': 0.6,
                'metalness': 0.0
            },
            'position': [ilot.get('x', 0), ilot.get('y', 0), 0.375],
            'rotation': [0, 0, 0],
            'userData': {
                'id': furniture_id,
                'type': 'furniture',
                'movable': True
            }
        }
    
    def _create_corridor_object(self, corridor: Dict, corridor_id: int) -> Dict:
        """Create corridor object configuration"""
        start_x = corridor.get('start_x', 0)
        start_y = corridor.get('start_y', 0)
        end_x = corridor.get('end_x', 0)
        end_y = corridor.get('end_y', 0)
        
        length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        angle = math.atan2(end_y - start_y, end_x - start_x)
        
        return {
            'type': 'corridor',
            'geometry': 'BoxGeometry',
            'width': length,
            'height': 0.05,  # Thin corridor marker
            'depth': 1.2,    # Corridor width
            'material': {
                'type': 'MeshStandardMaterial',
                'color': '#FF4444',
                'roughness': 0.2,
                'metalness': 0.8,
                'transparent': True,
                'opacity': 0.7
            },
            'position': [center_x, center_y, 0.025],
            'rotation': [0, 0, angle]
        }
    
    def _extract_wall_coordinates(self, wall: Any) -> Optional[List[List[float]]]:
        """Extract wall coordinates from various formats"""
        if isinstance(wall, dict):
            if 'points' in wall:
                return wall['points']
            elif 'coordinates' in wall:
                return wall['coordinates']
            elif 'start' in wall and 'end' in wall:
                return [wall['start'], wall['end']]
        elif isinstance(wall, (list, tuple)):
            if len(wall) >= 2:
                return wall
        return None
    
    def _create_threejs_html(self, scene_config: Dict, container_id: str) -> str:
        """Create HTML with Three.js implementation"""
        
        config_json = json.dumps(scene_config, indent=2)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>3D CAD Visualization</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    background-color: #f0f0f0;
                    font-family: 'Inter', sans-serif;
                }}
                #{container_id} {{
                    width: 100%;
                    height: 100vh;
                    position: relative;
                }}
                .controls-panel {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    z-index: 1000;
                }}
                .controls-panel h3 {{
                    margin: 0 0 10px 0;
                    color: #333;
                    font-size: 14px;
                }}
                .control-button {{
                    background: #4F46E5;
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    margin: 2px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                }}
                .control-button:hover {{
                    background: #4338CA;
                }}
                .info-panel {{
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    z-index: 1000;
                    font-size: 12px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div id="{container_id}">
                <div class="controls-panel">
                    <h3>3D Controls</h3>
                    <button class="control-button" onclick="resetCamera()">Reset View</button>
                    <button class="control-button" onclick="toggleWireframe()">Wireframe</button>
                    <button class="control-button" onclick="toggleShadows()">Shadows</button>
                </div>
                <div class="info-panel">
                    🖱️ Left Click: Rotate<br>
                    🖱️ Right Click: Pan<br>
                    🔍 Scroll: Zoom
                </div>
            </div>
            
            <script type="importmap">
            {{
                "imports": {{
                    "three": "https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.155.0/examples/jsm/"
                }}
            }}
            </script>
            
            <script type="module">
                import * as THREE from 'three';
                import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
                
                // Scene configuration
                const sceneConfig = {config_json};
                
                // Global variables
                let scene, camera, renderer, controls;
                let meshes = [];
                let wireframeMode = false;
                let shadowsEnabled = true;
                
                // Initialize the 3D scene
                function init() {{
                    const container = document.getElementById('{container_id}');
                    const rect = container.getBoundingClientRect();
                    
                    // Create scene
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(sceneConfig.scene.background);
                    
                    // Create camera
                    camera = new THREE.PerspectiveCamera(
                        sceneConfig.camera.fov,
                        rect.width / rect.height,
                        0.1,
                        1000
                    );
                    camera.position.set(...sceneConfig.camera.position);
                    
                    // Create renderer
                    renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(rect.width, rect.height);
                    renderer.setPixelRatio(window.devicePixelRatio);
                    renderer.shadowMap.enabled = shadowsEnabled;
                    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                    container.appendChild(renderer.domElement);
                    
                    // Create controls
                    controls = new OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = sceneConfig.controls.enableDamping;
                    controls.dampingFactor = sceneConfig.controls.dampingFactor;
                    controls.enableZoom = sceneConfig.controls.enableZoom;
                    controls.enablePan = sceneConfig.controls.enablePan;
                    controls.enableRotate = sceneConfig.controls.enableRotate;
                    controls.maxPolarAngle = sceneConfig.controls.maxPolarAngle;
                    
                    // Add lighting
                    addLighting();
                    
                    // Add objects
                    addObjects();
                    
                    // Start animation loop
                    animate();
                    
                    // Handle window resize
                    window.addEventListener('resize', onWindowResize);
                }}
                
                function addLighting() {{
                    // Ambient light
                    const ambientLight = new THREE.AmbientLight(
                        sceneConfig.lighting.ambient.color,
                        sceneConfig.lighting.ambient.intensity
                    );
                    scene.add(ambientLight);
                    
                    // Directional light
                    const directionalLight = new THREE.DirectionalLight(
                        sceneConfig.lighting.directional.color,
                        sceneConfig.lighting.directional.intensity
                    );
                    directionalLight.position.set(...sceneConfig.lighting.directional.position);
                    directionalLight.castShadow = sceneConfig.lighting.directional.castShadow;
                    directionalLight.shadow.mapSize.width = sceneConfig.lighting.directional.shadowMapSize;
                    directionalLight.shadow.mapSize.height = sceneConfig.lighting.directional.shadowMapSize;
                    scene.add(directionalLight);
                    
                    // Point lights
                    sceneConfig.lighting.point_lights.forEach(lightConfig => {{
                        const pointLight = new THREE.PointLight(
                            lightConfig.color,
                            lightConfig.intensity,
                            lightConfig.distance,
                            lightConfig.decay
                        );
                        pointLight.position.set(...lightConfig.position);
                        scene.add(pointLight);
                    }});
                }}
                
                function addObjects() {{
                    sceneConfig.objects.forEach(objConfig => {{
                        let geometry, material, mesh;
                        
                        // Create geometry
                        switch (objConfig.geometry) {{
                            case 'BoxGeometry':
                                geometry = new THREE.BoxGeometry(
                                    objConfig.width,
                                    objConfig.height,
                                    objConfig.depth
                                );
                                break;
                            case 'PlaneGeometry':
                                geometry = new THREE.PlaneGeometry(
                                    objConfig.width,
                                    objConfig.height
                                );
                                break;
                        }}
                        
                        // Create material
                        const materialConfig = objConfig.material;
                        material = new THREE.MeshStandardMaterial({{
                            color: materialConfig.color,
                            roughness: materialConfig.roughness,
                            metalness: materialConfig.metalness,
                            transparent: materialConfig.transparent || false,
                            opacity: materialConfig.opacity || 1.0
                        }});
                        
                        // Create mesh
                        mesh = new THREE.Mesh(geometry, material);
                        mesh.position.set(...objConfig.position);
                        if (objConfig.rotation) {{
                            mesh.rotation.set(...objConfig.rotation);
                        }}
                        
                        // Enable shadows
                        mesh.castShadow = true;
                        mesh.receiveShadow = true;
                        
                        // Store reference
                        mesh.userData = objConfig.userData || {{ type: objConfig.type }};
                        
                        scene.add(mesh);
                        meshes.push(mesh);
                    }});
                }}
                
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                
                function onWindowResize() {{
                    const container = document.getElementById('{container_id}');
                    const rect = container.getBoundingClientRect();
                    
                    camera.aspect = rect.width / rect.height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(rect.width, rect.height);
                }}
                
                // Control functions
                window.resetCamera = function() {{
                    camera.position.set(...sceneConfig.camera.position);
                    controls.target.set(...sceneConfig.camera.target);
                    controls.update();
                }}
                
                window.toggleWireframe = function() {{
                    wireframeMode = !wireframeMode;
                    meshes.forEach(mesh => {{
                        mesh.material.wireframe = wireframeMode;
                    }});
                }}
                
                window.toggleShadows = function() {{
                    shadowsEnabled = !shadowsEnabled;
                    renderer.shadowMap.enabled = shadowsEnabled;
                    scene.traverse(child => {{
                        if (child instanceof THREE.Light) {{
                            child.castShadow = shadowsEnabled;
                        }}
                    }});
                }}
                
                // Initialize when page loads
                init();
            </script>
        </body>
        </html>
        """