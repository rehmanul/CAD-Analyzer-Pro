"""
Simple SVG Renderer
A simplified approach to render DXF floor plans as SVG with proper color coding
"""

import ezdxf
from ezdxf import recover
import tempfile
import os
from typing import Dict, List, Any, Optional
import streamlit as st
from streamlit.components.v1 import html

class SimpleSVGRenderer:
    """Simple SVG renderer for DXF floor plans"""
    
    def __init__(self):
        self.color_map = {
            'walls': '#666666',    # Gray
            'doors': '#FF0000',    # Red
            'restricted': '#0066CC' # Blue
        }
        
    def render_dxf_to_svg(self, file_content: bytes, filename: str, target_bounds: Dict[str, float] = None) -> str:
        """Render DXF to SVG with proper color coding"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load DXF document
                doc, auditor = recover.readfile(tmp_file_path)
                msp = doc.modelspace()
                
                # Extract entities with bounds filtering
                entities = self._extract_entities_with_bounds(msp, target_bounds)
                
                # Create SVG
                svg_content = self._create_svg_from_entities(entities, target_bounds)
                
                return svg_content
                
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error rendering SVG: {str(e)}")
            return self._create_error_svg(str(e))
    
    def _extract_entities_with_bounds(self, msp, target_bounds: Dict[str, float] = None) -> List[Dict]:
        """Extract entities with proper categorization"""
        entities = []
        
        for entity in msp:
            try:
                # Check if entity is within bounds
                if target_bounds and not self._is_entity_in_bounds(entity, target_bounds):
                    continue
                
                entity_data = self._process_entity(entity)
                if entity_data:
                    entities.append(entity_data)
                    
            except Exception as e:
                print(f"Error processing entity: {str(e)}")
                continue
                
        return entities
    
    def _process_entity(self, entity) -> Optional[Dict]:
        """Process a single entity and return its data"""
        try:
            entity_type = entity.dxftype()
            
            if entity_type == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                
                return {
                    'type': 'line',
                    'category': 'walls',
                    'start': (float(start.x), float(start.y)),
                    'end': (float(end.x), float(end.y)),
                    'layer': entity.dxf.layer
                }
                
            elif entity_type == 'LWPOLYLINE':
                points = list(entity.get_points())
                if len(points) >= 2:
                    return {
                        'type': 'polyline',
                        'category': 'walls',
                        'points': [(float(p[0]), float(p[1])) for p in points],
                        'closed': entity.closed,
                        'layer': entity.dxf.layer
                    }
                    
            elif entity_type == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                
                # Check if it's a door arc
                if 0.3 <= radius <= 3.0:
                    return {
                        'type': 'arc',
                        'category': 'doors',
                        'center': (float(center.x), float(center.y)),
                        'radius': float(radius),
                        'start_angle': float(entity.dxf.start_angle),
                        'end_angle': float(entity.dxf.end_angle),
                        'layer': entity.dxf.layer
                    }
                    
            elif entity_type == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                
                return {
                    'type': 'circle',
                    'category': 'restricted',
                    'center': (float(center.x), float(center.y)),
                    'radius': float(radius),
                    'layer': entity.dxf.layer
                }
                
        except Exception as e:
            print(f"Error processing entity: {str(e)}")
            
        return None
    
    def _is_entity_in_bounds(self, entity, bounds: Dict[str, float]) -> bool:
        """Check if entity is within bounds"""
        try:
            entity_type = entity.dxftype()
            
            if entity_type == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                
                # Check if any point is within bounds
                points = [start, end]
                for point in points:
                    if (bounds['min_x'] <= point.x <= bounds['max_x'] and
                        bounds['min_y'] <= point.y <= bounds['max_y']):
                        return True
                        
            elif entity_type == 'LWPOLYLINE':
                points = list(entity.get_points())
                for point in points:
                    if (bounds['min_x'] <= point[0] <= bounds['max_x'] and
                        bounds['min_y'] <= point[1] <= bounds['max_y']):
                        return True
                        
            elif entity_type in ['ARC', 'CIRCLE']:
                center = entity.dxf.center
                if (bounds['min_x'] <= center.x <= bounds['max_x'] and
                    bounds['min_y'] <= center.y <= bounds['max_y']):
                    return True
                    
        except Exception as e:
            print(f"Error checking bounds: {str(e)}")
            
        return False
    
    def _create_svg_from_entities(self, entities: List[Dict], bounds: Dict[str, float] = None) -> str:
        """Create SVG from entities"""
        try:
            if not entities:
                return self._create_empty_svg()
            
            # Calculate bounds if not provided
            if not bounds:
                bounds = self._calculate_bounds_from_entities(entities)
            
            # SVG dimensions
            width = bounds['max_x'] - bounds['min_x']
            height = bounds['max_y'] - bounds['min_y']
            
            # Add padding
            padding = max(width, height) * 0.05
            svg_width = width + 2 * padding
            svg_height = height + 2 * padding
            
            # Create SVG header
            svg_lines = [
                f'<svg viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; background: white;">',
                f'<g transform="translate({padding}, {padding}) scale(1, -1) translate({-bounds["min_x"]}, {-bounds["max_y"]})">'
            ]
            
            # Process entities by category
            wall_entities = [e for e in entities if e['category'] == 'walls']
            door_entities = [e for e in entities if e['category'] == 'doors']
            restricted_entities = [e for e in entities if e['category'] == 'restricted']
            
            # Add walls
            for entity in wall_entities:
                svg_lines.append(self._entity_to_svg(entity, self.color_map['walls']))
            
            # Add doors
            for entity in door_entities:
                svg_lines.append(self._entity_to_svg(entity, self.color_map['doors']))
                
            # Add restricted areas
            for entity in restricted_entities:
                svg_lines.append(self._entity_to_svg(entity, self.color_map['restricted']))
            
            # Close SVG
            svg_lines.extend(['</g>', '</svg>'])
            
            return '\n'.join(svg_lines)
            
        except Exception as e:
            print(f"Error creating SVG: {str(e)}")
            return self._create_error_svg(str(e))
    
    def _entity_to_svg(self, entity: Dict, color: str) -> str:
        """Convert entity to SVG element"""
        try:
            if entity['type'] == 'line':
                start = entity['start']
                end = entity['end']
                return f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" stroke="{color}" stroke-width="0.5" />'
                
            elif entity['type'] == 'polyline':
                points = entity['points']
                points_str = ' '.join([f"{p[0]},{p[1]}" for p in points])
                fill = 'none' if not entity.get('closed', False) else color
                return f'<polyline points="{points_str}" stroke="{color}" stroke-width="0.5" fill="{fill}" />'
                
            elif entity['type'] == 'arc':
                center = entity['center']
                radius = entity['radius']
                start_angle = entity['start_angle']
                end_angle = entity['end_angle']
                
                # Convert angles to radians
                import math
                start_rad = math.radians(start_angle)
                end_rad = math.radians(end_angle)
                
                # Calculate arc points
                center_x, center_y = center
                x1 = center_x + radius * math.cos(start_rad)
                center_x, center_y = center
                y1 = center_y + radius * math.sin(start_rad)
                center_x, center_y = center
                x2 = center_x + radius * math.cos(end_rad)
                center_x, center_y = center
                y2 = center_y + radius * math.sin(end_rad)
                
                # Create path
                large_arc = 1 if abs(end_angle - start_angle) > 180 else 0
                return f'<path d="M {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2}" stroke="{color}" stroke-width="0.5" fill="none" />'
                
            elif entity['type'] == 'circle':
                center = entity['center']
                radius = entity['radius']
                center_x, center_y = center
                return f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" stroke="{color}" stroke-width="0.5" fill="none" />'
                
        except Exception as e:
            print(f"Error converting entity to SVG: {str(e)}")
            
        return ''
    
    def _calculate_bounds_from_entities(self, entities: List[Dict]) -> Dict[str, float]:
        """Calculate bounds from entities"""
        try:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for entity in entities:
                if entity['type'] == 'line':
                    start, end = entity['start'], entity['end']
                    min_x = min(min_x, start[0], end[0])
                    max_x = max(max_x, start[0], end[0])
                    min_y = min(min_y, start[1], end[1])
                    max_y = max(max_y, start[1], end[1])
                    
                elif entity['type'] == 'polyline':
                    for point in entity['points']:
                        min_x = min(min_x, point[0])
                        max_x = max(max_x, point[0])
                        min_y = min(min_y, point[1])
                        max_y = max(max_y, point[1])
                        
                elif entity['type'] in ['arc', 'circle']:
                    center = entity['center']
                    radius = entity['radius']
                    center_x, center_y = center
                    min_x = min(min_x, center_x - radius)
                    center_x, center_y = center
                    max_x = max(max_x, center_x + radius)
                    center_x, center_y = center
                    min_y = min(min_y, center_y - radius)
                    center_x, center_y = center
                    max_y = max(max_y, center_y + radius)
            
            return {
                'min_x': min_x if min_x != float('inf') else 0,
                'max_x': max_x if max_x != float('-inf') else 100,
                'min_y': min_y if min_y != float('inf') else 0,
                'max_y': max_y if max_y != float('-inf') else 100
            }
            
        except Exception as e:
            print(f"Error calculating bounds: {str(e)}")
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
    
    def _create_empty_svg(self) -> str:
        """Create empty SVG"""
        return '''
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; background: white;">
            <text x="200" y="150" text-anchor="middle" fill="#666" font-family="Arial" font-size="16">
                No floor plan data found
            </text>
        </svg>
        '''
    
    def _create_error_svg(self, error_msg: str) -> str:
        """Create error SVG"""
        return f'''
        <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; background: white;">
            <rect width="400" height="200" fill="#f8f9fa" stroke="#dee2e6"/>
            <text x="200" y="100" text-anchor="middle" fill="#dc3545" font-family="Arial" font-size="14">
                Error rendering floor plan
            </text>
            <text x="200" y="120" text-anchor="middle" fill="#6c757d" font-family="Arial" font-size="12">
                {error_msg}
            </text>
        </svg>
        '''
    
    def create_legend(self) -> str:
        """Create legend HTML"""
        return '''
        <div style="display: flex; align-items: center; gap: 20px; padding: 15px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-weight: 600; color: #495057; margin-right: 10px;">LÉGENDE:</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 25px; height: 4px; background: #666666; border-radius: 2px;"></div>
                <span style="font-size: 14px; color: #495057; font-weight: 500;">MUR</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 25px; height: 4px; background: #FF0000; border-radius: 2px;"></div>
                <span style="font-size: 14px; color: #495057; font-weight: 500;">ENTRÉE/SORTIE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 25px; height: 4px; background: #0066CC; border-radius: 2px;"></div>
                <span style="font-size: 14px; color: #495057; font-weight: 500;">NO ENTREE</span>
            </div>
        </div>
        '''
    
    def embed_svg_with_controls(self, svg_content: str, height: int = 600):
        """Embed SVG with pan/zoom controls"""
        try:
            # Display legend
            st.markdown(self.create_legend(), unsafe_allow_html=True)
            
            # Create interactive SVG container
            html_content = f'''
            <div id="svg-container" style="border: 2px solid #ddd; border-radius: 8px; overflow: hidden; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                {svg_content}
            </div>
            <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
            <script>
                const container = document.getElementById('svg-container');
                const svg = container.querySelector('svg');
                
                if (svg) {{
                    const panzoomInstance = panzoom(svg, {{
                        maxZoom: 20,
                        minZoom: 0.1,
                        smoothScroll: false,
                        bounds: true,
                        boundsPadding: 0.2
                    }});
                    
                    svg.addEventListener('dblclick', function() {{
                        panzoomInstance.reset();
                    }});
                }}
            </script>
            <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 12px; color: #666;">
                💡 <strong>Contrôles:</strong> Glisser pour déplacer • Molette pour zoomer • Double-clic pour réinitialiser
            </div>
            '''
            
            html(html_content, height=height + 100)
            
        except Exception as e:
            st.error(f"Erreur d'affichage SVG: {str(e)}")