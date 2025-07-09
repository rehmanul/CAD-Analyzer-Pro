"""
SVG Floor Plan Renderer
Converts DXF floor plans to high-quality SVG with proper color coding and interactive features
"""

import ezdxf
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.svg import SVGBackend
import re
import tempfile
import os
from typing import Dict, List, Any, Optional
import streamlit as st
from streamlit.components.v1 import html

class SVGFloorPlanRenderer:
    """Renders DXF floor plans as high-quality SVG with proper color coding"""
    
    def __init__(self):
        self.layer_color_map = {
            # Wall layers - Gray
            'WALLS': '#666666',
            'WALL': '#666666', 
            'MUR': '#666666',
            'MURS': '#666666',
            '0': '#666666',
            'DEFPOINTS': '#666666',
            
            # Door/Entry layers - Red
            'DOORS': '#FF0000',
            'DOOR': '#FF0000',
            'PORTE': '#FF0000',
            'PORTES': '#FF0000',
            'ENTRY': '#FF0000',
            'ENTREE': '#FF0000',
            'ENTREE_SORTIE': '#FF0000',
            
            # Restricted areas - Blue
            'RESTRICTED': '#0066CC',
            'BLOCK': '#0066CC',
            'NO_ENTREE': '#0066CC',
            'STAIR': '#0066CC',
            'ELEVATOR': '#0066CC',
            
            # Default
            'DEFAULT': '#333333'
        }
    
    def render_floor_plan_svg(self, file_content: bytes, filename: str, target_bounds: Dict[str, float] = None) -> str:
        """Render DXF floor plan as SVG with proper color coding"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load DXF document
                doc, auditor = recover.readfile(tmp_file_path)
                
                # Apply color mapping to layers
                self._apply_color_mapping(doc)
                
                # Create render context
                ctx = RenderContext(doc)
                
                # Configure layer properties
                self._configure_layer_properties(ctx)
                
                # Create SVG backend
                backend = SVGBackend()
                
                # Set up modelspace
                msp = doc.modelspace()
                
                # If target bounds specified, filter entities
                if target_bounds:
                    msp = self._filter_entities_by_bounds(msp, target_bounds)
                
                # Render to SVG
                frontend = Frontend(ctx, backend)
                frontend.draw_layout(msp, finalize=True)
                
                # Get SVG string
                svg_code = backend.get_string()
                
                # Clean up SVG for responsive scaling
                svg_code = self._cleanup_svg(svg_code)
                
                return svg_code
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error rendering SVG: {str(e)}")
            return self._create_error_svg(str(e))
    
    def _apply_color_mapping(self, doc):
        """Apply color mapping to DXF layers"""
        try:
            for layer_name in doc.layers:
                layer = doc.layers.get(layer_name)
                if layer:
                    # Map layer names to colors
                    color = self.layer_color_map.get(layer_name.upper(), self.layer_color_map['DEFAULT'])
                    
                    # Convert hex color to ACI color index (approximation)
                    aci_color = self._hex_to_aci_color(color)
                    layer.dxf.color = aci_color
                    
        except Exception as e:
            print(f"Error applying color mapping: {str(e)}")
    
    def _configure_layer_properties(self, ctx):
        """Configure layer properties for proper rendering"""
        try:
            # Set line weights and colors for different layer types
            for layer_name, color in self.layer_color_map.items():
                try:
                    # Set color for layer
                    rgb_color = self._hex_to_rgb(color)
                    ctx.layer_properties.set_color(layer_name, rgb_color)
                    
                    # Set line weight based on layer type
                    if 'WALL' in layer_name.upper() or 'MUR' in layer_name.upper():
                        ctx.layer_properties.set_lineweight(layer_name, 0.5)  # Thicker walls
                    else:
                        ctx.layer_properties.set_lineweight(layer_name, 0.25)  # Thinner lines
                        
                except Exception as e:
                    print(f"Error configuring layer {layer_name}: {str(e)}")
                    
        except Exception as e:
            print(f"Error configuring layer properties: {str(e)}")
    
    def _filter_entities_by_bounds(self, msp, bounds: Dict[str, float]):
        """Filter entities to only include those within target bounds"""
        try:
            # Create a new modelspace with filtered entities
            filtered_entities = []
            
            for entity in msp:
                if self._is_entity_in_bounds(entity, bounds):
                    filtered_entities.append(entity)
            
            # Return filtered modelspace
            return filtered_entities
            
        except Exception as e:
            print(f"Error filtering entities: {str(e)}")
            return msp
    
    def _is_entity_in_bounds(self, entity, bounds: Dict[str, float]) -> bool:
        """Check if entity is within bounds"""
        try:
            # Get entity bounding box
            bbox = self._get_entity_bbox(entity)
            if not bbox:
                return False
            
            # Check if entity overlaps with bounds
            return not (bbox['max_x'] < bounds['min_x'] or 
                       bbox['min_x'] > bounds['max_x'] or
                       bbox['max_y'] < bounds['min_y'] or
                       bbox['min_y'] > bounds['max_y'])
        except:
            return True  # Include if can't determine
    
    def _get_entity_bbox(self, entity) -> Optional[Dict]:
        """Get bounding box for an entity"""
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                return {
                    'min_x': min(start.x, end.x),
                    'max_x': max(start.x, end.x),
                    'min_y': min(start.y, end.y),
                    'max_y': max(start.y, end.y)
                }
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points())
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    return {
                        'min_x': min(xs),
                        'max_x': max(xs),
                        'min_y': min(ys),
                        'max_y': max(ys)
                    }
            elif entity.dxftype() in ['ARC', 'CIRCLE']:
                center = entity.dxf.center
                radius = entity.dxf.radius
                return {
                    'min_x': center.x - radius,
                    'max_x': center.x + radius,
                    'min_y': center.y - radius,
                    'max_y': center.y + radius
                }
        except:
            pass
        return None
    
    def _cleanup_svg(self, svg_code: str) -> str:
        """Clean up SVG for responsive scaling"""
        try:
            # Remove fixed width/height attributes for responsive scaling
            svg_code = re.sub(r' (width|height)="[^"]+"', '', svg_code, count=2)
            
            # Add viewBox if not present
            if 'viewBox=' not in svg_code:
                # Extract dimensions and add viewBox
                svg_code = re.sub(r'<svg', '<svg viewBox="0 0 800 600"', svg_code, count=1)
            
            # Ensure proper styling
            svg_code = svg_code.replace('<svg', '<svg style="width: 100%; height: 100%;"')
            
            return svg_code
            
        except Exception as e:
            print(f"Error cleaning up SVG: {str(e)}")
            return svg_code
    
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (51, 51, 51)  # Default gray
    
    def _hex_to_aci_color(self, hex_color: str) -> int:
        """Convert hex color to ACI color index (approximation)"""
        try:
            rgb = self._hex_to_rgb(hex_color)
            
            # Simple mapping to common ACI colors
            if rgb == (102, 102, 102):  # Gray walls
                return 8
            elif rgb == (255, 0, 0):    # Red doors
                return 1
            elif rgb == (0, 102, 204):  # Blue restricted
                return 5
            else:
                return 7  # Default white
                
        except:
            return 7
    
    def _create_error_svg(self, error_msg: str) -> str:
        """Create error SVG when rendering fails"""
        return f'''
        <svg viewBox="0 0 400 200" style="width: 100%; height: 100%;">
            <rect width="400" height="200" fill="#f8f9fa" stroke="#dee2e6"/>
            <text x="200" y="100" text-anchor="middle" fill="#dc3545" font-family="Arial" font-size="14">
                Error rendering floor plan
            </text>
            <text x="200" y="120" text-anchor="middle" fill="#6c757d" font-family="Arial" font-size="12">
                {error_msg}
            </text>
        </svg>
        '''
    
    def embed_svg_with_controls(self, svg_code: str, height: int = 600):
        """Embed SVG in Streamlit with pan/zoom controls"""
        try:
            # Create legend
            legend_html = '''
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 20px; height: 3px; background: #666666;"></div>
                    <span style="font-size: 12px;">MUR</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 20px; height: 3px; background: #FF0000;"></div>
                    <span style="font-size: 12px;">ENTRÉE/SORTIE</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 20px; height: 3px; background: #0066CC;"></div>
                    <span style="font-size: 12px;">NO ENTREE</span>
                </div>
            </div>
            '''
            
            # Display legend
            st.markdown(legend_html, unsafe_allow_html=True)
            
            # Embed SVG with pan/zoom
            html_content = f'''
            <div id="svg-container" style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; background: white;">
                {svg_code}
            </div>
            <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
            <script>
                const container = document.getElementById('svg-container');
                const svg = container.querySelector('svg');
                
                if (svg) {{
                    // Apply panzoom
                    const panzoomInstance = panzoom(svg, {{
                        zoomDoubleClickSpeed: 1,
                        maxZoom: 10,
                        minZoom: 0.1,
                        smoothScroll: false,
                        bounds: true,
                        boundsPadding: 0.1
                    }});
                    
                    // Add reset button functionality
                    svg.addEventListener('dblclick', function() {{
                        panzoomInstance.reset();
                    }});
                }}
            </script>
            <div style="margin-top: 10px; font-size: 12px; color: #666;">
                💡 <strong>Controls:</strong> Drag to pan • Mouse wheel to zoom • Double-click to reset
            </div>
            '''
            
            html(html_content, height=height + 50)
            
        except Exception as e:
            st.error(f"Error embedding SVG: {str(e)}")
            
    def create_legend(self) -> str:
        """Create HTML legend for the floor plan"""
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