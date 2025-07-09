"""
Production SVG Renderer
Professional CAD floor plan rendering with accurate SVG, layer mapping, and interactive controls
"""

import ezdxf
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.svg import SVGBackend
import tempfile
import os
import re
from typing import Dict, List, Any, Optional
import streamlit as st
from streamlit.components.v1 import html
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

class ProductionSVGRenderer:
    """Production-grade SVG renderer for CAD floor plans"""
    
    def __init__(self):
        self.LAYER_COLORS = {
            "WALL": "#888888",           # Gray for walls
            "WALLS": "#888888",          # Gray for walls
            "MUR": "#888888",            # Gray for walls (French)
            "MURS": "#888888",           # Gray for walls (French)
            "0": "#888888",              # Default layer
            "DEFPOINTS": "#888888",      # Construction points
            
            "ENTRY": "#d9534f",          # Red for entry/exit
            "DOOR": "#d9534f",           # Red for doors
            "DOORS": "#d9534f",          # Red for doors
            "PORTE": "#d9534f",          # Red for doors (French)
            "PORTES": "#d9534f",         # Red for doors (French)
            "ENTREE": "#d9534f",         # Red for entry (French)
            "SORTIE": "#d9534f",         # Red for exit (French)
            
            "BLOCK": "#337ab7",          # Blue for no entry
            "RESTRICTED": "#337ab7",     # Blue for restricted areas
            "NO_ENTREE": "#337ab7",      # Blue for no entry (French)
            "STAIR": "#337ab7",          # Blue for stairs
            "ELEVATOR": "#337ab7",       # Blue for elevator
        }
        
        self.PANZOOM_JS = "https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"
        self.SUBJX_JS = "https://unpkg.com/subjx@2.0.0/dist/js/subjx.min.js"
    
    def dxf_to_svg(self, file_content: bytes, filename: str, target_bounds: Dict[str, float] = None) -> str:
        """Convert DXF to SVG with proper layer colors and vector accuracy"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load DXF document
                doc, auditor = recover.readfile(tmp_file_path)
                msp = doc.modelspace()
                
                # Create render context
                ctx = RenderContext(doc)
                
                # Set custom colors for layers
                for layer_name, color in self.LAYER_COLORS.items():
                    try:
                        if hasattr(ctx, 'layer_properties') and hasattr(ctx.layer_properties, 'set_color'):
                            rgb_color = self._hex_to_rgb(color)
                            ctx.layer_properties.set_color(layer_name, rgb_color)
                    except Exception as e:
                        print(f"Error setting color for layer {layer_name}: {str(e)}")
                
                # Create SVG backend
                backend = SVGBackend()
                
                # Filter entities if target bounds specified
                if target_bounds:
                    filtered_entities = []
                    for entity in msp:
                        if self._is_entity_in_bounds(entity, target_bounds):
                            filtered_entities.append(entity)
                    
                    # Create filtered modelspace
                    temp_doc = ezdxf.new()
                    temp_msp = temp_doc.modelspace()
                    for entity in filtered_entities:
                        temp_msp.add_entity(entity.copy())
                    
                    msp = temp_msp
                
                # Render to SVG
                frontend = Frontend(ctx, backend)
                frontend.draw_layout(msp, finalize=True)
                
                # Get SVG string
                svg_code = backend.get_string()
                
                # Remove width/height for responsive scaling
                svg_code = re.sub(r' (width|height)="[^"]+"', '', svg_code, count=2)
                
                # Add viewBox if not present
                if 'viewBox=' not in svg_code:
                    svg_code = re.sub(r'<svg', '<svg viewBox="0 0 800 600"', svg_code, count=1)
                
                # Ensure proper styling
                svg_code = svg_code.replace('<svg', '<svg style="width: 100%; height: 100%; background: white;"')
                
                return svg_code
                
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error converting DXF to SVG: {str(e)}")
            return self._create_error_svg(str(e))
    
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (136, 136, 136)  # Default gray
    
    def _is_entity_in_bounds(self, entity, bounds: Dict[str, float]) -> bool:
        """Check if entity is within bounds"""
        try:
            entity_type = entity.dxftype()
            
            if entity_type == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                return (self._point_in_bounds(start, bounds) or 
                       self._point_in_bounds(end, bounds))
                       
            elif entity_type == 'LWPOLYLINE':
                points = list(entity.get_points())
                return any(self._point_in_bounds_xy(point, bounds) for point in points)
                
            elif entity_type in ['ARC', 'CIRCLE']:
                center = entity.dxf.center
                return self._point_in_bounds(center, bounds)
                
        except Exception as e:
            print(f"Error checking bounds: {str(e)}")
            
        return True  # Include if can't determine
    
    def _point_in_bounds(self, point, bounds: Dict[str, float]) -> bool:
        """Check if point is within bounds"""
        return (bounds['min_x'] <= point.x <= bounds['max_x'] and
                bounds['min_y'] <= point.y <= bounds['max_y'])
    
    def _point_in_bounds_xy(self, point, bounds: Dict[str, float]) -> bool:
        """Check if XY point is within bounds"""
        return (bounds['min_x'] <= point[0] <= bounds['max_x'] and
                bounds['min_y'] <= point[1] <= bounds['max_y'])
    
    def get_legend_html(self) -> str:
        """Generate professional legend HTML"""
        return """
        <div style="display: flex; align-items: center; gap: 24px; padding: 16px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 8px; margin: 16px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); border: 1px solid #dee2e6;">
            <div style="font-weight: 600; color: #495057; font-size: 16px; margin-right: 8px;">LÉGENDE:</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #337ab7; width: 24px; height: 24px; display: inline-block; border-radius: 3px; border: 1px solid #2e6da4;"></span>
                <span style="font-size: 14px; color: #495057; font-weight: 500;">NO ENTREE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #d9534f; width: 24px; height: 24px; display: inline-block; border-radius: 3px; border: 1px solid #c9302c;"></span>
                <span style="font-size: 14px; color: #495057; font-weight: 500;">ENTRÉE/SORTIE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #888888; width: 24px; height: 24px; display: inline-block; border-radius: 3px; border: 1px solid #777;"></span>
                <span style="font-size: 14px; color: #495057; font-weight: 500;">MUR</span>
            </div>
        </div>
        """
    
    def place_ilots(self, floor_bounds: Dict[str, float], ilot_config: Dict) -> List[Dict]:
        """Place îlots in the floor plan with proper distribution"""
        ilots = []
        
        try:
            # Get floor dimensions
            width = floor_bounds['max_x'] - floor_bounds['min_x']
            height = floor_bounds['max_y'] - floor_bounds['min_y']
            
            # Calculate îlot sizes based on configuration
            total_ilots = ilot_config.get('total_count', 20)
            clearance = ilot_config.get('clearance', 2.0)
            
            # Size distribution: 10% small, 25% medium, 30% large, 35% extra-large
            sizes = {
                'small': (3, 2),      # 10%
                'medium': (4, 3),     # 25%
                'large': (5, 4),      # 30%
                'extra_large': (6, 5) # 35%
            }
            
            distribution = {
                'small': int(total_ilots * 0.10),
                'medium': int(total_ilots * 0.25),
                'large': int(total_ilots * 0.30),
                'extra_large': int(total_ilots * 0.35)
            }
            
            # Place îlots in grid pattern
            x = floor_bounds['min_x'] + clearance
            y = floor_bounds['min_y'] + clearance
            row_height = 0
            
            for size_type, count in distribution.items():
                w, h = sizes[size_type]
                
                for _ in range(count):
                    # Check if îlot fits in current row
                    if x + w > floor_bounds['max_x'] - clearance:
                        x = floor_bounds['min_x'] + clearance
                        y += row_height + clearance
                        row_height = 0
                    
                    # Check if îlot fits in floor plan
                    if y + h <= floor_bounds['max_y'] - clearance:
                        ilot = {
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'type': size_type
                        }
                        ilots.append(ilot)
                        
                        x += w + clearance
                        row_height = max(row_height, h)
                    else:
                        break  # No more space
                        
        except Exception as e:
            print(f"Error placing îlots: {str(e)}")
            
        return ilots
    
    def ilots_to_svg(self, ilots: List[Dict]) -> str:
        """Convert îlots to SVG rectangles"""
        svg_rects = ""
        
        colors = {
            'small': '#f0ad4e',      # Orange
            'medium': '#5bc0de',     # Light blue
            'large': '#5cb85c',      # Green
            'extra_large': '#d9534f' # Red
        }
        
        for ilot in ilots:
            color = colors.get(ilot['type'], '#f0ad4e')
            svg_rects += f'''<rect x="{ilot['x']}" y="{ilot['y']}" width="{ilot['width']}" height="{ilot['height']}" 
                           fill="{color}" stroke="#333" stroke-width="0.5" class="ilot" opacity="0.8"/>'''
        
        return svg_rects
    
    def embed_interactive_svg(self, svg_code: str, ilots: List[Dict] = None, height: int = 700):
        """Embed SVG with interactive pan/zoom/drag controls"""
        try:
            # Add îlots to SVG if provided
            if ilots:
                ilot_svg = self.ilots_to_svg(ilots)
                svg_code = svg_code.replace('</svg>', f'{ilot_svg}</svg>')
            
            # Create interactive HTML
            html_content = f'''
            <div style="margin-bottom: 12px;">
                {self.get_legend_html()}
            </div>
            <div id="svg-wrap" style="border: 2px solid #ddd; border-radius: 8px; overflow: hidden; background: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                {svg_code}
            </div>
            
            <script src="{self.PANZOOM_JS}"></script>
            <script>
                const svg = document.querySelector("#svg-wrap svg");
                if (svg) {{
                    const panzoomInstance = panzoom(svg, {{
                        zoomDoubleClickSpeed: 1,
                        maxZoom: 20,
                        minZoom: 0.1,
                        smoothScroll: false,
                        bounds: true,
                        boundsPadding: 0.2
                    }});
                    
                    // Reset on double click
                    svg.addEventListener('dblclick', function(e) {{
                        e.preventDefault();
                        panzoomInstance.reset();
                    }});
                    
                    // Enable drag for ilots
                    const ilots = svg.querySelectorAll('.ilot');
                    ilots.forEach(ilot => {{
                        ilot.style.cursor = 'move';
                        let isDragging = false;
                        let startX, startY, startTransform;
                        
                        ilot.addEventListener('mousedown', function(e) {{
                            e.stopPropagation();
                            isDragging = true;
                            startX = e.clientX;
                            startY = e.clientY;
                            const transform = ilot.getAttribute('transform') || '';
                            startTransform = transform;
                            ilot.style.opacity = '0.6';
                        }});
                        
                        document.addEventListener('mousemove', function(e) {{
                            if (isDragging) {{
                                const dx = e.clientX - startX;
                                const dy = e.clientY - startY;
                                ilot.setAttribute('transform', `${{startTransform}} translate(${{dx}}, ${{dy}})`);
                            }}
                        }});
                        
                        document.addEventListener('mouseup', function() {{
                            if (isDragging) {{
                                isDragging = false;
                                ilot.style.opacity = '0.8';
                            }}
                        }});
                    }});
                }}
            </script>
            
            <div style="margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 6px; font-size: 13px; color: #666;">
                <strong>Contrôles:</strong> 
                Glisser pour déplacer la vue • Molette pour zoomer • Double-clic pour réinitialiser • 
                Glisser les îlots pour les repositionner
            </div>
            '''
            
            html(html_content, height=height + 150)
            
        except Exception as e:
            st.error(f"Erreur d'affichage SVG: {str(e)}")
    
    def export_svg(self, svg_code: str, filename: str = "floor_plan.svg"):
        """Export SVG with download button"""
        try:
            st.download_button(
                label="📥 Télécharger SVG",
                data=svg_code,
                file_name=filename,
                mime="image/svg+xml",
                help="Télécharger le plan au format SVG vectoriel"
            )
        except Exception as e:
            st.error(f"Erreur d'export SVG: {str(e)}")
    
    def export_png(self, svg_code: str, filename: str = "floor_plan.png"):
        """Export PNG using cairosvg"""
        try:
            import cairosvg
            
            if st.button("📥 Télécharger PNG", help="Télécharger le plan au format PNG"):
                png_bytes = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
                st.download_button(
                    label="💾 Télécharger PNG",
                    data=png_bytes,
                    file_name=filename,
                    mime="image/png"
                )
        except ImportError:
            st.warning("cairosvg non installé - export PNG indisponible")
        except Exception as e:
            st.error(f"Erreur d'export PNG: {str(e)}")
    
    def _create_error_svg(self, error_msg: str) -> str:
        """Create error SVG"""
        return f'''
        <svg viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; background: white;">
            <rect width="500" height="300" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
            <text x="250" y="130" text-anchor="middle" fill="#dc3545" font-family="Arial, sans-serif" font-size="18" font-weight="bold">
                Erreur de rendu du plan
            </text>
            <text x="250" y="160" text-anchor="middle" fill="#6c757d" font-family="Arial, sans-serif" font-size="14">
                {error_msg}
            </text>
            <text x="250" y="180" text-anchor="middle" fill="#6c757d" font-family="Arial, sans-serif" font-size="12">
                Vérifiez le format du fichier DXF
            </text>
        </svg>
        '''