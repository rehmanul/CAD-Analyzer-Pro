"""
Final Production CAD Renderer
Complete professional CAD floor plan rendering system
"""

import ezdxf
from ezdxf import recover
import tempfile
import os
import re
from typing import Dict, List, Any, Optional
import streamlit as st
from streamlit.components.v1 import html
import math

class FinalProductionRenderer:
    """Final production-grade CAD renderer with complete functionality"""
    
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
    
    def process_dxf_to_professional_svg(self, file_content: bytes, filename: str, target_bounds: Dict[str, float] = None) -> str:
        """Convert DXF to professional SVG with proper layer colors"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load DXF document
                doc, auditor = recover.readfile(tmp_file_path)
                msp = doc.modelspace()
                
                # Extract entities with proper categorization
                entities = self._extract_and_categorize_entities(msp, target_bounds)
                
                # Create professional SVG
                svg_code = self._create_professional_svg(entities, target_bounds)
                
                return svg_code
                
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error processing DXF: {str(e)}")
            return self._create_error_svg(str(e))
    
    def _extract_and_categorize_entities(self, msp, target_bounds: Dict[str, float] = None) -> List[Dict]:
        """Extract and categorize entities from DXF"""
        entities = []
        
        for entity in msp:
            try:
                # Skip unsupported entities
                if entity.dxftype() in ['ACAD_PROXY_ENTITY', 'UNKNOWN', 'XREF']:
                    continue
                
                # Check bounds if specified
                if target_bounds and not self._is_entity_in_bounds(entity, target_bounds):
                    continue
                
                entity_data = self._process_entity_with_category(entity)
                if entity_data:
                    entities.append(entity_data)
                    
            except Exception as e:
                print(f"Error processing entity {entity.dxftype()}: {str(e)}")
                continue
                
        return entities
    
    def _process_entity_with_category(self, entity) -> Optional[Dict]:
        """Process entity and determine its category"""
        try:
            entity_type = entity.dxftype()
            
            # Skip unsupported entities
            if entity_type in ['ACAD_PROXY_ENTITY', 'UNKNOWN', 'XREF', 'BLOCK_REFERENCE']:
                return None
                
            layer = getattr(entity.dxf, 'layer', '0').upper()
            
            # Determine category based on layer
            category = 'walls'  # Default
            if any(door_layer in layer for door_layer in ['DOOR', 'PORTE', 'ENTRY', 'ENTREE']):
                category = 'doors'
            elif any(restricted_layer in layer for restricted_layer in ['BLOCK', 'RESTRICTED', 'NO_ENTREE', 'STAIR']):
                category = 'restricted'
            
            if entity_type == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                return {
                    'type': 'line',
                    'category': category,
                    'start': (float(start.x), float(start.y)),
                    'end': (float(end.x), float(end.y)),
                    'layer': layer
                }
                
            elif entity_type == 'LWPOLYLINE':
                points = list(entity.get_points())
                if len(points) >= 2:
                    return {
                        'type': 'polyline',
                        'category': category,
                        'points': [(float(p[0]), float(p[1])) for p in points],
                        'closed': entity.closed,
                        'layer': layer
                    }
                    
            elif entity_type == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                return {
                    'type': 'arc',
                    'category': 'doors' if 0.3 <= radius <= 3.0 else category,
                    'center': (float(center.x), float(center.y)),
                    'radius': float(radius),
                    'start_angle': float(entity.dxf.start_angle),
                    'end_angle': float(entity.dxf.end_angle),
                    'layer': layer
                }
                
            elif entity_type == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                return {
                    'type': 'circle',
                    'category': category,
                    'center': (float(center.x), float(center.y)),
                    'radius': float(radius),
                    'layer': layer
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
            
        return True
    
    def _point_in_bounds(self, point, bounds: Dict[str, float]) -> bool:
        """Check if point is within bounds"""
        return (bounds['min_x'] <= point.x <= bounds['max_x'] and
                bounds['min_y'] <= point.y <= bounds['max_y'])
    
    def _point_in_bounds_xy(self, point, bounds: Dict[str, float]) -> bool:
        """Check if XY point is within bounds"""
        return (bounds['min_x'] <= point[0] <= bounds['max_x'] and
                bounds['min_y'] <= point[1] <= bounds['max_y'])
    
    def _create_professional_svg(self, entities: List[Dict], bounds: Dict[str, float] = None) -> str:
        """Create professional SVG from entities"""
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
            
            # Process entities by category with proper colors
            wall_entities = [e for e in entities if e['category'] == 'walls']
            door_entities = [e for e in entities if e['category'] == 'doors']
            restricted_entities = [e for e in entities if e['category'] == 'restricted']
            
            # Add walls (gray)
            for entity in wall_entities:
                svg_lines.append(self._entity_to_svg(entity, '#888888', 1.5))
            
            # Add doors (red)
            for entity in door_entities:
                svg_lines.append(self._entity_to_svg(entity, '#d9534f', 1.0))
                
            # Add restricted areas (blue)
            for entity in restricted_entities:
                svg_lines.append(self._entity_to_svg(entity, '#337ab7', 1.0))
            
            # Close SVG
            svg_lines.extend(['</g>', '</svg>'])
            
            return '\n'.join(svg_lines)
            
        except Exception as e:
            print(f"Error creating SVG: {str(e)}")
            return self._create_error_svg(str(e))
    
    def _entity_to_svg(self, entity: Dict, color: str, stroke_width: float = 1.0) -> str:
        """Convert entity to SVG element"""
        try:
            if entity['type'] == 'line':
                start = entity['start']
                end = entity['end']
                return f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" stroke="{color}" stroke-width="{stroke_width}" />'
                
            elif entity['type'] == 'polyline':
                points = entity['points']
                points_str = ' '.join([f"{p[0]},{p[1]}" for p in points])
                fill = 'none' if not entity.get('closed', False) else f'{color}40'  # Semi-transparent fill
                return f'<polyline points="{points_str}" stroke="{color}" stroke-width="{stroke_width}" fill="{fill}" />'
                
            elif entity['type'] == 'arc':
                center = entity['center']
                radius = entity['radius']
                start_angle = entity['start_angle']
                end_angle = entity['end_angle']
                
                # Convert angles to radians
                start_rad = math.radians(start_angle)
                end_rad = math.radians(end_angle)
                
                # Calculate arc points
                x1 = center[0] + radius * math.cos(start_rad)
                y1 = center[1] + radius * math.sin(start_rad)
                x2 = center[0] + radius * math.cos(end_rad)
                y2 = center[1] + radius * math.sin(end_rad)
                
                # Create path
                large_arc = 1 if abs(end_angle - start_angle) > 180 else 0
                return f'<path d="M {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2}" stroke="{color}" stroke-width="{stroke_width}" fill="none" />'
                
            elif entity['type'] == 'circle':
                center = entity['center']
                radius = entity['radius']
                return f'<circle cx="{center[0]}" cy="{center[1]}" r="{radius}" stroke="{color}" stroke-width="{stroke_width}" fill="none" />'
                
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
                    min_x = min(min_x, center[0] - radius)
                    max_x = max(max_x, center[0] + radius)
                    min_y = min(min_y, center[1] - radius)
                    max_y = max(max_y, center[1] + radius)
            
            return {
                'min_x': min_x if min_x != float('inf') else 0,
                'max_x': max_x if max_x != float('-inf') else 100,
                'min_y': min_y if min_y != float('inf') else 0,
                'max_y': max_y if max_y != float('-inf') else 100
            }
            
        except Exception as e:
            print(f"Error calculating bounds: {str(e)}")
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
    
    def create_professional_legend(self) -> str:
        """Create professional legend"""
        return '''
        <div style="display: flex; align-items: center; gap: 24px; padding: 20px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 12px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #dee2e6;">
            <div style="font-weight: 700; color: #2c3e50; font-size: 18px; margin-right: 12px;">LÉGENDE PROFESSIONNELLE:</div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 6px; background: #337ab7; border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                <span style="font-size: 15px; color: #2c3e50; font-weight: 600;">NO ENTREE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 6px; background: #d9534f; border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                <span style="font-size: 15px; color: #2c3e50; font-weight: 600;">ENTRÉE/SORTIE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 6px; background: #888888; border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                <span style="font-size: 15px; color: #2c3e50; font-weight: 600;">MUR</span>
            </div>
        </div>
        '''
    
    def embed_final_interactive_svg(self, svg_code: str, height: int = 700):
        """Embed final interactive SVG with all controls"""
        try:
            # Create complete interactive HTML
            html_content = f'''
            <div style="margin-bottom: 20px;">
                {self.create_professional_legend()}
            </div>
            
            <div id="cad-container" style="position: relative; border: 3px solid #007bff; border-radius: 12px; overflow: hidden; background: white; box-shadow: 0 6px 20px rgba(0,0,0,0.15);">
                <div style="position: absolute; top: 10px; right: 10px; z-index: 1000; background: rgba(0,0,0,0.7); color: white; padding: 8px 12px; border-radius: 6px; font-size: 12px;">
                    <strong>CAD Analyzer Pro</strong> - Plan Professionnel
                </div>
                {svg_code}
            </div>
            
            <script src="{self.PANZOOM_JS}"></script>
            <script>
                const svg = document.querySelector("#cad-container svg");
                if (svg) {{
                    // Enhanced panzoom with professional settings
                    const panzoomInstance = panzoom(svg, {{
                        zoomDoubleClickSpeed: 1.5,
                        maxZoom: 50,
                        minZoom: 0.05,
                        smoothScroll: true,
                        bounds: true,
                        boundsPadding: 0.3,
                        filterKey: function() {{ return true; }},
                        beforeWheel: function(e) {{
                            const shouldIgnore = e.ctrlKey || e.shiftKey || e.altKey || e.metaKey;
                            return shouldIgnore;
                        }}
                    }});
                    
                    // Professional reset on double click
                    svg.addEventListener('dblclick', function(e) {{
                        e.preventDefault();
                        panzoomInstance.reset();
                        console.log('Plan réinitialisé');
                    }});
                    
                    // Add mouse wheel zoom with ctrl key
                    svg.addEventListener('wheel', function(e) {{
                        if (e.ctrlKey) {{
                            e.preventDefault();
                            const delta = e.deltaY > 0 ? 0.9 : 1.1;
                            panzoomInstance.zoomTo(e.clientX, e.clientY, delta);
                        }}
                    }});
                    
                    // Professional cursor management
                    svg.style.cursor = 'grab';
                    svg.addEventListener('mousedown', function() {{
                        svg.style.cursor = 'grabbing';
                    }});
                    svg.addEventListener('mouseup', function() {{
                        svg.style.cursor = 'grab';
                    }});
                    
                    console.log('CAD Analyzer Pro: Rendu professionnel activé');
                }}
            </script>
            
            <div style="margin-top: 16px; padding: 16px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
                <div style="font-size: 14px; color: #495057; line-height: 1.6;">
                    <strong>🎯 Contrôles Professionnels:</strong><br>
                    • <strong>Déplacer:</strong> Clic + glisser pour naviguer<br>
                    • <strong>Zoomer:</strong> Ctrl + molette ou molette simple<br>
                    • <strong>Réinitialiser:</strong> Double-clic pour revenir à la vue initiale<br>
                    • <strong>Zoom précis:</strong> Jusqu'à 50x pour les détails architecturaux
                </div>
            </div>
            '''
            
            html(html_content, height=height + 200)
            
        except Exception as e:
            st.error(f"Erreur d'affichage: {str(e)}")
    
    def create_export_options(self, svg_code: str, filename: str):
        """Create professional export options"""
        try:
            st.markdown("### 📥 Options d'Export Professionnel")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 Télécharger SVG",
                    data=svg_code,
                    file_name=f"{filename.replace('.dxf', '')}_plan_professionnel.svg",
                    mime="image/svg+xml",
                    help="Format vectoriel haute qualité pour impression et CAO"
                )
            
            with col2:
                try:
                    import cairosvg
                    if st.button("🖼️ Générer PNG", help="Convertir en image PNG haute résolution"):
                        with st.spinner("Génération PNG..."):
                            png_bytes = cairosvg.svg2png(
                                bytestring=svg_code.encode("utf-8"),
                                output_width=2400,
                                output_height=1800
                            )
                            st.download_button(
                                label="💾 Télécharger PNG",
                                data=png_bytes,
                                file_name=f"{filename.replace('.dxf', '')}_plan_professionnel.png",
                                mime="image/png"
                            )
                except ImportError:
                    st.info("cairosvg requis pour l'export PNG")
            
            with col3:
                st.button("📊 Rapport Technique", help="Générer rapport d'analyse complet")
                
        except Exception as e:
            st.error(f"Erreur d'export: {str(e)}")
    
    def _create_empty_svg(self) -> str:
        """Create empty SVG"""
        return '''
        <svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; background: white;">
            <rect width="500" height="400" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
            <text x="250" y="180" text-anchor="middle" fill="#6c757d" font-family="Arial, sans-serif" font-size="18" font-weight="bold">
                Aucune donnée de plan trouvée
            </text>
            <text x="250" y="210" text-anchor="middle" fill="#6c757d" font-family="Arial, sans-serif" font-size="14">
                Vérifiez le format du fichier DXF
            </text>
        </svg>
        '''
    
    def _create_error_svg(self, error_msg: str) -> str:
        """Create error SVG"""
        return f'''
        <svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; background: white;">
            <rect width="600" height="400" fill="#f8f9fa" stroke="#dc3545" stroke-width="3"/>
            <text x="300" y="180" text-anchor="middle" fill="#dc3545" font-family="Arial, sans-serif" font-size="20" font-weight="bold">
                Erreur de Rendu CAD
            </text>
            <text x="300" y="210" text-anchor="middle" fill="#6c757d" font-family="Arial, sans-serif" font-size="14">
                {error_msg}
            </text>
            <text x="300" y="240" text-anchor="middle" fill="#6c757d" font-family="Arial, sans-serif" font-size="12">
                Contactez le support technique si le problème persiste
            </text>
        </svg>
        '''