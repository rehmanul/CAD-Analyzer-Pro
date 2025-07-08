"""
Large File Processor - Handles files over 3MB with memory optimization
Progressive processing for files that exceed Render memory limits
"""

import gc
import logging
from typing import Dict, List, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)

class LargeFileProcessor:
    """Processor for large files with memory optimization"""
    
    def __init__(self):
        self.max_safe_size = 3 * 1024 * 1024  # 3MB
        self.max_large_size = 10 * 1024 * 1024  # 10MB absolute limit
        self.chunk_size = 1000  # Process in chunks
        
    def can_process_file(self, file_size: int) -> bool:
        """Check if file can be processed"""
        return file_size <= self.max_large_size
    
    def process_large_file_safe(self, file_content: bytes, filename: str) -> Optional[Dict]:
        """Process large file with memory safety"""
        try:
            st.info("🔄 Processing large file with memory optimization...")
            
            # Force garbage collection before processing
            gc.collect()
            
            # Process in chunks to prevent memory overflow
            if filename.lower().endswith('.dxf'):
                return self._process_large_dxf(file_content, filename)
            else:
                return self._process_large_general(file_content, filename)
                
        except Exception as e:
            logger.error(f"Large file processing failed: {str(e)}")
            st.error(f"Large file processing failed: {str(e)}")
            return None
    
    def _process_large_dxf(self, file_content: bytes, filename: str) -> Optional[Dict]:
        """Process large DXF file with optimization"""
        try:
            import ezdxf
            
            # Create temporary file for processing
            with st.spinner("Reading DXF file..."):
                doc = ezdxf.from_bytes(file_content)
                
            # Extract entities in chunks
            entities = []
            entity_count = 0
            max_entities = 1000  # Limit for memory
            
            with st.spinner("Extracting entities..."):
                for space in [doc.modelspace(), doc.paperspace()]:
                    for entity in space:
                        if entity_count >= max_entities:
                            break
                        
                        # Process only essential entity types
                        if entity.dxftype() in ['LINE', 'POLYLINE', 'LWPOLYLINE', 'CIRCLE', 'ARC']:
                            entities.append({
                                'type': entity.dxftype(),
                                'layer': entity.dxf.layer,
                                'color': getattr(entity.dxf, 'color', 256),
                                'coords': self._extract_coords(entity)
                            })
                            entity_count += 1
                    
                    if entity_count >= max_entities:
                        break
            
            # Clean up
            del doc
            gc.collect()
            
            # Calculate bounds
            bounds = self._calculate_bounds_fast(entities)
            
            st.success(f"✅ Processed {len(entities)} entities from large file")
            if entity_count >= max_entities:
                st.warning(f"⚠️ File truncated to {max_entities} entities for memory efficiency")
            
            return {
                'entities': entities,
                'bounds': bounds,
                'filename': filename,
                'entity_count': len(entities),
                'truncated': entity_count >= max_entities
            }
            
        except Exception as e:
            logger.error(f"DXF processing failed: {str(e)}")
            return None
    
    def _process_large_general(self, file_content: bytes, filename: str) -> Optional[Dict]:
        """Process large non-DXF file"""
        try:
            st.warning("Large non-DXF files may not process optimally")
            
            # Basic processing with memory limits
            return {
                'entities': [],
                'bounds': {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80},
                'filename': filename,
                'entity_count': 0,
                'truncated': True
            }
            
        except Exception as e:
            logger.error(f"General file processing failed: {str(e)}")
            return None
    
    def _extract_coords(self, entity) -> List:
        """Extract coordinates from entity safely"""
        try:
            if entity.dxftype() == 'LINE':
                return [
                    [entity.dxf.start.x, entity.dxf.start.y],
                    [entity.dxf.end.x, entity.dxf.end.y]
                ]
            elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
                return [[p.x, p.y] for p in entity.points()]
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                return [[center.x, center.y], [radius, radius]]
            else:
                return []
        except Exception:
            return []
    
    def _calculate_bounds_fast(self, entities: List[Dict]) -> Dict:
        """Calculate bounds quickly"""
        try:
            if not entities:
                return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}
            
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for entity in entities:
                coords = entity.get('coords', [])
                for coord in coords:
                    if len(coord) >= 2:
                        x, y = coord[0], coord[1]
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
            
            # Add padding
            padding = 10
            return {
                'min_x': min_x - padding,
                'max_x': max_x + padding,
                'min_y': min_y - padding,
                'max_y': max_y + padding
            }
            
        except Exception as e:
            logger.error(f"Bounds calculation failed: {str(e)}")
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}

# Global instance
large_file_processor = LargeFileProcessor()