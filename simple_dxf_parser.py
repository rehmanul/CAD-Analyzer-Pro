import ezdxf
import re

def parse_dxf_simple(file_content, filename):
    """Simple, working DXF parser"""
    try:
        # Try ezdxf first
        import io
        doc = ezdxf.read(io.BytesIO(file_content))
        
        walls = []
        for entity in doc.modelspace():
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                walls.append([(start.x, start.y), (end.x, end.y)])
            elif entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0], p[1]) for p in entity.get_points()]
                if len(points) > 1:
                    walls.append(points)
        
        if walls:
            all_points = [p for wall in walls for p in wall]
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            bounds = {
                'min_x': min(x_coords), 'min_y': min(y_coords),
                'max_x': max(x_coords), 'max_y': max(y_coords)
            }
            return {'success': True, 'walls': walls, 'bounds': bounds, 'method': 'ezdxf'}
    except:
        pass
    
    # Manual parsing
    try:
        content = file_content.decode('utf-8', errors='ignore')
        walls = []
        
        # Find LINE entities
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            if lines[i].strip() == 'LINE':
                x1 = y1 = x2 = y2 = 0
                j = i + 1
                while j < len(lines) and j < i + 30:
                    if lines[j].strip() == '10' and j+1 < len(lines):
                        x1 = float(lines[j+1].strip())
                    elif lines[j].strip() == '20' and j+1 < len(lines):
                        y1 = float(lines[j+1].strip())
                    elif lines[j].strip() == '11' and j+1 < len(lines):
                        x2 = float(lines[j+1].strip())
                    elif lines[j].strip() == '21' and j+1 < len(lines):
                        y2 = float(lines[j+1].strip())
                        walls.append([(x1, y1), (x2, y2)])
                        break
                    j += 1
            i += 1
        
        if walls:
            all_points = [p for wall in walls for p in wall]
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            bounds = {
                'min_x': min(x_coords), 'min_y': min(y_coords),
                'max_x': max(x_coords), 'max_y': max(y_coords)
            }
            return {'success': True, 'walls': walls, 'bounds': bounds, 'method': 'manual'}
    except:
        pass
    
    return {'success': False, 'error': 'Could not parse DXF'}