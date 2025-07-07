import numpy as np
import random

def place_ilots_simple(bounds, config):
    """Simple, working ilot placement"""
    if not bounds:
        return []
    
    width = bounds['max_x'] - bounds['min_x']
    height = bounds['max_y'] - bounds['min_y']
    area = width * height * 0.6  # 60% utilization
    
    ilots = []
    categories = [
        ('size_0_1', 0.75, config.get('size_0_1_percent', 10)),
        ('size_1_3', 2.0, config.get('size_1_3_percent', 25)),
        ('size_3_5', 4.0, config.get('size_3_5_percent', 30)),
        ('size_5_10', 7.5, config.get('size_5_10_percent', 35))
    ]
    
    for category, avg_size, percentage in categories:
        count = max(1, int(area * percentage / 100 / avg_size))
        
        for i in range(count):
            x = bounds['min_x'] + random.uniform(width * 0.1, width * 0.9)
            y = bounds['min_y'] + random.uniform(height * 0.1, height * 0.9)
            size = avg_size * random.uniform(0.8, 1.2)
            side = np.sqrt(size)
            
            ilots.append({
                'id': f'ilot_{len(ilots)}',
                'x': x, 'y': y,
                'width': side, 'height': side,
                'area': size, 'size_category': category
            })
    
    return ilots

def generate_corridors_simple(ilots):
    """Simple corridor generation"""
    corridors = []
    
    # Group by Y coordinate (rows)
    rows = {}
    for ilot in ilots:
        y_key = int(ilot['y'] / 20) * 20
        if y_key not in rows:
            rows[y_key] = []
        rows[y_key].append(ilot)
    
    # Create corridors between rows
    row_keys = sorted(rows.keys())
    for i in range(len(row_keys) - 1):
        if len(rows[row_keys[i]]) > 1 and len(rows[row_keys[i+1]]) > 1:
            y_mid = (row_keys[i] + row_keys[i+1]) / 2
            x_start = min(ilot['x'] for ilot in rows[row_keys[i]] + rows[row_keys[i+1]])
            x_end = max(ilot['x'] for ilot in rows[row_keys[i]] + rows[row_keys[i+1]])
            
            corridors.append({
                'id': f'corridor_{len(corridors)}',
                'start_point': (x_start, y_mid),
                'end_point': (x_end, y_mid),
                'width': 1.5,
                'path_points': [(x_start, y_mid), (x_end, y_mid)]
            })
    
    return corridors