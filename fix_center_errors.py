#!/usr/bin/env python3
"""
Fix all 'center[' access patterns in utils files
This script fixes the Application error: 'center' by replacing all center[0], center[1] 
with proper coordinate unpacking
"""

import os
import re

def fix_center_access(file_path):
    """Fix center[0], center[1] access patterns in a file"""
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Count how many fixes we'll make
    center_pattern = r'center\[(\d+)\]'
    matches = re.findall(center_pattern, content)
    if not matches:
        print(f"  No center[] patterns found in {file_path}")
        return False
    
    print(f"  Found {len(matches)} center[] patterns to fix")
    
    # Strategy: Replace center[0] and center[1] with center_x, center_y variables
    # First, find lines that have center[0] or center[1] and add unpacking before them
    
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has center[0] or center[1]
        if 'center[0]' in line or 'center[1]' in line:
            # Get the indentation of this line
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # Check if we already have unpacking before this line
            if i > 0 and 'center_x, center_y = center' in lines[i-1]:
                # Already have unpacking, just replace the access
                line = line.replace('center[0]', 'center_x')
                line = line.replace('center[1]', 'center_y')
                new_lines.append(line)
            else:
                # Add unpacking line before this line
                new_lines.append(indent_str + 'center_x, center_y = center')
                # Replace the access in current line
                line = line.replace('center[0]', 'center_x')
                line = line.replace('center[1]', 'center_y')
                new_lines.append(line)
        else:
            new_lines.append(line)
        
        i += 1
    
    # Write the fixed content back
    fixed_content = '\n'.join(new_lines)
    
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"  Fixed {file_path}")
    return True

def main():
    """Fix all center access errors in utils directory"""
    utils_dir = 'utils'
    files_with_center = [
        'architectural_room_visualizer.py',
        'client_expected_visualizer.py', 
        'empty_plan_visualizer.py',
        'exact_reference_visualizer.py',
        'fast_architectural_visualizer.py',
        'final_production_renderer.py',
        'optimized_corridor_generator.py',
        'phase3_pixel_perfect_visualizer.py',
        'phase4_export_integration.py',
        'real_dxf_processor.py',
        'reference_floor_plan_visualizer.py',
        'simple_svg_renderer.py',
        'smart_ilot_placer.py',
        'ultra_high_performance_analyzer.py'
    ]
    
    fixed_count = 0
    for filename in files_with_center:
        file_path = os.path.join(utils_dir, filename)
        if os.path.exists(file_path):
            if fix_center_access(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {fixed_count} files with center[] access patterns")
    print("All 'center' coordinate access errors should now be resolved!")

if __name__ == '__main__':
    main()