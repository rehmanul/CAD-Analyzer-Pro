#!/usr/bin/env python3
"""
Comprehensive KeyError Detection Script
Finds ALL potential KeyError sources in the CAD processing pipeline
"""

import os
import re
import ast

def find_dict_access_patterns(file_path):
    """Find all dictionary access patterns that could cause KeyErrors"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Patterns that could cause KeyError
        patterns = [
            r"(\w+)\['([^']+)'\]",  # dict['key'] patterns
            r"(\w+)\[\"([^\"]+)\"\]",  # dict["key"] patterns
        ]
        
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for var_name, key_name in matches:
                    # Skip if there's a .get() call nearby
                    if '.get(' not in line and f'{var_name}.get(' not in line:
                        issues.append({
                            'file': file_path,
                            'line': i,
                            'code': line.strip(),
                            'variable': var_name,
                            'key': key_name,
                            'pattern': f"{var_name}['{key_name}']"
                        })
        
        return issues
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    """Find all potential KeyError sources"""
    all_issues = []
    
    # Scan utils directory
    for root, dirs, files in os.walk('utils'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                issues = find_dict_access_patterns(file_path)
                all_issues.extend(issues)
    
    # Group by file
    by_file = {}
    for issue in all_issues:
        file_name = issue['file']
        if file_name not in by_file:
            by_file[file_name] = []
        by_file[file_name].append(issue)
    
    print("COMPREHENSIVE KEYERROR ANALYSIS")
    print("=" * 50)
    
    # Focus on the most critical files
    critical_keys = ['center', 'type', 'coordinates', 'radius', 'start', 'end']
    
    for file_path, issues in by_file.items():
        critical_issues = [i for i in issues if i['key'] in critical_keys]
        if critical_issues:
            print(f"\n🚨 CRITICAL FILE: {file_path}")
            print(f"   Found {len(critical_issues)} critical KeyError risks")
            
            for issue in critical_issues:
                print(f"   Line {issue['line']}: {issue['pattern']}")
                print(f"      Code: {issue['code']}")
    
    print(f"\nTOTAL CRITICAL ISSUES FOUND: {sum(len([i for i in issues if i['key'] in critical_keys]) for issues in by_file.values())}")

if __name__ == '__main__':
    main()