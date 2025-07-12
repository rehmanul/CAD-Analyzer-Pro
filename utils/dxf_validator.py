
class DXFValidator:
    """Utility to validate and diagnose DXF files"""
    
    def __init__(self):
        pass
    
    def validate_dxf_file(self, file_content: bytes, filename: str) -> dict:
        """Validate DXF file and return diagnostic information"""
        result = {
            'is_valid': False,
            'file_size': len(file_content),
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        try:
            # Check file size
            if len(file_content) < 100:
                result['errors'].append("File too small to be a valid DXF file")
                return result
            
            # Check encoding
            try:
                content_sample = file_content[:5000].decode('utf-8', errors='strict')
                result['info'].append("File uses UTF-8 encoding")
            except UnicodeDecodeError:
                try:
                    content_sample = file_content[:5000].decode('latin-1', errors='strict')
                    result['warnings'].append("File uses Latin-1 encoding")
                except UnicodeDecodeError:
                    result['errors'].append("File has encoding issues")
                    return result
            
            # Check DXF markers
            dxf_markers = ['0\nSECTION', 'HEADER', 'ENTITIES', 'ENDSEC', 'EOF']
            found_markers = []
            
            for marker in dxf_markers:
                if marker in content_sample:
                    found_markers.append(marker)
            
            if len(found_markers) >= 1:  # More lenient validation
                result['is_valid'] = True
                result['info'].append(f"Found DXF markers: {', '.join(found_markers)}")
            else:
                # Check for basic DXF patterns more broadly
                if any(pattern in content_sample for pattern in ['0\n', 'SECTION', 'HEADER', 'ENTITIES']):
                    result['is_valid'] = True
                    result['info'].append("Found basic DXF structure")
                else:
                    result['errors'].append("Missing required DXF structure markers")
            
            # Check for AutoCAD signature
            if 'AutoCAD' in content_sample:
                result['info'].append("AutoCAD signature found")
            
            # Estimate entity count
            entity_count = content_sample.count('\n0\n')
            if entity_count > 0:
                result['info'].append(f"Estimated {entity_count} entities in sample")
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def format_validation_report(self, validation_result: dict) -> str:
        """Format validation result as readable report"""
        report = f"DXF File Validation Report\n"
        report += f"File Size: {validation_result['file_size']} bytes\n"
        report += f"Valid: {'Yes' if validation_result['is_valid'] else 'No'}\n\n"
        
        if validation_result['errors']:
            report += "ERRORS:\n"
            for error in validation_result['errors']:
                report += f"  ❌ {error}\n"
            report += "\n"
        
        if validation_result['warnings']:
            report += "WARNINGS:\n"
            for warning in validation_result['warnings']:
                report += f"  ⚠️ {warning}\n"
            report += "\n"
        
        if validation_result['info']:
            report += "INFORMATION:\n"
            for info in validation_result['info']:
                report += f"  ℹ️ {info}\n"
        
        return report
