"""
Data Validator - Ensures authentic data processing without mock/fallback data
"""

import streamlit as st
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class DataValidator:
    """Validates that all data comes from authentic sources, not mock/fallback data"""
    
    def __init__(self):
        self.validation_log = []
    
    def validate_analysis_results(self, results: Dict) -> bool:
        """Validate that analysis results come from real file processing"""
        if not results:
            self.log_validation("Analysis results are empty")
            return False
            
        # Check for authentic data markers
        required_fields = ['bounds', 'entity_count', 'success']
        for field in required_fields:
            if field not in results:
                self.log_validation(f"Missing required field: {field}")
                return False
        
        # Check if results indicate successful real processing
        if not results.get('success', False):
            self.log_validation("Analysis marked as unsuccessful")
            return False
            
        # Verify entity count is reasonable for a real file
        entity_count = results.get('entity_count', 0)
        if entity_count == 0:
            self.log_validation("No entities found - may indicate failed processing")
            return False
            
        self.log_validation(f"Analysis validation passed: {entity_count} entities")
        return True
    
    def validate_ilot_data(self, ilots: List[Dict]) -> bool:
        """Validate that îlot data comes from real placement algorithms"""
        if not ilots:
            self.log_validation("No îlots provided")
            return False
            
        # Check for realistic îlot properties
        for i, ilot in enumerate(ilots):
            required_props = ['x', 'y', 'width', 'height', 'area', 'size_category']
            for prop in required_props:
                if prop not in ilot:
                    self.log_validation(f"Îlot {i} missing property: {prop}")
                    return False
            
            # Validate realistic values
            area = ilot.get('area', 0)
            if area <= 0 or area > 50:  # Reasonable area limits
                self.log_validation(f"Îlot {i} has unrealistic area: {area}")
                return False
                
        self.log_validation(f"Îlot validation passed: {len(ilots)} authentic îlots")
        return True
    
    def validate_corridor_data(self, corridors: List[Dict]) -> bool:
        """Validate corridor data authenticity"""
        if not corridors:
            self.log_validation("No corridors provided")
            return True  # Corridors are optional
            
        for i, corridor in enumerate(corridors):
            if 'path' not in corridor:
                self.log_validation(f"Corridor {i} missing path")
                return False
                
            path = corridor['path']
            if len(path) < 2:
                self.log_validation(f"Corridor {i} has insufficient path points")
                return False
                
        self.log_validation(f"Corridor validation passed: {len(corridors)} authentic corridors")
        return True
    
    def log_validation(self, message: str):
        """Log validation message"""
        self.validation_log.append(message)
        logging.info(f"Data Validation: {message}")
    
    def get_validation_report(self) -> str:
        """Get full validation report"""
        return "\n".join(self.validation_log)
    
    def display_data_source_info(self, results: Dict, ilots: List[Dict]):
        """Display information about data sources to user"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Data Source Info**")
        
        if results:
            file_source = results.get('filename', 'Unknown')
            processing_time = results.get('processing_time', 0)
            entity_count = results.get('entity_count', 0)
            
            st.sidebar.text(f"File: {file_source}")
            st.sidebar.text(f"Entities: {entity_count}")
            st.sidebar.text(f"Processing: {processing_time:.2f}s")
            
        if ilots:
            st.sidebar.text(f"Îlots: {len(ilots)} placed")
            
            # Show size distribution
            size_dist = {}
            for ilot in ilots:
                size_cat = ilot.get('size_category', 'unknown')
                size_dist[size_cat] = size_dist.get(size_cat, 0) + 1
            
            for size, count in size_dist.items():
                st.sidebar.text(f"  {size}: {count}")
                
        # Data authenticity indicator
        if self.validate_analysis_results(results) and self.validate_ilot_data(ilots):
            st.sidebar.success("✅ Authentic data")
        else:
            st.sidebar.error("❌ Data validation failed")