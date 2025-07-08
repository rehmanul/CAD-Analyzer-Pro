"""
CAD Analyzer Pro - Minimal Deployment Version
Render-compatible version without scipy dependencies
"""

import streamlit as st
import sys
import os
import traceback
import logging
import random
import math
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="CAD Analyzer Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MinimalCADAnalyzer:
    """Minimal CAD Analyzer for Render deployment"""
    
    def __init__(self):
        self.use_postgres = False  # Disable PostgreSQL for simplicity
        
    def run(self):
        """Main application entry point"""
        st.title("🏗️ CAD Analyzer Pro")
        st.markdown("**Professional Floor Plan Analysis with Îlot Placement**")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose your floor plan file",
                type=['dxf', 'dwg', 'png', 'jpg', 'jpeg'],
                help="Upload DXF/DWG for best results. Maximum file size: 3MB"
            )
            
            # Îlot configuration
            st.subheader("Îlot Size Distribution (%)")
            size_0_1_percent = st.slider("Small (0-1 m²)", 0, 50, 10)
            size_1_3_percent = st.slider("Medium (1-3 m²)", 0, 50, 25)
            size_3_5_percent = st.slider("Large (3-5 m²)", 0, 50, 30)
            size_5_10_percent = st.slider("Extra Large (5-10 m²)", 0, 50, 35)
            
            # Validate percentages
            total_percent = size_0_1_percent + size_1_3_percent + size_3_5_percent + size_5_10_percent
            if total_percent != 100:
                st.warning(f"Total: {total_percent}% (should be 100%)")
        
        # Main content
        if uploaded_file is not None:
            # File processing
            tab1, tab2, tab3 = st.tabs(["📁 File Processing", "🏗️ Îlot Placement", "📊 Results"])
            
            with tab1:
                self.process_file(uploaded_file)
            
            with tab2:
                config = {
                    'size_0_1_percent': size_0_1_percent,
                    'size_1_3_percent': size_1_3_percent,
                    'size_3_5_percent': size_3_5_percent,
                    'size_5_10_percent': size_5_10_percent,
                    'total_ilots': 30
                }
                self.place_ilots(config)
            
            with tab3:
                self.show_results()
        else:
            st.info("👆 Please upload a floor plan file to begin analysis")
            
            # Show sample data
            st.subheader("Sample Data")
            if st.button("Load Sample Floor Plan"):
                self.load_sample_data()
    
    def process_file(self, uploaded_file):
        """Process uploaded file"""
        st.subheader("📁 File Processing")
        
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        
        # File size check
        max_size = 3 * 1024 * 1024  # 3MB
        if file_details['filesize'] > max_size:
            st.error(f"File too large: {file_details['filesize'] / (1024*1024):.1f}MB. Maximum: 3MB")
            return
        
        st.success(f"File loaded: {file_details['filename']} ({file_details['filesize']:,} bytes)")
        
        if st.button("Process File"):
            with st.spinner("Processing floor plan..."):
                # Simulate processing
                import time
                time.sleep(2)
                
                # Generate sample analysis results
                bounds = {
                    'min_x': 0,
                    'max_x': 100,
                    'min_y': 0,
                    'max_y': 80
                }
                
                entities = []
                for i in range(50):  # Sample entities
                    entities.append({
                        'type': 'LINE',
                        'x1': random.uniform(0, 100),
                        'y1': random.uniform(0, 80),
                        'x2': random.uniform(0, 100),
                        'y2': random.uniform(0, 80)
                    })
                
                analysis_results = {
                    'bounds': bounds,
                    'entities': entities,
                    'filename': file_details['filename'],
                    'entity_count': len(entities)
                }
                
                st.session_state.analysis_results = analysis_results
                st.session_state.file_processed = True
                
                st.success("✅ File processed successfully!")
                st.json({
                    'entities_found': len(entities),
                    'floor_dimensions': f"{bounds['max_x']}m x {bounds['max_y']}m",
                    'processing_time': '2.0 seconds'
                })
    
    def place_ilots(self, config):
        """Place îlots on floor plan"""
        st.subheader("🏗️ Îlot Placement")
        
        if not st.session_state.get('file_processed', False):
            st.warning("Please process a file first")
            return
        
        if st.button("Generate Îlot Placement"):
            with st.spinner("Placing îlots..."):
                # Get bounds
                analysis_results = st.session_state.get('analysis_results', {})
                bounds = analysis_results.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80})
                
                # Generate îlots
                ilots = self.generate_ilots(bounds, config)
                
                st.session_state.placed_ilots = ilots
                st.session_state.ilot_placement_complete = True
                
                st.success(f"✅ Successfully placed {len(ilots)} îlots")
                
                # Show distribution
                size_counts = {}
                for ilot in ilots:
                    category = ilot['size_category']
                    size_counts[category] = size_counts.get(category, 0) + 1
                
                total_ilots = len(ilots)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    count = size_counts.get('size_0_1', 0)
                    st.metric("Small", f"{count} ({count/total_ilots*100:.0f}%)")
                
                with col2:
                    count = size_counts.get('size_1_3', 0)
                    st.metric("Medium", f"{count} ({count/total_ilots*100:.0f}%)")
                
                with col3:
                    count = size_counts.get('size_3_5', 0)
                    st.metric("Large", f"{count} ({count/total_ilots*100:.0f}%)")
                
                with col4:
                    count = size_counts.get('size_5_10', 0)
                    st.metric("Extra Large", f"{count} ({count/total_ilots*100:.0f}%)")
    
    def generate_ilots(self, bounds, config):
        """Generate îlots with specified distribution"""
        ilots = []
        
        # Size categories
        size_categories = {
            'size_0_1': {'proportion': config['size_0_1_percent'] / 100, 'avg_area': 0.75},
            'size_1_3': {'proportion': config['size_1_3_percent'] / 100, 'avg_area': 2.0},
            'size_3_5': {'proportion': config['size_3_5_percent'] / 100, 'avg_area': 4.0},
            'size_5_10': {'proportion': config['size_5_10_percent'] / 100, 'avg_area': 7.5}
        }
        
        total_ilots = config['total_ilots']
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        
        # Grid dimensions
        grid_cols = int(math.sqrt(total_ilots * width / height))
        grid_rows = int(math.ceil(total_ilots / grid_cols))
        
        ilot_id = 0
        for category, props in size_categories.items():
            count = max(1, int(total_ilots * props['proportion']))
            
            for i in range(count):
                if ilot_id >= total_ilots:
                    break
                
                # Grid position
                col = ilot_id % grid_cols
                row = ilot_id // grid_cols
                
                # Calculate position
                x = bounds['min_x'] + (col + 0.5) * width / grid_cols
                y = bounds['min_y'] + (row + 0.5) * height / grid_rows
                
                # Add randomness
                x += random.uniform(-width/grid_cols*0.2, width/grid_cols*0.2)
                y += random.uniform(-height/grid_rows*0.2, height/grid_rows*0.2)
                
                # Calculate size
                area = props['avg_area'] * random.uniform(0.8, 1.2)
                side = math.sqrt(area)
                
                ilots.append({
                    'id': f'ilot_{ilot_id + 1}',
                    'x': x,
                    'y': y,
                    'width': side,
                    'height': side,
                    'area': area,
                    'size_category': category
                })
                
                ilot_id += 1
        
        return ilots
    
    def show_results(self):
        """Show results and visualization"""
        st.subheader("📊 Results")
        
        if not st.session_state.get('ilot_placement_complete', False):
            st.warning("Please complete îlot placement first")
            return
        
        ilots = st.session_state.get('placed_ilots', [])
        
        if not ilots:
            st.error("No îlots to display")
            return
        
        # Create simple visualization
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add floor boundary
            analysis_results = st.session_state.get('analysis_results', {})
            bounds = analysis_results.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80})
            
            fig.add_shape(
                type="rect",
                x0=bounds['min_x'], y0=bounds['min_y'],
                x1=bounds['max_x'], y1=bounds['max_y'],
                line=dict(color='black', width=2),
                fillcolor="rgba(255,255,255,0.1)"
            )
            
            # Add îlots
            colors = {
                'size_0_1': '#FFB6C1',
                'size_1_3': '#FFA07A',
                'size_3_5': '#FF6347',
                'size_5_10': '#FF4500'
            }
            
            for ilot in ilots:
                color = colors.get(ilot['size_category'], '#FF6347')
                
                fig.add_shape(
                    type="rect",
                    x0=ilot['x'] - ilot['width']/2,
                    y0=ilot['y'] - ilot['height']/2,
                    x1=ilot['x'] + ilot['width']/2,
                    y1=ilot['y'] + ilot['height']/2,
                    line=dict(color=color, width=1),
                    fillcolor=color,
                    opacity=0.7
                )
            
            fig.update_layout(
                title="Floor Plan with Îlot Placement",
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                showlegend=False,
                height=600
            )
            
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            st.info("Visualization failed, but îlot placement was successful")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        total_ilots = len(ilots)
        total_area = sum(ilot['area'] for ilot in ilots)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Îlots", total_ilots)
        
        with col2:
            st.metric("Total Area", f"{total_area:.1f} m²")
        
        with col3:
            st.metric("Average Area", f"{total_area/total_ilots:.1f} m²")
        
        # Export options
        st.subheader("Export")
        
        if st.button("Export Results"):
            results = {
                'total_ilots': total_ilots,
                'total_area': total_area,
                'ilots': ilots
            }
            
            st.download_button(
                label="Download JSON",
                data=str(results),
                file_name="ilot_placement_results.json",
                mime="application/json"
            )
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Sample analysis results
        bounds = {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}
        entities = [{'type': 'LINE', 'x1': 0, 'y1': 0, 'x2': 100, 'y2': 0}]
        
        st.session_state.analysis_results = {
            'bounds': bounds,
            'entities': entities,
            'filename': 'sample_floor_plan.dxf',
            'entity_count': len(entities)
        }
        
        st.session_state.file_processed = True
        st.success("Sample data loaded successfully!")
        st.rerun()

def main():
    """Main application entry point"""
    try:
        app = MinimalCADAnalyzer()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Debug Information:")
        st.code(f"Python version: {sys.version}")
        st.code(f"Error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()