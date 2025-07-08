
#!/usr/bin/env python3
"""
CAD Analyzer Pro - Complete Application
Production-ready with all sections: Analysis, Îlot Placement, Corridor Generation, Results & Export
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import io
import base64
from typing import Dict, List, Any, Optional
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# Import production modules
from production_floor_analyzer import ProductionFloorAnalyzer
from production_ilot_system import ProductionIlotSystem
from corridor_generator import AdvancedCorridorGenerator
from production_visualizer import ProductionVisualizer

# Page configuration
st.set_page_config(
    page_title="CAD Analyzer Pro",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling - Dark theme compatible
st.markdown("""
<style>
    .section-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #60a5fa;
        margin: 1rem 0;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background: rgba(59, 130, 246, 0.1);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border-left: 4px solid #3b82f6;
        backdrop-filter: blur(10px);
    }
    .success-message {
        background: rgba(34, 197, 94, 0.2);
        border: 2px solid #22c55e;
        color: #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .warning-message {
        background: rgba(251, 146, 60, 0.2);
        border: 2px solid #fb923c;
        color: #fb923c;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    /* Enhanced sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(30, 58, 138, 0.1);
        backdrop-filter: blur(10px);
    }
    /* Better contrast for text */
    h1, h2, h3, p {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

class CADAnalyzerApp:
    def __init__(self):
        self.floor_analyzer = ProductionFloorAnalyzer()
        self.ilot_system = ProductionIlotSystem()
        self.corridor_generator = AdvancedCorridorGenerator()
        self.visualizer = ProductionVisualizer()
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'placed_ilots' not in st.session_state:
            st.session_state.placed_ilots = []
        if 'corridors' not in st.session_state:
            st.session_state.corridors = []
        if 'file_processed' not in st.session_state:
            st.session_state.file_processed = False

    def run(self):
        """Run the main application"""
        # Sidebar with settings
        with st.sidebar:
            st.markdown("## 🎛️ Settings & Controls")
            
            # Îlot Size Distribution Settings
            st.markdown("### 📊 Îlot Size Distribution")
            st.markdown("**Client Requirements:**")
            size_0_1_pct = st.slider("0-1 m² (Small - Yellow)", 5, 20, 10, key="size_0_1")
            size_1_3_pct = st.slider("1-3 m² (Medium - Orange)", 15, 35, 25, key="size_1_3") 
            size_3_5_pct = st.slider("3-5 m² (Large - Green)", 20, 40, 30, key="size_3_5")
            size_5_10_pct = st.slider("5-10 m² (XL - Purple)", 25, 50, 35, key="size_5_10")
            
            total_pct = size_0_1_pct + size_1_3_pct + size_3_5_pct + size_5_10_pct
            if total_pct != 100:
                st.error(f"Total must be 100%. Current: {total_pct}%")
            
            st.markdown("### 🛤️ Spacing Settings")
            min_spacing = st.slider("Minimum Spacing (m)", 0.5, 3.0, 1.0, key="min_spacing")
            wall_clearance = st.slider("Wall Clearance (m)", 0.3, 2.0, 0.5, key="wall_clearance")
            corridor_width = st.slider("Corridor Width (m)", 1.0, 3.0, 1.5, key="corridor_width")
            
            st.markdown("### 🎯 Optimization")
            utilization_target = st.slider("Space Utilization (%)", 50, 90, 70, key="utilization")
            
            # Store settings in session state
            st.session_state.ilot_config = {
                'size_0_1_percent': size_0_1_pct,
                'size_1_3_percent': size_1_3_pct, 
                'size_3_5_percent': size_3_5_pct,
                'size_5_10_percent': size_5_10_pct,
                'min_spacing': min_spacing,
                'wall_clearance': wall_clearance,
                'corridor_width': corridor_width,
                'utilization_target': utilization_target / 100
            }

        # Main header with better visibility
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    color: white; 
                    text-align: center; 
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h1 style="margin: 0; font-size: 2.5em;">🏨 CAD Analyzer Pro</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Professional Hotel Floor Plan Analysis with Îlot Placement & Corridor Generation
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Floor Plan Analysis", 
            "🏢 Îlot Placement", 
            "🛤️ Corridor Generation", 
            "📊 Results & Export"
        ])

        with tab1:
            self.render_analysis_tab()

        with tab2:
            self.render_ilot_placement_tab()

        with tab3:
            self.render_corridor_generation_tab()

        with tab4:
            self.render_results_export_tab()

    def render_analysis_tab(self):
        """Render floor plan analysis interface"""
        st.markdown('<div class="section-header"><h2>📋 Floor Plan Analysis</h2></div>', unsafe_allow_html=True)

        # File upload section
        st.subheader("Upload Floor Plan")
        uploaded_file = st.file_uploader(
            "Choose a floor plan file",
            type=['dxf', 'dwg', 'pdf', 'png', 'jpg', 'jpeg'],
            help="Upload your floor plan file for analysis"
        )

        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                file_content = uploaded_file.read()
                
                # Process based on file type
                if uploaded_file.name.lower().endswith(('.dxf', '.dwg')):
                    result = self.floor_analyzer.process_dxf_file(file_content, uploaded_file.name)
                elif uploaded_file.name.lower().endswith('.pdf'):
                    st.error("PDF processing not yet implemented. Please use DXF/DWG files.")
                    return
                else:
                    result = self.floor_analyzer.process_image_file(file_content, uploaded_file.name)

                if result.get('success'):
                    st.session_state.analysis_results = result
                    st.session_state.file_processed = True
                    
                    st.markdown('<div class="success-message">✅ Floor plan processed successfully!</div>', unsafe_allow_html=True)
                    
                    # Display analysis results
                    self.display_analysis_results(result)
                else:
                    st.error(f"Error processing file: {result.get('error', 'Unknown error')}")

    def display_analysis_results(self, result):
        """Display analysis results"""
        st.subheader("Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", result.get('entity_count', 0))
        with col2:
            st.metric("Walls", result.get('wall_count', 0))
        with col3:
            st.metric("Restricted Areas", result.get('restricted_count', 0))
        with col4:
            st.metric("Entrances", result.get('entrance_count', 0))

        # Bounds information
        bounds = result.get('bounds', {})
        if bounds:
            st.subheader("Floor Plan Dimensions")
            width = bounds.get('max_x', 0) - bounds.get('min_x', 0)
            height = bounds.get('max_y', 0) - bounds.get('min_y', 0)
            area = width * height
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", f"{width:.1f} m")
            with col2:
                st.metric("Height", f"{height:.1f} m")
            with col3:
                st.metric("Total Area", f"{area:.1f} m²")

        # Visualization
        if result.get('walls') or result.get('entities'):
            st.subheader("Floor Plan Visualization")
            fig = self.create_floor_plan_visualization(result)
            st.plotly_chart(fig, use_container_width=True, height=600)

    def create_floor_plan_visualization(self, result):
        """Create floor plan visualization"""
        fig = go.Figure()
        
        bounds = result.get('bounds', {})
        
        # Add walls (black lines)
        walls = result.get('walls', [])
        for wall in walls:
            if len(wall) >= 2:
                x_coords = [point[0] for point in wall] + [wall[0][0]]
                y_coords = [point[1] for point in wall] + [wall[0][1]]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Walls',
                    showlegend=False
                ))

        # Add restricted areas (light blue)
        restricted = result.get('restricted_areas', [])
        for area in restricted:
            if len(area) >= 3:
                x_coords = [point[0] for point in area] + [area[0][0]]
                y_coords = [point[1] for point in area] + [area[0][1]]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor='lightblue',
                    line=dict(color='blue', width=1),
                    name='Restricted Areas',
                    showlegend=False
                ))

        # Add entrances (red)
        entrances = result.get('entrances', [])
        for entrance in entrances:
            if len(entrance) >= 3:
                x_coords = [point[0] for point in entrance] + [entrance[0][0]]
                y_coords = [point[1] for point in entrance] + [entrance[0][1]]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor='red',
                    line=dict(color='darkred', width=1),
                    name='Entrances',
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            title="Floor Plan Analysis",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            showlegend=True,
            height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def render_ilot_placement_tab(self):
        """Render îlot placement interface"""
        st.markdown('<div class="section-header"><h2>🏢 Îlot Placement</h2></div>', unsafe_allow_html=True)

        if not st.session_state.file_processed:
            st.warning("Please complete floor plan analysis first.")
            return

        st.markdown("""
        **Îlot Size Distribution (Client Requirements):**
        - 0-1 m²: Small îlots (Yellow)
        - 1-3 m²: Medium îlots (Orange) 
        - 3-5 m²: Large îlots (Green)
        - 5-10 m²: Extra Large îlots (Purple)
        """)

        # Configuration from sidebar
        if 'ilot_config' not in st.session_state:
            st.warning("⚠️ Please configure îlot settings in the sidebar first!")
            return
        
        config = st.session_state.ilot_config
        total_percent = config['size_0_1_percent'] + config['size_1_3_percent'] + config['size_3_5_percent'] + config['size_5_10_percent']
        
        if total_percent != 100:
            st.error(f"⚠️ Size percentages must total 100%. Current: {total_percent}%. Please adjust in sidebar.")
            return
        
        # Show current configuration
        st.markdown("### 📋 Current Configuration")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Small îlots", f"{config['size_0_1_percent']}%")
        with col2:
            st.metric("Medium îlots", f"{config['size_1_3_percent']}%")
        with col3:
            st.metric("Large îlots", f"{config['size_3_5_percent']}%")
        with col4:
            st.metric("XL îlots", f"{config['size_5_10_percent']}%")

        # Placement button
        if st.button("🚀 Place Îlots", type="primary", use_container_width=True):
            self.place_ilots(config)

        # Display placement results
        if st.session_state.placed_ilots:
            self.display_ilot_results()

    def place_ilots(self, config):
        """Place îlots based on configuration"""
        with st.spinner("Placing îlots..."):
            try:
                # Load floor plan data
                result = st.session_state.analysis_results
                self.ilot_system.load_floor_plan_data(
                    walls=result.get('walls', []),
                    restricted_areas=result.get('restricted_areas', []),
                    entrances=result.get('entrances', []),
                    zones={},
                    bounds=result.get('bounds', {})
                )

                # Process placement
                placement_result = self.ilot_system.process_full_placement(config)
                
                st.session_state.placed_ilots = placement_result.get('ilots', [])
                
                if st.session_state.placed_ilots:
                    st.markdown(f'<div class="success-message">✅ Successfully placed {len(st.session_state.placed_ilots)} îlots!</div>', unsafe_allow_html=True)
                else:
                    st.warning("No îlots were placed. Please check your configuration.")
                    
            except Exception as e:
                st.error(f"Error placing îlots: {str(e)}")

    def display_ilot_results(self):
        """Display îlot placement results"""
        st.subheader("Îlot Placement Results")
        
        # Summary metrics
        total_ilots = len(st.session_state.placed_ilots)
        total_area = sum(ilot.get('area', 0) for ilot in st.session_state.placed_ilots)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Îlots", total_ilots)
        with col2:
            st.metric("Total Area", f"{total_area:.1f} m²")
        with col3:
            avg_area = total_area / total_ilots if total_ilots > 0 else 0
            st.metric("Average Area", f"{avg_area:.1f} m²")

        # Size distribution
        size_counts = {}
        for ilot in st.session_state.placed_ilots:
            size_cat = ilot.get('size_category', 'unknown')
            size_counts[size_cat] = size_counts.get(size_cat, 0) + 1

        if size_counts:
            st.subheader("Size Distribution")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("0-1 m²", size_counts.get('size_0_1', 0))
            with col2:
                st.metric("1-3 m²", size_counts.get('size_1_3', 0))
            with col3:
                st.metric("3-5 m²", size_counts.get('size_3_5', 0))
            with col4:
                st.metric("5-10 m²", size_counts.get('size_5_10', 0))

        # Visualization with îlots
        st.subheader("Floor Plan with Îlots")
        fig = self.create_complete_visualization()
        st.plotly_chart(fig, use_container_width=True, height=700)

    def render_corridor_generation_tab(self):
        """Render corridor generation interface"""
        st.markdown('<div class="section-header"><h2>🛤️ Corridor Generation</h2></div>', unsafe_allow_html=True)

        if not st.session_state.placed_ilots:
            st.warning("Please complete îlot placement first.")
            return

        st.markdown("""
        **Corridor Network Features:**
        - Mandatory corridors between facing îlot rows
        - Main corridors from entrances
        - Secondary corridors for connectivity
        - Access corridors for isolated îlots
        """)

        # Configuration from sidebar
        if 'ilot_config' not in st.session_state:
            st.warning("⚠️ Please configure settings in the sidebar first!")
            return
            
        config = st.session_state.ilot_config
        
        # Show current corridor settings
        st.markdown("### 📋 Current Corridor Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Corridor Width", f"{config['corridor_width']:.1f} m")
        with col2:
            st.metric("Main Width", f"{config['corridor_width'] * 1.5:.1f} m")
        with col3:
            generate_facing = st.checkbox("Force Between Facing Rows", value=True, key="force_facing")

        # Generation button
        if st.button("🛤️ Generate Corridors", type="primary", use_container_width=True):
            self.generate_corridors({
                'corridor_width': config['corridor_width'],
                'main_width': config['corridor_width'] * 1.5,
                'secondary_width': config['corridor_width'],
                'access_width': config['corridor_width'],
                'force_between_facing': generate_facing,
                'generate_main': True,
                'generate_secondary': True
            })

        # Display corridor results
        if st.session_state.corridors:
            self.display_corridor_results()

    def generate_corridors(self, config):
        """Generate corridors based on configuration"""
        with st.spinner("Generating corridors..."):
            try:
                # Load data into corridor generator
                result = st.session_state.analysis_results
                self.corridor_generator.load_floor_plan_data(
                    ilots=st.session_state.placed_ilots,
                    walls=result.get('walls', []),
                    restricted_areas=result.get('restricted_areas', []),
                    entrances=result.get('entrances', []),
                    bounds=result.get('bounds', {})
                )

                # Generate corridor network
                corridor_result = self.corridor_generator.generate_complete_corridor_network(config)
                
                st.session_state.corridors = corridor_result.get('corridors', [])
                
                if st.session_state.corridors:
                    st.markdown(f'<div class="success-message">✅ Generated {len(st.session_state.corridors)} corridors!</div>', unsafe_allow_html=True)
                else:
                    st.warning("No corridors were generated.")
                    
            except Exception as e:
                st.error(f"Error generating corridors: {str(e)}")

    def display_corridor_results(self):
        """Display corridor generation results"""
        st.subheader("Corridor Network Results")
        
        # Summary metrics
        total_corridors = len(st.session_state.corridors)
        total_length = sum(corridor.get('length', 0) for corridor in st.session_state.corridors)
        mandatory_count = len([c for c in st.session_state.corridors if c.get('is_mandatory', False)])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Corridors", total_corridors)
        with col2:
            st.metric("Total Length", f"{total_length:.1f} m")
        with col3:
            st.metric("Mandatory Corridors", mandatory_count)

        # Corridor types
        corridor_types = {}
        for corridor in st.session_state.corridors:
            corridor_type = corridor.get('type', 'unknown')
            corridor_types[corridor_type] = corridor_types.get(corridor_type, 0) + 1

        if corridor_types:
            st.subheader("Corridor Types")
            cols = st.columns(len(corridor_types))
            for i, (corridor_type, count) in enumerate(corridor_types.items()):
                with cols[i]:
                    st.metric(corridor_type.title(), count)

    def render_results_export_tab(self):
        """Render results and export options"""
        st.markdown('<div class="section-header"><h2>📊 Results & Export</h2></div>', unsafe_allow_html=True)

        if not st.session_state.placed_ilots:
            st.warning("Please complete îlot placement and corridor generation first.")
            return

        # Final visualization
        st.subheader("Complete Floor Plan Layout")
        fig = self.create_complete_visualization()
        st.plotly_chart(fig, use_container_width=True, height=700)

        # Project summary
        st.subheader("Project Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Îlots", len(st.session_state.placed_ilots))
        with col2:
            st.metric("Total Corridors", len(st.session_state.corridors))
        with col3:
            total_ilot_area = sum(ilot.get('area', 0) for ilot in st.session_state.placed_ilots)
            st.metric("Îlot Area", f"{total_ilot_area:.1f} m²")
        with col4:
            total_corridor_length = sum(corridor.get('length', 0) for corridor in st.session_state.corridors)
            st.metric("Corridor Length", f"{total_corridor_length:.1f} m")

        # Export options
        st.subheader("Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export as JSON", type="secondary"):
                self.export_json()
        
        with col2:
            if st.button("Export Summary", type="secondary"):
                self.export_summary()
        
        with col3:
            if st.button("Export Image", type="secondary"):
                st.info("Use the camera icon in the plot toolbar above to save the visualization as an image.")

    def create_complete_visualization(self):
        """Create complete visualization with all elements"""
        fig = go.Figure()
        
        result = st.session_state.analysis_results
        
        # Add walls (black lines)
        walls = result.get('walls', [])
        for wall in walls:
            if len(wall) >= 2:
                x_coords = [point[0] for point in wall] + [wall[0][0]]
                y_coords = [point[1] for point in wall] + [wall[0][1]]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Walls',
                    showlegend=False
                ))

        # Add restricted areas (light blue)
        restricted = result.get('restricted_areas', [])
        for area in restricted:
            if len(area) >= 3:
                x_coords = [point[0] for point in area] + [area[0][0]]
                y_coords = [point[1] for point in area] + [area[0][1]]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor='lightblue',
                    line=dict(color='blue', width=1),
                    name='Restricted Areas',
                    showlegend=False
                ))

        # Add entrances (red)
        entrances = result.get('entrances', [])
        for entrance in entrances:
            if len(entrance) >= 3:
                x_coords = [point[0] for point in entrance] + [entrance[0][0]]
                y_coords = [point[1] for point in entrance] + [entrance[0][1]]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor='red',
                    line=dict(color='darkred', width=1),
                    name='Entrances',
                    showlegend=False
                ))

        # Add îlots with size-based colors
        color_map = {
            'size_0_1': 'yellow',
            'size_1_3': 'orange', 
            'size_3_5': 'green',
            'size_5_10': 'purple'
        }

        for ilot in st.session_state.placed_ilots:
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 1)
            height = ilot.get('height', 1)
            size_cat = ilot.get('size_category', 'size_1_3')
            
            # Create rectangle for îlot
            x_coords = [x - width/2, x + width/2, x + width/2, x - width/2, x - width/2]
            y_coords = [y - height/2, y - height/2, y + height/2, y + height/2, y - height/2]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                fill='toself',
                fillcolor=color_map.get(size_cat, 'gray'),
                line=dict(color='black', width=1),
                name=f'Îlot {size_cat}',
                showlegend=False
            ))

        # Add corridors (blue lines)
        for corridor in st.session_state.corridors:
            path_points = corridor.get('path_points', [])
            if len(path_points) >= 2:
                x_coords = [point[0] for point in path_points]
                y_coords = [point[1] for point in path_points]
                
                color = 'blue' if corridor.get('is_mandatory', False) else 'cyan'
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name='Corridors',
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            title="Complete Floor Plan Layout",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            showlegend=True,
            height=700,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def export_json(self):
        """Export results as JSON"""
        export_data = {
            'analysis_results': st.session_state.analysis_results,
            'placed_ilots': st.session_state.placed_ilots,
            'corridors': st.session_state.corridors,
            'export_timestamp': datetime.now().isoformat()
        }
        
        json_string = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="Download JSON Data",
            data=json_string,
            file_name=f"floor_plan_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def export_summary(self):
        """Export summary report"""
        summary = f"""
CAD Analyzer Pro - Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FLOOR PLAN ANALYSIS:
- Total Entities: {st.session_state.analysis_results.get('entity_count', 0)}
- Walls: {st.session_state.analysis_results.get('wall_count', 0)}
- Restricted Areas: {st.session_state.analysis_results.get('restricted_count', 0)}
- Entrances: {st.session_state.analysis_results.get('entrance_count', 0)}

ÎLOT PLACEMENT:
- Total Îlots: {len(st.session_state.placed_ilots)}
- Total Area: {sum(ilot.get('area', 0) for ilot in st.session_state.placed_ilots):.1f} m²

CORRIDOR NETWORK:
- Total Corridors: {len(st.session_state.corridors)}
- Total Length: {sum(corridor.get('length', 0) for corridor in st.session_state.corridors):.1f} m
- Mandatory Corridors: {len([c for c in st.session_state.corridors if c.get('is_mandatory', False)])}

SIZE DISTRIBUTION:
"""
        
        size_counts = {}
        for ilot in st.session_state.placed_ilots:
            size_cat = ilot.get('size_category', 'unknown')
            size_counts[size_cat] = size_counts.get(size_cat, 0) + 1
        
        for size_cat, count in size_counts.items():
            summary += f"- {size_cat}: {count} îlots\n"
        
        st.download_button(
            label="Download Summary Report",
            data=summary,
            file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Initialize and run the app
if __name__ == "__main__":
    app = CADAnalyzerApp()
    app.run()
else:
    app = CADAnalyzerApp()
    app.run()
