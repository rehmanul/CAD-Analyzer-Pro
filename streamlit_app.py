
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

# Import ultra-high performance modules
from ultra_high_performance_analyzer import UltraHighPerformanceAnalyzer
from optimized_dxf_processor import OptimizedDXFProcessor
from optimized_ilot_placer import OptimizedIlotPlacer
from simple_ilot_placer import SimpleIlotPlacer
from client_expected_visualizer import ClientExpectedVisualizer
from optimized_corridor_generator import OptimizedCorridorGenerator
from professional_floor_plan_visualizer import ProfessionalFloorPlanVisualizer
from reference_style_visualizer import ReferenceStyleVisualizer
from architectural_floor_plan_visualizer import ArchitecturalFloorPlanVisualizer
from architectural_room_visualizer import ArchitecturalRoomVisualizer
from exact_reference_visualizer import ExactReferenceVisualizer
from proper_dxf_processor import ProperDXFProcessor
from fast_dxf_processor import FastDXFProcessor
from real_dxf_processor import RealDXFProcessor
from fast_architectural_visualizer import FastArchitecturalVisualizer
from empty_plan_visualizer import EmptyPlanVisualizer
from data_validator import DataValidator

# Page configuration
st.set_page_config(
    page_title="CAD Analyzer Pro",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(67, 56, 202, 0.9), rgba(99, 102, 241, 0.9)), 
                    url('https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1200');
        background-size: cover;
        background-position: center;
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 0;
        line-height: 1.6;
    }
    
    /* Modern Upload Section */
    .upload-section {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border: 2px dashed #d1d5db;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Sidebar */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(229, 231, 235, 0.5);
    }
    
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Modern Cards and Metrics */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f9fafb);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Success and Error Messages */
    .success-message {
        background: linear-gradient(145deg, #ecfdf5, #d1fae5);
        border: 2px solid #10b981;
        color: #065f46;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
    }
    
    .warning-message {
        background: linear-gradient(145deg, #fffbeb, #fef3c7);
        border: 2px solid #f59e0b;
        color: #92400e;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.1);
    }
    
    .error-message {
        background: linear-gradient(145deg, #fef2f2, #fecaca);
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(145deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #4f46e5, #4338ca);
        transform: translateY(-1px);
        box-shadow: 0 10px 25px -3px rgba(99, 102, 241, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(249, 250, 251, 0.8);
        padding: 0.5rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #6366f1, #4f46e5);
        color: white;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
    }
    
    /* Professional Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1f2937;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Modern Plotly Container */
    .plot-container {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(229, 231, 235, 0.5);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CADAnalyzerApp:
    def __init__(self):
        self.floor_analyzer = UltraHighPerformanceAnalyzer()
        self.dxf_processor = OptimizedDXFProcessor()
        self.proper_dxf_processor = ProperDXFProcessor()  # For proper architectural extraction
        self.fast_dxf_processor = FastDXFProcessor(timeout_seconds=8)  # For large files
        self.real_dxf_processor = RealDXFProcessor()  # For real architectural data
        self.fast_visualizer = FastArchitecturalVisualizer()  # For fast rendering
        self.empty_plan_visualizer = EmptyPlanVisualizer()  # For clean empty plan view
        self.ilot_placer = OptimizedIlotPlacer()
        self.simple_placer = SimpleIlotPlacer()  # Backup placer
        self.corridor_generator = OptimizedCorridorGenerator()
        self.visualizer = ClientExpectedVisualizer()
        self.professional_visualizer = ProfessionalFloorPlanVisualizer()
        self.reference_visualizer = ReferenceStyleVisualizer()  # Matches your reference images
        self.architectural_visualizer = ArchitecturalFloorPlanVisualizer()  # Exact match to your reference
        self.exact_visualizer = ExactReferenceVisualizer()  # EXACT match to your reference images
        self.room_visualizer = ArchitecturalRoomVisualizer()  # For proper room structure
        self.data_validator = DataValidator()
        
        # Initialize session state with visualization modes
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'placed_ilots' not in st.session_state:
            st.session_state.placed_ilots = []
        if 'corridors' not in st.session_state:
            st.session_state.corridors = []
        if 'file_processed' not in st.session_state:
            st.session_state.file_processed = False
        # Add visualization mode tracking
        if 'visualization_mode' not in st.session_state:
            st.session_state.visualization_mode = "none"  # none -> base -> with_ilots -> detailed

    def run(self):
        """Run the main application"""
        # Enhanced Sidebar with modern styling
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">🎛️ Settings & Controls</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Îlot Size Distribution Settings
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">📊 Îlot Size Distribution</div>
                <p style="color: #6b7280; margin-bottom: 1rem;"><strong>Client Requirements:</strong></p>
            </div>
            """, unsafe_allow_html=True)
            size_0_1_pct = st.slider("0-1 m² (Small - Yellow)", 5, 20, 10, key="size_0_1")
            size_1_3_pct = st.slider("1-3 m² (Medium - Orange)", 15, 35, 25, key="size_1_3") 
            size_3_5_pct = st.slider("3-5 m² (Large - Green)", 20, 40, 30, key="size_3_5")
            size_5_10_pct = st.slider("5-10 m² (XL - Purple)", 25, 50, 35, key="size_5_10")
            
            total_pct = size_0_1_pct + size_1_3_pct + size_3_5_pct + size_5_10_pct
            if total_pct != 100:
                st.error(f"Total must be 100%. Current: {total_pct}%")
            
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">🛤️ Spacing Settings</div>
            </div>
            """, unsafe_allow_html=True)
            min_spacing = st.slider("Minimum Spacing (m)", 0.5, 3.0, 1.0, key="min_spacing")
            wall_clearance = st.slider("Wall Clearance (m)", 0.3, 2.0, 0.5, key="wall_clearance")
            corridor_width = st.slider("Corridor Width (m)", 1.0, 3.0, 1.5, key="corridor_width")
            
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">🎯 Optimization</div>
            </div>
            """, unsafe_allow_html=True)
            utilization_target = st.slider("Space Utilization (%)", 50, 90, 70, key="utilization")
            
            # Display data source validation info
            if st.session_state.analysis_results and st.session_state.placed_ilots:
                self.data_validator.display_data_source_info(
                    st.session_state.analysis_results, 
                    st.session_state.placed_ilots
                )
            
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

        # Modern Hero Section
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🏨 CAD Analyzer Pro</h1>
            <p class="hero-subtitle">
                Professional Floor Plan Analysis & Hotel Layout Optimization
                <br>Advanced îlot placement • Intelligent corridor generation • Export capabilities
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


        
        uploaded_file = st.file_uploader(
            "Choose a floor plan file",
            type=['dxf', 'dwg', 'pdf', 'png', 'jpg', 'jpeg'],
            help="Supported formats: DXF, DWG, PDF, PNG, JPG • Max size: 50MB"
        )

        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    file_content = uploaded_file.read()
                    
                    # Use real DXF processor for authentic architectural data
                    if uploaded_file.name.lower().endswith('.dxf'):
                        file_size_mb = len(file_content) / (1024 * 1024)
                        st.info(f"Processing DXF file ({file_size_mb:.1f}MB) - extracting real architectural data...")
                        result = self.real_dxf_processor.process_dxf_file(file_content, uploaded_file.name)
                    else:
                        # Use ultra-high performance analyzer for other files
                        result = self.floor_analyzer.process_file_ultra_fast(file_content, uploaded_file.name)
                    
                    if not result.get('success'):
                        st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
                        return
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    return

                if result.get('success'):
                    st.session_state.analysis_results = result
                    st.session_state.file_processed = True
                    # Set visualization mode to show base floor plan (Image 1 style)
                    st.session_state.visualization_mode = "base"
                    
                    st.markdown('<div class="success-message">✅ Floor plan processed successfully! Showing empty floor plan (walls, entrances, restricted areas).</div>', unsafe_allow_html=True)
                    
                    # Display analysis results
                    self.display_analysis_results(result)
                else:
                    st.error(f"Error processing file: {result.get('error', 'Unknown error')}")

    def display_analysis_results(self, result):
        """Display analysis results"""
        st.subheader("Analysis Results")
        

        
        # Analysis Metrics
        st.markdown("### 📊 Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", result.get('entity_count', 0))
        with col2:
            st.metric("Walls", len(result.get('walls', [])))
        with col3:
            st.metric("Restricted Areas", len(result.get('restricted_areas', [])))
        with col4:
            st.metric("Entrances", len(result.get('entrances', [])))

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
                st.metric("Height", f"{height:.1f} m")
            with col3:
                st.metric("Total Area", f"{area:.1f} m²")

        # Visualization with status indicator
        if result.get('walls') or result.get('entities'):
            # Show current visualization mode
            mode = st.session_state.get('visualization_mode', 'base')
            mode_messages = {
                'base': '📋 Stage 1: Empty Floor Plan (walls, entrances, restricted areas)',
                'with_ilots': '🏢 Stage 2: Floor Plan with Îlots Placed',
                'detailed': '🛤️ Stage 3: Complete Layout with Corridors'
            }
            
            st.markdown(f"""
            <div class="success-message">
                <strong>Current View:</strong> {mode_messages.get(mode, 'Floor Plan')}
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Floor Plan Visualization")
            fig = self.create_architectural_floor_plan_visualization(result)
            st.plotly_chart(fig, use_container_width=True, height=600)

    def create_architectural_floor_plan_visualization(self, result):
        """Create authentic architectural floor plan visualization from real DXF data"""
        mode = st.session_state.get('visualization_mode', 'base')
        
        # Always use fast architectural visualizer for authentic DXF data
        if mode == 'base':
            # Clean empty floor plan with authentic architectural data
            fig = self.fast_visualizer.create_fast_floor_plan(result)
        elif mode == 'with_ilots':
            # Floor plan with red rectangular îlots
            ilots = st.session_state.get('placed_ilots', [])
            fig = self.fast_visualizer.create_floor_plan_with_ilots(result, ilots)
        elif mode == 'detailed':
            # Complete layout with corridors
            ilots = st.session_state.get('placed_ilots', [])
            corridors = st.session_state.get('corridors', [])
            fig = self.fast_visualizer.create_complete_floor_plan(result, ilots, corridors)
        else:
            # Default to authentic architectural visualization
            fig = self.fast_visualizer.create_fast_floor_plan(result)
        
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
        """Place îlots using reliable placement algorithm"""
        with st.spinner("Placing îlots with guaranteed placement algorithm..."):
            try:
                # Get analysis results
                result = st.session_state.analysis_results
                
                # Calculate target count from bounds and configuration
                bounds = result.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
                area = (bounds['max_x'] - bounds['min_x']) * (bounds['max_y'] - bounds['min_y'])
                target_count = max(8, min(int(area / 12), 40))  # More conservative target
                
                # Try optimized placer first
                placed_ilots = []
                try:
                    placed_ilots = self.ilot_placer.generate_optimal_ilot_placement(
                        analysis_data=result,
                        target_count=target_count
                    )
                except Exception:
                    pass
                
                # Use simple placer if optimized fails or returns no results
                if not placed_ilots:
                    placed_ilots = self.simple_placer.place_ilots_guaranteed(
                        analysis_data=result,
                        target_count=target_count
                    )
                
                st.session_state.placed_ilots = placed_ilots
                
                if placed_ilots:
                    # Update visualization mode to show îlots (Image 2 style)
                    st.session_state.visualization_mode = "with_ilots"
                    st.markdown(f'<div class="success-message">✅ Successfully placed {len(placed_ilots)} îlots with {sum(ilot.get("area", 0) for ilot in placed_ilots):.1f} m² total area! Visualization updated to show îlots.</div>', unsafe_allow_html=True)
                else:
                    st.error("Unable to place îlots. Please check the floor plan has sufficient open space.")
                    
            except Exception as e:
                st.error(f"Error placing îlots: {str(e)}")
                # No fallback - show error message
                st.error("Unable to place îlots. Please check that the uploaded file contains valid floor plan data.")

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

        # Show updated visualization with îlots
        st.subheader("Updated Floor Plan with Îlots")
        
        # Display the updated visualization based on current mode
        if st.session_state.analysis_results:
            fig = self.create_architectural_floor_plan_visualization(st.session_state.analysis_results)
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
            
            # Show final visualization with corridors
            st.subheader("Complete Floor Plan with Corridors")
            if st.session_state.analysis_results:
                fig = self.create_architectural_floor_plan_visualization(st.session_state.analysis_results)
                st.plotly_chart(fig, use_container_width=True, height=700)

    def generate_corridors(self, config):
        """Generate corridors based on configuration"""
        with st.spinner("Generating corridors..."):
            try:
                # Generate optimized corridor network
                result = st.session_state.analysis_results
                
                st.session_state.corridors = self.corridor_generator.generate_optimized_corridors(
                    analysis_data=result,
                    ilots=st.session_state.placed_ilots
                )
                
                if st.session_state.corridors:
                    # Update visualization mode to show complete layout (Image 3 style)
                    st.session_state.visualization_mode = "detailed"
                    st.markdown(f'<div class="success-message">✅ Generated {len(st.session_state.corridors)} corridors! Visualization updated to show complete layout.</div>', unsafe_allow_html=True)
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

        # Final visualization with controls
        st.subheader("Complete Floor Plan Layout")
        
        # Visualization controls
        col1, col2 = st.columns([3, 1])
        with col2:
            view_3d = st.checkbox("3D View", key="export_3d_toggle")
            professional_style = st.checkbox("Professional Style", value=True, key="export_prof_style")
        
        fig = self.create_complete_visualization(use_professional=professional_style, show_3d=view_3d)
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

    def create_complete_visualization(self, use_professional=True, show_3d=False):
        """Create complete visualization matching your reference images"""
        result = st.session_state.analysis_results
        ilots = st.session_state.placed_ilots
        corridors = st.session_state.corridors
        
        if show_3d:
            # Use professional visualizer for 3D view
            fig = self.professional_visualizer.create_professional_floor_plan(
                analysis_data=result,
                ilots=ilots,
                corridors=corridors,
                show_3d=True
            )
        else:
            # Use reference style visualizer to match your exact images
            if corridors:
                # Image 3: With corridors
                fig = self.reference_visualizer.create_floor_plan_with_corridors(
                    analysis_data=result,
                    ilots=ilots,
                    corridors=corridors
                )
            elif ilots:
                # Image 2: With îlots
                fig = self.reference_visualizer.create_floor_plan_with_ilots(
                    analysis_data=result,
                    ilots=ilots
                )
            else:
                # Image 1: Empty plan
                fig = self.reference_visualizer.create_empty_floor_plan(result)
        
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
