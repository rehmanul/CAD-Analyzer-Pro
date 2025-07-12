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
from smart_floor_plan_detector import SmartFloorPlanDetector
from floor_plan_extractor import FloorPlanExtractor
from targeted_floor_plan_extractor import TargetedFloorPlanExtractor
from svg_floor_plan_renderer import SVGFloorPlanRenderer
from simple_svg_renderer import SimpleSVGRenderer
from production_svg_renderer import ProductionSVGRenderer
from final_production_renderer import FinalProductionRenderer
from fast_architectural_visualizer import FastArchitecturalVisualizer
from empty_plan_visualizer import EmptyPlanVisualizer
from data_validator import DataValidator
from reference_floor_plan_visualizer import ReferenceFloorPlanVisualizer
from smart_ilot_placer import SmartIlotPlacer
from advanced_3d_renderer import Advanced3DRenderer
from webgl_3d_renderer import WebGL3DRenderer

# Import Enhanced Phase Components
try:
    from phase1_integration_layer import Phase1IntegrationLayer
    phase1_processor = Phase1IntegrationLayer()
except ImportError:
    phase1_processor = None

try:
    from dxf_validator import DXFValidator
    dxf_validator = DXFValidator()
except ImportError:
    dxf_validator = None

try:
    from phase2_integration_layer import phase2_processor, Phase2Configuration
except ImportError:
    phase2_processor = None
    Phase2Configuration = None

try:
    from phase3_integration_layer import phase3_processor, Phase3Configuration
    from phase4_integration_layer import phase4_processor, Phase4Configuration  
except ImportError:
    phase3_processor = None
    phase4_processor = None
    Phase3Configuration = None
    Phase4Configuration = None

# Page configuration
st.set_page_config(
    page_title="CAD Analyzer Pro",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Theme-Aware CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styling with Theme Support */
    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Light Theme Variables */
    :root {
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-light: #9ca3af;
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --border-color: #e5e7eb;
    }
    
    /* Dark Theme Variables */
    .stApp[data-theme="dark"] {
        --text-primary: #ffffff;
        --text-secondary: #d1d5db;
        --text-light: #9ca3af;
        --bg-primary: #1f2937;
        --bg-secondary: #374151;
        --border-color: #4b5563;
    }

    /* ULTIMATE DARK THEME FIX - Maximum Specificity */
    .stApp[data-theme="dark"], 
    .stApp[data-theme="dark"] *, 
    .stApp[data-theme="dark"] *::before, 
    .stApp[data-theme="dark"] *::after {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    
    /* Special exceptions for elements that should remain dark text */
    .stApp[data-theme="dark"] .stAlert,
    .stApp[data-theme="dark"] .stAlert *,
    .stApp[data-theme="dark"] .stException,
    .stApp[data-theme="dark"] .stException *,
    .stApp[data-theme="dark"] .stSuccess,
    .stApp[data-theme="dark"] .stSuccess *,
    .stApp[data-theme="dark"] .stWarning,
    .stApp[data-theme="dark"] .stWarning *,
    .stApp[data-theme="dark"] .stError,
    .stApp[data-theme="dark"] .stError *,
    .stApp[data-theme="dark"] .stInfo,
    .stApp[data-theme="dark"] .stInfo * {
        color: #000000 !important;
    }
    
    /* Input fields need special handling */
    .stApp[data-theme="dark"] input[type="text"],
    .stApp[data-theme="dark"] input[type="number"],
    .stApp[data-theme="dark"] input[type="email"],
    .stApp[data-theme="dark"] input[type="password"],
    .stApp[data-theme="dark"] textarea,
    .stApp[data-theme="dark"] select {
        color: #ffffff !important;
        background-color: #374151 !important;
        border-color: #6b7280 !important;
    }
    
    /* Plot elements should remain default */
    .stApp[data-theme="dark"] .js-plotly-plot,
    .stApp[data-theme="dark"] .js-plotly-plot *,
    .stApp[data-theme="dark"] .plotly,
    .stApp[data-theme="dark"] .plotly * {
        color: unset !important;
        fill: unset !important;
    }

    /* Modern Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(67, 56, 202, 0.9), rgba(99, 102, 241, 0.9)), 
                    url('https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1200');
        background-size: cover;
        background-position: center;
        color: white !important;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        text-align: center;
    }

    .hero-section * {
        color: white !important;
    }
    
    /* Professional section headers */
    .section-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    .section-header h2 {
        color: white !important;
        margin: 0;
        font-weight: 600;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        border-left: 4px solid #34d399;
    }
    
    /* Enhanced button styling */
    .stButton button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 3D visualization controls */
    .viz-controls {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* WebGL container styling */
    .webgl-container {
        border: 2px solid #4f46e5;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(79, 70, 229, 0.2);
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
        color: white !important;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 0;
        line-height: 1.6;
        color: white !important;
    }

    /* Modern Upload Section - NO BACKGROUND CHANGES */
    .upload-section {
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

    /* Enhanced Sidebar - NO BACKGROUND CHANGES */
    .sidebar-section {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }

    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    /* Modern Cards and Metrics */
    .metric-card {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: transform 0.2s ease;
        color: var(--text-primary);
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
    }

    /* Success and Error Messages */
    .success-message {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.2));
        border: 2px solid #10b981;
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
    }

    .warning-message {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.2));
        border: 2px solid #f59e0b;
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.1);
    }

    .error-message {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.2));
        border: 2px solid #ef4444;
        color: var(--text-primary);
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


    def initialize_session_state(self):
        """Initialize all session state variables to prevent SessionInfo errors"""
        # Core application state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'placed_ilots' not in st.session_state:
            st.session_state.placed_ilots = []
        if 'corridors' not in st.session_state:
            st.session_state.corridors = []
        if 'file_processed' not in st.session_state:
            st.session_state.file_processed = False
        
        # Visualization mode tracking
        if 'visualization_mode' not in st.session_state:
            st.session_state.visualization_mode = "none"  # none -> base -> with_ilots -> detailed
        
        # Configuration state
        if 'ilot_config' not in st.session_state:
            st.session_state.ilot_config = {
                'size_0_1_percent': 10,
                'size_1_3_percent': 25, 
                'size_3_5_percent': 30,
                'size_5_10_percent': 35,
                'min_spacing': 1.0,
                'wall_clearance': 0.5,
                'corridor_width': 1.5,
                'utilization_target': 0.7
            }
        
        # Processing state
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'last_error' not in st.session_state:
            st.session_state.last_error = None


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
        color: var(--text-primary) !important;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Fix Streamlit default text colors with better specificity */
    .stMarkdown, .stText, p, span, div, label {
        color: var(--text-primary) !important;
    }

    /* Fix metric labels and values */
    [data-testid="metric-container"] {
        color: var(--text-primary) !important;
    }

    [data-testid="metric-container"] > div {
        color: var(--text-primary) !important;
    }
    
    /* Ensure sidebar text visibility */
    .stSidebar .stMarkdown,
    .stSidebar .stText,
    .stSidebar p,
    .stSidebar span,
    .stSidebar div,
    .stSidebar label {
        color: var(--text-primary) !important;
    }
    
    /* Input field labels */
    .stNumberInput label,
    .stSlider label,
    .stCheckbox label,
    .stSelectbox label {
        color: var(--text-primary) !important;
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
        # Initialize session state first thing
        self.initialize_session_state()
        self.floor_analyzer = UltraHighPerformanceAnalyzer()
        self.dxf_processor = OptimizedDXFProcessor()
        self.proper_dxf_processor = ProperDXFProcessor()  # For proper architectural extraction
        self.fast_dxf_processor = FastDXFProcessor(timeout_seconds=8)  # For large files
        self.real_dxf_processor = RealDXFProcessor()  # For real architectural data
        self.smart_floor_detector = SmartFloorPlanDetector()  # For floor plan detection from multi-view DXF
        self.floor_plan_extractor = FloorPlanExtractor()  # For precise floor plan extraction
        self.targeted_extractor = TargetedFloorPlanExtractor()  # For extracting specific floor plan section
        self.svg_renderer = SVGFloorPlanRenderer()  # For high-quality SVG rendering
        self.simple_svg_renderer = SimpleSVGRenderer()  # For simple SVG rendering
        self.production_svg_renderer = ProductionSVGRenderer()  # For production SVG rendering
        self.final_renderer = FinalProductionRenderer()  # Final production renderer
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
        self.reference_floor_plan_visualizer = ReferenceFloorPlanVisualizer()  # Clean reference style
        self.smart_ilot_placer = SmartIlotPlacer()  # Intelligent îlot placement

        # Initialize all session state variables at startup
        self.initialize_session_state()

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

            # Update settings in session state
            st.session_state.ilot_config.update({
                'size_0_1_percent': size_0_1_pct,
                'size_1_3_percent': size_1_3_pct, 
                'size_3_5_percent': size_3_5_pct,
                'size_5_10_percent': size_5_10_pct,
                'min_spacing': min_spacing,
                'wall_clearance': wall_clearance,
                'corridor_width': corridor_width,
                'utilization_target': utilization_target / 100
            })

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
            help="Supported formats: DXF, DWG, PDF, PNG, JPG • Max size: 200MB"
        )

        if uploaded_file is not None:
            # Validate file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 200MB")
                return

            # Validate file extension
            file_ext = uploaded_file.name.lower().split('.')[-1] if '.' in uploaded_file.name else ''
            if file_ext not in ['dxf', 'dwg', 'pdf', 'png', 'jpg', 'jpeg']:
                st.error(f"Unsupported file format: .{file_ext}. Please upload DXF, DWG, PDF, PNG, or JPG files.")
                return

            with st.spinner(f"Processing {uploaded_file.name} ({file_size_mb:.1f}MB)..."):
                try:
                    # Reset uploaded file pointer
                    uploaded_file.seek(0)
                    file_content = uploaded_file.read()
                    
                    # Validate file content
                    if not file_content:
                        st.error("File appears to be empty or corrupted")
                        return
                    
                    # Additional DXF file validation
                    if file_ext == 'dxf':
                        try:
                            if dxf_validator:
                                validation_result = dxf_validator.validate_dxf_file(file_content, uploaded_file.name)
                                
                                if not validation_result['is_valid']:
                                    st.error("❌ Invalid DXF file detected!")
                                    
                                    with st.expander("🔍 Detailed Validation Report"):
                                        report = dxf_validator.format_validation_report(validation_result)
                                        st.text(report)
                                    
                                    st.info("💡 Try these solutions:")
                                    st.info("• Re-export the DXF from your CAD software")
                                    st.info("• Ensure the file was not corrupted during transfer")
                                    st.info("• Check that the file is actually a DXF file (not renamed)")
                                    return
                                else:
                                    st.success("✅ DXF file validation passed")
                                    with st.expander("📊 File Information"):
                                        report = dxf_validator.format_validation_report(validation_result)
                                        st.text(report)
                            else:
                                st.info("📋 Processing DXF file without validation (validator not available)")
                        except Exception as validation_error:
                            st.warning(f"⚠️ DXF validation failed: {str(validation_error)}")
                            st.info("📋 Continuing with standard DXF processing...")

                    # Enhanced Processing Options
                    st.markdown("### 🚀 Enhanced Processing Mode")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        use_phase1_enhanced = st.checkbox(
                            "Phase 1 Enhanced", 
                            value=False,
                            help="Advanced CAD parsing with smart floor plan detection and geometric element recognition"
                        )
                    with col2:
                        use_phase2_advanced = st.checkbox(
                            "Phase 2 Advanced", 
                            value=False,
                            help="Advanced îlot placement with multiple algorithms and intelligent corridor generation"
                        )
                    with col3:
                        use_phase3_visualization = st.checkbox(
                            "Phase 3 Visualization", 
                            value=False,
                            help="Pixel-perfect visualization with exact reference matching and multi-stage rendering"
                        )
                    with col4:
                        use_phase4_export = st.checkbox(
                            "Phase 4 Export", 
                            value=False,
                            help="Comprehensive export with multiple formats and system integration"
                        )
                    
                    if use_phase1_enhanced:
                        st.info("🔬 Using Phase 1 Enhanced Processing: Advanced CAD Parser + Smart Floor Plan Detector + Geometric Element Recognizer")
                        
                        # Use Phase 1 enhanced processing
                        result = phase1_processor.process_cad_file_enhanced(file_content, uploaded_file.name)
                        
                        if result and result.get('processing_metadata', {}).get('phase1_complete'):
                            st.success("✅ Phase 1 Enhanced Processing Complete!")
                            
                            # Display enhanced processing metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                            with col2:
                                st.metric("Elements Detected", result.get('element_counts', {}).get('walls', 0) + 
                                         result.get('element_counts', {}).get('doors', 0) + 
                                         result.get('element_counts', {}).get('windows', 0))
                            with col3:
                                st.metric("Quality Score", f"{result.get('quality_metrics', {}).get('overall_quality_score', 0):.1f}%")
                            
                            # Display enhancement details
                            enhancement_level = result.get('performance_metrics', {}).get('enhancement_level', 'Unknown')
                            detection_confidence = result.get('processing_metadata', {}).get('detection_confidence', 0)
                            st.info(f"Enhancement: {enhancement_level} | Detection Confidence: {detection_confidence:.1f}")
                            
                            # Phase 2 Advanced Processing (if enabled)
                            if use_phase2_advanced:
                                st.markdown("---")
                                st.info("🚀 Starting Phase 2 Advanced Processing: Advanced Îlot Placement + Intelligent Corridor Generation")
                                
                                # Configure Phase 2 with sidebar settings
                                phase2_config = Phase2Configuration(
                                    ilot_size_distribution=st.session_state.get('ilot_size_distribution', {
                                        'Small (0-1m²)': 30,
                                        'Medium (1-3m²)': 40, 
                                        'Large (3-5m²)': 20,
                                        'Extra Large (5-10m²)': 10
                                    }),
                                    ilot_min_spacing=st.session_state.get('min_spacing', 1.0),
                                    ilot_wall_clearance=0.5,
                                    ilot_utilization_target=0.7,
                                    ilot_placement_strategy="hybrid",
                                    corridor_main_width=1.5,
                                    corridor_secondary_width=1.2,
                                    corridor_access_width=1.0,
                                    corridor_pathfinding_algorithm="a_star",
                                    corridor_optimize_traffic=True,
                                    enable_iterative_optimization=True,
                                    max_optimization_iterations=3,
                                    quality_threshold=0.8
                                )
                                
                                # Run Phase 2 processing
                                phase2_result = phase2_processor.process_ilot_placement_and_corridors(result, phase2_config)
                                
                                if phase2_result.get('success') and phase2_result.get('phase2_complete'):
                                    st.success("✅ Phase 2 Advanced Processing Complete!")
                                    
                                    # Display Phase 2 metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Îlots Placed", phase2_result.get('ilot_metrics', {}).get('total_ilots_placed', 0))
                                    with col2:
                                        st.metric("Corridors Created", phase2_result.get('corridor_metrics', {}).get('total_corridors', 0))
                                    with col3:
                                        st.metric("Space Utilization", f"{phase2_result.get('combined_metrics', {}).get('overall_utilization', 0)*100:.1f}%")
                                    with col4:
                                        st.metric("Quality Score", f"{phase2_result.get('overall_quality_score', 0)*100:.1f}%")
                                    
                                    # Display processing details
                                    processing_info = phase2_result.get('processing_info', {})
                                    config_summary = phase2_result.get('configuration_summary', {})
                                    st.info(f"Strategy: {config_summary.get('ilot_placement_strategy', 'Unknown')} | "
                                           f"Algorithm: {config_summary.get('corridor_pathfinding_algorithm', 'Unknown')} | "
                                           f"Processing Time: {processing_info.get('total_processing_time', 0):.2f}s")
                                    
                                    # Update result with Phase 2 data
                                    result.update(phase2_result)
                                    
                                    # Phase 3 Pixel-Perfect Visualization (if enabled)
                                    if use_phase3_visualization:
                                        st.markdown("---")
                                        st.info("🎨 Starting Phase 3 Pixel-Perfect Visualization: Multi-Stage Rendering + Reference Matching")
                                        
                                        phase3_config = Phase3Configuration(
                                            visualization_stage="corridors_added",
                                            visualization_style="reference_match",
                                            canvas_size=(1800, 1800),
                                            show_labels=True,
                                            enable_multi_stage=True,
                                            export_ready=True
                                        )
                                        
                                        phase3_result = phase3_processor.create_advanced_visualizations(result, phase3_config)
                                        
                                        if phase3_result.get('success') and phase3_result.get('phase3_complete'):
                                            st.success("✅ Phase 3 Pixel-Perfect Visualization Complete!")
                                            
                                            # Display Phase 3 metrics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Visualizations Created", phase3_result.get('metadata', {}).get('total_visualizations', 0))
                                            with col2:
                                                st.metric("Canvas Resolution", phase3_result.get('metadata', {}).get('canvas_resolution', '1800x1800'))
                                            with col3:
                                                st.metric("Style Applied", phase3_result.get('configuration_used', {}).get('style', 'reference_match'))
                                            
                                            # Update result with Phase 3 data
                                            result['phase3_visualizations'] = phase3_result.get('visualizations', {})
                                            result['phase3_metadata'] = phase3_result.get('metadata', {})
                                            
                                            # Phase 4 Export & Integration (if enabled)
                                            if use_phase4_export:
                                                st.markdown("---")
                                                st.info("📦 Starting Phase 4 Export & Integration: Multi-Format Export + System Integration")
                                                
                                                phase4_config = Phase4Configuration(
                                                    export_formats=['json', 'csv', 'html', 'png', 'zip_package'],
                                                    summary_level="comprehensive",
                                                    include_visualizations=True,
                                                    include_metrics=True,
                                                    file_prefix="cad_analysis",
                                                    create_zip_package=True
                                                )
                                                
                                                phase4_result = phase4_processor.process_export_and_integration(
                                                    result, 
                                                    phase3_result.get('visualizations', {}), 
                                                    phase4_config
                                                )
                                                
                                                if phase4_result.get('success') and phase4_result.get('phase4_complete'):
                                                    st.success("✅ Phase 4 Export & Integration Complete!")
                                                    
                                                    # Display Phase 4 metrics
                                                    export_package = phase4_result.get('export_package', {})
                                                    processing_info = export_package.get('processing_info', {})
                                                    
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Formats Generated", processing_info.get('formats_generated', 0))
                                                    with col2:
                                                        st.metric("Export Time", f"{processing_info.get('export_time', 0):.2f}s")
                                                    with col3:
                                                        st.metric("Download Ready", "Yes" if phase4_result.get('download_ready') else "No")
                                                    
                                                    # Create download buttons
                                                    downloads = phase4_processor.create_streamlit_downloads(export_package)
                                                    
                                                    if downloads:
                                                        st.markdown("### 📥 Download Export Files")
                                                        download_cols = st.columns(len(downloads))
                                                        
                                                        for i, (format_name, download_data) in enumerate(downloads.items()):
                                                            with download_cols[i % len(download_cols)]:
                                                                if download_data['encoding'] == 'base64':
                                                                    # For binary files
                                                                    import base64
                                                                    file_data = base64.b64decode(download_data['data'])
                                                                else:
                                                                    # For text files
                                                                    file_data = download_data['data']
                                                                
                                                                st.download_button(
                                                                    label=f"📄 {format_name.upper()}",
                                                                    data=file_data,
                                                                    file_name=download_data['filename'],
                                                                    mime=download_data['mime'],
                                                                    help=f"Download {format_name} format ({download_data['size_mb']:.1f}MB)"
                                                                )
                                                    
                                                    # Update result with Phase 4 data
                                                    result['phase4_export_package'] = export_package
                                                    result['phase4_downloads'] = downloads
                                                else:
                                                    st.warning("Phase 4 Export & Integration encountered issues.")
                                                    st.info(f"Reason: {phase4_result.get('error', 'Unknown error')}")
                                        else:
                                            st.warning("Phase 3 Pixel-Perfect Visualization encountered issues.")
                                            st.info(f"Reason: {phase3_result.get('error', 'Unknown error')}")
                                else:
                                    st.warning("Phase 2 Advanced Processing encountered issues. Continuing with Phase 1 results.")
                                    st.info(f"Reason: {phase2_result.get('reason', 'Unknown error')}")
                        else:
                            st.error("Phase 1 Enhanced Processing failed. Falling back to standard processing.")
                            # Fallback to standard processing
                            if uploaded_file.name.lower().endswith('.dxf'):
                                st.info("Processing DXF file with standard processors...")
                                processors = [
                                    ("Fast Processor", self.fast_dxf_processor),
                                    ("Real Processor", self.real_dxf_processor),
                                    ("Optimized Processor", self.dxf_processor)
                                ]
                                
                                result = None
                                for processor_name, processor in processors:
                                    try:
                                        if hasattr(processor, 'process_dxf_file'):
                                            result = processor.process_dxf_file(file_content, uploaded_file.name)
                                        else:
                                            result = processor.process_file_ultra_fast(file_content, uploaded_file.name)
                                        
                                        if result and result.get('success'):
                                            st.success(f"Successfully processed with {processor_name}")
                                            break
                                            
                                    except Exception as e:
                                        st.warning(f"{processor_name} failed: {str(e)}")
                                        continue
                            else:
                                # Use ultra-high performance analyzer for other files
                                result = self.floor_analyzer.process_file_ultra_fast(file_content, uploaded_file.name)
                    else:
                        # Standard processing with fallback processors
                        st.info("Using standard processing with multiple fallback processors...")
                        
                        # Process based on file type with improved error handling
                        if uploaded_file.name.lower().endswith('.dxf'):
                            st.info("Processing DXF file - extracting floor plan...")
                            
                            # Try multiple processors for better success rate
                            processors = [
                                ("Real DXF Processor", self.real_dxf_processor),
                                ("Proper DXF Processor", self.proper_dxf_processor),
                                ("Fast DXF Processor", self.fast_dxf_processor),
                                ("Optimized DXF Processor", self.dxf_processor)
                            ]
                            
                            result = None
                            last_error = None
                            
                            for processor_name, processor in processors:
                                try:
                                    st.info(f"Trying {processor_name}...")
                                    
                                    if hasattr(processor, 'process_dxf_file'):
                                        result = processor.process_dxf_file(file_content, uploaded_file.name)
                                    else:
                                        result = processor.process_file_ultra_fast(file_content, uploaded_file.name)
                                    
                                    if result and result.get('success'):
                                        st.success(f"✅ Successfully processed with {processor_name}")
                                        break
                                        
                                except Exception as e:
                                    last_error = str(e)
                                    st.warning(f"❌ {processor_name} failed: {str(e)}")
                                    continue
                            
                            if not result or not result.get('success'):
                                st.error("❌ All DXF processors failed.")
                                if last_error:
                                    st.error(f"Last error: {last_error}")
                                st.info("💡 Troubleshooting tips:")
                                st.info("• Ensure the DXF file is not corrupted")
                                st.info("• Try re-exporting the DXF from your CAD software")
                                st.info("• Check that the file contains actual geometric data")
                                return
                        else:
                            # Use ultra-high performance analyzer for other files
                            try:
                                result = self.floor_analyzer.process_file_ultra_fast(file_content, uploaded_file.name)
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)}")
                                return

                    if not result or not result.get('success'):
                        st.error(f"Processing failed: {result.get('error', 'Unknown error') if result else 'No result'}")
                        return

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please try uploading the file again or check the file format.")
                    return

                # Success handling
                st.session_state.analysis_results = result
                st.session_state.file_processed = True
                # Set visualization mode to show base floor plan (Image 1 style)
                st.session_state.visualization_mode = "base"

                st.markdown('<div class="success-message">✅ Floor plan processed successfully! Clean architectural visualization ready.</div>', unsafe_allow_html=True)

                # Display analysis results with proper visualization
                self.display_analysis_results(result)

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
            st.plotly_chart(fig, use_container_width=True, height=1800)

    def create_architectural_floor_plan_visualization(self, result):
        """Create advanced floor plan visualization with 3D rendering capabilities"""
        mode = st.session_state.get('visualization_mode', 'base')
        
        # Get current tab context for unique keys
        import inspect
        frame = inspect.currentframe()
        caller_name = frame.f_back.f_code.co_name
        unique_prefix = f"{caller_name}_{mode}"
        
        # Add 3D visualization option
        col1, col2 = st.columns([3, 1])
        with col2:
            view_3d = st.checkbox("🎛️ 3D View", value=False, key=f"{unique_prefix}_3d_view_toggle")
            if view_3d:
                render_quality = st.selectbox(
                    "Quality",
                    ["Standard", "High", "Ultra"],
                    index=1,
                    key=f"{unique_prefix}_3d_quality"
                )
                show_wireframe = st.checkbox("Wireframe", value=False, key=f"{unique_prefix}_wireframe_toggle")
                enable_shadows = st.checkbox("Shadows", value=True, key=f"{unique_prefix}_shadows_toggle")

        try:
            # Check if 3D view is enabled
            if view_3d:
                # Use advanced 3D renderer
                ilots = st.session_state.get('placed_ilots', [])
                corridors = st.session_state.get('corridors', [])
                
                # Create 3D visualization
                advanced_3d = Advanced3DRenderer()
                fig = advanced_3d.create_advanced_3d_visualization(
                    result, ilots, corridors, 
                    show_wireframe=show_wireframe,
                    enable_shadows=enable_shadows
                )
                
                # Add 3D-specific styling
                fig.update_layout(
                    title={
                        'text': '🎛️ Advanced 3D Floor Plan Visualization',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20}
                    },
                    height=1800
                )
                
                return fig
            else:
                # Use standard 2D visualization
                if mode == 'base':
                    # Clean empty floor plan with gray walls, blue restricted areas, red entrances
                    fig = self.architectural_visualizer.create_empty_floor_plan(result)
                elif mode == 'with_ilots':
                    # Floor plan with red rectangular îlots
                    ilots = st.session_state.get('placed_ilots', [])
                    fig = self.architectural_visualizer.create_floor_plan_with_ilots(result, ilots)
                elif mode == 'detailed':
                    # Complete layout with corridors
                    ilots = st.session_state.get('placed_ilots', [])
                    corridors = st.session_state.get('corridors', [])
                    fig = self.architectural_visualizer.create_complete_floor_plan(result, ilots, corridors)
                else:
                    # Default to clean visualization
                    fig = self.architectural_visualizer.create_empty_floor_plan(result)

                # Ensure the visualization has proper styling
                fig.update_layout(
                    plot_bgcolor='#f7fafc',  # Light gray background
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        x=1.02, y=1,
                        bgcolor='white',
                        bordercolor='#4a5568',
                        borderwidth=1
                    )
                )

                return fig

        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            # Create a simple fallback that works
            fig = go.Figure()

            # Add basic walls if available
            walls = result.get('walls', [])
            entities = result.get('entities', [])

            if walls or entities:
                # Add walls as gray lines
                for wall in (walls if walls else entities[:50]):  # Limit to first 50 entities
                    if isinstance(wall, list) and len(wall) >= 2:
                        x_coords = [point[0] for point in wall]
                        y_coords = [point[1] for point in wall]
                        fig.add_trace(go.Scatter(
                            x=x_coords, y=y_coords,
                            mode='lines',
                            line=dict(color='#4a5568', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

            # Set basic layout
            bounds = result.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
            padding = 5

            fig.update_layout(
                title="Floor Plan Analysis",
                xaxis=dict(
                    range=[bounds['min_x'] - padding, bounds['max_x'] + padding],
                    showgrid=False, showticklabels=False,
                    scaleanchor="y", scaleratio=1
                ),
                yaxis=dict(
                    range=[bounds['min_y'] - padding, bounds['max_y'] + padding],
                    showgrid=False, showticklabels=False
                ),
                plot_bgcolor='#f7fafc',
                paper_bgcolor='white'
            )

        return fig

    def _validate_dxf_content(self, file_content: bytes) -> bool:
        """Validate DXF file content"""
        try:
            # Check for DXF header
            content_str = file_content.decode('utf-8', errors='ignore')[:1000]
            dxf_indicators = ['0\nSECTION', 'HEADER', 'ENTITIES', 'AutoCAD']
            
            return any(indicator in content_str for indicator in dxf_indicators)
        except:
            return False

    def create_fallback_visualization(self, result):
        """Fallback visualization using simple plotly traces"""
        try:
            fig = go.Figure()

            # Get bounds
            bounds = result.get('bounds', {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})

            # Add walls from entities if available
            entities = result.get('entities', [])
            walls = result.get('walls', [])

            wall_count = 0

            # Process walls first
            for wall in walls:
                if isinstance(wall, list) and len(wall) >= 2:
                    x_coords = [p[0] for p in wall if len(p) >= 2]
                    y_coords = [p[1] for p in wall if len(p) >= 2]

                    if len(x_coords) >= 2:
                        fig.add_trace(go.Scatter(
                            x=x_coords, y=y_coords,
                            mode='lines',
                            line=dict(color='#666666', width=2),
                            name='Walls' if wall_count == 0 else None,
                            showlegend=(wall_count == 0)
                        ))
                        wall_count += 1

            # Process entities if no walls
            if wall_count == 0:
                for entity in entities[:200]:  # Limit for performance
                    if isinstance(entity, dict):
                        points = entity.get('points', [])
                        if points and len(points) >= 2:
                            x_coords = [p[0] for p in points if len(p) >= 2]
                            y_coords = [p[1] for p in points if len(p) >= 2]

                            if len(x_coords) >= 2:
                                fig.add_trace(go.Scatter(
                                    x=x_coords, y=y_coords,
                                    mode='lines',
                                    line=dict(color='#666666', width=1),
                                    showlegend=False
                                ))
                                wall_count += 1

            # Set layout
            fig.update_layout(
                title="Floor Plan (Fallback View)",
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(),
                plot_bgcolor='#F5F5F5',
                paper_bgcolor='white',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True, height=600)
            st.success(f"Fallback visualization rendered with {wall_count} elements")

        except Exception as e:
            st.error(f"Fallback visualization failed: {str(e)}")
            st.info("Try uploading a different DXF file or check the file format.")

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
            with st.spinner("Placing îlots intelligently..."):
                # Use smart îlot placer for intelligent placement
                analysis_results = st.session_state.analysis_results
                placed_ilots = self.smart_ilot_placer.place_ilots_smart(analysis_results, config)

                if placed_ilots:
                    st.session_state.placed_ilots = placed_ilots
                    st.session_state.visualization_mode = "with_ilots"

                    # Calculate placement statistics
                    stats = self.smart_ilot_placer.calculate_placement_stats(placed_ilots)

                    st.markdown('<div class="success-message">✅ Îlots placed successfully! Showing floor plan with green rectangular îlots.</div>', unsafe_allow_html=True)

                    # Display placement statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Îlots", stats['total_ilots'])
                    with col2:
                        st.metric("Total Area", f"{stats['total_area']:.1f} m²")
                    with col3:
                        st.metric("Average Size", f"{stats['average_size']:.1f} m²")
                    with col4:
                        coverage = (stats['total_area'] / (analysis_results['bounds']['max_x'] * analysis_results['bounds']['max_y'])) * 100
                        st.metric("Coverage", f"{coverage:.1f}%")
                else:
                    st.error("❌ Failed to place îlots. Please check the configuration and try again.")

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
            st.plotly_chart(fig, use_container_width=True, height=1800)

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
                st.plotly_chart(fig, use_container_width=True, height=1800)

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
        """Render results and export options with advanced 3D capabilities"""
        st.markdown('<div class="section-header"><h2>📊 Results & Export</h2></div>', unsafe_allow_html=True)

        if not st.session_state.placed_ilots:
            st.warning("Please complete îlot placement and corridor generation first.")
            return

        # Advanced Visualization Options
        st.markdown("### 🎨 Advanced Visualization Options")
        
        # Visualization mode selection
        viz_mode = st.selectbox(
            "Select Visualization Mode",
            ["2D Professional", "3D Interactive (Plotly)", "3D WebGL Real-Time", "All Views"],
            index=0
        )
        
        if viz_mode == "2D Professional":
            fig = self.create_complete_visualization(use_professional=True, show_3d=False)
            st.plotly_chart(fig, use_container_width=True, height=1800)
            
        elif viz_mode == "3D Interactive (Plotly)":
            advanced_3d = Advanced3DRenderer()
            fig = advanced_3d.create_advanced_3d_visualization(
                st.session_state.analysis_results, 
                st.session_state.placed_ilots, 
                st.session_state.corridors,
                show_wireframe=st.checkbox("Show Wireframe", value=False, key="results_export_wireframe"),
                enable_shadows=st.checkbox("Enable Shadows", value=True, key="results_export_shadows")
            )
            st.plotly_chart(fig, use_container_width=True, height=1800)
            
        elif viz_mode == "3D WebGL Real-Time":
            st.markdown("#### 🎛️ Real-Time 3D WebGL Visualization")
            st.info("Interactive 3D visualization with real-time manipulation capabilities")
            
            webgl_renderer = WebGL3DRenderer()
            webgl_renderer.render_3d_scene(
                st.session_state.analysis_results,
                st.session_state.placed_ilots,
                st.session_state.corridors,
                container_id="webgl-3d-scene"
            )
            
        elif viz_mode == "All Views":
            st.markdown("#### 📋 2D Professional View")
            fig_2d = self.create_complete_visualization(use_professional=True, show_3d=False)
            st.plotly_chart(fig_2d, use_container_width=True, height=1200)
            
            st.markdown("#### 🎛️ 3D Interactive View")
            advanced_3d = Advanced3DRenderer()
            fig_3d = advanced_3d.create_advanced_3d_visualization(
                st.session_state.analysis_results, 
                st.session_state.placed_ilots, 
                st.session_state.corridors,
                show_wireframe=False,
                enable_shadows=True
            )
            st.plotly_chart(fig_3d, use_container_width=True, height=1200)
            
            st.markdown("#### 🎮 WebGL Real-Time View")
            webgl_renderer = WebGL3DRenderer()
            webgl_renderer.render_3d_scene(
                st.session_state.analysis_results,
                st.session_state.placed_ilots,
                st.session_state.corridors,
                container_id="webgl-all-views"
            )

        # Project summary
        st.markdown("### 🎯 Project Summary")

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
        st.markdown("### 💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export as JSON", type="secondary"):
                self.export_json()

        with col2:
            if st.button("Export Summary", type="secondary"):
                self.export_summary()

        with col3:
            if st.button("📐 Export 3D Model", type="secondary"):
                st.info("3D model export functionality - IFC, OBJ, GLTF formats coming soon!")

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
            # Use architectural visualizer to match your exact images
            if corridors:
                # Image 3: With corridors
                fig = self.architectural_visualizer.create_complete_floor_plan(
                    analysis_data=result,
                    ilots=ilots,
                    corridors=corridors
                )
            elif ilots:
                # Image 2: With îlots
                fig = self.architectural_visualizer.create_floor_plan_with_ilots(
                    analysis_data=result,
                    ilots=ilots
                )
            else:
                # Image 1: Empty plan
                fig = self.architectural_visualizer.create_empty_floor_plan(result)

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