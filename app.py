import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import io
import tempfile
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import psycopg2
from psycopg2.pool import SimpleConnectionPool

# Import utility modules
from utils.advanced_dxf_parser import AdvancedDXFParser
from utils.geometric_analyzer import GeometricAnalyzer
from utils.production_ilot_system import ProductionIlotSystem
from utils.corridor_generator import AdvancedCorridorGenerator
from utils.visualization import FloorPlanVisualizer
from utils.report_generator import ReportGenerator
from utils.production_database import ProductionDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database connection
@st.cache_resource
def init_database():
    """Initialize PostgreSQL database connection"""
    try:
        db_config = {
            'host': 'dpg-d1h53rffte5s739b1i40-a.oregon-postgres.render.com',
            'port': '5432',
            'database': 'dwg_analyzer_pro',
            'user': 'de_de',
            'password': 'PUPB8V0s2b3bvNZUblolz7d6UM9bcBzb'
        }

        db = ProductionDatabase(db_config)
        db.initialize_schema()
        st.success("✅ PostgreSQL connection pool initialized")
        st.success("✅ Production database schema initialized")
        return db
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        return None

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'zones_analyzed' not in st.session_state:
        st.session_state.zones_analyzed = None
    if 'ilots_placed' not in st.session_state:
        st.session_state.ilots_placed = None
    if 'corridors_generated' not in st.session_state:
        st.session_state.corridors_generated = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

# Custom CSS for professional hotel UI
def load_custom_css():
    """Load custom CSS for professional hotel application styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .sub-header {
        color: #2E86AB;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .client-view-button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        margin: 1rem 0;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def process_uploaded_file(uploaded_file):
    """Process uploaded floor plan file with comprehensive format support"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if file_extension in ['dxf', 'dwg']:
                parser = AdvancedDXFParser()
                file_data = parser.parse_file(tmp_file_path)

                if file_data and 'entities' in file_data:
                    # Enhance with geometric analysis
                    analyzer = GeometricAnalyzer()
                    enhanced_data = analyzer.analyze_floor_plan_structure(file_data)
                    return enhanced_data
                else:
                    st.error("Failed to parse DXF/DWG file. Please check the file format.")
                    return None

            elif file_extension in ['jpg', 'jpeg', 'png', 'pdf']:
                # For now, create a sample data structure for image files
                # In production, this would use computer vision analysis
                return create_sample_floor_plan_data()

        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
        return None

def create_sample_floor_plan_data():
    """Create sample floor plan data for demonstration"""
    return {
        'entities': [
            # Walls (black lines)
            {'type': 'wall', 'layer': 'walls', 'color': 'black', 
             'points': [0, 0, 50, 0, 50, 30, 0, 30, 0, 0]},
            {'type': 'wall', 'layer': 'walls', 'color': 'black', 
             'points': [10, 5, 40, 5, 40, 25, 10, 25, 10, 5]},

            # Restricted areas (blue zones - stairs, elevators)
            {'type': 'restricted', 'layer': 'restricted', 'color': 'blue',
             'points': [5, 10, 8, 10, 8, 15, 5, 15], 'zone_type': 'stairs'},
            {'type': 'restricted', 'layer': 'restricted', 'color': 'blue',
             'points': [42, 10, 45, 10, 45, 15, 42, 15], 'zone_type': 'elevator'},

            # Entrances (red zones)
            {'type': 'entrance', 'layer': 'entrances', 'color': 'red',
             'points': [23, 0, 27, 0, 27, 2, 23, 2], 'zone_type': 'main_entrance'},
        ],
        'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 50, 'max_y': 30},
        'metadata': {
            'file_type': 'sample',
            'layers': ['walls', 'restricted', 'entrances'],
            'units': 'meters'
        }
    }

def show_welcome_screen():
    """Display welcome screen with hotel-specific features"""
    st.markdown("""
    <div class="main-header">
        <h1>🏨 Professional Hotel Floor Plan Analyzer</h1>
        <p>Advanced AI-powered îlot placement and corridor generation system</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature overview with hotel-specific capabilities
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Hotel-Specific Intelligence</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <h4>🏗️ Zone Detection:</h4>
                <ul>
                    <li>✅ Walls (black lines) - structural boundaries</li>
                    <li>✅ Restricted areas (blue) - stairs, elevators, utilities</li>
                    <li>✅ Entrances/Exits (red) - no îlot placement zones</li>
                </ul>
            </div>
            <div>
                <h4>🎨 Intelligent Îlot Placement:</h4>
                <ul>
                    <li>✅ Configurable size distribution (0-1m², 1-3m², 3-5m², 5-10m²)</li>
                    <li>✅ Automatic placement avoiding restricted zones</li>
                    <li>✅ Mandatory corridors between facing îlot rows</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File upload section with enhanced styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #2E86AB;
        margin: 2rem 0;
        text-align: center;
    ">
        <h3 style="color: #2E86AB; margin-bottom: 1rem;">📁 Upload Your Hotel Floor Plan</h3>
        <p style="color: #495057; margin-bottom: 1.5rem;">
            Supports DXF, DWG, JPG, PNG, PDF formats with intelligent zone detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a floor plan file",
        type=['dxf', 'dwg', 'jpg', 'jpeg', 'png', 'pdf', 'ifc'],
        help="📁 Supported formats: DXF (full support), DWG (with conversion), JPG/PNG (computer vision), PDF (vector extraction), IFC (BIM integration)"
    )

    if uploaded_file is not None:
        with st.spinner("🔄 Processing your hotel floor plan..."):
            file_data = process_uploaded_file(uploaded_file)
            if file_data:
                st.session_state.uploaded_file_data = file_data

                st.markdown("""
                <div class="success-box">
                    <h4>✅ Hotel Floor Plan Successfully Processed!</h4>
                    <p>Your floor plan has been analyzed and is ready for intelligent îlot placement and corridor generation.</p>
                </div>
                """, unsafe_allow_html=True)

                # Show file information with hotel-specific metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 File Size", f"{len(uploaded_file.getvalue())/1024:.1f} KB")
                with col2:
                    st.metric("🏗️ Elements", len(file_data.get('entities', [])))
                with col3:
                    st.metric("📏 Bounds", f"{file_data.get('bounds', {}).get('max_x', 0):.1f}m")
                with col4:
                    st.metric("🎯 Layers", len(file_data.get('metadata', {}).get('layers', [])))

                # Quick preview
                st.markdown("### 👀 Hotel Floor Plan Preview")
                create_quick_preview()

def create_quick_preview():
    """Create a quick preview of the uploaded floor plan"""
    if not st.session_state.uploaded_file_data:
        return

    fig = go.Figure()

    entities = st.session_state.uploaded_file_data.get('entities', [])
    bounds = st.session_state.uploaded_file_data.get('bounds', {})

    for entity in entities:
        if 'points' in entity and len(entity['points']) >= 4:
            points = entity['points']
            x_coords = [points[i] for i in range(0, len(points), 2)]
            y_coords = [points[i] for i in range(1, len(points), 2)]

            color = entity.get('color', 'black')
            if entity.get('type') == 'wall':
                color = 'black'
            elif entity.get('type') == 'restricted':
                color = 'blue'
            elif entity.get('type') == 'entrance':
                color = 'red'

            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(color=color, width=3),
                name=entity.get('type', 'Unknown').title(),
                showlegend=True
            ))

    fig.update_layout(
        title="Hotel Floor Plan Preview - Zone Detection",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def show_zone_analysis():
    """Display zone analysis with hotel-specific requirements"""
    st.markdown('<h2 class="sub-header">🏗️ Hotel Zone Analysis & Detection</h2>', unsafe_allow_html=True)

    if not st.session_state.uploaded_file_data:
        st.warning("Please upload a hotel floor plan first.")
        return

    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Hotel Zone Detection Parameters</h4>
        <p>Configure thresholds for detecting walls, restricted areas, and entrances in your hotel floor plan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Zone detection parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏗️ Wall Detection**")
        wall_threshold = st.slider("Wall Threshold", 0.0, 1.0, 0.1, 0.1,
                                  help="Sensitivity for detecting walls (black lines)")

    with col2:
        st.markdown("**🚫 Restricted Areas**")
        restricted_threshold = st.slider("Restricted Threshold", 0.0, 1.0, 0.8, 0.1,
                                       help="Threshold for stairs, elevators (blue zones)")

    with col3:
        st.markdown("**🚪 Entrances/Exits**")
        entrance_threshold = st.slider("Entrance Threshold", 0.0, 1.0, 0.3, 0.1,
                                     help="Sensitivity for entrance detection (red zones)")

    if st.button("🔍 Analyze Hotel Zones", type="primary", use_container_width=True):
        with st.spinner("🏗️ Analyzing hotel zones..."):
            zones = analyze_zones(wall_threshold, restricted_threshold, entrance_threshold)
            if zones:
                st.session_state.zones_analyzed = zones

                st.markdown("""
                <div class="success-box">
                    <h4>✅ Hotel Zones Successfully Analyzed!</h4>
                    <p>Zone detection complete. Ready for îlot placement configuration.</p>
                </div>
                """, unsafe_allow_html=True)

                # Display zone statistics
                display_zone_statistics(zones)

                # Visualize detected zones
                visualize_detected_zones(zones)

def analyze_zones(wall_threshold, restricted_threshold, entrance_threshold):
    """Analyze zones in the hotel floor plan"""
    if not st.session_state.uploaded_file_data:
        return None

    try:
        analyzer = GeometricAnalyzer()
        zones = analyzer.analyze_zones(
            st.session_state.uploaded_file_data,
            wall_threshold,
            restricted_threshold,
            entrance_threshold
        )
        return zones
    except Exception as e:
        st.error(f"Zone analysis failed: {e}")
        return None

def display_zone_statistics(zones):
    """Display statistics about detected zones"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        walls_count = len(zones.get('walls', []))
        st.metric("🏗️ Walls", walls_count)

    with col2:
        restricted_count = len(zones.get('restricted_areas', []))
        st.metric("🚫 Restricted", restricted_count)

    with col3:
        entrances_count = len(zones.get('entrances', []))
        st.metric("🚪 Entrances", entrances_count)

    with col4:
        open_spaces = zones.get('open_spaces', [])
        total_area = sum(space.get('area', 0) for space in open_spaces)
        st.metric("📐 Available Area", f"{total_area:.1f}m²")

def visualize_detected_zones(zones):
    """Visualize the detected zones"""
    fig = go.Figure()

    # Plot walls (black)
    for wall in zones.get('walls', []):
        if 'geometry' in wall and 'coordinates' in wall['geometry']:
            coords = wall['geometry']['coordinates']
            if len(coords) >= 4:
                x_coords = [coords[i] for i in range(0, len(coords), 2)]
                y_coords = [coords[i] for i in range(1, len(coords), 2)]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color='black', width=4),
                    name='Walls',
                    showlegend=True
                ))

    # Plot restricted areas (blue)
    for restricted in zones.get('restricted_areas', []):
        if 'geometry' in restricted and 'coordinates' in restricted['geometry']:
            coords = restricted['geometry']['coordinates']
            if len(coords) >= 4:
                x_coords = [coords[i] for i in range(0, len(coords), 2)]
                y_coords = [coords[i] for i in range(1, len(coords), 2)]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines+markers',
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.3)',
                    line=dict(color='blue', width=3),
                    name='Restricted Areas',
                    showlegend=True
                ))

    # Plot entrances (red)
    for entrance in zones.get('entrances', []):
        if 'geometry' in entrance and 'coordinates' in entrance['geometry']:
            coords = entrance['geometry']['coordinates']
            if len(coords) >= 4:
                x_coords = [coords[i] for i in range(0, len(coords), 2)]
                y_coords = [coords[i] for i in range(1, len(coords), 2)]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines+markers',
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=3),
                    name='Entrances/Exits',
                    showlegend=True
                ))

    fig.update_layout(
        title="🏨 Hotel Zone Detection Results",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def show_ilot_configuration():
    """Display îlot configuration interface with hotel requirements"""
    st.markdown('<h2 class="sub-header">🎨 Hotel Îlot Placement Configuration</h2>', unsafe_allow_html=True)

    if not st.session_state.zones_analyzed:
        st.warning("Please analyze hotel zones first.")
        return

    st.markdown("""
    <div class="feature-card">
        <h4>🏨 Hotel Îlot Size Distribution</h4>
        <p>Configure the percentage distribution of îlot sizes according to your hotel layout requirements.</p>
    </div>
    """, unsafe_allow_html=True)

    # Îlot size distribution configuration
    st.markdown("### 📊 Size Distribution Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Îlot Size Categories:**")
        size_0_1 = st.slider("0-1 m² îlots (%)", 0, 50, 10, 5)
        size_1_3 = st.slider("1-3 m² îlots (%)", 0, 50, 25, 5)

    with col2:
        st.markdown("**📐 Larger Îlot Sizes:**")
        size_3_5 = st.slider("3-5 m² îlots (%)", 0, 50, 30, 5)
        size_5_10 = st.slider("5-10 m² îlots (%)", 0, 50, 35, 5)

    # Validate total percentage
    total_percentage = size_0_1 + size_1_3 + size_3_5 + size_5_10

    if total_percentage != 100:
        st.warning(f"⚠️ Total percentage: {total_percentage}% (should be 100%)")
    else:
        st.success(f"✅ Perfect! Total percentage: {total_percentage}%")

    # Advanced placement options
    st.markdown("### ⚙️ Advanced Placement Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_distance_walls = st.slider("Min distance from walls (m)", 0.1, 2.0, 0.5, 0.1)
        min_distance_restricted = st.slider("Min distance from restricted (m)", 0.5, 5.0, 2.0, 0.1)

    with col2:
        min_distance_entrances = st.slider("Min distance from entrances (m)", 1.0, 5.0, 3.0, 0.1)
        optimization_iterations = st.slider("Optimization iterations", 10, 100, 50, 10)

    with col3:
        corridor_width = st.slider("Corridor width (m)", 1.0, 3.0, 1.5, 0.1)
        allow_wall_touching = st.checkbox("Allow îlots to touch walls", True)

    # Placement configuration summary
    if total_percentage == 100:
        st.markdown("### 📋 Placement Configuration Summary")

        config_data = {
            'Size Range': ['0-1 m²', '1-3 m²', '3-5 m²', '5-10 m²'],
            'Percentage': [f"{size_0_1}%", f"{size_1_3}%", f"{size_3_5}%", f"{size_5_10}%"]
        }

        df = pd.DataFrame(config_data)
        st.table(df)

        if st.button("🎯 Place Hotel Îlots", type="primary", use_container_width=True):
            with st.spinner("🏨 Placing îlots in your hotel floor plan..."):
                ilot_config = {
                    'size_distribution': {
                        '0-1': size_0_1 / 100,
                        '1-3': size_1_3 / 100,
                        '3-5': size_3_5 / 100,
                        '5-10': size_5_10 / 100
                    },
                    'constraints': {
                        'min_distance_walls': min_distance_walls,
                        'min_distance_restricted': min_distance_restricted,
                        'min_distance_entrances': min_distance_entrances,
                        'allow_wall_touching': allow_wall_touching
                    },
                    'optimization': {
                        'iterations': optimization_iterations,
                        'corridor_width': corridor_width
                    }
                }

                ilots = place_ilots(ilot_config)

                if ilots:
                    st.session_state.ilots_placed = ilots

                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Hotel Îlots Successfully Placed!</h4>
                        <p>Îlot placement complete. Ready for corridor generation.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    display_ilot_statistics(ilots)
                    visualize_placed_ilots(ilots)

def place_ilots(config):
    """Place îlots according to configuration"""
    try:
        ilot_placer = ProductionIlotSystem()
        ilots = ilot_placer.place_ilots(
            st.session_state.uploaded_file_data,
            st.session_state.zones_analyzed,
            config
        )
        return ilots
    except Exception as e:
        st.error(f"Îlot placement failed: {e}")
        return None

def display_ilot_statistics(ilots):
    """Display statistics about placed îlots"""
    if not ilots:
        return

    total_ilots = len(ilots.get('placed_ilots', []))
    total_area = sum(ilot.get('area', 0) for ilot in ilots.get('placed_ilots', []))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Total Îlots", total_ilots)

    with col2:
        st.metric("📐 Total Area", f"{total_area:.1f}m²")

    with col3:
        efficiency = ilots.get('metrics', {}).get('space_utilization', 0)
        st.metric("⚡ Efficiency", f"{efficiency:.1f}%")

    with col4:
        coverage = ilots.get('metrics', {}).get('area_coverage', 0)
        st.metric("📊 Coverage", f"{coverage:.1f}%")

def visualize_placed_ilots(ilots):
    """Visualize the placed îlots"""
    fig = go.Figure()

    # Plot zones first (from previous analysis)
    if st.session_state.zones_analyzed:
        zones = st.session_state.zones_analyzed

        # Walls
        for wall in zones.get('walls', []):
            if 'geometry' in wall and 'coordinates' in wall['geometry']:
                coords = wall['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        line=dict(color='black', width=5),
                        name='Walls',
                        showlegend=True
                    ))

        # Restricted areas
        for restricted in zones.get('restricted_areas', []):
            if 'geometry' in restricted and 'coordinates' in restricted['geometry']:
                coords = restricted['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(0, 0, 255, 0.3)',
                        line=dict(color='blue', width=3),
                        name='Restricted Areas',
                        showlegend=True
                    ))

        # Entrances
        for entrance in zones.get('entrances', []):
            if 'geometry' in entrance and 'coordinates' in entrance['geometry']:
                coords = entrance['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.4)',
                        line=dict(color='red', width=3),
                        name='Entrances/Exits',
                        showlegend=True
                    ))

    # Plot îlots with size-based coloring
    placed_ilots = ilots.get('placed_ilots', [])
    for i, ilot in enumerate(placed_ilots):
        if 'geometry' in ilot and 'coordinates' in ilot['geometry']:
            coords = ilot['geometry']['coordinates']
            if len(coords) >= 4:
                x_coords = [coords[i] for i in range(0, len(coords), 2)]
                y_coords = [coords[i] for i in range(1, len(coords), 2)]

                # Color by size category
                area = ilot.get('area', 0)
                if area <= 1:
                    color = 'lightblue'
                elif area <= 3:
                    color = 'lightgreen'
                elif area <= 5:
                    color = 'orange'
                else:
                    color = 'pink'

                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines+markers',
                    fill='toself',
                    fillcolor=color,
                    line=dict(color='darkblue', width=2),
                    name=f'Îlot {i+1} ({area:.1f}m²)',
                    showlegend=False
                    ))

    fig.update_layout(
        title="🏨 Hotel Îlot Placement Results",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def show_corridor_generation():
    """Display corridor generation interface"""
    st.markdown('<h2 class="sub-header">🛤️ Hotel Corridor Generation</h2>', unsafe_allow_html=True)

    if not st.session_state.ilots_placed:
        st.warning("Please place îlots first.")
        return

    st.markdown("""
    <div class="feature-card">
        <h4>🛤️ Automatic Corridor Generation</h4>
        <p>Generate mandatory corridors between facing îlot rows for optimal hotel circulation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Corridor generation parameters
    col1, col2 = st.columns(2)

    with col1:
        corridor_width = st.slider("Corridor Width (m)", 1.0, 4.0, 1.5, 0.1)
        min_corridor_length = st.slider("Min Corridor Length (m)", 2.0, 10.0, 3.0, 0.5)

    with col2:
        max_corridor_turns = st.slider("Max Corridor Turns", 0, 5, 2, 1)
        corridor_clearance = st.slider("Corridor Clearance (m)", 0.1, 1.0, 0.3, 0.1)

    if st.button("🛤️ Generate Hotel Corridors", type="primary", use_container_width=True):
        with st.spinner("🏨 Generating hotel corridor network..."):
            corridor_config = {
                'width': corridor_width,
                'min_length': min_corridor_length,
                'max_turns': max_corridor_turns,
                'clearance': corridor_clearance
            }

            corridors = generate_corridors(corridor_config)

            if corridors:
                st.session_state.corridors_generated = corridors

                st.markdown("""
                <div class="success-box">
                    <h4>✅ Hotel Corridors Successfully Generated!</h4>
                    <p>Corridor network complete. Ready for final visualization and export.</p>
                </div>
                """, unsafe_allow_html=True)

                display_corridor_statistics(corridors)
                visualize_complete_layout()

def generate_corridors(config):
    """Generate corridors between îlot rows"""
    try:
        corridor_gen = AdvancedCorridorGenerator()
        
        # Load floor plan data into generator
        ilots = st.session_state.ilots_placed.get('placed_ilots', [])
        zones = st.session_state.zones_analyzed
        
        walls = [z.get('geometry', {}).get('coordinates', []) for z in zones.get('walls', [])]
        restricted_areas = [z.get('geometry', {}).get('coordinates', []) for z in zones.get('restricted_areas', [])]
        entrances = [z.get('geometry', {}).get('coordinates', []) for z in zones.get('entrances', [])]
        bounds = st.session_state.uploaded_file_data.get('bounds', {})
        
        corridor_gen.load_floor_plan_data(ilots, walls, restricted_areas, entrances, bounds)
        
        # Generate complete corridor network
        corridors = corridor_gen.generate_complete_corridor_network(config)
        return corridors
    except Exception as e:
        st.error(f"Corridor generation failed: {e}")
        return None

def display_corridor_statistics(corridors):
    """Display corridor statistics"""
    total_corridors = len(corridors.get('corridors', []))
    total_length = sum(c.get('length', 0) for c in corridors.get('corridors', []))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🛤️ Corridors", total_corridors)

    with col2:
        st.metric("📏 Total Length", f"{total_length:.1f}m")

    with col3:
        connectivity = corridors.get('metrics', {}).get('connectivity_score', 0)
        st.metric("🔗 Connectivity", f"{connectivity:.1f}%")

    with col4:
        accessibility = corridors.get('metrics', {}).get('accessibility_score', 0)
        st.metric("♿ Accessibility", f"{accessibility:.1f}%")

def visualize_complete_layout():
    """Visualize the complete hotel layout"""
    fig = go.Figure()

    # Plot all components together
    if st.session_state.zones_analyzed:
        zones = st.session_state.zones_analyzed

        # Walls
        for wall in zones.get('walls', []):
            if 'geometry' in wall and 'coordinates' in wall['geometry']:
                coords = wall['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        line=dict(color='black', width=5),
                        name='Walls',
                        showlegend=True
                    ))

        # Restricted areas
        for restricted in zones.get('restricted_areas', []):
            if 'geometry' in restricted and 'coordinates' in restricted['geometry']:
                coords = restricted['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(0, 0, 255, 0.3)',
                        line=dict(color='blue', width=3),
                        name='Restricted Areas',
                        showlegend=True
                    ))

        # Entrances
        for entrance in zones.get('entrances', []):
            if 'geometry' in entrance and 'coordinates' in entrance['geometry']:
                coords = entrance['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.4)',
                        line=dict(color='red', width=3),
                        name='Entrances/Exits',
                        showlegend=True
                    ))

    # Plot îlots
    if st.session_state.ilots_placed:
        placed_ilots = st.session_state.ilots_placed.get('placed_ilots', [])
        for i, ilot in enumerate(placed_ilots):
            if 'geometry' in ilot and 'coordinates' in ilot['geometry']:
                coords = ilot['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]

                    area = ilot.get('area', 0)
                    if area <= 1:
                        color = 'lightblue'
                    elif area <= 3:
                        color = 'lightgreen'
                    elif area <= 5:
                        color = 'orange'
                    else:
                        color = 'pink'

                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines+markers',
                        fill='toself',
                        fillcolor=color,
                        line=dict(color='darkblue', width=2),
                        name=f'Îlot {i+1} ({area:.1f}m²)',
                        showlegend=False
                    ))

    # Plot corridors
    if st.session_state.corridors_generated:
        corridors = st.session_state.corridors_generated.get('corridors', [])
        for corridor in corridors:
            if 'geometry' in corridor and 'coordinates' in corridor['geometry']:
                coords = corridor['geometry']['coordinates']
                if len(coords) >= 4:
                    x_coords = [coords[i] for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] for i in range(1, len(coords), 2)]
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 255, 0, 0.3)',
                        line=dict(color='gold', width=3),
                        name='Corridors',
                        showlegend=True
                    ))

    fig.update_layout(
        title="🏨 Complete Hotel Floor Plan with Îlots and Corridors",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def show_performance_metrics():
    """Display comprehensive performance metrics"""
    st.markdown('<h2 class="sub-header">📊 Hotel Performance Analytics</h2>', unsafe_allow_html=True)

    if not all([st.session_state.zones_analyzed, st.session_state.ilots_placed, st.session_state.corridors_generated]):
        st.warning("Please complete the full analysis pipeline first.")
        return

    # Compile all metrics
    metrics = compile_performance_metrics()

    # Overall performance score
    st.markdown("### 🏆 Overall Performance Score")
    overall_score = metrics.get('overall_score', 0)

    progress_col, score_col = st.columns([3, 1])
    with progress_col:
        st.progress(overall_score / 100)
    with score_col:
        st.metric("Score", f"{overall_score:.1f}/100")

    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Space Utilization", f"{metrics.get('space_utilization', 0):.1f}%")

    with col2:
        st.metric("♿ Accessibility", f"{metrics.get('accessibility_score', 0):.1f}%")

    with col3:
        st.metric("🔄 Circulation", f"{metrics.get('circulation_efficiency', 0):.1f}%")

    with col4:
        st.metric("🛡️ Safety", f"{metrics.get('safety_compliance', 0):.1f}%")

    # Radar chart for performance visualization
    display_performance_radar(metrics)

def compile_performance_metrics():
    """Compile comprehensive performance metrics"""
    metrics = {}

    if st.session_state.ilots_placed:
        ilot_metrics = st.session_state.ilots_placed.get('metrics', {})
        metrics.update(ilot_metrics)

    if st.session_state.corridors_generated:
        corridor_metrics = st.session_state.corridors_generated.get('metrics', {})
        metrics.update(corridor_metrics)

    # Calculate overall score
    scores = [
        metrics.get('space_utilization', 0),
        metrics.get('accessibility_score', 0),
        metrics.get('circulation_efficiency', 0),
        metrics.get('safety_compliance', 0)
    ]

    metrics['overall_score'] = sum(scores) / len(scores) if scores else 0

    return metrics

def display_performance_radar(metrics):
    """Display performance metrics as radar chart"""
    categories = ['Space Utilization', 'Accessibility', 'Circulation', 'Safety']
    values = [
        metrics.get('space_utilization', 0),
        metrics.get('accessibility_score', 0),
        metrics.get('circulation_efficiency', 0),
        metrics.get('safety_compliance', 0)
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='#2E86AB'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="Hotel Performance Metrics Analysis",
        height=400
    )

    st.plotly_chart(fig_radar, use_container_width=True)

def show_export_options():
    """Display export and reporting options"""
    st.markdown('<h2 class="sub-header">📄 Export & Reporting</h2>', unsafe_allow_html=True)

    if not all([st.session_state.zones_analyzed, st.session_state.ilots_placed, st.session_state.corridors_generated]):
        st.warning("Please complete the full analysis pipeline first.")
        return

    st.markdown("""
    <div class="feature-card">
        <h4>📊 Professional Hotel Reports</h4>
        <p>Generate comprehensive reports and export your hotel floor plan analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Report Options")
        include_metrics = st.checkbox("Include Performance Metrics", True)
        include_images = st.checkbox("Include Visualizations", True)
        include_recommendations = st.checkbox("Include Recommendations", True)

        report_format = st.selectbox("Report Format", ["PDF", "Excel", "Word"])

    with col2:
        st.markdown("### 🖼️ Image Export")
        image_format = st.selectbox("Image Format", ["PNG", "SVG", "JPG"])
        image_resolution = st.selectbox("Resolution", ["Low (72 DPI)", "Medium (150 DPI)", "High (300 DPI)"])

        export_format = st.selectbox("Export Data", ["JSON", "CSV", "DXF"])

    if st.button("📄 Generate Hotel Report", type="primary", use_container_width=True):
        with st.spinner("📊 Generating comprehensive hotel report..."):
            report_config = {
                'include_metrics': include_metrics,
                'include_images': include_images,
                'include_recommendations': include_recommendations,
                'format': report_format.lower()
            }

            generate_report(report_config)

def generate_report(config):
    """Generate comprehensive report"""
    try:
        report_gen = ReportGenerator()

        # Compile all analysis data
        analysis_data = {
            'floor_plan': st.session_state.uploaded_file_data,
            'zones': st.session_state.zones_analyzed,
            'ilots': st.session_state.ilots_placed,
            'corridors': st.session_state.corridors_generated,
            'metrics': compile_performance_metrics()
        }

        report = report_gen.generate_comprehensive_report(analysis_data, config)

        if report:
            st.success("✅ Hotel report generated successfully!")

            # Provide download link
            st.download_button(
                label="📥 Download Hotel Report",
                data=report['content'],
                file_name=report['filename'],
                mime=report['mime_type']
            )

    except Exception as e:
        st.error(f"Report generation failed: {e}")

def main():
    """Main application function"""
    # Initialize
    init_session_state()
    load_custom_css()

    # Initialize database
    db = init_database()

    # Sidebar navigation
    st.sidebar.title("🏨 Hotel Floor Plan Analyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Choose Analysis Step",
        [
            "📁 Upload Floor Plan",
            "🏗️ Zone Analysis",
            "🎨 Îlot Configuration",
            "🛤️ Corridor Generation",
            "📊 Performance Metrics",
            "📄 Export & Reports"
        ]
    )

    # Main content area
    if page == "📁 Upload Floor Plan":
        show_welcome_screen()
    elif page == "🏗️ Zone Analysis":
        show_zone_analysis()
    elif page == "🎨 Îlot Configuration":
        show_ilot_configuration()
    elif page == "🛤️ Corridor Generation":
        show_corridor_generation()
    elif page == "📊 Performance Metrics":
        show_performance_metrics()
    elif page == "📄 Export & Reports":
        show_export_options()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("🏨 **Professional Hotel Floor Plan Analyzer**")
    st.sidebar.markdown("Advanced AI-powered spatial optimization")

if __name__ == "__main__":
    main()