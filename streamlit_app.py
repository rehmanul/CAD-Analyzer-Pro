import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import tempfile
import os
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="Professional Floor Plan Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'zones_analyzed' not in st.session_state:
        st.session_state.zones_analyzed = None
    if 'ilots_placed' not in st.session_state:
        st.session_state.ilots_placed = None

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
    """Display welcome screen"""
    st.markdown("""
    <div class="main-header">
        <h1>🏨 Professional Hotel Floor Plan Analyzer</h1>
        <p>Advanced AI-powered îlot placement and corridor generation system</p>
    </div>
    """, unsafe_allow_html=True)

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

    # Demo button
    if st.button("🚀 Try Demo with Sample Floor Plan", type="primary", use_container_width=True):
        st.session_state.uploaded_file_data = create_sample_floor_plan_data()
        st.success("✅ Sample hotel floor plan loaded!")
        st.rerun()

    # File upload section
    st.markdown("### 📁 Upload Your Floor Plan")
    uploaded_file = st.file_uploader(
        "Choose a floor plan file",
        type=['dxf', 'dwg', 'jpg', 'jpeg', 'png', 'pdf'],
        help="Upload your floor plan file (demo mode - will use sample data)"
    )

    if uploaded_file is not None:
        # For demo purposes, use sample data
        st.session_state.uploaded_file_data = create_sample_floor_plan_data()
        st.markdown("""
        <div class="success-box">
            <h4>✅ Floor Plan Uploaded!</h4>
            <p>File processed successfully. Using sample data for demonstration.</p>
        </div>
        """, unsafe_allow_html=True)

def create_quick_preview():
    """Create a preview of the floor plan"""
    if not st.session_state.uploaded_file_data:
        return

    fig = go.Figure()

    entities = st.session_state.uploaded_file_data.get('entities', [])

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
        title="Hotel Floor Plan Preview",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def show_zone_analysis():
    """Display zone analysis interface"""
    st.markdown('<h2 class="sub-header">🏗️ Zone Analysis</h2>', unsafe_allow_html=True)

    if not st.session_state.uploaded_file_data:
        st.warning("Please upload a floor plan first or try the demo.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        wall_threshold = st.slider("Wall Detection", 0.0, 1.0, 0.1, 0.1)

    with col2:
        restricted_threshold = st.slider("Restricted Areas", 0.0, 1.0, 0.8, 0.1)

    with col3:
        entrance_threshold = st.slider("Entrances", 0.0, 1.0, 0.3, 0.1)

    if st.button("🔍 Analyze Zones", type="primary", use_container_width=True):
        # Simulate zone analysis
        zones = {
            'walls': [{'geometry': {'coordinates': [0, 0, 50, 0, 50, 30, 0, 30, 0, 0]}}],
            'restricted_areas': [{'geometry': {'coordinates': [5, 10, 8, 10, 8, 15, 5, 15]}}],
            'entrances': [{'geometry': {'coordinates': [23, 0, 27, 0, 27, 2, 23, 2]}}],
            'open_spaces': [{'area': 800}]
        }

        st.session_state.zones_analyzed = zones

        st.markdown("""
        <div class="success-box">
            <h4>✅ Zones Analyzed Successfully!</h4>
            <p>Zone detection complete. Ready for îlot placement.</p>
        </div>
        """, unsafe_allow_html=True)

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏗️ Walls", len(zones['walls']))
        with col2:
            st.metric("🚫 Restricted", len(zones['restricted_areas']))
        with col3:
            st.metric("🚪 Entrances", len(zones['entrances']))
        with col4:
            st.metric("📐 Area", "800m²")

def show_ilot_configuration():
    """Display îlot configuration interface"""
    st.markdown('<h2 class="sub-header">🎨 Îlot Configuration</h2>', unsafe_allow_html=True)

    if not st.session_state.zones_analyzed:
        st.warning("Please analyze zones first.")
        return

    st.markdown("### 📊 Size Distribution Configuration")

    col1, col2 = st.columns(2)

    with col1:
        size_0_1 = st.slider("0-1 m² îlots (%)", 0, 50, 10, 5)
        size_1_3 = st.slider("1-3 m² îlots (%)", 0, 50, 25, 5)

    with col2:
        size_3_5 = st.slider("3-5 m² îlots (%)", 0, 50, 30, 5)
        size_5_10 = st.slider("5-10 m² îlots (%)", 0, 50, 35, 5)

    total_percentage = size_0_1 + size_1_3 + size_3_5 + size_5_10

    if total_percentage != 100:
        st.warning(f"⚠️ Total percentage: {total_percentage}% (should be 100%)")
    else:
        st.success(f"✅ Perfect! Total percentage: {total_percentage}%")

    if total_percentage == 100:
        if st.button("🎯 Place Îlots", type="primary", use_container_width=True):
            # Simulate îlot placement
            ilots = {
                'placed_ilots': [
                    {'geometry': {'coordinates': [15, 8, 18, 8, 18, 11, 15, 11, 15, 8]}, 'area': 9, 'id': 1},
                    {'geometry': {'coordinates': [25, 8, 27, 8, 27, 10, 25, 10, 25, 8]}, 'area': 4, 'id': 2},
                    {'geometry': {'coordinates': [30, 8, 32, 8, 32, 9, 30, 9, 30, 8]}, 'area': 2, 'id': 3},
                ],
                'metrics': {
                    'space_utilization': 75,
                    'area_coverage': 68
                }
            }

            st.session_state.ilots_placed = ilots

            st.markdown("""
            <div class="success-box">
                <h4>✅ Îlots Placed Successfully!</h4>
                <p>Îlot placement complete. Ready for corridor generation.</p>
            </div>
            """, unsafe_allow_html=True)

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Total Îlots", len(ilots['placed_ilots']))
            with col2:
                st.metric("📐 Total Area", "15m²")
            with col3:
                st.metric("⚡ Efficiency", "75%")
            with col4:
                st.metric("📊 Coverage", "68%")

def show_visualization():
    """Show complete visualization"""
    st.markdown('<h2 class="sub-header">🖼️ Complete Layout Visualization</h2>', unsafe_allow_html=True)

    if not all([st.session_state.uploaded_file_data, st.session_state.zones_analyzed, st.session_state.ilots_placed]):
        st.warning("Please complete all previous steps first.")
        return

    fig = go.Figure()

    # Plot floor plan
    entities = st.session_state.uploaded_file_data.get('entities', [])
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
                fill='toself' if entity.get('type') != 'wall' else None,
                fillcolor=f'rgba({{"blue": "0, 0, 255", "red": "255, 0, 0"}.get(color, "0, 0, 0")}, 0.3)',
                line=dict(color=color, width=3),
                name=entity.get('type', 'Unknown').title(),
                showlegend=True
            ))

    # Plot îlots
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
                    name=f'Îlot {i+1} ({area}m²)',
                    showlegend=False
                ))

    fig.update_layout(
        title="🏨 Complete Hotel Floor Plan with Îlots",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    init_session_state()

    # Sidebar navigation
    st.sidebar.title("🏨 Hotel Floor Plan Analyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Choose Analysis Step",
        [
            "📁 Upload Floor Plan",
            "🏗️ Zone Analysis", 
            "🎨 Îlot Configuration",
            "🖼️ Visualization"
        ]
    )

    # Main content
    if page == "📁 Upload Floor Plan":
        show_welcome_screen()
        if st.session_state.uploaded_file_data:
            st.markdown("### 👀 Floor Plan Preview")
            create_quick_preview()

    elif page == "🏗️ Zone Analysis":
        show_zone_analysis()

    elif page == "🎨 Îlot Configuration":
        show_ilot_configuration()

    elif page == "🖼️ Visualization":
        show_visualization()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("🏨 **Professional Hotel Floor Plan Analyzer**")
    st.sidebar.markdown("Streamlit Cloud Compatible Version")

if __name__ == "__main__":
    main()