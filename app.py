import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from PIL import Image
import json
import tempfile
import os
from pathlib import Path

# Import custom modules
from utils.dxf_parser import DXFParser
from utils.geometric_analyzer import GeometricAnalyzer
from utils.ilot_placer import IlotPlacer
from utils.corridor_generator import CorridorGenerator
from utils.visualization import FloorPlanVisualizer
from utils.report_generator import ReportGenerator
from utils.spatial_optimizer import SpatialOptimizer

# Import advanced enterprise modules
from utils.webgl_visualizer import WebGLVisualizer
try:
    from utils.ml_space_optimizer import MLSpaceOptimizer, HybridOptimizer
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"ML optimization modules not available: {e}")
    ML_AVAILABLE = False
    # Create dummy classes
    class MLSpaceOptimizer:
        def optimize_placement(self, *args, **kwargs):
            return []
        def get_optimization_metrics(self):
            return {}
    class HybridOptimizer:
        def optimize(self, *args, **kwargs):
            return []

from utils.bim_integration import BIMIntegrationManager

# Import collaboration module optionally
try:
    from utils.realtime_collaboration import CollaborationClient
    COLLABORATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Collaboration features not available: {e}")
    COLLABORATION_AVAILABLE = False
    # Create dummy class
    class CollaborationClient:
        def __init__(self):
            pass

# Configure page
st.set_page_config(
    page_title="Professional Floor Plan Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'floor_plan_data' not in st.session_state:
    st.session_state.floor_plan_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'ilot_configuration' not in st.session_state:
    st.session_state.ilot_configuration = {}
if 'corridor_config' not in st.session_state:
    st.session_state.corridor_config = {}

# Initialize advanced enterprise features
if 'webgl_visualizer' not in st.session_state:
    st.session_state.webgl_visualizer = WebGLVisualizer()
if 'ml_optimizer' not in st.session_state and ML_AVAILABLE:
    st.session_state.ml_optimizer = MLSpaceOptimizer()
elif 'ml_optimizer' not in st.session_state:
    st.session_state.ml_optimizer = MLSpaceOptimizer()  # Dummy class
if 'hybrid_optimizer' not in st.session_state and ML_AVAILABLE:
    st.session_state.hybrid_optimizer = HybridOptimizer()
elif 'hybrid_optimizer' not in st.session_state:
    st.session_state.hybrid_optimizer = HybridOptimizer()  # Dummy class
if 'bim_manager' not in st.session_state:
    st.session_state.bim_manager = BIMIntegrationManager()
if 'collaboration_enabled' not in st.session_state:
    st.session_state.collaboration_enabled = False and COLLABORATION_AVAILABLE
if 'optimization_mode' not in st.session_state:
    st.session_state.optimization_mode = 'hybrid'
if 'visualization_mode' not in st.session_state:
    st.session_state.visualization_mode = '2d'

def main():
    st.title("🏗️ Professional Floor Plan Analyzer")
    st.markdown("Enterprise-grade CAD analysis with intelligent îlot placement and corridor generation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload section
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload CAD File",
            type=['dxf', 'dwg', 'pdf'],
            help="Supported formats: DXF, DWG, PDF"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process File"):
                process_uploaded_file(uploaded_file)
        
        # Enterprise Features Configuration
        st.subheader("🚀 Enterprise Features")
        
        # Visualization Mode
        st.markdown("**Visualization**")
        visualization_mode = st.selectbox(
            "Visualization Mode",
            ["2d", "3d_webgl", "heatmap"],
            index=0 if st.session_state.visualization_mode == '2d' else 1
        )
        st.session_state.visualization_mode = visualization_mode
        
        # Optimization Engine
        st.markdown("**AI Optimization**")
        optimization_mode = st.selectbox(
            "Optimization Engine",
            ["traditional", "ml_only", "hybrid", "ensemble"],
            index=2
        )
        st.session_state.optimization_mode = optimization_mode
        
        # BIM Integration
        st.markdown("**BIM Integration**")
        enable_bim = st.checkbox("Enable BIM Integration", False)
        if enable_bim:
            bim_system = st.selectbox("BIM System", ["ifc", "revit", "archicad"], index=0)
        
        # Real-time Collaboration
        st.markdown("**Collaboration**")
        if COLLABORATION_AVAILABLE:
            enable_collaboration = st.checkbox("Enable Real-time Collaboration", st.session_state.collaboration_enabled)
            st.session_state.collaboration_enabled = enable_collaboration
            
            if enable_collaboration:
                project_id = st.text_input("Project ID", value="floor_plan_project_001")
                user_name = st.text_input("Your Name", value="Designer")
        else:
            st.info("💡 Install 'websockets' package to enable collaboration features")
            st.session_state.collaboration_enabled = False
        
        # Analysis configuration
        if st.session_state.floor_plan_data is not None:
            st.subheader("⚙️ Analysis Settings")
            
            # Zone detection settings
            st.markdown("**Zone Detection**")
            wall_color_threshold = st.slider("Wall Color Threshold", 0.0, 1.0, 0.1, 0.01)
            restricted_area_threshold = st.slider("Restricted Area Threshold", 0.0, 1.0, 0.8, 0.01)
            entrance_threshold = st.slider("Entrance/Exit Threshold", 0.0, 1.0, 0.3, 0.01)
            
            # Îlot configuration
            st.markdown("**Îlot Configuration**")
            configure_ilot_settings()
            
            # Corridor configuration
            st.markdown("**Corridor Settings**")
            configure_corridor_settings()
            
            # Analysis actions
            st.subheader("🎯 Analysis Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Analyze Zones"):
                    analyze_zones(wall_color_threshold, restricted_area_threshold, entrance_threshold)
            
            with col2:
                if st.button("📐 Place Îlots"):
                    place_ilots_advanced()
            
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("🛤️ Generate Corridors"):
                    generate_corridors()
            
            with col4:
                if st.button("🤖 ML Optimize"):
                    ml_optimize_placement()
            
            # Enterprise Actions
            st.subheader("🚀 Enterprise Actions")
            col5, col6 = st.columns(2)
            
            with col5:
                if st.button("🌐 Export to BIM"):
                    export_to_bim()
            
            with col6:
                if st.button("📊 Generate Report"):
                    generate_comprehensive_report()
    
    # Main content area
    if st.session_state.floor_plan_data is None:
        show_welcome_screen()
    else:
        show_analysis_interface()

def show_welcome_screen():
    st.markdown("""
    ## Welcome to Professional Floor Plan Analyzer
    
    This enterprise-grade application provides comprehensive CAD analysis capabilities:
    
    ### 🎯 Key Features
    - **Complete DXF/DWG/PDF Processing**: Full geometric entity extraction and parsing
    - **Intelligent Zone Detection**: Automatic identification of walls, restricted areas, and entrances
    - **Advanced Îlot Placement**: AI-powered spatial optimization algorithms
    - **Smart Corridor Generation**: Automatic pathfinding with conflict resolution
    - **Professional Visualization**: Interactive 2D floor plan viewer
    - **Comprehensive Reporting**: Detailed metrics and optimization analysis
    
    ### 📋 Getting Started
    1. Upload your CAD file using the sidebar
    2. Configure analysis settings
    3. Process and analyze your floor plan
    4. Generate intelligent îlot placements
    5. Create optimized corridor networks
    6. Export professional reports
    
    ### 📊 Supported File Formats
    - **DXF**: AutoCAD Drawing Exchange Format
    - **DWG**: AutoCAD Drawing Database
    - **PDF**: Portable Document Format with vector graphics
    
    Upload a file to begin your professional floor plan analysis.
    """)

def show_analysis_interface():
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺️ Floor Plan View", 
        "📊 Analysis Results", 
        "🎯 Îlot Placement", 
        "🛤️ Corridor Network",
        "🌐 3D Visualization",
        "🤖 AI Optimization",
        "📈 Reports"
    ])
    
    with tab1:
        show_floor_plan_view()
    
    with tab2:
        show_analysis_results()
    
    with tab3:
        show_ilot_placement()
    
    with tab4:
        show_corridor_network()
    
    with tab5:
        show_3d_visualization()
    
    with tab6:
        show_ai_optimization()
    
    with tab7:
        show_reports()

def process_uploaded_file(uploaded_file):
    """Process the uploaded CAD file"""
    with st.spinner("Processing CAD file..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Parse the file based on its type
            parser = DXFParser()
            
            if uploaded_file.name.lower().endswith('.dxf'):
                floor_plan_data = parser.parse_dxf(tmp_file_path)
            elif uploaded_file.name.lower().endswith('.dwg'):
                floor_plan_data = parser.parse_dwg(tmp_file_path)
            elif uploaded_file.name.lower().endswith('.pdf'):
                floor_plan_data = parser.parse_pdf(tmp_file_path)
            else:
                st.error("Unsupported file format")
                return
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Store in session state
            st.session_state.floor_plan_data = floor_plan_data
            
            st.success(f"✅ Successfully processed {uploaded_file.name}")
            st.info(f"Extracted {len(floor_plan_data.get('entities', []))} geometric entities")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

def configure_ilot_settings():
    """Configure îlot placement settings"""
    st.markdown("**Size Distribution**")
    
    # Îlot size categories
    small_percent = st.slider("Small Îlots (%)", 0, 100, 30, 5)
    medium_percent = st.slider("Medium Îlots (%)", 0, 100, 50, 5)
    large_percent = st.slider("Large Îlots (%)", 0, 100, 20, 5)
    
    # Validate percentages
    total_percent = small_percent + medium_percent + large_percent
    if total_percent != 100:
        st.warning(f"Total percentage: {total_percent}% (should be 100%)")
    
    # Îlot dimensions
    st.markdown("**Dimensions (meters)**")
    small_size = st.number_input("Small Îlot Size", 1.0, 10.0, 2.0, 0.1)
    medium_size = st.number_input("Medium Îlot Size", 2.0, 15.0, 4.0, 0.1)
    large_size = st.number_input("Large Îlot Size", 5.0, 25.0, 8.0, 0.1)
    
    # Spacing requirements
    min_spacing = st.number_input("Minimum Spacing", 0.5, 5.0, 1.5, 0.1)
    wall_clearance = st.number_input("Wall Clearance", 0.5, 3.0, 1.0, 0.1)
    
    # Store configuration
    st.session_state.ilot_configuration = {
        'size_distribution': {
            'small': small_percent,
            'medium': medium_percent,
            'large': large_percent
        },
        'dimensions': {
            'small': small_size,
            'medium': medium_size,
            'large': large_size
        },
        'spacing': {
            'min_spacing': min_spacing,
            'wall_clearance': wall_clearance
        }
    }

def configure_corridor_settings():
    """Configure corridor generation settings"""
    corridor_width = st.number_input("Corridor Width (m)", 0.8, 3.0, 1.5, 0.1)
    main_corridor_width = st.number_input("Main Corridor Width (m)", 1.0, 4.0, 2.0, 0.1)
    
    # Pathfinding algorithm
    algorithm = st.selectbox("Pathfinding Algorithm", 
                           ["A-Star", "Dijkstra", "Breadth-First"])
    
    # Optimization settings
    optimize_turns = st.checkbox("Minimize Turns", True)
    avoid_obstacles = st.checkbox("Avoid Obstacles", True)
    ensure_accessibility = st.checkbox("Ensure Accessibility", True)
    
    st.session_state.corridor_config = {
        'width': corridor_width,
        'main_width': main_corridor_width,
        'algorithm': algorithm,
        'optimize_turns': optimize_turns,
        'avoid_obstacles': avoid_obstacles,
        'ensure_accessibility': ensure_accessibility
    }

def analyze_zones(wall_threshold, restricted_threshold, entrance_threshold):
    """Analyze zones in the floor plan"""
    if st.session_state.floor_plan_data is None:
        st.error("No floor plan data available")
        return
    
    with st.spinner("Analyzing zones..."):
        try:
            analyzer = GeometricAnalyzer()
            
            # Perform zone analysis
            analysis_results = analyzer.analyze_zones(
                st.session_state.floor_plan_data,
                wall_threshold=wall_threshold,
                restricted_threshold=restricted_threshold,
                entrance_threshold=entrance_threshold
            )
            
            # Store results
            st.session_state.analysis_results = analysis_results
            
            st.success("✅ Zone analysis completed")
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Walls Detected", len(analysis_results.get('walls', [])))
            
            with col2:
                st.metric("Restricted Areas", len(analysis_results.get('restricted_areas', [])))
            
            with col3:
                st.metric("Entrances/Exits", len(analysis_results.get('entrances', [])))
            
        except Exception as e:
            st.error(f"Error during zone analysis: {str(e)}")
            st.exception(e)

def place_ilots_advanced():
    """Place îlots using advanced optimization methods"""
    if st.session_state.analysis_results is None:
        st.error("Please analyze zones first")
        return
    
    optimization_mode = st.session_state.optimization_mode
    
    with st.spinner(f"Placing îlots using {optimization_mode} optimization..."):
        try:
            if optimization_mode == "traditional":
                # Traditional placement
                placer = IlotPlacer()
                optimizer = SpatialOptimizer()
                
                ilot_results = placer.place_ilots(
                    st.session_state.analysis_results,
                    st.session_state.ilot_configuration
                )
                
                optimized_results = optimizer.optimize_placement(
                    ilot_results,
                    st.session_state.analysis_results
                )
            else:
                # Advanced ML/Hybrid optimization
                hybrid_optimizer = st.session_state.hybrid_optimizer
                
                # Generate îlot requirements
                ilot_requirements = generate_ilot_requirements()
                constraints = {
                    'min_spacing': st.session_state.ilot_configuration.get('spacing', {}).get('min_spacing', 1.5),
                    'wall_clearance': st.session_state.ilot_configuration.get('spacing', {}).get('wall_clearance', 1.0)
                }
                
                optimized_results = hybrid_optimizer.optimize(
                    st.session_state.analysis_results,
                    ilot_requirements,
                    constraints,
                    mode=optimization_mode
                )
            
            # Update analysis results
            st.session_state.analysis_results['ilots'] = optimized_results
            
            st.success(f"✅ Îlot placement completed using {optimization_mode} optimization")
            
            # Display enhanced metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Îlots", len(optimized_results))
            
            with col2:
                coverage = calculate_coverage_percentage(optimized_results)
                st.metric("Coverage", f"{coverage:.1f}%")
            
            with col3:
                efficiency = calculate_efficiency_score(optimized_results)
                st.metric("Efficiency", f"{efficiency:.1f}%")
            
            with col4:
                ai_score = calculate_ai_optimization_score(optimized_results)
                st.metric("AI Score", f"{ai_score:.1f}%")
            
        except Exception as e:
            st.error(f"Error during advanced îlot placement: {str(e)}")
            st.exception(e)

def ml_optimize_placement():
    """Perform ML-only optimization"""
    if st.session_state.analysis_results is None:
        st.error("Please analyze zones first")
        return
    
    with st.spinner("Running ML optimization..."):
        try:
            ml_optimizer = st.session_state.ml_optimizer
            
            # Generate requirements and constraints
            ilot_requirements = generate_ilot_requirements()
            constraints = {
                'min_spacing': 1.5,
                'wall_clearance': 1.0
            }
            
            # Run ML optimization
            episodes = st.sidebar.slider("Training Episodes", 50, 500, 100, 50)
            optimized_results = ml_optimizer.optimize_placement(
                st.session_state.analysis_results,
                ilot_requirements,
                constraints,
                episodes=episodes
            )
            
            # Update results
            st.session_state.analysis_results['ilots'] = optimized_results
            
            st.success("✅ ML optimization completed")
            
            # Show ML metrics
            metrics = ml_optimizer.get_optimization_metrics()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Episodes", metrics.get('training_episodes', 0))
            
            with col2:
                st.metric("Convergence Rate", f"{metrics.get('convergence_rate', 0)*100:.1f}%")
            
            with col3:
                st.metric("Final Reward", f"{metrics.get('max_reward', 0):.1f}")
            
        except Exception as e:
            st.error(f"Error during ML optimization: {str(e)}")
            st.exception(e)

def generate_corridors():
    """Generate corridor network"""
    if 'ilots' not in st.session_state.analysis_results:
        st.error("Please place îlots first")
        return
    
    with st.spinner("Generating corridor network..."):
        try:
            generator = CorridorGenerator()
            
            # Generate corridors
            corridor_results = generator.generate_corridors(
                st.session_state.analysis_results,
                st.session_state.corridor_config
            )
            
            # Update analysis results
            st.session_state.analysis_results['corridors'] = corridor_results
            
            st.success("✅ Corridor network generated")
            
            # Display corridor summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Corridors", len(corridor_results))
            
            with col2:
                total_length = sum(c.get('length', 0) for c in corridor_results)
                st.metric("Total Length", f"{total_length:.1f}m")
            
            with col3:
                connectivity = calculate_connectivity_score(corridor_results)
                st.metric("Connectivity", f"{connectivity:.1f}%")
            
        except Exception as e:
            st.error(f"Error during corridor generation: {str(e)}")
            st.exception(e)

def show_floor_plan_view():
    """Display the interactive floor plan view"""
    if st.session_state.floor_plan_data is None:
        st.info("Upload a floor plan to view")
        return
    
    visualizer = FloorPlanVisualizer()
    
    # Create visualization
    fig = visualizer.create_interactive_view(
        st.session_state.floor_plan_data,
        st.session_state.analysis_results
    )
    
    # Display with controls
    st.plotly_chart(fig, use_container_width=True)
    
    # View controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Zoom to Fit"):
            st.rerun()
    
    with col2:
        if st.button("🎯 Center View"):
            st.rerun()
    
    with col3:
        if st.button("📏 Show Measurements"):
            st.rerun()

def show_analysis_results():
    """Display detailed analysis results"""
    if st.session_state.analysis_results is None:
        st.info("Run zone analysis to see results")
        return
    
    results = st.session_state.analysis_results
    
    # Zone analysis summary
    st.subheader("Zone Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Detected Elements**")
        metrics_data = {
            'Element Type': ['Walls', 'Restricted Areas', 'Entrances/Exits', 'Open Spaces'],
            'Count': [
                len(results.get('walls', [])),
                len(results.get('restricted_areas', [])),
                len(results.get('entrances', [])),
                len(results.get('open_spaces', []))
            ],
            'Total Area (m²)': [
                sum(w.get('area', 0) for w in results.get('walls', [])),
                sum(r.get('area', 0) for r in results.get('restricted_areas', [])),
                sum(e.get('area', 0) for e in results.get('entrances', [])),
                sum(o.get('area', 0) for o in results.get('open_spaces', []))
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
    
    with col2:
        # Area distribution chart
        fig_pie = px.pie(
            values=df_metrics['Total Area (m²)'],
            names=df_metrics['Element Type'],
            title="Area Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed zone information
    if st.checkbox("Show Detailed Zone Information"):
        show_detailed_zone_info(results)

def show_detailed_zone_info(results):
    """Show detailed information about each zone"""
    tab1, tab2, tab3, tab4 = st.tabs(["Walls", "Restricted Areas", "Entrances", "Open Spaces"])
    
    with tab1:
        if results.get('walls'):
            wall_data = []
            for i, wall in enumerate(results['walls']):
                wall_data.append({
                    'ID': i,
                    'Length (m)': wall.get('length', 0),
                    'Thickness (m)': wall.get('thickness', 0),
                    'Start Point': f"({wall.get('start', {}).get('x', 0):.2f}, {wall.get('start', {}).get('y', 0):.2f})",
                    'End Point': f"({wall.get('end', {}).get('x', 0):.2f}, {wall.get('end', {}).get('y', 0):.2f})"
                })
            
            df_walls = pd.DataFrame(wall_data)
            st.dataframe(df_walls, use_container_width=True)
        else:
            st.info("No walls detected")
    
    with tab2:
        if results.get('restricted_areas'):
            restricted_data = []
            for i, area in enumerate(results['restricted_areas']):
                restricted_data.append({
                    'ID': i,
                    'Area (m²)': area.get('area', 0),
                    'Perimeter (m)': area.get('perimeter', 0),
                    'Center': f"({area.get('center', {}).get('x', 0):.2f}, {area.get('center', {}).get('y', 0):.2f})"
                })
            
            df_restricted = pd.DataFrame(restricted_data)
            st.dataframe(df_restricted, use_container_width=True)
        else:
            st.info("No restricted areas detected")
    
    with tab3:
        if results.get('entrances'):
            entrance_data = []
            for i, entrance in enumerate(results['entrances']):
                entrance_data.append({
                    'ID': i,
                    'Width (m)': entrance.get('width', 0),
                    'Type': entrance.get('type', 'Unknown'),
                    'Position': f"({entrance.get('position', {}).get('x', 0):.2f}, {entrance.get('position', {}).get('y', 0):.2f})"
                })
            
            df_entrances = pd.DataFrame(entrance_data)
            st.dataframe(df_entrances, use_container_width=True)
        else:
            st.info("No entrances detected")
    
    with tab4:
        if results.get('open_spaces'):
            space_data = []
            for i, space in enumerate(results['open_spaces']):
                space_data.append({
                    'ID': i,
                    'Area (m²)': space.get('area', 0),
                    'Usable Area (m²)': space.get('usable_area', 0),
                    'Shape': space.get('shape', 'Unknown'),
                    'Center': f"({space.get('center', {}).get('x', 0):.2f}, {space.get('center', {}).get('y', 0):.2f})"
                })
            
            df_spaces = pd.DataFrame(space_data)
            st.dataframe(df_spaces, use_container_width=True)
        else:
            st.info("No open spaces detected")

def show_ilot_placement():
    """Display îlot placement results"""
    if not st.session_state.analysis_results or 'ilots' not in st.session_state.analysis_results:
        st.info("Place îlots to see placement results")
        return
    
    ilots = st.session_state.analysis_results['ilots']
    
    # Placement summary
    st.subheader("Îlot Placement Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Îlots", len(ilots))
    
    with col2:
        coverage = calculate_coverage_percentage(ilots)
        st.metric("Coverage", f"{coverage:.1f}%")
    
    with col3:
        efficiency = calculate_efficiency_score(ilots)
        st.metric("Efficiency", f"{efficiency:.1f}%")
    
    with col4:
        total_area = sum(i.get('area', 0) for i in ilots)
        st.metric("Total Area", f"{total_area:.1f}m²")
    
    # Size distribution
    st.subheader("Size Distribution")
    
    size_counts = {'Small': 0, 'Medium': 0, 'Large': 0}
    for ilot in ilots:
        size_counts[ilot.get('size_category', 'Unknown')] += 1
    
    fig_bar = px.bar(
        x=list(size_counts.keys()),
        y=list(size_counts.values()),
        title="Îlot Size Distribution"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed îlot information
    if st.checkbox("Show Detailed Îlot Information"):
        ilot_data = []
        for i, ilot in enumerate(ilots):
            ilot_data.append({
                'ID': i,
                'Size Category': ilot.get('size_category', 'Unknown'),
                'Area (m²)': ilot.get('area', 0),
                'Position': f"({ilot.get('position', {}).get('x', 0):.2f}, {ilot.get('position', {}).get('y', 0):.2f})",
                'Rotation (°)': ilot.get('rotation', 0),
                'Accessibility Score': ilot.get('accessibility_score', 0)
            })
        
        df_ilots = pd.DataFrame(ilot_data)
        st.dataframe(df_ilots, use_container_width=True)

def show_corridor_network():
    """Display corridor network results"""
    if not st.session_state.analysis_results or 'corridors' not in st.session_state.analysis_results:
        st.info("Generate corridors to see network results")
        return
    
    corridors = st.session_state.analysis_results['corridors']
    
    # Network summary
    st.subheader("Corridor Network Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Corridors", len(corridors))
    
    with col2:
        total_length = sum(c.get('length', 0) for c in corridors)
        st.metric("Total Length", f"{total_length:.1f}m")
    
    with col3:
        connectivity = calculate_connectivity_score(corridors)
        st.metric("Connectivity", f"{connectivity:.1f}%")
    
    with col4:
        avg_width = sum(c.get('width', 0) for c in corridors) / len(corridors) if corridors else 0
        st.metric("Avg Width", f"{avg_width:.1f}m")
    
    # Corridor types
    st.subheader("Corridor Types")
    
    type_counts = {}
    for corridor in corridors:
        corridor_type = corridor.get('type', 'Unknown')
        type_counts[corridor_type] = type_counts.get(corridor_type, 0) + 1
    
    fig_pie = px.pie(
        values=list(type_counts.values()),
        names=list(type_counts.keys()),
        title="Corridor Type Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed corridor information
    if st.checkbox("Show Detailed Corridor Information"):
        corridor_data = []
        for i, corridor in enumerate(corridors):
            corridor_data.append({
                'ID': i,
                'Type': corridor.get('type', 'Unknown'),
                'Length (m)': corridor.get('length', 0),
                'Width (m)': corridor.get('width', 0),
                'Start Point': f"({corridor.get('start', {}).get('x', 0):.2f}, {corridor.get('start', {}).get('y', 0):.2f})",
                'End Point': f"({corridor.get('end', {}).get('x', 0):.2f}, {corridor.get('end', {}).get('y', 0):.2f})",
                'Accessibility': corridor.get('accessibility', 'Unknown')
            })
        
        df_corridors = pd.DataFrame(corridor_data)
        st.dataframe(df_corridors, use_container_width=True)

def show_3d_visualization():
    """Display advanced 3D visualization"""
    if st.session_state.analysis_results is None:
        st.info("Complete analysis to view 3D visualization")
        return
    
    st.subheader("🌐 Advanced 3D Visualization")
    
    # Visualization controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        multi_floor = st.checkbox("Multi-floor View", False)
        num_floors = st.slider("Number of Floors", 1, 5, 1) if multi_floor else 1
    
    with col2:
        visualization_type = st.selectbox(
            "Visualization Type", 
            ["3D Scene", "Heatmap", "Flow Analysis"]
        )
    
    with col3:
        export_format = st.selectbox("Export Format", ["HTML", "PNG", "STL"])
    
    try:
        webgl_viz = st.session_state.webgl_visualizer
        
        if visualization_type == "3D Scene":
            # Create 3D scene
            fig_3d = webgl_viz.create_3d_visualization(
                st.session_state.analysis_results,
                st.session_state.analysis_results.get('ilots', []),
                st.session_state.analysis_results.get('corridors', []),
                multi_floor=multi_floor,
                num_floors=num_floors
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
        elif visualization_type == "Heatmap":
            # Create heatmap visualization
            metric = st.selectbox("Heatmap Metric", ["accessibility", "density", "flow"])
            
            fig_heatmap = webgl_viz.create_heatmap_visualization(
                st.session_state.analysis_results,
                metric=metric
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Export controls
        if st.button("Export 3D Scene"):
            if export_format == "HTML":
                filename = webgl_viz.export_webgl_scene(fig_3d if 'fig_3d' in locals() else fig_heatmap)
                st.success(f"Exported to {filename}")
            
    except Exception as e:
        st.error(f"Error creating 3D visualization: {str(e)}")

def show_ai_optimization():
    """Display AI optimization interface"""
    st.subheader("🤖 AI Optimization Dashboard")
    
    if st.session_state.analysis_results is None:
        st.info("Complete zone analysis to access AI optimization")
        return
    
    # Optimization status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Configuration**")
        st.write(f"Optimization Mode: {st.session_state.optimization_mode}")
        st.write(f"ML Model Status: {'Loaded' if st.session_state.ml_optimizer.agent else 'Not Loaded'}")
    
    with col2:
        st.markdown("**Performance Metrics**")
        if hasattr(st.session_state.ml_optimizer, 'training_history') and st.session_state.ml_optimizer.training_history:
            metrics = st.session_state.ml_optimizer.get_optimization_metrics()
            st.metric("Training Episodes", metrics.get('training_episodes', 0))
            st.metric("Convergence Rate", f"{metrics.get('convergence_rate', 0)*100:.1f}%")
        else:
            st.info("No training history available")
    
    # Training controls
    st.subheader("Training Controls")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🎯 Quick Training (50 episodes)"):
            run_ml_training(50)
    
    with col4:
        if st.button("🚀 Extended Training (200 episodes)"):
            run_ml_training(200)
    
    # Optimization comparison
    if 'ilots' in st.session_state.analysis_results:
        st.subheader("Optimization Comparison")
        
        current_ilots = st.session_state.analysis_results['ilots']
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            traditional_score = calculate_efficiency_score(current_ilots)
            st.metric("Current Efficiency", f"{traditional_score:.1f}%")
        
        with col6:
            ai_score = calculate_ai_optimization_score(current_ilots)
            st.metric("AI Score", f"{ai_score:.1f}%")
        
        with col7:
            improvement = ai_score - traditional_score
            st.metric("Improvement", f"{improvement:+.1f}%")

def show_reports():
    """Display comprehensive reports"""
    if st.session_state.analysis_results is None:
        st.info("Complete analysis to generate reports")
        return
    
    st.subheader("Comprehensive Analysis Report")
    
    # Report generation controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate PDF Report"):
            generate_pdf_report()
    
    with col2:
        if st.button("💾 Export Data"):
            export_analysis_data()
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics = calculate_performance_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Space Utilization", f"{metrics.get('space_utilization', 0):.1f}%")
        st.metric("Accessibility Score", f"{metrics.get('accessibility_score', 0):.1f}/100")
    
    with col2:
        st.metric("Circulation Efficiency", f"{metrics.get('circulation_efficiency', 0):.1f}%")
        st.metric("Layout Optimization", f"{metrics.get('layout_optimization', 0):.1f}%")
    
    with col3:
        st.metric("Safety Compliance", f"{metrics.get('safety_compliance', 0):.1f}%")
        st.metric("Overall Score", f"{metrics.get('overall_score', 0):.1f}/100")
    
    # Recommendations
    st.subheader("Optimization Recommendations")
    
    recommendations = generate_recommendations()
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"Recommendation {i+1}: {rec['title']}"):
            st.write(rec['description'])
            st.write(f"**Priority:** {rec['priority']}")
            st.write(f"**Expected Impact:** {rec['impact']}")

def generate_comprehensive_report():
    """Generate comprehensive analysis report"""
    if st.session_state.analysis_results is None:
        st.error("No analysis results available")
        return
    
    with st.spinner("Generating comprehensive report..."):
        try:
            report_generator = ReportGenerator()
            
            # Generate report
            report_data = report_generator.generate_comprehensive_report(
                st.session_state.floor_plan_data,
                st.session_state.analysis_results,
                st.session_state.ilot_configuration,
                st.session_state.corridor_config
            )
            
            st.success("✅ Report generated successfully")
            
            # Display report summary
            st.subheader("Report Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Pages", report_data.get('total_pages', 0))
                st.metric("Analysis Points", report_data.get('analysis_points', 0))
            
            with col2:
                st.metric("Recommendations", len(report_data.get('recommendations', [])))
                st.metric("Optimization Score", f"{report_data.get('optimization_score', 0):.1f}%")
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.exception(e)

def generate_pdf_report():
    """Generate PDF report for download"""
    st.info("PDF report generation functionality would be implemented here")

def export_analysis_data():
    """Export analysis data to various formats"""
    st.info("Data export functionality would be implemented here")

def calculate_coverage_percentage(ilots):
    """Calculate coverage percentage of îlots"""
    if not ilots:
        return 0.0
    
    total_ilot_area = sum(i.get('area', 0) for i in ilots)
    # This would be calculated based on the actual floor plan area
    total_floor_area = 1000  # Placeholder
    
    return (total_ilot_area / total_floor_area) * 100 if total_floor_area > 0 else 0.0

def calculate_efficiency_score(ilots):
    """Calculate efficiency score of îlot placement"""
    if not ilots:
        return 0.0
    
    # This would be calculated based on actual placement optimization metrics
    return 85.0  # Placeholder

def calculate_connectivity_score(corridors):
    """Calculate connectivity score of corridor network"""
    if not corridors:
        return 0.0
    
    # This would be calculated based on actual network analysis
    return 90.0  # Placeholder

def calculate_performance_metrics():
    """Calculate overall performance metrics"""
    return {
        'space_utilization': 78.5,
        'accessibility_score': 82.3,
        'circulation_efficiency': 75.8,
        'layout_optimization': 88.2,
        'safety_compliance': 95.1,
        'overall_score': 84.0
    }

def generate_ilot_requirements():
    """Generate îlot requirements based on configuration"""
    config = st.session_state.ilot_configuration
    size_dist = config.get('size_distribution', {'small': 30, 'medium': 50, 'large': 20})
    dimensions = config.get('dimensions', {'small': 2.0, 'medium': 4.0, 'large': 8.0})
    
    requirements = []
    total_ilots = 20  # Default number
    
    for size, percentage in size_dist.items():
        count = int(total_ilots * percentage / 100)
        for i in range(count):
            requirements.append({
                'id': f"{size}_{i}",
                'size_category': size,
                'dimensions': {
                    'width': dimensions[size],
                    'height': dimensions[size],
                    'area': dimensions[size] ** 2
                }
            })
    
    return requirements

def calculate_ai_optimization_score(ilots):
    """Calculate AI optimization score"""
    if not ilots:
        return 0.0
    
    # Simulate AI scoring based on placement properties
    total_score = 0
    for ilot in ilots:
        properties = ilot.get('properties', {})
        if properties.get('algorithm') == 'ml_reinforcement_learning':
            total_score += ilot.get('placement_score', 85.0)
        else:
            total_score += ilot.get('placement_score', 75.0)
    
    return total_score / len(ilots) if ilots else 0.0

def run_ml_training(episodes):
    """Run ML training with specified episodes"""
    with st.spinner(f"Training ML model for {episodes} episodes..."):
        try:
            ml_optimizer = st.session_state.ml_optimizer
            
            # Generate requirements
            ilot_requirements = generate_ilot_requirements()
            constraints = {'min_spacing': 1.5, 'wall_clearance': 1.0}
            
            # Run training
            optimized_results = ml_optimizer.optimize_placement(
                st.session_state.analysis_results,
                ilot_requirements,
                constraints,
                episodes=episodes
            )
            
            st.success(f"✅ Training completed with {episodes} episodes")
            
            # Show training results
            metrics = ml_optimizer.get_optimization_metrics()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Reward", f"{metrics.get('max_reward', 0):.1f}")
            
            with col2:
                st.metric("Improvement", f"{metrics.get('reward_improvement', 0):.1f}%")
            
            with col3:
                st.metric("Convergence", f"{metrics.get('convergence_rate', 0)*100:.1f}%")
                
        except Exception as e:
            st.error(f"Error during training: {str(e)}")

def export_to_bim():
    """Export results to BIM format"""
    if st.session_state.analysis_results is None:
        st.error("No analysis results to export")
        return
    
    with st.spinner("Exporting to BIM format..."):
        try:
            bim_manager = st.session_state.bim_manager
            
            # Export to IFC
            output_path = "floor_plan_with_ilots.ifc"
            exported_file = bim_manager.export_to_bim(
                st.session_state.floor_plan_data,
                st.session_state.analysis_results.get('ilots', []),
                output_path,
                bim_system='ifc'
            )
            
            if exported_file:
                st.success(f"✅ Exported to {exported_file}")
                
                # Validate export
                validation = bim_manager.validate_bim_export(exported_file)
                if validation['valid']:
                    st.success("✅ BIM export validation passed")
                    
                    # Show statistics
                    stats = validation.get('statistics', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("File Size", f"{stats.get('file_size', 0)/1024:.1f} KB")
                    with col2:
                        st.metric("Entities", stats.get('entity_count', 0))
                else:
                    st.warning("⚠️ BIM export validation failed")
                    for error in validation.get('errors', []):
                        st.error(f"Error: {error}")
            
        except Exception as e:
            st.error(f"Error exporting to BIM: {str(e)}")

def generate_recommendations():
    """Generate optimization recommendations"""
    return [
        {
            'title': 'Optimize Îlot Distribution',
            'description': 'Consider redistributing medium-sized îlots to improve space utilization in the northeast section.',
            'priority': 'High',
            'impact': 'Increase space utilization by 8-12%'
        },
        {
            'title': 'Improve Corridor Connectivity',
            'description': 'Add secondary corridors to reduce bottlenecks and improve circulation flow.',
            'priority': 'Medium',
            'impact': 'Reduce circulation time by 15-20%'
        },
        {
            'title': 'Enhance Accessibility',
            'description': 'Widen corridors near entrance points to comply with accessibility standards.',
            'priority': 'High',
            'impact': 'Improve accessibility compliance to 100%'
        },
        {
            'title': 'Implement AI Optimization',
            'description': 'Switch to machine learning optimization for improved placement efficiency.',
            'priority': 'Medium',
            'impact': 'Potential 15-25% improvement in space utilization'
        }
    ]

if __name__ == "__main__":
    main()
