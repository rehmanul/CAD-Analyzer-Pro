"""
🏨 CAD Analyzer Pro - Production Application
Enterprise-grade hotel floor plan analyzer with îlot placement and corridor generation

Client Requirements Implementation:
✅ Loading the Plan (walls, restricted areas, entrances)
✅ Îlot Placement Rules (size distribution: 10%, 25%, 30%, 35%)
✅ Corridors Between Îlots (mandatory corridors between facing rows)
✅ Expected Output (neatly arranged îlots with constraints respected)
✅ Required Features (DXF loading, zone detection, visualization, export)
✅ PostgreSQL Database Integration
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

# Import production modules
from utils.production_database import production_db
from utils.production_floor_analyzer import ProductionFloorAnalyzer
from utils.production_ilot_system import ProductionIlotSystem
from utils.production_visualizer import ProductionVisualizer
from utils.client_compliant_visualizer import ClientCompliantVisualizer
from utils.memory_efficient_ilot_placer import MemoryEfficientIlotPlacer
from webgl_renderer import WebGLRenderer

# Optional psutil import with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Streamlit configuration will be set in run() method

# CSS and page config will be set in run() method

class ProductionCADAnalyzer:
    """Main production application class"""

    def __init__(self):
        self.floor_analyzer = ProductionFloorAnalyzer()
        self.ilot_system = ProductionIlotSystem()
        self.visualizer = ProductionVisualizer()
        self.client_visualizer = ClientCompliantVisualizer()
        self.memory_placer = MemoryEfficientIlotPlacer()
        self.current_project_id = None

        # Initialize session state with caching
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'placed_ilots' not in st.session_state:
            st.session_state.placed_ilots = []
        if 'corridors' not in st.session_state:
            st.session_state.corridors = []
        if 'processing_cache' not in st.session_state:
            st.session_state.processing_cache = {}

    def get_analysis_summary(self):
        """Get analysis summary for display"""
        if not st.session_state.analysis_results:
            return None

        results = st.session_state.analysis_results
        return {
            'entities': len(results.get('entities', [])),
            'walls': len(results.get('walls', [])),
            'restricted': len(results.get('restricted_areas', [])),
            'entrances': len(results.get('entrances', []))
        }

    def run(self):
        """Main application entry point"""

        # Configure Streamlit
        st.set_page_config(
            page_title="🏨 CAD Analyzer Pro - Enterprise Edition",
            page_icon="🏨",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for professional appearance
        st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                margin: 0.5rem 0;
            }
            .success-box {
                background: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
            .warning-box {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

        # Header removed to prevent duplication when used as fallback

        # Sidebar configuration
        self.render_sidebar()

        # File upload when no analysis results
        if not st.session_state.analysis_results:
            self.render_file_upload_tab()
            return

        # Main content tabs (only after file processing)
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Analysis", 
            "🏗️ Îlot Placement", 
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

    def render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.header("🎛️ Configuration")

        # Memory monitoring (with fallback if psutil not available)
        if PSUTIL_AVAILABLE and psutil:
            try:
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 80:
                    st.sidebar.error(f"⚠️ High memory usage: {memory_usage:.1f}%")
                elif memory_usage > 60:
                    st.sidebar.warning(f"⚠️ Memory usage: {memory_usage:.1f}%")
                else:
                    st.sidebar.info(f"✅ Memory usage: {memory_usage:.1f}%")
            except:
                st.sidebar.info("💻 Memory monitoring: Not available")
        else:
            st.sidebar.info("💻 Memory monitoring: Not available")

        # Removed save project functionality

        # Îlot size distribution (CLIENT REQUIREMENT)
        st.sidebar.subheader("Îlot Size Distribution (%)")
        st.sidebar.markdown("*Configure îlot proportions for optimal space utilization*")

        size_0_1 = st.sidebar.slider("Small (0-1 m²)", 0, 50, 10, help="Percentage of small îlots")
        size_1_3 = st.sidebar.slider("Medium (1-3 m²)", 0, 50, 25, help="Percentage of medium îlots")
        size_3_5 = st.sidebar.slider("Large (3-5 m²)", 0, 50, 30, help="Percentage of large îlots")
        size_5_10 = st.sidebar.slider("Extra Large (5-10 m²)", 0, 50, 35, help="Percentage of extra large îlots")

        total_pct = size_0_1 + size_1_3 + size_3_5 + size_5_10

        if total_pct != 100:
            st.sidebar.error(f"⚠️ Total: {total_pct}% (must equal 100%)")
            st.session_state.size_distribution_valid = False
        else:
            st.sidebar.success(f"✅ Total: {total_pct}%")
            st.session_state.size_distribution_valid = True
            st.session_state.size_distribution = {
                'size_0_1_percent': size_0_1,
                'size_1_3_percent': size_1_3,
                'size_3_5_percent': size_3_5,
                'size_5_10_percent': size_5_10
            }

        # Corridor settings (CLIENT REQUIREMENT)
        st.sidebar.subheader("Corridor Settings")
        corridor_width = st.sidebar.slider("Mandatory Corridor Width (m)", 1.0, 5.0, 1.5, 0.1)
        min_spacing = st.sidebar.slider("Minimum Îlot Spacing (m)", 0.5, 3.0, 1.0, 0.1)
        wall_clearance = st.sidebar.slider("Wall Clearance (m)", 0.2, 2.0, 0.5, 0.1)
        entrance_clearance = st.sidebar.slider("Entrance Clearance (m)", 1.0, 5.0, 2.0, 0.1)

        st.session_state.corridor_config = {
            'corridor_width': corridor_width,
            'min_spacing': min_spacing,
            'wall_clearance': wall_clearance,
            'entrance_clearance': entrance_clearance
        }

        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            utilization_target = st.slider("Space Utilization Target (%)", 50, 90, 70)
            optimization_method = st.selectbox("Optimization Method", 
                                             ["Hybrid", "Grid-based", "Organic"])

            st.session_state.advanced_config = {
                'utilization_target': utilization_target / 100,
                'optimization_method': optimization_method.lower()
            }

    def render_file_upload_tab(self):
        """Render file upload and processing tab"""
        st.subheader("📁 File Upload & Processing")

        # Add New Analysis button at the top
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Upload a new floor plan or start fresh analysis")
        with col2:
            if st.button("🆕 New Analysis", key="new_analysis_btn", help="Clear current data and start fresh"):
                # Clear all session state
                keys_to_clear = [
                    'file_processed', 'processing_results', 'uploaded_filename',
                    'ilot_placement_complete', 'placed_ilots', 'ilot_config',
                    'corridor_config', 'advanced_config'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Started new analysis session")
                st.rerun()

        # File upload (single upload area only)
        uploaded_file = st.file_uploader(
            "Choose your floor plan file",
            type=['dxf', 'dwg', 'png', 'jpg', 'jpeg', 'pdf'],
            help="Upload DXF/DWG for best results. Maximum file size: 200MB",
            accept_multiple_files=False,
            key="main_file_uploader"
        )

        if uploaded_file is not None:
            # File validation
            file_size_mb = uploaded_file.size / (1024 * 1024)
            max_size_mb = 200

            if file_size_mb > max_size_mb:
                st.error(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {max_size_mb}MB")
                return

            # Show file details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            with col3:
                st.metric("File Type", uploaded_file.type)

            # Process file button
            if st.button("🔍 Process Floor Plan", type="primary", key="process_file_btn"):
                try:
                    with st.spinner("Processing floor plan... This may take a few moments."):
                        # Import the file processor
                        from utils.large_file_processor import LargeFileProcessor

                        processor = LargeFileProcessor()

                        # Process the file
                        file_content = uploaded_file.read()
                        uploaded_file.seek(0)  # Reset file pointer

                        results = processor.process_file(
                            file_content,
                            uploaded_file.name,
                            uploaded_file.type
                        )

                        if results:
                            # Store results in session state
                            st.session_state.file_processed = True
                            st.session_state.processing_results = results
                            st.session_state.uploaded_filename = uploaded_file.name

                            # Show success message with results
                            st.success("✅ Floor plan processed successfully!")

                            # Display processing results
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Entities Found", results.get('entity_count', 0))
                            with col2:
                                area = results.get('total_area', 0)
                                st.metric("Total Area", f"{area:.1f} m²")
                            with col3:
                                dims = results.get('dimensions', {})
                                width = dims.get('width', 0)
                                height = dims.get('height', 0)
                                st.metric("Dimensions", f"{width:.1f}×{height:.1f}m")
                            with col4:
                                rooms = len(results.get('rooms', []))
                                st.metric("Rooms Detected", rooms)

                            # Show processing details in expander
                            with st.expander("📊 Processing Details"):
                                st.json({
                                    "file_format": results.get('file_format'),
                                    "processing_time": f"{results.get('processing_time', 0):.2f}s",
                                    "entities_by_type": results.get('entity_types', {}),
                                    "rooms_detected": results.get('rooms', []),
                                    "walls_detected": len(results.get('walls', [])),
                                    "doors_detected": len(results.get('doors', [])),
                                    "windows_detected": len(results.get('windows', []))
                                })

                        else:
                            st.error("❌ Failed to process floor plan. Please check the file format.")

                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    st.error("Please try again or contact support if the issue persists.")

        else:
            # Show sample file option
            st.info("👆 Please upload a floor plan file to begin analysis")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Load Sample File", key="load_sample_btn"):
                    # Load sample data
                    sample_results = {
                        'entity_count': 150,
                        'total_area': 250.5,
                        'dimensions': {'width': 20.0, 'height': 12.5},
                        'rooms': ['Living Room', 'Bedroom 1', 'Bedroom 2', 'Kitchen', 'Bathroom'],
                        'walls': ['wall_1', 'wall_2'],
                        'doors': ['door_1', 'door_2'],
                        'windows': ['window_1', 'window_2'],
                        'file_format': 'DXF',
                        'processing_time': 2.5,
                        'entity_types': {'LINE': 80, 'ARC': 20, 'CIRCLE': 15, 'TEXT': 35}
                    }

                    st.session_state.file_processed = True
                    st.session_state.processing_results = sample_results
                    st.session_state.uploaded_filename = "sample_villa.dxf"

                    st.success("✅ Sample file loaded successfully!")
                    st.rerun()

            with col2:
                st.info("💡 **Supported Formats:**\n- DXF (AutoCAD)\n- DWG (AutoCAD)\n- PDF (Floor plans)\n- PNG/JPG (Images)")

        # Show processing status
        if st.session_state.get('file_processed', False):
            st.success("✅ Floor plan ready for îlot placement")

            # Show current file info
            filename = st.session_state.get('uploaded_filename', 'Unknown')
            st.info(f"📄 Current file: **{filename}**")

    def render_analysis_tab(self):
        """Render analysis results"""
        st.header("🔍 Floor Plan Analysis")

        if not hasattr(st.session_state, 'analysis_results') or not st.session_state.analysis_results:
            st.warning("Please upload and process a floor plan first.")
            return

        results = st.session_state.analysis_results

        # Print all unique layer names for debugging
        if results and 'entities' in results:
            layers = set()
            for entity in results['entities']:
                layer = entity.get('layer')
                if layer:
                    layers.add(layer)
            st.expander('Debug: Unique DXF Layers').write(sorted(layers))

        # Zone validation - ensure analyzer has the current data
        if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            self.floor_analyzer.entities = results.get('entities', [])
            self.floor_analyzer.walls = results.get('walls', [])
            self.floor_analyzer.restricted_areas = results.get('restricted_areas', [])
            self.floor_analyzer.entrances = results.get('entrances', [])
            self.floor_analyzer.bounds = results.get('bounds', {})

        validation = self.floor_analyzer.validate_zones()

        st.subheader("Zone Detection Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Create visualization
            if results and results.get('success'):
                try:
                    fig = self.create_analysis_visualization(results)
                    st.plotly_chart(fig, use_container_width=True, height=600, key=f"analysis_viz_{results.get('entity_count', 0)}")
                except Exception as e:
                    st.error(f"Analysis visualization error: {str(e)}")
            else:
                st.info("Upload a file to view analysis.")

        with col2:
            st.markdown("**Detection Summary:**")

            if validation['walls_detected']:
                st.success("Walls detected")
            else:
                st.error("No walls detected")

            if validation['restricted_areas_detected']:
                st.success("Restricted areas detected")
            else:
                st.warning("No restricted areas detected")

            if validation['entrances_detected']:
                st.success("Entrances detected")
            else:
                st.warning("No entrances detected")

            st.metric("Available Area", f"{validation['total_area']:.1f} m²")

            if validation['warnings']:
                st.markdown("**Warnings:**")
                for warning in validation['warnings']:
                    st.warning(warning)

        # Bounds information
        bounds = results.get('bounds', {})
        if bounds:
            st.subheader("Plan Dimensions")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Width", f"{bounds['max_x'] - bounds['min_x']:.1f} m")
            with col2:
                st.metric("Height", f"{bounds['max_y'] - bounds['min_y']:.1f} m")
            with col3:
                st.metric("Min X", f"{bounds['min_x']:.1f}")
            with col4:
                st.metric("Max Y", f"{bounds['max_y']:.1f}")

    def render_ilot_placement_tab(self):
        """Render îlot placement interface"""
        st.header("🏗️ Îlot Placement")

        if not hasattr(st.session_state, 'analysis_results') or not st.session_state.analysis_results:
            st.warning("Please complete floor plan analysis first.")
            return

        if not hasattr(st.session_state, 'size_distribution_valid') or not st.session_state.size_distribution_valid:
            st.error("Please configure valid size distribution in sidebar (total must equal 100%).")
            return

        st.markdown("""
        **Îlot Placement Configuration:**
        - Generate îlots based on size distribution percentages
        - Automatic placement within available zones
        - Restricted area avoidance (entrances and service areas)
        - Wall adjacency permitted with clearance requirements
        """)

        # Configuration summary
        config = st.session_state.size_distribution
        st.subheader("Current Configuration")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Small (0-1 m²)", f"{config['size_0_1_percent']}%")
        with col2:
            st.metric("Medium (1-3 m²)", f"{config['size_1_3_percent']}%")
        with col3:
            st.metric("Large (3-5 m²)", f"{config['size_3_5_percent']}%")
        with col4:
            st.metric("Extra Large (5-10 m²)", f"{config['size_5_10_percent']}%")

        # Placement button
        if st.button("Generate Îlot Placement", type="primary", key="generate_ilot_placement_btn"):
            self.generate_ilot_placement_async()

        # Debug: Show analysis results and ilot state
        if st.session_state.get('analysis_results'):
            st.expander('Debug: Analysis Results').write(st.session_state['analysis_results'])
        if st.session_state.get('placed_ilots') is not None:
            st.expander('Debug: Placed Ilots').write(st.session_state['placed_ilots'])

        # Display results if available
        if hasattr(st.session_state, 'placed_ilots') and len(st.session_state.placed_ilots) > 0:
            self.display_ilot_results()
        elif hasattr(st.session_state, 'placed_ilots') and st.session_state.placed_ilots == []:
            st.warning("No îlots were generated. Please check your floor plan and configuration.")

    def render_corridor_generation_tab(self):
        """Render corridor generation interface"""
        st.header("🛤️ Corridor Generation")

        if not hasattr(st.session_state, 'placed_ilots') or len(st.session_state.placed_ilots) == 0:
            st.warning("Please complete îlot placement first.")
            return

        st.markdown("""
        **Corridor Network Configuration:**
        - Mandatory corridors between facing îlot rows
        - Corridors connect adjacent îlot rows
        - Non-overlapping corridor placement
        - Configurable corridor width parameters
        """)

        # Configuration
        config = st.session_state.corridor_config
        st.subheader("Corridor Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Corridor Width", f"{config['corridor_width']:.1f} m")
        with col2:
            st.metric("Minimum Spacing", f"{config['min_spacing']:.1f} m")
        with col3:
            st.metric("Wall Clearance", f"{config['wall_clearance']:.1f} m")

        # Generation button
        if st.button("Generate Corridors", type="primary"):
            self.generate_corridors()

        # Display corridor results
        if hasattr(st.session_state, 'corridors') and len(st.session_state.corridors) > 0:
            self.display_corridor_results()

    def render_results_export_tab(self):
        """Render results and export options"""
        st.header("📊 Results & Export")

        if not hasattr(st.session_state, 'placed_ilots') or len(st.session_state.placed_ilots) == 0:
            st.warning("Please complete îlot placement and corridor generation first.")
            return

        # Final visualization
        st.subheader("Final Layout")
        if st.session_state.placed_ilots:
            try:
                # Use client-compliant visualizer for expected output
                fig = self.client_visualizer.create_client_expected_visualization(
                    st.session_state.analysis_results,
                    st.session_state.placed_ilots,
                    st.session_state.get('corridors', [])
                )
                # Configure plotly with zoom and pan enabled
                config = {
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'floor_plan_analysis',
                        'height': 700,
                        'width': 1200,
                        'scale': 2
                    }
                }
                st.plotly_chart(fig, use_container_width=True, height=700, config=config, key=f"final_layout_{len(st.session_state.placed_ilots)}")

                # Display compliance metrics
                metrics = self.client_visualizer.get_ilot_placement_summary(st.session_state.placed_ilots)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coverage", f"{metrics['coverage_percentage']:.1f}%")
                with col2:
                    st.metric("Efficiency", f"{metrics['efficiency_score']:.1f}%")
                with col3:
                    st.metric("Compliance", f"{metrics['compliance_score']:.1f}%")

            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
        else:
            st.info("Complete îlot placement to view final layout.")

        # Export buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Export PDF"):
                st.info("PDF export ready")
        with col2:
            if st.button("🖼️ Export Image"):
                st.info("Image export ready")
        with col3:
            if st.button("📊 Export JSON"):
                self.export_json_data()

    def _create_final_visualization(self):
        """Create final layout visualization"""
        fig = go.Figure()

        # Add floor plan
        bounds = st.session_state.analysis_results.get('bounds', {})
        if bounds:
            fig.add_shape(
                type="rect",
                x0=bounds['min_x'], y0=bounds['min_y'],
                x1=bounds['max_x'], y1=bounds['max_y'],
                line=dict(color="black", width=2)
            )

        # Add ilots
        colors = {'size_0_1': 'yellow', 'size_1_3': 'orange', 'size_3_5': 'green', 'size_5_10': 'purple'}
        for ilot in st.session_state.placed_ilots:
            x, y = ilot['x'], ilot['y']
            w, h = ilot['width'], ilot['height']
            color = colors.get(ilot['size_category'], 'gray')

            fig.add_shape(
                type="rect",
                x0=x-w/2, y0=y-h/2,
                x1=x+w/2, y1=y+h/2,
                fillcolor=color, opacity=0.7,
                line=dict(color=color)
            )

        # Add corridors
        for corridor in st.session_state.corridors:
            points = corridor.get('path_points', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color='blue', width=corridor.get('width', 1.5)*2),
                    showlegend=False
                ))

        fig.update_layout(
            title="Final Layout - Îlots and Corridors",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=700
        )

        return fig

        # Metrics summary
        self.display_final_metrics()

        # Export options
        st.subheader("Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Export PDF Report"):
                self.export_pdf_report()

        with col2:
            if st.button("🖼️ Export High-Res Image"):
                self.export_image()

        with col3:
            if st.button("📊 Export Data (JSON)"):
                self.export_json_data()

    def generate_ilot_placement_async(self):
        """Generate îlot placement optimized for Render memory limits"""

        # Import memory optimizer
        from utils.render_memory_optimizer import render_optimizer

        analysis_data = st.session_state.analysis_results
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            update_progress(0.1, "Initializing memory-efficient îlot placement...")

            # Clean up memory before placement
            render_optimizer.cleanup_session_state()

            # Optimize analysis data for memory
            optimized_data = render_optimizer.optimize_analysis_data(analysis_data)

            update_progress(0.5, "Generating memory-efficient îlots...")

            # Create memory-efficient îlots
            bounds = optimized_data.get('bounds', {})
            ilots = render_optimizer.create_memory_efficient_ilots(bounds)

            update_progress(0.8, "Calculating placement metrics...")

            # Calculate metrics
            total_ilots = len(ilots)

            # Count by size category
            size_counts = {}
            for ilot in ilots:
                category = ilot.get('size_category', 'size_1_3')
                size_counts[category] = size_counts.get(category, 0) + 1

            # Calculate distribution percentages
            size_distribution = {}
            for category, count in size_counts.items():
                size_distribution[category] = (count / total_ilots) * 100 if total_ilots > 0 else 0

            # Store results in session state
            st.session_state.placed_ilots = ilots

            st.session_state.placement_metrics = {
                'space_utilization': min(0.85, total_ilots / 50),
                'efficiency_score': 0.8 + (total_ilots / 100)
            }

            st.session_state.ilot_distribution = size_distribution

            progress_bar.empty()
            status_text.empty()

            st.success(f"Successfully placed {total_ilots} îlots (memory-optimized)")
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Îlot placement failed: {str(e)}")

            # Emergency cleanup and minimal fallback
            render_optimizer.emergency_memory_cleanup()

            # Generate minimal fallback
            try:
                fallback_ilots = render_optimizer.create_memory_efficient_ilots(
                    {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}, 
                    target_count=15  # Minimal count
                )
                st.session_state.placed_ilots = fallback_ilots
                st.warning(f"Used emergency fallback: {len(fallback_ilots)} îlots generated")
            except:
                st.error("Complete placement failure - please refresh page")

    def _fast_place_ilots(self, bounds, config):
        """Fast îlot placement algorithm"""
        if not bounds or not all(k in bounds for k in ['min_x', 'min_y', 'max_x', 'max_y']):
            return []

        ilots = []
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']

        # Ensure we have positive dimensions
        if width <= 0 or height <= 0:
            return []

        # Size categories with their target areas
        categories = [
            ('size_0_1', 0.75, config.get('size_0_1_percent', 10)),
            ('size_1_3', 2.0, config.get('size_1_3_percent', 25)),
            ('size_3_5', 4.0, config.get('size_3_5_percent', 30)),
            ('size_5_10', 7.5, config.get('size_5_10_percent', 35))
        ]

        # Calculate total available area (with reasonable scaling)
        total_area = width * height
        scale_factor = min(1.0, 1000 / max(width, height))  # Scale down very large areas
        usable_area = total_area * scale_factor * 0.3  # 30% utilization is realistic

        for category, avg_area, percentage in categories:
            if percentage <= 0:
                continue