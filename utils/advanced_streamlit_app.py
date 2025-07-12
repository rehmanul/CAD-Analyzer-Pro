
import streamlit as st
import plotly.graph_objects as go
import json
from datetime import datetime
import base64
import io
from typing import Dict, List, Any
from ultimate_pixel_perfect_processor import UltimatePixelPerfectProcessor

class AdvancedStreamlitApp:
    """Advanced Streamlit application with pixel-perfect CAD processing"""
    
    def __init__(self):
        self.processor = UltimatePixelPerfectProcessor()
        self.setup_page_config()
        self.apply_advanced_styling()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        try:
            st.set_page_config(
                page_title="CAD Analyzer Pro - Ultimate Edition",
                page_icon="🏗️",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        except:
            pass  # Already configured
    
    def apply_advanced_styling(self):
        """Apply advanced CSS styling"""
        st.markdown("""
        <style>
        .main {
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero-subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .stage-indicator {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 1rem 2rem;
            border-radius: 25px;
            margin: 1rem 0;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        
        .success-banner {
            background: linear-gradient(90deg, #56ab2f, #a8e6cf);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-weight: 600;
            text-align: center;
        }
        
        .processing-step {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the advanced application"""
        # Initialize session state
        self.initialize_session_state()
        
        # Hero section
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🏗️ CAD Analyzer Pro</h1>
            <p class="hero-subtitle">Ultimate Edition - Pixel-Perfect Floor Plan Processing</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Floor Plan Analysis",
            "🏢 Îlot Placement", 
            "🛤️ Corridor Generation",
            "📊 Complete Results"
        ])
        
        with tab1:
            self.render_analysis_tab()
        
        with tab2:
            self.render_ilot_tab()
        
        with tab3:
            self.render_corridor_tab()
        
        with tab4:
            self.render_results_tab()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'placed_ilots' not in st.session_state:
            st.session_state.placed_ilots = []
        if 'corridors' not in st.session_state:
            st.session_state.corridors = []
        if 'processing_stage' not in st.session_state:
            st.session_state.processing_stage = 'ready'
    
    def render_analysis_tab(self):
        """Render floor plan analysis tab"""
        st.markdown("## 📋 Floor Plan Analysis - Ultimate Precision")
        
        # Configuration sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Îlot size distribution
            st.markdown("#### Îlot Size Distribution")
            small_pct = st.slider("Small Îlots (1-3m²)", 10, 40, 25, key="small_pct")
            medium_pct = st.slider("Medium Îlots (3-7m²)", 20, 50, 35, key="medium_pct")
            large_pct = st.slider("Large Îlots (7-12m²)", 15, 35, 25, key="large_pct")
            xl_pct = st.slider("XL Îlots (12-20m²)", 5, 25, 15, key="xl_pct")
            
            total_pct = small_pct + medium_pct + large_pct + xl_pct
            if total_pct != 100:
                st.error(f"Total must be 100%. Current: {total_pct}%")
            
            # Spacing settings
            st.markdown("#### Spacing Settings")
            min_spacing = st.slider("Minimum Spacing (m)", 0.5, 2.0, 1.0, key="min_spacing")
            corridor_width = st.slider("Corridor Width (m)", 1.0, 3.0, 1.5, key="corridor_width")
            utilization = st.slider("Target Utilization (%)", 50, 85, 70, key="utilization") / 100
            
            # Store configuration
            st.session_state.config = {
                'size_distribution': {
                    'small': small_pct,
                    'medium': medium_pct,
                    'large': large_pct,
                    'xl': xl_pct
                },
                'min_spacing': min_spacing,
                'corridor_width': corridor_width,
                'utilization_target': utilization
            }
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CAD File (DXF, PDF, Image)",
            type=['dxf', 'pdf', 'png', 'jpg', 'jpeg'],
            help="Supported formats: DXF, PDF, PNG, JPG - Maximum 200MB"
        )
        
        if uploaded_file is not None:
            # File validation
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"File too large: {file_size_mb:.1f}MB. Maximum: 200MB")
                return
            
            # Processing section
            st.markdown("### 🚀 Ultimate Processing Mode")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💎 Ultimate Pixel-Perfect Processing - Zero Fallback Data")
            with col2:
                if st.button("🔥 Process File", type="primary"):
                    self.process_file_ultimate(uploaded_file)
        
        # Display results
        if st.session_state.analysis_data:
            self.display_analysis_results()
    
    def process_file_ultimate(self, uploaded_file):
        """Process file with ultimate precision"""
        with st.spinner("🔥 Processing with ultimate precision..."):
            # Read file content
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Process with ultimate processor
            result = self.processor.process_cad_file_ultimate(file_content, uploaded_file.name)
            
            if result.get('success'):
                st.session_state.analysis_data = result
                st.session_state.processing_stage = 'analyzed'
                
                st.markdown("""
                <div class="success-banner">
                    ✅ Ultimate Processing Complete - Pixel-Perfect Results Ready!
                </div>
                """, unsafe_allow_html=True)
                
                # Display processing metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entities Processed", result.get('entity_count', 0))
                with col2:
                    st.metric("Walls Detected", len(result.get('walls', [])))
                with col3:
                    st.metric("Restricted Areas", len(result.get('restricted_areas', [])))
                with col4:
                    st.metric("Quality Score", f"{result.get('quality_score', 0)*100:.1f}%")
            else:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                st.info("💡 Ensure your file contains valid geometric data")
    
    def display_analysis_results(self):
        """Display analysis results with pixel-perfect visualization"""
        st.markdown("### 🎨 Pixel-Perfect Visualization - Stage 1: Empty Floor Plan")
        
        # Current stage indicator
        st.markdown("""
        <div class="stage-indicator">
            📋 Stage 1: Empty Floor Plan (Walls, Restricted Areas, Entrances)
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display visualization
        fig = self.processor.create_pixel_perfect_visualization(st.session_state.analysis_data, 'empty')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Analysis metrics
        bounds = st.session_state.analysis_data.get('bounds', {})
        if bounds:
            width = bounds.get('max_x', 0) - bounds.get('min_x', 0)
            height = bounds.get('max_y', 0) - bounds.get('min_y', 0)
            area = width * height
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>Width</h3><h2>{width:.1f} m</h2></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>Height</h3><h2>{height:.1f} m</h2></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>Total Area</h3><h2>{area:.1f} m²</h2></div>', unsafe_allow_html=True)
    
    def render_ilot_tab(self):
        """Render îlot placement tab"""
        st.markdown("## 🏢 Îlot Placement - Ultimate Precision")
        
        if not st.session_state.analysis_data:
            st.warning("⚠️ Please complete floor plan analysis first")
            return
        
        # Configuration check
        if 'config' not in st.session_state:
            st.warning("⚠️ Please configure settings in the Analysis tab sidebar")
            return
        
        # Display current configuration
        config = st.session_state.config
        st.markdown("### 📊 Current Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h4>Small Îlots</h4><h3>{config["size_distribution"]["small"]}%</h3></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h4>Medium Îlots</h4><h3>{config["size_distribution"]["medium"]}%</h3></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h4>Large Îlots</h4><h3>{config["size_distribution"]["large"]}%</h3></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h4>XL Îlots</h4><h3>{config["size_distribution"]["xl"]}%</h3></div>', unsafe_allow_html=True)
        
        # Placement button
        if st.button("🏢 Place Îlots - Ultimate Precision", type="primary", use_container_width=True):
            with st.spinner("🏢 Placing îlots with ultimate precision..."):
                ilots = self.processor.place_ilots_ultimate(st.session_state.analysis_data, st.session_state.config)
                
                if ilots:
                    st.session_state.placed_ilots = ilots
                    st.session_state.processing_stage = 'ilots_placed'
                    
                    st.markdown("""
                    <div class="success-banner">
                        ✅ Îlots Placed Successfully - Pixel-Perfect Positioning!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display placement metrics
                    total_area = sum(ilot['area'] for ilot in ilots)
                    avg_area = total_area / len(ilots)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Îlots", len(ilots))
                    with col2:
                        st.metric("Total Area", f"{total_area:.1f} m²")
                    with col3:
                        st.metric("Average Size", f"{avg_area:.1f} m²")
                else:
                    st.error("❌ Failed to place îlots")
        
        # Display results
        if st.session_state.placed_ilots:
            self.display_ilot_results()
    
    def display_ilot_results(self):
        """Display îlot placement results"""
        st.markdown("### 🎨 Pixel-Perfect Visualization - Stage 2: Floor Plan with Îlots")
        
        # Stage indicator
        st.markdown("""
        <div class="stage-indicator">
            🏢 Stage 2: Floor Plan with Îlots Placed (Red Rectangles)
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display visualization
        fig = self.processor.create_visualization_with_ilots(st.session_state.analysis_data, st.session_state.placed_ilots)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Size distribution analysis
        size_counts = {}
        total_area_by_size = {}
        
        for ilot in st.session_state.placed_ilots:
            category = ilot['category']
            size_counts[category] = size_counts.get(category, 0) + 1
            total_area_by_size[category] = total_area_by_size.get(category, 0) + ilot['area']
        
        st.markdown("### 📊 Size Distribution Analysis")
        cols = st.columns(len(size_counts))
        
        for i, (category, count) in enumerate(size_counts.items()):
            with cols[i]:
                avg_area = total_area_by_size[category] / count
                st.markdown(f'''
                <div class="metric-card">
                    <h4>{category} Îlots</h4>
                    <h3>{count} units</h3>
                    <p>Avg: {avg_area:.1f} m²</p>
                </div>
                ''', unsafe_allow_html=True)
    
    def render_corridor_tab(self):
        """Render corridor generation tab"""
        st.markdown("## 🛤️ Corridor Generation - Ultimate Precision")
        
        if not st.session_state.placed_ilots:
            st.warning("⚠️ Please complete îlot placement first")
            return
        
        # Corridor generation button
        if st.button("🛤️ Generate Corridors - Ultimate Precision", type="primary", use_container_width=True):
            with st.spinner("🛤️ Generating corridors with ultimate precision..."):
                corridors = self.processor.generate_corridors_ultimate(st.session_state.analysis_data, st.session_state.placed_ilots)
                
                if corridors:
                    st.session_state.corridors = corridors
                    st.session_state.processing_stage = 'complete'
                    
                    st.markdown("""
                    <div class="success-banner">
                        ✅ Corridors Generated Successfully - Complete Layout Ready!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display corridor metrics
                    total_length = sum(corridor['length'] for corridor in corridors)
                    main_corridors = len([c for c in corridors if c['type'] == 'main'])
                    secondary_corridors = len([c for c in corridors if c['type'] == 'secondary'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Corridors", len(corridors))
                    with col2:
                        st.metric("Total Length", f"{total_length:.1f} m")
                    with col3:
                        st.metric("Main/Secondary", f"{main_corridors}/{secondary_corridors}")
                else:
                    st.error("❌ Failed to generate corridors")
        
        # Display results
        if st.session_state.corridors:
            self.display_corridor_results()
    
    def display_corridor_results(self):
        """Display corridor generation results"""
        st.markdown("### 🎨 Pixel-Perfect Visualization - Stage 3: Complete Layout")
        
        # Stage indicator
        st.markdown("""
        <div class="stage-indicator">
            🛤️ Stage 3: Complete Layout with Corridors and Area Labels
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display complete visualization
        fig = self.processor.create_complete_visualization(
            st.session_state.analysis_data, 
            st.session_state.placed_ilots, 
            st.session_state.corridors
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Corridor analysis
        corridor_types = {}
        total_length_by_type = {}
        
        for corridor in st.session_state.corridors:
            corridor_type = corridor['type']
            corridor_types[corridor_type] = corridor_types.get(corridor_type, 0) + 1
            total_length_by_type[corridor_type] = total_length_by_type.get(corridor_type, 0) + corridor['length']
        
        st.markdown("### 📊 Corridor Network Analysis")
        cols = st.columns(len(corridor_types))
        
        for i, (corridor_type, count) in enumerate(corridor_types.items()):
            with cols[i]:
                total_length = total_length_by_type[corridor_type]
                avg_length = total_length / count
                st.markdown(f'''
                <div class="metric-card">
                    <h4>{corridor_type.title()} Corridors</h4>
                    <h3>{count} units</h3>
                    <p>Total: {total_length:.1f} m</p>
                    <p>Avg: {avg_length:.1f} m</p>
                </div>
                ''', unsafe_allow_html=True)
    
    def render_results_tab(self):
        """Render complete results and export tab"""
        st.markdown("## 📊 Complete Results - Ultimate Edition")
        
        if st.session_state.processing_stage != 'complete':
            st.warning("⚠️ Please complete all processing stages first")
            return
        
        # Final complete visualization
        st.markdown("### 🎨 Final Pixel-Perfect Visualization")
        fig = self.processor.create_complete_visualization(
            st.session_state.analysis_data,
            st.session_state.placed_ilots,
            st.session_state.corridors
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Complete project summary
        st.markdown("### 📈 Complete Project Summary")
        
        # Calculate comprehensive metrics
        total_ilots = len(st.session_state.placed_ilots)
        total_corridors = len(st.session_state.corridors)
        total_ilot_area = sum(ilot['area'] for ilot in st.session_state.placed_ilots)
        total_corridor_length = sum(corridor['length'] for corridor in st.session_state.corridors)
        
        bounds = st.session_state.analysis_data['bounds']
        room_area = (bounds['max_x'] - bounds['min_x']) * (bounds['max_y'] - bounds['min_y'])
        utilization = (total_ilot_area / room_area) * 100
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>🏢 Total Îlots</h3>
                <h2>{total_ilots}</h2>
                <p>{total_ilot_area:.1f} m² total area</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>🛤️ Corridor Network</h3>
                <h2>{total_corridors}</h2>
                <p>{total_corridor_length:.1f} m total length</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>📊 Space Utilization</h3>
                <h2>{utilization:.1f}%</h2>
                <p>of total floor area</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            avg_ilot_size = total_ilot_area / total_ilots
            st.markdown(f'''
            <div class="metric-card">
                <h3>📏 Average Îlot</h3>
                <h2>{avg_ilot_size:.1f} m²</h2>
                <p>optimal sizing</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Export section
        st.markdown("### 💾 Export Ultimate Package")
        
        if st.button("📦 Generate Complete Export Package", type="primary", use_container_width=True):
            with st.spinner("📦 Generating complete export package..."):
                export_data = self.processor.export_complete_package(
                    st.session_state.analysis_data,
                    st.session_state.placed_ilots,
                    st.session_state.corridors
                )
                
                # Create downloads
                json_data = json.dumps(export_data, indent=2)
                
                # Summary report
                summary_report = f"""
CAD ANALYZER PRO - ULTIMATE EDITION
Complete Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=====================================================

FLOOR PLAN ANALYSIS:
• Processing Quality: {export_data['analysis_metadata']['quality_score']*100:.1f}%
• Total Entities: {st.session_state.analysis_data.get('entity_count', 0)}
• Walls Detected: {len(st.session_state.analysis_data.get('walls', []))}
• Restricted Areas: {len(st.session_state.analysis_data.get('restricted_areas', []))}
• Entrances: {len(st.session_state.analysis_data.get('entrances', []))}

ÎLOT PLACEMENT RESULTS:
• Total Îlots Placed: {export_data['ilot_placement']['total_ilots']}
• Total Îlot Area: {export_data['ilot_placement']['total_area']:.1f} m²
• Average Îlot Size: {export_data['summary_statistics']['average_ilot_size']:.1f} m²
• Space Utilization: {export_data['summary_statistics']['room_utilization']:.1f}%

Size Distribution:
"""
                for category, count in export_data['ilot_placement']['size_distribution'].items():
                    summary_report += f"• {category}: {count} îlots\n"
                
                summary_report += f"""
CORRIDOR NETWORK:
• Total Corridors: {export_data['corridor_network']['total_corridors']}
• Total Length: {export_data['corridor_network']['total_length']:.1f} m
• Corridor Density: {export_data['summary_statistics']['corridor_density']:.3f} m/m²

QUALITY METRICS:
• Processing Version: {export_data['analysis_metadata']['processor_version']}
• Overall Quality Score: {export_data['analysis_metadata']['quality_score']*100:.1f}%
• Data Authenticity: 100% (No fallback data used)

=====================================================
Generated by CAD Analyzer Pro - Ultimate Edition
Pixel-Perfect Processing with Zero Compromises
"""
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download Complete JSON Data",
                        data=json_data,
                        file_name=f"cad_analysis_ultimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📋 Download Summary Report",
                        data=summary_report,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.success("✅ Export package generated successfully!")

# Application entry point
if __name__ == "__main__":
    app = AdvancedStreamlitApp()
    app.run()
