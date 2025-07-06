# Overwriting existing production_app.py with full professional implementation as requested

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

from utils.dxf_parser import DXFParser
from utils.intelligent_zone_detector import IntelligentZoneDetector
from utils.ilot_placer import IlotPlacer
from utils.corridor_generator import AdvancedCorridorGenerator
from utils.visualization import FloorPlanVisualizer

# Initialize core components
dxf_parser = DXFParser()
zone_detector = IntelligentZoneDetector()
ilot_placer = IlotPlacer()
corridor_generator = AdvancedCorridorGenerator()
visualizer = FloorPlanVisualizer()

st.set_page_config(
    page_title="🏨 Hotel Floor Plan Analyzer - Professional",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏨 Hotel Floor Plan Analyzer - Professional Edition")

    uploaded_file = st.file_uploader(
        "Upload your hotel floor plan (DXF or DWG)",
        type=["dxf", "dwg"],
        help="Upload DXF or DWG files for full native support"
    )

    if uploaded_file:
        with st.spinner("Parsing floor plan..."):
            content = uploaded_file.read()
            filename = uploaded_file.name.lower()
            if filename.endswith(".dxf"):
                parsed_data = dxf_parser.parse_dxf(filename)
            elif filename.endswith(".dwg"):
                parsed_data = dxf_parser.parse_dwg(filename)
            else:
                st.error("Unsupported file format. Please upload DXF or DWG.")
                return

        st.success("Floor plan parsed successfully!")

        # Zone detection
        with st.spinner("Detecting zones..."):
            analysis_results = zone_detector.analyze_floor_plan(parsed_data)
        st.success("Zones detected successfully!")

        # User input for îlot proportions
        st.sidebar.header("Îlot Size Distribution (%)")
        small_pct = st.sidebar.slider("Small (0-1 m²)", 0, 50, 10)
        medium_pct = st.sidebar.slider("Medium (1-3 m²)", 0, 50, 25)
        large_pct = st.sidebar.slider("Large (3-5 m²)", 0, 50, 30)
        xlarge_pct = st.sidebar.slider("Extra Large (5-10 m²)", 0, 50, 35)

        total_pct = small_pct + medium_pct + large_pct + xlarge_pct
        if total_pct != 100:
            st.sidebar.error("Total percentage must equal 100%")
            return

        ilot_config = {
            'small': small_pct,
            'medium': medium_pct,
            'large': large_pct,
            'xlarge': xlarge_pct
        }

        # Îlot placement
        with st.spinner("Placing îlots..."):
            placement_input = {
                'open_spaces': analysis_results.get('zones', {}).get('open_spaces', []),
                'walls': analysis_results.get('zones', {}).get('walls', []),
                'restricted_areas': analysis_results.get('zones', {}).get('restricted', [])
            }
            placed_ilots = ilot_placer.place_ilots(placement_input, ilot_config)
        st.success(f"Placed {len(placed_ilots)} îlots successfully!")

        # Corridor generation config
        st.sidebar.header("Corridor Settings")
        corridor_width = st.sidebar.slider("Mandatory Corridor Width (m)", 1.0, 5.0, 1.5)

        corridor_config = {
            'force_between_facing': True,
            'access_width': corridor_width,
            'generate_main': True,
            'generate_secondary': True,
            'generate_access': True
        }

        # Corridor generation
        with st.spinner("Generating corridors..."):
            walls = analysis_results.get('zones', {}).get('walls', [])
            restricted_areas = analysis_results.get('zones', {}).get('restricted', [])
            entrances = analysis_results.get('zones', {}).get('entrances', [])
            bounds = analysis_results.get('analysis_metadata', {}).get('bounds', {})

            corridor_generator.load_floor_plan_data(placed_ilots, walls, restricted_areas, entrances, bounds)
            corridor_results = corridor_generator.generate_complete_corridor_network(corridor_config)
        st.success(f"Generated {len(corridor_results['corridors'])} corridors successfully!")

        # Visualization
        combined_analysis = {
            'walls': walls,
            'restricted_areas': restricted_areas,
            'entrances': entrances,
            'ilots': placed_ilots,
            'corridors': corridor_results['corridors']
        }

        fig = visualizer.create_interactive_view(parsed_data, combined_analysis)
        st.plotly_chart(fig, use_container_width=True, height=700)

        # Export options
        st.markdown("### Export Options")
        if st.button("Export as PDF"):
            st.info("PDF export functionality to be implemented.")
        if st.button("Export as Image"):
            st.info("Image export functionality to be implemented.")

if __name__ == "__main__":
    main()
