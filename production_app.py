"""
Production DWG Analyzer Pro - Complete Implementation
Matching client requirements exactly with PostgreSQL integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import uuid
from datetime import datetime
import base64
import io
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import os
from pathlib import Path
import hashlib
warnings.filterwarnings('ignore')

# Production system imports
from utils.production_database import production_db
from utils.production_floor_analyzer import ProductionFloorAnalyzer
from utils.production_ilot_system import ProductionIlotPlacer

# Configure page
st.set_page_config(
    page_title="DWG Analyzer Pro - Production",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for production UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}

.zone-legend {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
}

.ilot-config {
    background: #e8f4fd;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.status-success {
    background: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 5px;
    border-left: 4px solid #28a745;
}

.status-warning {
    background: #fff3cd;
    color: #856404;
    padding: 0.75rem;
    border-radius: 5px;
    border-left: 4px solid #ffc107;
}

.status-error {
    background: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 5px;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main production application"""
    # Initialize session state
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = None
    if 'floor_analyzer' not in st.session_state:
        st.session_state.floor_analyzer = None
    if 'ilot_placer' not in st.session_state:
        st.session_state.ilot_placer = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Main header
    st.markdown('<h1 class="main-header">🏢 DWG Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Production Floor Plan Analysis with Intelligent Îlot Placement</p>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col2:
        if st.button("📁 Projects", use_container_width=True):
            st.session_state.current_page = 'projects'
            st.rerun()
    
    with col3:
        if st.button("🔄 Analysis", use_container_width=True):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col4:
        if st.button("📊 Results", use_container_width=True):
            st.session_state.current_page = 'results'
            st.rerun()
    
    with col5:
        if st.button("📋 Reports", use_container_width=True):
            st.session_state.current_page = 'reports'
            st.rerun()
    
    st.markdown("---")
    
    # Route to pages
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'projects':
        show_project_management()
    elif st.session_state.current_page == 'analysis':
        show_floor_plan_analysis()
    elif st.session_state.current_page == 'results':
        show_results_visualization()
    elif st.session_state.current_page == 'reports':
        show_export_reports()

def show_home_page():
    """Display home page with file upload"""
    st.markdown("## 🚀 Quick Start")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Floor Plan")
        uploaded_file = st.file_uploader(
            "Choose a floor plan file",
            type=['dxf', 'dwg', 'jpg', 'jpeg', 'png', 'pdf'],
            help="Supported formats: DXF, DWG, JPG, PNG, PDF"
        )
        
        if uploaded_file is not None:
            # Process file
            process_uploaded_file(uploaded_file)
    
    with col2:
        st.markdown('<div class="zone-legend">', unsafe_allow_html=True)
        st.markdown("### 🎨 Zone Color Legend")
        st.markdown("**As per client requirements:**")
        st.markdown("🔴 **Red**: Entrances/Exits - NO îlots allowed")
        st.markdown("🔵 **Light Blue**: Restricted areas (stairs, elevators)")
        st.markdown("⚫ **Black**: Walls - îlots CAN touch (except near entrances)")
        st.markdown("⚪ **White**: Available space for îlot placement")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent projects
    st.markdown("## 📁 Recent Projects")
    try:
        projects = production_db.get_all_projects()
        if projects:
            for project in projects[:5]:
                with st.expander(f"📄 {project['name']} - {project['created_at'].strftime('%Y-%m-%d')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Îlots", project.get('ilot_count', 0))
                    with col2:
                        st.metric("Corridors", project.get('corridor_count', 0))
                    with col3:
                        if st.button(f"Open", key=f"open_{project['id']}"):
                            st.session_state.current_project_id = str(project['id'])
                            st.session_state.current_page = 'analysis'
                            st.rerun()
        else:
            st.info("No projects found. Upload a floor plan to get started!")
    except Exception as e:
        st.error(f"Database connection error: {e}")

def process_uploaded_file(uploaded_file):
    """Process uploaded floor plan file"""
    try:
        # Read file content
        file_content = uploaded_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Create project
        project_name = f"Project {uploaded_file.name}"
        project_id = production_db.create_project(
            name=project_name,
            description=f"Floor plan analysis from {uploaded_file.name}",
            metadata={
                'filename': uploaded_file.name,
                'file_size': len(file_content),
                'file_hash': file_hash
            }
        )
        
        # Initialize analyzer
        analyzer = ProductionFloorAnalyzer()
        
        # Process based on file type
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if file_ext in ['dxf', 'dwg']:
            result = analyzer.process_dxf_file(file_content, uploaded_file.name)
        elif file_ext in ['jpg', 'jpeg', 'png']:
            result = analyzer.process_image_file(file_content, uploaded_file.name)
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return
        
        if result['success']:
            # Store floor plan data
            floor_plan_id = production_db.store_floor_plan(
                project_id=project_id,
                entities=result['entities'],
                zones={},
                walls=result['walls'],
                restricted_areas=result['restricted_areas'],
                entrances=result['entrances'],
                bounds=result['bounds']
            )
            
            # Store in session
            st.session_state.current_project_id = project_id
            st.session_state.floor_analyzer = analyzer
            
            # Show success
            st.markdown('<div class="status-success">', unsafe_allow_html=True)
            st.markdown(f"✅ **File processed successfully!**")
            st.markdown(f"- **Walls detected**: {result['wall_count']}")
            st.markdown(f"- **Restricted areas**: {result['restricted_count']}")
            st.markdown(f"- **Entrances**: {result['entrance_count']}")
            st.markdown(f"- **Total entities**: {result['entity_count']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Start Analysis", type="primary"):
                st.session_state.current_page = 'analysis'
                st.rerun()
        else:
            st.error(f"Failed to process file: {result['error']}")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")

def show_project_management():
    """Display project management interface"""
    st.markdown("## 📁 Project Management")
    
    try:
        projects = production_db.get_all_projects()
        
        if not projects:
            st.info("No projects found. Upload a floor plan to create your first project!")
            return
        
        # Projects table
        df = pd.DataFrame(projects)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display projects
        for i, project in enumerate(projects):
            with st.expander(f"📄 {project['name']}", expanded=i==0):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"**Created**: {project['created_at'].strftime('%Y-%m-%d')}")
                    st.markdown(f"**Status**: {project['status']}")
                
                with col2:
                    st.metric("Îlots", project.get('ilot_count', 0))
                    st.metric("Corridors", project.get('corridor_count', 0))
                
                with col3:
                    if st.button(f"📊 Open Project", key=f"open_project_{project['id']}"):
                        st.session_state.current_project_id = str(project['id'])
                        st.session_state.current_page = 'analysis'
                        st.rerun()
                
                with col4:
                    if st.button(f"🗑️ Delete", key=f"delete_{project['id']}"):
                        production_db.delete_project(str(project['id']))
                        st.rerun()
                
                # Show description if available
                if project.get('description'):
                    st.markdown(f"**Description**: {project['description']}")
    
    except Exception as e:
        st.error(f"Error loading projects: {e}")

def show_floor_plan_analysis():
    """Display floor plan analysis interface - EXACT CLIENT REQUIREMENTS"""
    if not st.session_state.current_project_id:
        st.warning("Please select a project first")
        return
    
    st.markdown("## 🔄 Floor Plan Analysis")
    
    try:
        # Load project data
        project_data = production_db.get_project_data(st.session_state.current_project_id)
        
        if not project_data or not project_data.get('floor_plan'):
            st.error("No floor plan data found for this project")
            return
        
        floor_plan = project_data['floor_plan']
        
        # Initialize analyzer with stored data
        if not st.session_state.floor_analyzer:
            analyzer = ProductionFloorAnalyzer()
            analyzer.entities = floor_plan.get('original_entities', [])
            analyzer.walls = floor_plan.get('walls', [])
            analyzer.restricted_areas = floor_plan.get('restricted_areas', [])
            analyzer.entrances = floor_plan.get('entrances', [])
            analyzer.bounds = floor_plan.get('bounds', {})
            st.session_state.floor_analyzer = analyzer
        
        analyzer = st.session_state.floor_analyzer
        
        # Show floor plan preview
        st.markdown("### 📋 Floor Plan Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Walls", len(analyzer.walls))
        with col2:
            st.metric("Restricted Areas", len(analyzer.restricted_areas))
        with col3:
            st.metric("Entrances", len(analyzer.entrances))
        with col4:
            available_area = analyzer.calculate_available_area()
            st.metric("Available Area", f"{available_area:.1f} m²")
        
        # Îlot Configuration - CLIENT REQUIREMENTS
        st.markdown("### 🏠 Îlot Layout Configuration")
        st.markdown('<div class="ilot-config">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Size Distribution (as per client requirements):**")
            size_0_1 = st.slider("0-1 m² îlots (%)", 0, 100, 10, help="Small îlots")
            size_1_3 = st.slider("1-3 m² îlots (%)", 0, 100, 25, help="Medium îlots")
            size_3_5 = st.slider("3-5 m² îlots (%)", 0, 100, 30, help="Large îlots")
            size_5_10 = st.slider("5-10 m² îlots (%)", 0, 100, 35, help="Extra large îlots")
        
        with col2:
            st.markdown("**Placement Rules:**")
            min_spacing = st.number_input("Minimum spacing between îlots (m)", 0.5, 5.0, 1.0, 0.1)
            wall_clearance = st.number_input("Wall clearance (m)", 0.1, 2.0, 0.5, 0.1)
            entrance_clearance = st.number_input("Entrance clearance (m)", 1.0, 5.0, 2.0, 0.1)
            corridor_width = st.number_input("Corridor width (m)", 1.0, 3.0, 1.5, 0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation
        total_percentage = size_0_1 + size_1_3 + size_3_5 + size_5_10
        if total_percentage != 100:
            st.warning(f"Size percentages must sum to 100%. Current total: {total_percentage}%")
        
        # Configuration object
        config = {
            'size_0_1_percent': size_0_1,
            'size_1_3_percent': size_1_3,
            'size_3_5_percent': size_3_5,
            'size_5_10_percent': size_5_10,
            'min_spacing': min_spacing,
            'wall_clearance': wall_clearance,
            'entrance_clearance': entrance_clearance,
            'corridor_width': corridor_width,
            'total_area': available_area
        }
        
        # Generate îlots button
        if st.button("🏗️ Generate Îlot Placement", type="primary", disabled=(total_percentage != 100)):
            with st.spinner("Analyzing floor plan and placing îlots..."):
                # Initialize îlot placer
                placer = ProductionIlotPlacer()
                placer.load_floor_plan_data(
                    walls=analyzer.walls,
                    restricted_areas=analyzer.restricted_areas,
                    entrances=analyzer.entrances,
                    zones={},
                    bounds=analyzer.bounds
                )
                
                # Process placement
                placement_result = placer.process_full_placement(config)
                
                if placement_result['ilots']:
                    # Store configuration
                    config_id = production_db.store_ilot_configuration(
                        st.session_state.current_project_id, config
                    )
                    
                    # Store îlot placements
                    production_db.store_ilot_placements(
                        st.session_state.current_project_id,
                        config_id,
                        placement_result['ilots']
                    )
                    
                    # Store corridors
                    production_db.store_corridors(
                        st.session_state.current_project_id,
                        placement_result['corridors']
                    )
                    
                    # Store analysis results
                    production_db.store_analysis_results(
                        st.session_state.current_project_id,
                        placement_result['metrics']
                    )
                    
                    # Update session state
                    st.session_state.analysis_results = placement_result
                    st.session_state.ilot_placer = placer
                    
                    st.success(f"✅ Successfully placed {len(placement_result['ilots'])} îlots with {len(placement_result['corridors'])} corridors!")
                    
                    # Show quick metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Îlots Placed", len(placement_result['ilots']))
                    with col2:
                        st.metric("Corridors", len(placement_result['corridors']))
                    with col3:
                        st.metric("Space Utilization", f"{placement_result['metrics']['space_utilization']*100:.1f}%")
                    with col4:
                        st.metric("Efficiency Score", f"{placement_result['metrics']['efficiency_score']*100:.1f}%")
                    
                    if st.button("📊 View Results", type="primary"):
                        st.session_state.current_page = 'results'
                        st.rerun()
                else:
                    st.error("Failed to place îlots. Please check your configuration and try again.")
    
    except Exception as e:
        st.error(f"Analysis error: {e}")

def show_results_visualization():
    """Display results with visualization matching client images"""
    if not st.session_state.current_project_id:
        st.warning("Please select a project and run analysis first")
        return
    
    st.markdown("## 📊 Analysis Results")
    
    try:
        # Load project data
        project_data = production_db.get_project_data(st.session_state.current_project_id)
        
        if not project_data.get('ilots'):
            st.warning("No îlot placement data found. Please run analysis first.")
            return
        
        # Show metrics
        if project_data.get('analysis'):
            analysis = project_data['analysis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Space Utilization", f"{analysis.get('space_utilization', 0)*100:.1f}%")
            with col2:
                st.metric("Coverage", f"{analysis.get('coverage_percentage', 0)*100:.1f}%")
            with col3:
                st.metric("Efficiency Score", f"{analysis.get('efficiency_score', 0)*100:.1f}%")
            with col4:
                st.metric("Safety Compliance", f"{analysis.get('safety_compliance', 0)*100:.1f}%")
        
        # Visualization - MATCHING CLIENT REQUIREMENTS
        st.markdown("### 🎨 Floor Plan Visualization")
        
        # Create visualization
        fig = create_production_visualization(project_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Îlot breakdown
        st.markdown("### 📋 Îlot Summary")
        
        ilots = project_data['ilots']
        ilot_df = pd.DataFrame(ilots)
        
        if not ilot_df.empty:
            # Group by size category
            summary = ilot_df.groupby('size_category').agg({
                'area': ['count', 'sum', 'mean']
            }).round(2)
            
            summary.columns = ['Count', 'Total Area (m²)', 'Average Area (m²)']
            st.dataframe(summary, use_container_width=True)
            
            # Corridor summary
            if project_data.get('corridors'):
                st.markdown("### 🛤️ Corridor Network")
                corridors = project_data['corridors']
                
                corridor_df = pd.DataFrame(corridors)
                corridor_summary = corridor_df.groupby('corridor_type').agg({
                    'width': 'mean',
                    'is_mandatory': 'sum'
                }).round(2)
                
                st.dataframe(corridor_summary, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying results: {e}")

def create_production_visualization(project_data):
    """Create visualization matching client reference images"""
    fig = go.Figure()
    
    # Floor plan bounds
    floor_plan = project_data.get('floor_plan', {})
    bounds = floor_plan.get('bounds', {})
    
    if not bounds:
        return fig
    
    # Add walls (black lines)
    walls = floor_plan.get('walls', [])
    for wall in walls:
        if len(wall) >= 2:
            x_coords = [point[0] for point in wall] + [wall[0][0]]
            y_coords = [point[1] for point in wall] + [wall[0][1]]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(color='black', width=3),
                name='Walls',
                showlegend=False
            ))
    
    # Add restricted areas (light blue)
    restricted_areas = floor_plan.get('restricted_areas', [])
    for area in restricted_areas:
        if len(area) >= 3:
            x_coords = [point[0] for point in area] + [area[0][0]]
            y_coords = [point[1] for point in area] + [area[0][1]]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.5)',
                line=dict(color='#3498DB', width=2),
                name='Restricted Areas',
                showlegend=False
            ))
    
    # Add entrances (red)
    entrances = floor_plan.get('entrances', [])
    for entrance in entrances:
        if len(entrance) >= 3:
            x_coords = [point[0] for point in entrance] + [entrance[0][0]]
            y_coords = [point[1] for point in entrance] + [entrance[0][1]]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.7)',
                line=dict(color='#E74C3C', width=2),
                name='Entrances/Exits',
                showlegend=False
            ))
    
    # Add îlots with area labels - MATCHING CLIENT IMAGE 3
    ilots = project_data.get('ilots', [])
    size_colors = {
        'size_0_1': '#FF6B6B',    # Red for 0-1 m²
        'size_1_3': '#4ECDC4',    # Teal for 1-3 m²
        'size_3_5': '#45B7D1',    # Blue for 3-5 m²
        'size_5_10': '#96CEB4'    # Green for 5-10 m²
    }
    
    for ilot in ilots:
        x = ilot['x']
        y = ilot['y']
        w = ilot['width']
        h = ilot['height']
        area = ilot['area']
        category = ilot['size_category']
        
        # Create rectangle
        x_rect = [x-w/2, x+w/2, x+w/2, x-w/2, x-w/2]
        y_rect = [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2]
        
        color = size_colors.get(category, '#95A5A6')
        
        fig.add_trace(go.Scatter(
            x=x_rect, y=y_rect,
            fill='toself',
            fillcolor=color,
            line=dict(color='#2C3E50', width=1),
            name=f'{category}',
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Area: {area:.1f} m²<br>Category: {category}"
        ))
        
        # Add area label - MATCHING CLIENT REQUIREMENT
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[f"{area:.1f}m²"],
            textfont=dict(size=10, color='#2C3E50'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add corridors
    corridors = project_data.get('corridors', [])
    for corridor in corridors:
        path_points = corridor.get('path_points', [])
        if len(path_points) >= 2:
            width = corridor.get('width', 1.5)
            
            # Create corridor polygon
            x_coords = [point[0] for point in path_points]
            y_coords = [point[1] for point in path_points]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(color='#F39C12', width=width*5),  # Scale for visibility
                name='Corridors',
                showlegend=False
            ))
    
    # Set layout
    fig.update_layout(
        title="Floor Plan with Îlot Placement",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    return fig

def show_export_reports():
    """Display export and reporting options"""
    st.markdown("## 📋 Export & Reports")
    
    if not st.session_state.current_project_id:
        st.warning("Please select a project first")
        return
    
    try:
        project_data = production_db.get_project_data(st.session_state.current_project_id)
        
        if not project_data:
            st.error("No project data found")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Export Options")
            
            if st.button("📋 Generate PDF Report"):
                st.info("PDF report generation would be implemented here")
            
            if st.button("🖼️ Export Visualization"):
                st.info("Image export would be implemented here")
            
            if st.button("📄 Export Data (JSON)"):
                # Export project data as JSON
                export_data = {
                    'project': project_data['project'],
                    'ilots': project_data['ilots'],
                    'corridors': project_data['corridors'],
                    'analysis': project_data['analysis']
                }
                
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=json_str,
                    file_name=f"project_{st.session_state.current_project_id}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("### 📈 Project Summary")
            
            if project_data.get('ilots'):
                st.metric("Total Îlots", len(project_data['ilots']))
                
                total_area = sum(ilot['area'] for ilot in project_data['ilots'])
                st.metric("Total Îlot Area", f"{total_area:.1f} m²")
            
            if project_data.get('corridors'):
                st.metric("Total Corridors", len(project_data['corridors']))
            
            if project_data.get('analysis'):
                analysis = project_data['analysis']
                st.metric("Space Utilization", f"{analysis.get('space_utilization', 0)*100:.1f}%")
    
    except Exception as e:
        st.error(f"Error loading export data: {e}")

if __name__ == "__main__":
    main()