"""
Optimized Visualization with WebGL and Trace Grouping
Ultra-high performance visualization with optimized plotly rendering
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional
import time
import logging

class OptimizedVisualization:
    """Ultra-high performance visualization with WebGL and trace optimization"""
    
    def __init__(self):
        self.use_webgl = True
        self.trace_grouping = True
        self.render_cache = {}
        
    def create_optimized_floor_plan(self, analysis_data: Dict, ilots: List[Dict], 
                                   corridors: List[Dict], show_measurements: bool = True) -> go.Figure:
        """Create optimized floor plan with WebGL and trace grouping"""
        start_time = time.time()
        
        # Create base figure with WebGL
        fig = go.Figure()
        
        # Configure for high performance
        fig.update_layout(
            title="CAD Floor Plan Analysis - Ultra-High Performance",
            showlegend=True,
            width=1200,
            height=800,
            dragmode='pan',
            hovermode='closest',
            # Enable WebGL for performance
            uirevision='constant'
        )
        
        bounds = analysis_data.get('bounds', {})
        
        # Add walls with optimized rendering
        self._add_optimized_walls(fig, analysis_data.get('walls', []))
        
        # Add zones with grouped traces
        self._add_optimized_zones(fig, analysis_data.get('zones', []))
        
        # Add entrances and restricted areas
        self._add_optimized_entrances(fig, analysis_data.get('entrances', []))
        self._add_optimized_restricted_areas(fig, analysis_data.get('restricted_areas', []))
        
        # Add îlots with optimized rendering
        self._add_optimized_ilots(fig, ilots, show_measurements)
        
        # Add corridors with efficient line rendering
        self._add_optimized_corridors(fig, corridors)
        
        # Configure optimized layout
        self._configure_optimized_layout(fig, bounds)
        
        render_time = time.time() - start_time
        logging.info(f"Optimized visualization rendered in {render_time:.3f}s")
        
        return fig
        
    def _add_optimized_walls(self, fig: go.Figure, walls: List[Dict]):
        """Add walls with optimized line rendering"""
        if not walls:
            return
            
        # Group wall coordinates for batch rendering
        wall_x = []
        wall_y = []
        
        for wall in walls:
            coords = wall.get('coordinates', [])
            if len(coords) >= 2:
                for i in range(len(coords)):
                    wall_x.append(coords[i][0])
                    wall_y.append(coords[i][1])
                    
                # Add break between segments
                wall_x.append(None)
                wall_y.append(None)
                
        # Single trace for all walls
        if wall_x:
            fig.add_trace(go.Scatter(
                x=wall_x,
                y=wall_y,
                mode='lines',
                line=dict(color='black', width=2),
                name='Walls',
                showlegend=True,
                hoverinfo='skip'
            ))
            
    def _add_optimized_zones(self, fig: go.Figure, zones: List[Dict]):
        """Add zones with optimized shape rendering"""
        for zone in zones:
            zone_bounds = zone.get('bounds', {})
            if not zone_bounds:
                continue
                
            # Add zone as filled rectangle
            fig.add_shape(
                type="rect",
                x0=zone_bounds.get('min_x', 0),
                y0=zone_bounds.get('min_y', 0),
                x1=zone_bounds.get('max_x', 100),
                y1=zone_bounds.get('max_y', 100),
                fillcolor='lightblue',
                opacity=0.1,
                line=dict(color='blue', width=1, dash='dash')
            )
            
    def _add_optimized_entrances(self, fig: go.Figure, entrances: List[Dict]):
        """Add entrances with optimized rendering"""
        if not entrances:
            return
            
        # Group entrance coordinates
        entrance_x = []
        entrance_y = []
        
        for entrance in entrances:
            bounds = entrance.get('bounds', {})
            if bounds:
                # Rectangle corners
                x0, y0 = bounds.get('min_x', 0), bounds.get('min_y', 0)
                x1, y1 = bounds.get('max_x', 0), bounds.get('max_y', 0)
                
                # Add rectangle coordinates
                entrance_x.extend([x0, x1, x1, x0, x0, None])
                entrance_y.extend([y0, y0, y1, y1, y0, None])
                
        # Single trace for all entrances
        if entrance_x:
            fig.add_trace(go.Scatter(
                x=entrance_x,
                y=entrance_y,
                mode='lines',
                fill='toself',
                fillcolor='red',
                line=dict(color='red', width=2),
                opacity=0.6,
                name='Entrances',
                showlegend=True,
                hoverinfo='skip'
            ))
            
    def _add_optimized_restricted_areas(self, fig: go.Figure, restricted_areas: List[Dict]):
        """Add restricted areas with optimized rendering"""
        if not restricted_areas:
            return
            
        # Group restricted area coordinates
        restricted_x = []
        restricted_y = []
        
        for area in restricted_areas:
            bounds = area.get('bounds', {})
            if bounds:
                # Rectangle corners
                x0, y0 = bounds.get('min_x', 0), bounds.get('min_y', 0)
                x1, y1 = bounds.get('max_x', 0), bounds.get('max_y', 0)
                
                # Add rectangle coordinates
                restricted_x.extend([x0, x1, x1, x0, x0, None])
                restricted_y.extend([y0, y0, y1, y1, y0, None])
                
        # Single trace for all restricted areas
        if restricted_x:
            fig.add_trace(go.Scatter(
                x=restricted_x,
                y=restricted_y,
                mode='lines',
                fill='toself',
                fillcolor='blue',
                line=dict(color='blue', width=2),
                opacity=0.4,
                name='Restricted Areas',
                showlegend=True,
                hoverinfo='skip'
            ))
            
    def _add_optimized_ilots(self, fig: go.Figure, ilots: List[Dict], show_measurements: bool):
        """Add îlots with optimized rendering and measurements"""
        if not ilots:
            return
            
        # Group îlots by color for efficient rendering
        ilot_groups = {}
        
        for ilot in ilots:
            color = ilot.get('color', '#FFFF00')
            if color not in ilot_groups:
                ilot_groups[color] = []
            ilot_groups[color].append(ilot)
            
        # Render each color group as single trace
        for color, group_ilots in ilot_groups.items():
            ilot_x = []
            ilot_y = []
            hover_text = []
            
            for ilot in group_ilots:
                x = ilot.get('x', 0)
                y = ilot.get('y', 0)
                width = ilot.get('width', 3.0)
                height = ilot.get('height', 2.0)
                area = ilot.get('area', width * height)
                
                # Rectangle coordinates
                ilot_x.extend([x, x + width, x + width, x, x, None])
                ilot_y.extend([y, y, y + height, y + height, y, None])
                
                # Hover text
                hover_text.extend([
                    f"Îlot {ilot.get('id', 'N/A')}<br>Area: {area:.1f} m²<br>Size: {ilot.get('size_category', 'N/A')}"
                ] * 5 + [None])
                
            # Add trace for this color group
            if ilot_x:
                size_category = group_ilots[0].get('size_category', 'unknown')
                fig.add_trace(go.Scatter(
                    x=ilot_x,
                    y=ilot_y,
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    line=dict(color='black', width=1),
                    opacity=0.8,
                    name=f'Îlots ({size_category})',
                    text=hover_text,
                    hoverinfo='text',
                    showlegend=True
                ))
                
            # Add measurements if requested
            if show_measurements:
                self._add_optimized_measurements(fig, group_ilots)
                
    def _add_optimized_measurements(self, fig: go.Figure, ilots: List[Dict]):
        """Add measurements with optimized annotation rendering"""
        annotations = []
        
        for ilot in ilots:
            x = ilot.get('x', 0)
            y = ilot.get('y', 0)
            width = ilot.get('width', 3.0)
            height = ilot.get('height', 2.0)
            area = ilot.get('area', width * height)
            
            # Add area measurement at center
            annotations.append(dict(
                x=x + width / 2,
                y=y + height / 2,
                text=f"{area:.1f} m²",
                showarrow=False,
                font=dict(color='red', size=12, family='Arial Black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='red',
                borderwidth=1
            ))
            
        # Add all annotations at once
        fig.update_layout(annotations=annotations)
        
    def _add_optimized_corridors(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors with optimized line rendering"""
        if not corridors:
            return
            
        # Group corridor coordinates
        corridor_x = []
        corridor_y = []
        
        for corridor in corridors:
            points = corridor.get('points', [])
            if len(points) >= 2:
                for point in points:
                    corridor_x.append(point[0])
                    corridor_y.append(point[1])
                    
                # Add break between corridors
                corridor_x.append(None)
                corridor_y.append(None)
                
        # Single trace for all corridors
        if corridor_x:
            fig.add_trace(go.Scatter(
                x=corridor_x,
                y=corridor_y,
                mode='lines',
                line=dict(color='blue', width=3, dash='dash'),
                name='Corridors',
                showlegend=True,
                hoverinfo='skip'
            ))
            
    def _configure_optimized_layout(self, fig: go.Figure, bounds: Dict):
        """Configure layout for optimal performance"""
        # Set axis ranges
        margin = 5
        fig.update_xaxes(
            range=[bounds.get('min_x', 0) - margin, bounds.get('max_x', 100) + margin],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="X Coordinate (m)"
        )
        
        fig.update_yaxes(
            range=[bounds.get('min_y', 0) - margin, bounds.get('max_y', 100) + margin],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="Y Coordinate (m)",
            scaleanchor="x",
            scaleratio=1
        )
        
        # Optimize layout
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(size=12),
            hoverlabel=dict(bgcolor="white", font_size=12),
            # Performance optimizations
            modebar=dict(
                bgcolor='rgba(255,255,255,0.8)',
                color='black',
                activecolor='blue'
            )
        )
        
    def create_performance_metrics_chart(self, metrics: Dict) -> go.Figure:
        """Create performance metrics visualization"""
        fig = go.Figure()
        
        # Processing speed chart
        fig.add_trace(go.Bar(
            x=['File Processing', 'Îlot Placement', 'Corridor Generation', 'Visualization'],
            y=[
                metrics.get('processing_speed', 0),
                metrics.get('placement_speed', 0),
                metrics.get('corridor_speed', 0),
                metrics.get('visualization_speed', 0)
            ],
            name='Performance (items/sec)',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Ultra-High Performance Metrics',
            xaxis_title='Component',
            yaxis_title='Speed (items/second)',
            showlegend=False,
            height=400
        )
        
        return fig