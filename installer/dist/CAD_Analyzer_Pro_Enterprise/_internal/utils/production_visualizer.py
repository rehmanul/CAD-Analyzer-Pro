"""
Production Visualizer - Client Visual Requirements Implementation
Creates visualizations matching the expected output images with proper color coding
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json

class ProductionVisualizer:
    """Production-grade visualizer matching client requirements"""
    
    def __init__(self):
        # Color scheme matching client requirements
        self.colors = {
            'walls': '#000000',           # Black lines
            'restricted': '#ADD8E6',      # Light blue (stairs, elevators)
            'entrances': '#FF0000',       # Red (entrances/exits)
            'ilot_small': '#FFFF00',      # Yellow (0-1 m²)
            'ilot_medium': '#FFA500',     # Orange (1-3 m²)
            'ilot_large': '#008000',      # Green (3-5 m²)
            'ilot_xlarge': '#800080',     # Purple (5-10 m²)
            'corridor_mandatory': '#0000FF',  # Blue (mandatory corridors)
            'corridor_access': '#00FFFF',     # Cyan (access corridors)
            'background': '#FFFFFF',      # White background
            'grid': '#E0E0E0'            # Light gray grid
        }
        
        self.size_category_map = {
            'size_0_1': 'ilot_small',
            'size_1_3': 'ilot_medium', 
            'size_3_5': 'ilot_large',
            'size_5_10': 'ilot_xlarge'
        }
    
    def create_complete_floor_plan_view(self, analysis_data: Dict, ilots: List[Dict], 
                                      corridors: List[Dict]) -> go.Figure:
        """
        Create complete floor plan visualization matching client expected output
        Shows: walls (black), restricted areas (blue), entrances (red), 
               îlots (color-coded by size), corridors (blue)
        """
        
        fig = go.Figure()
        
        # Add background grid
        self._add_background_grid(fig, analysis_data.get('bounds', {}))
        
        # Add walls (black lines) - CLIENT REQUIREMENT
        self._add_walls_to_plot(fig, analysis_data.get('walls', []))
        
        # Add restricted areas (light blue) - CLIENT REQUIREMENT  
        self._add_restricted_areas_to_plot(fig, analysis_data.get('restricted_areas', []))
        
        # Add entrances (red zones) - CLIENT REQUIREMENT
        self._add_entrances_to_plot(fig, analysis_data.get('entrances', []))
        
        # Add îlots with size-based color coding - CLIENT REQUIREMENT
        self._add_ilots_to_plot(fig, ilots)
        
        # Add corridors - CLIENT REQUIREMENT
        self._add_corridors_to_plot(fig, corridors)
        
        # Configure layout to match expected output
        self._configure_layout(fig, analysis_data.get('bounds', {}))
        
        return fig
    
    def _add_background_grid(self, fig: go.Figure, bounds: Dict):
        """Add subtle background grid"""
        if not bounds:
            return
        
        # Create grid lines
        x_range = bounds['max_x'] - bounds['min_x']
        y_range = bounds['max_y'] - bounds['min_y']
        
        grid_spacing = max(x_range, y_range) / 20  # 20 grid lines
        
        # Vertical grid lines
        x_start = bounds['min_x']
        while x_start <= bounds['max_x']:
            fig.add_shape(
                type="line",
                x0=x_start, y0=bounds['min_y'],
                x1=x_start, y1=bounds['max_y'],
                line=dict(color=self.colors['grid'], width=0.5, dash="dot"),
                layer="below"
            )
            x_start += grid_spacing
        
        # Horizontal grid lines
        y_start = bounds['min_y']
        while y_start <= bounds['max_y']:
            fig.add_shape(
                type="line",
                x0=bounds['min_x'], y0=y_start,
                x1=bounds['max_x'], y1=y_start,
                line=dict(color=self.colors['grid'], width=0.5, dash="dot"),
                layer="below"
            )
            y_start += grid_spacing
    
    def _add_walls_to_plot(self, fig: go.Figure, walls: List):
        """Add walls as black lines - CLIENT REQUIREMENT"""
        for i, wall in enumerate(walls):
            if len(wall) >= 2:
                x_coords = [point[0] for point in wall]
                y_coords = [point[1] for point in wall]
                
                # Close polygon if needed
                if len(wall) > 2:
                    x_coords.append(wall[0][0])
                    y_coords.append(wall[0][1])
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color=self.colors['walls'], width=3),
                    name='Walls' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='walls',
                    hoverinfo='skip'
                ))
    
    def _add_restricted_areas_to_plot(self, fig: go.Figure, restricted_areas: List):
        """Add restricted areas as light blue zones - CLIENT REQUIREMENT"""
        for i, area in enumerate(restricted_areas):
            if len(area) >= 3:
                x_coords = [point[0] for point in area] + [area[0][0]]
                y_coords = [point[1] for point in area] + [area[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor=f'rgba(173, 216, 230, 0.6)',  # Light blue with transparency
                    line=dict(color=self.colors['restricted'], width=2),
                    name='Restricted Areas' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='restricted',
                    hovertemplate='<b>Restricted Area</b><br>Type: Stairs/Elevator<extra></extra>'
                ))
    
    def _add_entrances_to_plot(self, fig: go.Figure, entrances: List):
        """Add entrances as red zones - CLIENT REQUIREMENT"""
        for i, entrance in enumerate(entrances):
            if len(entrance) >= 3:
                x_coords = [point[0] for point in entrance] + [entrance[0][0]]
                y_coords = [point[1] for point in entrance] + [entrance[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.4)',  # Red with transparency
                    line=dict(color=self.colors['entrances'], width=2),
                    name='Entrances/Exits' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='entrances',
                    hovertemplate='<b>Entrance/Exit</b><br>No îlot placement allowed<extra></extra>'
                ))
    
    def _add_ilots_to_plot(self, fig: go.Figure, ilots: List[Dict]):
        """Add îlots with size-based color coding - CLIENT REQUIREMENT"""
        
        # Group îlots by size category for legend
        size_groups = {}
        for ilot in ilots:
            category = ilot.get('size_category', 'unknown')
            if category not in size_groups:
                size_groups[category] = []
            size_groups[category].append(ilot)
        
        # Add îlots by category
        for category, category_ilots in size_groups.items():
            color_key = self.size_category_map.get(category, 'ilot_small')
            color = self.colors[color_key]
            
            # Category name for legend
            category_names = {
                'size_0_1': 'Small Îlots (0-1 m²)',
                'size_1_3': 'Medium Îlots (1-3 m²)',
                'size_3_5': 'Large Îlots (3-5 m²)',
                'size_5_10': 'Extra Large Îlots (5-10 m²)'
            }
            
            for i, ilot in enumerate(category_ilots):
                # Create rectangular îlot
                x, y = ilot['x'], ilot['y']
                w, h = ilot['width'], ilot['height']
                
                x_coords = [x-w/2, x+w/2, x+w/2, x-w/2, x-w/2]
                y_coords = [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=1),
                    name=category_names.get(category, category) if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup=f'ilots_{category}',
                    hovertemplate=f'<b>Îlot {ilot["id"]}</b><br>' +
                                f'Size: {ilot["area"]:.1f} m²<br>' +
                                f'Category: {category}<br>' +
                                f'Position: ({x:.1f}, {y:.1f})<extra></extra>'
                ))
    
    def _add_corridors_to_plot(self, fig: go.Figure, corridors: List[Dict]):
        """Add corridors - CLIENT REQUIREMENT"""
        
        mandatory_corridors = []
        access_corridors = []
        
        for corridor in corridors:
            if corridor.get('is_mandatory', False):
                mandatory_corridors.append(corridor)
            else:
                access_corridors.append(corridor)
        
        # Add mandatory corridors (blue)
        for i, corridor in enumerate(mandatory_corridors):
            path_points = corridor.get('path_points', [])
            if len(path_points) >= 2:
                x_coords = [point[0] for point in path_points]
                y_coords = [point[1] for point in path_points]
                width = corridor.get('width', 1.5)
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color=self.colors['corridor_mandatory'], width=width*3),
                    name='Mandatory Corridors' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='corridors_mandatory',
                    hovertemplate=f'<b>Mandatory Corridor</b><br>' +
                                f'Width: {width:.1f} m<br>' +
                                f'Type: {corridor.get("type", "unknown")}<extra></extra>'
                ))
        
        # Add access corridors (cyan)
        for i, corridor in enumerate(access_corridors):
            path_points = corridor.get('path_points', [])
            if len(path_points) >= 2:
                x_coords = [point[0] for point in path_points]
                y_coords = [point[1] for point in path_points]
                width = corridor.get('width', 1.5)
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color=self.colors['corridor_access'], width=width*2),
                    name='Access Corridors' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='corridors_access',
                    hovertemplate=f'<b>Access Corridor</b><br>' +
                                f'Width: {width:.1f} m<br>' +
                                f'Type: {corridor.get("type", "unknown")}<extra></extra>'
                ))
    
    def _configure_layout(self, fig: go.Figure, bounds: Dict):
        """Configure layout to match expected output"""
        
        fig.update_layout(
            title={
                'text': "🏨 Hotel Floor Plan - Îlot Placement & Corridor Network",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title="X Coordinate (meters)",
            yaxis_title="Y Coordinate (meters)",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            height=700,
            margin=dict(l=50, r=150, t=80, b=50),
            
            # Equal aspect ratio for accurate representation
            xaxis=dict(
                scaleanchor="y", 
                scaleratio=1,
                showgrid=True,
                gridcolor=self.colors['grid'],
                gridwidth=1,
                zeroline=False
            ),
            yaxis=dict(
                scaleanchor="x", 
                scaleratio=1,
                showgrid=True,
                gridcolor=self.colors['grid'],
                gridwidth=1,
                zeroline=False
            ),
            
            # Hover settings
            hovermode='closest'
        )
        
        # Set axis ranges with padding
        if bounds:
            padding = max(bounds['max_x'] - bounds['min_x'], 
                         bounds['max_y'] - bounds['min_y']) * 0.1
            
            fig.update_xaxes(range=[bounds['min_x'] - padding, bounds['max_x'] + padding])
            fig.update_yaxes(range=[bounds['min_y'] - padding, bounds['max_y'] + padding])
    
    def create_analysis_summary_chart(self, metrics: Dict) -> go.Figure:
        """Create analysis summary chart"""
        
        # Metrics for radar chart
        categories = [
            'Space Utilization',
            'Coverage Efficiency', 
            'Distribution Quality',
            'Accessibility Score',
            'Circulation Efficiency',
            'Safety Compliance'
        ]
        
        values = [
            metrics.get('space_utilization', 0) * 100,
            metrics.get('coverage_percentage', 0) * 100,
            metrics.get('efficiency_score', 0) * 100,
            metrics.get('accessibility_score', 0) * 100,
            metrics.get('circulation_efficiency', 0) * 100,
            metrics.get('safety_compliance', 1.0) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 123, 255, 0.3)',
            line=dict(color='rgba(0, 123, 255, 1)', width=2),
            name='Performance Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Analysis Performance Summary",
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_size_distribution_chart(self, ilots: List[Dict], target_distribution: Dict) -> go.Figure:
        """Create size distribution comparison chart"""
        
        # Count actual distribution
        actual_counts = {'size_0_1': 0, 'size_1_3': 0, 'size_3_5': 0, 'size_5_10': 0}
        for ilot in ilots:
            category = ilot.get('size_category', 'size_0_1')
            if category in actual_counts:
                actual_counts[category] += 1
        
        total_ilots = sum(actual_counts.values())
        actual_percentages = {k: (v/total_ilots*100) if total_ilots > 0 else 0 
                            for k, v in actual_counts.items()}
        
        # Target percentages
        target_percentages = {
            'size_0_1': target_distribution.get('size_0_1_percent', 10),
            'size_1_3': target_distribution.get('size_1_3_percent', 25),
            'size_3_5': target_distribution.get('size_3_5_percent', 30),
            'size_5_10': target_distribution.get('size_5_10_percent', 35)
        }
        
        categories = ['Small (0-1 m²)', 'Medium (1-3 m²)', 'Large (3-5 m²)', 'Extra Large (5-10 m²)']
        target_values = [target_percentages['size_0_1'], target_percentages['size_1_3'], 
                        target_percentages['size_3_5'], target_percentages['size_5_10']]
        actual_values = [actual_percentages['size_0_1'], actual_percentages['size_1_3'],
                        actual_percentages['size_3_5'], actual_percentages['size_5_10']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Target Distribution',
            x=categories,
            y=target_values,
            marker_color='rgba(55, 128, 191, 0.7)',
            text=[f'{v:.1f}%' for v in target_values],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Actual Distribution',
            x=categories,
            y=actual_values,
            marker_color='rgba(219, 64, 82, 0.7)',
            text=[f'{v:.1f}%' for v in actual_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Îlot Size Distribution - Target vs Actual',
            xaxis_title='Size Categories',
            yaxis_title='Percentage (%)',
            barmode='group',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_corridor_network_analysis(self, corridors: List[Dict]) -> go.Figure:
        """Create corridor network analysis visualization"""
        
        if not corridors:
            # Empty state
            fig = go.Figure()
            fig.add_annotation(
                text="No corridors generated yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Corridor Network Analysis", height=400)
            return fig
        
        # Analyze corridor types
        corridor_types = {}
        total_length = 0
        
        for corridor in corridors:
            corridor_type = corridor.get('type', 'unknown')
            if corridor_type not in corridor_types:
                corridor_types[corridor_type] = {'count': 0, 'total_length': 0}
            
            corridor_types[corridor_type]['count'] += 1
            
            # Calculate length
            path_points = corridor.get('path_points', [])
            if len(path_points) >= 2:
                length = 0
                for i in range(len(path_points) - 1):
                    p1, p2 = path_points[i], path_points[i+1]
                    length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                corridor_types[corridor_type]['total_length'] += length
                total_length += length
        
        # Create pie chart
        labels = list(corridor_types.keys())
        values = [corridor_types[label]['count'] for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f'Corridor Network Analysis<br><sub>Total Length: {total_length:.1f}m</sub>',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def export_high_resolution_image(self, fig: go.Figure, filename: str = "floor_plan.png", 
                                   width: int = 1920, height: int = 1080, scale: int = 2) -> bytes:
        """Export high-resolution image"""
        
        # Convert plotly figure to high-res image
        img_bytes = fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,
            engine="kaleido"
        )
        
        return img_bytes
    
    def create_professional_report_layout(self, analysis_data: Dict, ilots: List[Dict], 
                                        corridors: List[Dict], metrics: Dict) -> Dict[str, go.Figure]:
        """Create complete set of professional report visualizations"""
        
        figures = {}
        
        # Main floor plan
        figures['main_plan'] = self.create_complete_floor_plan_view(
            analysis_data, ilots, corridors
        )
        
        # Performance metrics
        figures['metrics_radar'] = self.create_analysis_summary_chart(metrics)
        
        # Size distribution
        target_dist = {
            'size_0_1_percent': 10,
            'size_1_3_percent': 25, 
            'size_3_5_percent': 30,
            'size_5_10_percent': 35
        }
        figures['size_distribution'] = self.create_size_distribution_chart(ilots, target_dist)
        
        # Corridor analysis
        figures['corridor_analysis'] = self.create_corridor_network_analysis(corridors)
        
        return figures
    
    def generate_color_legend(self) -> Dict[str, str]:
        """Generate color legend for documentation"""
        return {
            'Walls (Black Lines)': self.colors['walls'],
            'Restricted Areas (Light Blue)': self.colors['restricted'],
            'Entrances/Exits (Red)': self.colors['entrances'],
            'Small Îlots 0-1m² (Yellow)': self.colors['ilot_small'],
            'Medium Îlots 1-3m² (Orange)': self.colors['ilot_medium'],
            'Large Îlots 3-5m² (Green)': self.colors['ilot_large'],
            'Extra Large Îlots 5-10m² (Purple)': self.colors['ilot_xlarge'],
            'Mandatory Corridors (Blue)': self.colors['corridor_mandatory'],
            'Access Corridors (Cyan)': self.colors['corridor_access']
        }