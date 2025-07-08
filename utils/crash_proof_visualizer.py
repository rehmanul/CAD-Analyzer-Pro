"""
Crash-Proof Visualizer - Prevents visualization crashes
Robust visualization with error handling and memory management
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import streamlit as st
from shapely.geometry import Polygon, Point
import traceback

logger = logging.getLogger(__name__)

class CrashProofVisualizer:
    """Crash-proof visualization system"""
    
    def __init__(self):
        self.max_elements = 100  # Reduced for memory efficiency
        self.colors = {
            'walls': '#000000',
            'restricted': '#0066CC',
            'entrances': '#FF0000',
            'ilots': {
                'size_0_1': '#FFB6C1',  # Light pink
                'size_1_3': '#FFA07A',  # Light salmon
                'size_3_5': '#FF6347',  # Tomato
                'size_5_10': '#FF4500'  # Orange red
            },
            'background': '#FFFFFF'
        }
    
    def create_safe_floor_plan(self, analysis_data: Dict, ilots: List[Dict]) -> Optional[go.Figure]:
        """Create floor plan visualization with error handling"""
        try:
            fig = go.Figure()
            
            # Set up safe bounds
            bounds = self._get_safe_bounds(analysis_data)
            
            # Add floor plan elements safely
            self._add_floor_plan_base(fig, bounds)
            
            # Add îlots with crash protection
            if ilots:
                self._add_ilots_safely(fig, ilots[:self.max_elements])
            
            # Configure layout safely
            self._configure_safe_layout(fig, bounds)
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            st.error(f"Visualization error: {str(e)}")
            return self._create_fallback_figure()
    
    def _get_safe_bounds(self, analysis_data: Dict) -> Dict:
        """Get safe bounds for visualization"""
        try:
            if 'bounds' in analysis_data:
                bounds = analysis_data['bounds']
                if all(key in bounds for key in ['min_x', 'max_x', 'min_y', 'max_y']):
                    return bounds
            
            # Fallback bounds
            return {
                'min_x': 0, 'max_x': 100,
                'min_y': 0, 'max_y': 80
            }
            
        except Exception as e:
            logger.error(f"Bounds extraction failed: {str(e)}")
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 80}
    
    def _add_floor_plan_base(self, fig: go.Figure, bounds: Dict):
        """Add floor plan base elements safely"""
        try:
            # Add floor boundary
            fig.add_shape(
                type="rect",
                x0=bounds['min_x'], y0=bounds['min_y'],
                x1=bounds['max_x'], y1=bounds['max_y'],
                line=dict(color=self.colors['walls'], width=2),
                fillcolor="rgba(255,255,255,0.1)"
            )
            
            # Add title annotation
            fig.add_annotation(
                x=(bounds['min_x'] + bounds['max_x']) / 2,
                y=bounds['max_y'] + 5,
                text="Floor Plan with Îlot Placement",
                showarrow=False,
                font=dict(size=16, color=self.colors['walls'])
            )
            
        except Exception as e:
            logger.error(f"Floor plan base creation failed: {str(e)}")
    
    def _add_ilots_safely(self, fig: go.Figure, ilots: List[Dict]):
        """Add îlots to visualization safely"""
        try:
            if not ilots:
                return
            
            # Group îlots by size category
            ilot_groups = {}
            for ilot in ilots:
                category = ilot.get('size_category', 'size_1_3')
                if category not in ilot_groups:
                    ilot_groups[category] = []
                ilot_groups[category].append(ilot)
            
            # Add each group
            for category, group_ilots in ilot_groups.items():
                self._add_ilot_group(fig, group_ilots, category)
            
        except Exception as e:
            logger.error(f"Îlot visualization failed: {str(e)}")
    
    def _add_ilot_group(self, fig: go.Figure, ilots: List[Dict], category: str):
        """Add a group of îlots safely"""
        try:
            color = self.colors['ilots'].get(category, '#FF6347')
            
            # Extract coordinates safely
            x_coords = []
            y_coords = []
            labels = []
            
            for ilot in ilots:
                try:
                    x = float(ilot.get('x', 0))
                    y = float(ilot.get('y', 0))
                    width = float(ilot.get('width', 1))
                    height = float(ilot.get('height', 1))
                    
                    # Add rectangle for îlot
                    fig.add_shape(
                        type="rect",
                        x0=x - width/2, y0=y - height/2,
                        x1=x + width/2, y1=y + height/2,
                        line=dict(color=color, width=1),
                        fillcolor=color,
                        opacity=0.7
                    )
                    
                    # Add center point for hover
                    x_coords.append(x)
                    y_coords.append(y)
                    labels.append(f"Îlot {ilot.get('id', 'N/A')}<br>Size: {category}<br>Area: {ilot.get('area', 0):.1f}m²")
                    
                except Exception as e:
                    logger.error(f"Individual îlot rendering failed: {str(e)}")
                    continue
            
            # Add scatter plot for hover information
            if x_coords and y_coords:
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=8,
                        opacity=0.8
                    ),
                    text=labels,
                    hoverinfo='text',
                    name=f"Îlots ({category})",
                    showlegend=True
                ))
            
        except Exception as e:
            logger.error(f"Îlot group rendering failed: {str(e)}")
    
    def _configure_safe_layout(self, fig: go.Figure, bounds: Dict):
        """Configure layout safely"""
        try:
            # Add margins
            margin = 10
            
            fig.update_layout(
                title="Floor Plan Analysis - Îlot Placement",
                xaxis=dict(
                    title="X Coordinate (m)",
                    range=[bounds['min_x'] - margin, bounds['max_x'] + margin],
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title="Y Coordinate (m)",
                    range=[bounds['min_y'] - margin, bounds['max_y'] + margin],
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                width=800,
                height=600,
                showlegend=True,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Ensure equal aspect ratio
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
        except Exception as e:
            logger.error(f"Layout configuration failed: {str(e)}")
    
    def _create_fallback_figure(self) -> go.Figure:
        """Create fallback visualization"""
        try:
            fig = go.Figure()
            
            # Add simple message
            fig.add_annotation(
                x=50, y=40,
                text="Visualization temporarily unavailable<br>Please try again",
                showarrow=False,
                font=dict(size=14, color='red')
            )
            
            fig.update_layout(
                title="Floor Plan Visualization",
                xaxis=dict(range=[0, 100]),
                yaxis=dict(range=[0, 80]),
                width=800,
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Fallback figure creation failed: {str(e)}")
            return go.Figure()
    
    def get_placement_metrics(self, ilots: List[Dict]) -> Dict:
        """Get placement metrics from ilots list"""
        try:
            if not ilots:
                return {
                    'total_ilots': 0,
                    'size_distribution': {},
                    'total_area': 0,
                    'average_size': 0
                }
            
            # Count by size category
            size_counts = {}
            total_area = 0
            
            for ilot in ilots:
                category = ilot.get('size_category', 'size_1_3')
                size_counts[category] = size_counts.get(category, 0) + 1
                total_area += ilot.get('area', 0)
            
            # Calculate distribution percentages
            total_count = len(ilots)
            size_distribution = {}
            for category, count in size_counts.items():
                size_distribution[category] = (count / total_count) * 100
            
            return {
                'total_ilots': total_count,
                'size_distribution': size_distribution,
                'total_area': total_area,
                'average_size': total_area / total_count if total_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            return {
                'total_ilots': 0,
                'size_distribution': {},
                'total_area': 0,
                'average_size': 0
            }
    
    def create_metrics_chart(self, metrics: Dict) -> Optional[go.Figure]:
        """Create metrics visualization safely"""
        try:
            if not metrics:
                return None
            
            # Size distribution chart
            size_dist = metrics.get('size_distribution', {})
            if size_dist:
                categories = list(size_dist.keys())
                values = list(size_dist.values())
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=categories,
                        y=values,
                        marker_color=['#FFB6C1', '#FFA07A', '#FF6347', '#FF4500']
                    )
                ])
                
                fig.update_layout(
                    title="Îlot Size Distribution",
                    xaxis_title="Size Category",
                    yaxis_title="Percentage (%)",
                    height=400
                )
                
                return fig
            
            return None
            
        except Exception as e:
            logger.error(f"Metrics chart creation failed: {str(e)}")
            return None

# Global instance
crash_proof_visualizer = CrashProofVisualizer()