import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class WebGLRenderer:
    def __init__(self):
        self.config = {
            'displayModeBar': False,
            'staticPlot': False,
            'responsive': True
        }
    
    def create_webgl_visualization(self, ilots, corridors, bounds):
        """Create WebGL-accelerated visualization"""
        fig = go.Figure()
        
        # Use WebGL for large datasets
        if len(ilots) > 100:
            self._add_webgl_ilots(fig, ilots)
        else:
            self._add_svg_ilots(fig, ilots)
        
        self._add_corridors(fig, corridors)
        self._add_boundaries(fig, bounds)
        
        fig.update_layout(
            title="Floor Plan - WebGL Accelerated",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=700,
            showlegend=False
        )
        
        return fig
    
    def _add_webgl_ilots(self, fig, ilots):
        """Add ilots using WebGL for performance"""
        colors = {'size_0_1': 'yellow', 'size_1_3': 'orange', 'size_3_5': 'green', 'size_5_10': 'purple'}
        
        # Group by category for batch rendering
        for category in colors:
            category_ilots = [i for i in ilots if i['size_category'] == category]
            if not category_ilots:
                continue
            
            x_coords = []
            y_coords = []
            
            for ilot in category_ilots:
                # Create rectangle vertices
                x, y = ilot['x'], ilot['y']
                w, h = ilot['width'], ilot['height']
                
                x_coords.extend([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2, None])
                y_coords.extend([y-h/2, y-h/2, y+h/2, y+h/2, y-h/2, None])
            
            fig.add_trace(go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode='lines',
                fill='toself',
                fillcolor=colors[category],
                line=dict(color=colors[category], width=1),
                name=category,
                hoverinfo='skip'
            ))
    
    def _add_svg_ilots(self, fig, ilots):
        """Add ilots using SVG for small datasets"""
        colors = {'size_0_1': 'yellow', 'size_1_3': 'orange', 'size_3_5': 'green', 'size_5_10': 'purple'}
        
        for ilot in ilots:
            x, y = ilot['x'], ilot['y']
            w, h = ilot['width'], ilot['height']
            color = colors.get(ilot['size_category'], 'gray')
            
            fig.add_shape(
                type="rect",
                x0=x-w/2, y0=y-h/2,
                x1=x+w/2, y1=y+h/2,
                fillcolor=color, opacity=0.7,
                line=dict(color=color, width=1)
            )
    
    def _add_corridors(self, fig, corridors):
        """Add corridors efficiently"""
        for corridor in corridors:
            points = corridor.get('path_points', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                fig.add_trace(go.Scattergl(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color='blue', width=corridor.get('width', 1.5)*2),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    def _add_boundaries(self, fig, bounds):
        """Add floor plan boundaries"""
        if bounds:
            fig.add_shape(
                type="rect",
                x0=bounds['min_x'], y0=bounds['min_y'],
                x1=bounds['max_x'], y1=bounds['max_y'],
                line=dict(color="black", width=3),
                fillcolor="rgba(0,0,0,0)"
            )

# Global renderer
webgl_renderer = WebGLRenderer()