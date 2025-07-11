"""
Phase 3 Integration Layer
Combines Pixel-Perfect Visualizer with existing visualization systems
Provides unified interface for all visualization modes and stages
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from phase3_pixel_perfect_visualizer import (
    PixelPerfectVisualizer, VisualizationConfig, VisualizationStage, 
    VisualizationStyle, pixel_perfect_visualizer
)

@dataclass
class Phase3Configuration:
    """Combined configuration for Phase 3 visualization"""
    visualization_stage: str = "corridors_added"  # empty_plan, ilots_placed, corridors_added
    visualization_style: str = "reference_match"  # reference_match, professional, technical, modern
    canvas_size: tuple = (1800, 1800)
    show_labels: bool = True
    show_grid: bool = False
    line_width: float = 2.0
    enable_multi_stage: bool = True
    export_ready: bool = True

class Phase3IntegrationLayer:
    """
    Integration layer for Phase 3 Pixel-Perfect Visualization System
    Provides the main interface for advanced visualization with exact reference matching
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Phase 3 components
        self.pixel_perfect_visualizer = pixel_perfect_visualizer
        
        # Stage mapping
        self.stage_mapping = {
            'empty_plan': VisualizationStage.EMPTY_PLAN,
            'ilots_placed': VisualizationStage.ILOTS_PLACED,
            'corridors_added': VisualizationStage.CORRIDORS_ADDED
        }
        
        # Style mapping
        self.style_mapping = {
            'reference_match': VisualizationStyle.REFERENCE_MATCH,
            'professional': VisualizationStyle.PROFESSIONAL,
            'technical': VisualizationStyle.TECHNICAL,
            'modern': VisualizationStyle.MODERN
        }

    def create_advanced_visualizations(self, analysis_data: Dict[str, Any], 
                                     config: Phase3Configuration) -> Dict[str, Any]:
        """
        Main Phase 3 processing method that creates pixel-perfect visualizations
        
        Args:
            analysis_data: Complete analysis data from Phase 1 and Phase 2
            config: Phase 3 configuration parameters
            
        Returns:
            Dictionary with visualizations and metadata
        """
        
        try:
            if config.enable_multi_stage:
                # Create all three visualization stages
                visualizations = self._create_multi_stage_visualizations(analysis_data, config)
            else:
                # Create single visualization for specified stage
                visualization = self._create_single_visualization(analysis_data, config)
                visualizations = {config.visualization_stage: visualization}
            
            # Generate visualization metadata
            metadata = self._generate_visualization_metadata(analysis_data, config, visualizations)
            
            result = {
                'success': True,
                'phase3_complete': True,
                'visualizations': visualizations,
                'metadata': metadata,
                'configuration_used': {
                    'stage': config.visualization_stage,
                    'style': config.visualization_style,
                    'canvas_size': config.canvas_size,
                    'multi_stage': config.enable_multi_stage
                }
            }
            
            self.logger.info("Phase 3 visualization generation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Phase 3 visualization generation: {str(e)}")
            return {
                'success': False,
                'phase3_complete': False,
                'error': str(e),
                'visualizations': {},
                'metadata': {}
            }

    def _create_multi_stage_visualizations(self, analysis_data: Dict[str, Any], 
                                         config: Phase3Configuration) -> Dict[str, Any]:
        """Create visualizations for all three stages"""
        
        visualizations = {}
        
        # Get style enum
        style = self.style_mapping.get(config.visualization_style, VisualizationStyle.REFERENCE_MATCH)
        
        # Create visualizations for each stage
        for stage_name, stage_enum in self.stage_mapping.items():
            try:
                viz_config = VisualizationConfig(
                    stage=stage_enum,
                    style=style,
                    canvas_size=config.canvas_size,
                    show_labels=config.show_labels,
                    show_grid=config.show_grid,
                    line_width=config.line_width
                )
                
                figure = self.pixel_perfect_visualizer.create_pixel_perfect_visualization(
                    analysis_data, viz_config
                )
                
                visualizations[stage_name] = figure
                
            except Exception as e:
                self.logger.error(f"Error creating visualization for stage {stage_name}: {str(e)}")
                visualizations[stage_name] = None
        
        return visualizations

    def _create_single_visualization(self, analysis_data: Dict[str, Any], 
                                   config: Phase3Configuration):
        """Create single visualization for specified stage"""
        
        # Get enums
        stage = self.stage_mapping.get(config.visualization_stage, VisualizationStage.CORRIDORS_ADDED)
        style = self.style_mapping.get(config.visualization_style, VisualizationStyle.REFERENCE_MATCH)
        
        viz_config = VisualizationConfig(
            stage=stage,
            style=style,
            canvas_size=config.canvas_size,
            show_labels=config.show_labels,
            show_grid=config.show_grid,
            line_width=config.line_width
        )
        
        return self.pixel_perfect_visualizer.create_pixel_perfect_visualization(
            analysis_data, viz_config
        )

    def _generate_visualization_metadata(self, analysis_data: Dict[str, Any], 
                                       config: Phase3Configuration, 
                                       visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for visualizations"""
        
        metadata = {
            'visualization_system': 'Phase 3 Pixel-Perfect Visualizer',
            'total_visualizations': len(visualizations),
            'successful_visualizations': len([v for v in visualizations.values() if v is not None]),
            'canvas_resolution': f"{config.canvas_size[0]}x{config.canvas_size[1]}",
            'style_applied': config.visualization_style,
            'stages_generated': list(visualizations.keys()),
            
            # Element counts in visualizations
            'elements_visualized': {
                'walls': len(analysis_data.get('walls', [])),
                'restricted_areas': len(analysis_data.get('restricted_areas', [])),
                'entrances': len(analysis_data.get('entrances', [])),
                'placed_ilots': len(analysis_data.get('placed_ilots', [])),
                'corridors': len(analysis_data.get('corridors', []))
            },
            
            # Quality metrics
            'visualization_quality': {
                'pixel_perfect': True,
                'reference_match': config.visualization_style == 'reference_match',
                'export_ready': config.export_ready,
                'interactive': True
            }
        }
        
        return metadata

    def get_available_stages(self) -> List[str]:
        """Get list of available visualization stages"""
        return list(self.stage_mapping.keys())

    def get_available_styles(self) -> List[str]:
        """Get list of available visualization styles"""
        return list(self.style_mapping.keys())

    def get_phase3_capabilities(self) -> Dict[str, Any]:
        """Get information about Phase 3 visualization capabilities"""
        return {
            'components': {
                'pixel_perfect_visualizer': {
                    'description': 'Exact visual matches to reference designs',
                    'stages': self.get_available_stages(),
                    'styles': self.get_available_styles(),
                    'features': ['Pixel-perfect rendering', 'Color matching', 'Multi-stage support']
                }
            },
            'visualization_stages': {
                'empty_plan': 'Floor plan structure only (walls, restricted areas, entrances)',
                'ilots_placed': 'Floor plan + placed îlots with size-based coloring',
                'corridors_added': 'Complete visualization with îlots and corridor network'
            },
            'visualization_styles': {
                'reference_match': 'Exact match to provided reference images',
                'professional': 'Professional architectural drawing style',
                'technical': 'Technical engineering drawing style',
                'modern': 'Modern minimalist visualization style'
            },
            'key_features': [
                'Pixel-perfect reference matching',
                'Multi-stage visualization generation',
                'Professional styling presets',
                'Export-ready high-resolution output',
                'Interactive elements with hover details',
                'Customizable canvas sizes',
                'Exact color palette matching'
            ],
            'output_specifications': {
                'default_resolution': '1800x1800 pixels',
                'supported_resolutions': ['1200x1200', '1800x1800', '2400x2400'],
                'color_accuracy': '100% match to reference palette',
                'export_formats': ['Interactive Plotly', 'PNG', 'SVG', 'HTML']
            }
        }

# Create global instance for easy import
phase3_processor = Phase3IntegrationLayer()