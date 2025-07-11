"""
Phase 4 Integration Layer
Combines Export & Integration System with complete pipeline
Provides unified interface for comprehensive data export and system integration
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from phase4_export_integration import (
    ExportIntegrationSystem, ExportConfiguration, ExportFormat, 
    DataSummaryLevel, export_integration_system
)

@dataclass
class Phase4Configuration:
    """Combined configuration for Phase 4 export and integration"""
    export_formats: List[str]  # List of format names
    summary_level: str = "detailed"  # basic, detailed, comprehensive, technical
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_metrics: bool = True
    file_prefix: str = "cad_analysis"
    create_zip_package: bool = True
    enable_api_integration: bool = False

class Phase4IntegrationLayer:
    """
    Integration layer for Phase 4 Export & Integration System
    Provides the main interface for comprehensive data export and system integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Phase 4 components
        self.export_system = export_integration_system
        
        # Format mapping
        self.format_mapping = {
            'json': ExportFormat.JSON,
            'csv': ExportFormat.CSV,
            'pdf': ExportFormat.PDF,
            'svg': ExportFormat.SVG,
            'png': ExportFormat.PNG,
            'html': ExportFormat.HTML,
            'excel': ExportFormat.EXCEL,
            'zip_package': ExportFormat.ZIP_PACKAGE
        }
        
        # Summary level mapping
        self.summary_mapping = {
            'basic': DataSummaryLevel.BASIC,
            'detailed': DataSummaryLevel.DETAILED,
            'comprehensive': DataSummaryLevel.COMPREHENSIVE,
            'technical': DataSummaryLevel.TECHNICAL
        }

    def process_export_and_integration(self, analysis_data: Dict[str, Any], 
                                     visualizations: Optional[Dict[str, Any]], 
                                     config: Phase4Configuration) -> Dict[str, Any]:
        """
        Main Phase 4 processing method that handles comprehensive export and integration
        
        Args:
            analysis_data: Complete analysis data from all previous phases
            visualizations: Phase 3 visualizations
            config: Phase 4 configuration parameters
            
        Returns:
            Complete export package with all requested formats
        """
        
        try:
            # Convert configuration
            export_config = self._create_export_configuration(config)
            
            # Process export with visualization integration
            export_result = self.export_system.export_analysis_results(
                analysis_data, visualizations, export_config
            )
            
            # Add integration features
            integration_result = self._add_integration_features(
                export_result, analysis_data, config
            )
            
            # Generate comprehensive result
            result = {
                'success': True,
                'phase4_complete': True,
                'export_package': integration_result,
                'download_ready': True,
                'api_integration': config.enable_api_integration,
                'configuration_used': {
                    'formats': config.export_formats,
                    'summary_level': config.summary_level,
                    'zip_package': config.create_zip_package
                }
            }
            
            self.logger.info("Phase 4 export and integration completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Phase 4 export and integration: {str(e)}")
            return {
                'success': False,
                'phase4_complete': False,
                'error': str(e),
                'export_package': {},
                'download_ready': False
            }

    def _create_export_configuration(self, config: Phase4Configuration) -> ExportConfiguration:
        """Create export configuration from Phase 4 config"""
        
        # Convert format names to enums
        export_formats = []
        for format_name in config.export_formats:
            if format_name in self.format_mapping:
                export_formats.append(self.format_mapping[format_name])
        
        # Add ZIP package if requested
        if config.create_zip_package and ExportFormat.ZIP_PACKAGE not in export_formats:
            export_formats.append(ExportFormat.ZIP_PACKAGE)
        
        # Convert summary level
        summary_level = self.summary_mapping.get(
            config.summary_level, DataSummaryLevel.DETAILED
        )
        
        return ExportConfiguration(
            formats=export_formats,
            summary_level=summary_level,
            include_visualizations=config.include_visualizations,
            include_raw_data=config.include_raw_data,
            include_metrics=config.include_metrics,
            file_prefix=config.file_prefix,
            timestamp_suffix=True
        )

    def _add_integration_features(self, export_result: Dict[str, Any], 
                                analysis_data: Dict[str, Any], 
                                config: Phase4Configuration) -> Dict[str, Any]:
        """Add integration features to export result"""
        
        # Add API integration if enabled
        if config.enable_api_integration:
            export_result['api_endpoints'] = self._generate_api_endpoints(analysis_data)
            export_result['webhook_data'] = self._generate_webhook_data(analysis_data)
        
        # Add download helpers
        export_result['download_helpers'] = self._generate_download_helpers(export_result)
        
        # Add sharing capabilities
        export_result['sharing_options'] = self._generate_sharing_options(export_result)
        
        # Add integration documentation
        export_result['integration_docs'] = self._generate_integration_documentation(export_result)
        
        return export_result

    def _generate_api_endpoints(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API integration endpoints"""
        
        return {
            'data_endpoint': '/api/v2/analysis/data',
            'summary_endpoint': '/api/v2/analysis/summary',
            'visualization_endpoint': '/api/v2/analysis/visualizations',
            'export_endpoint': '/api/v2/analysis/export',
            'authentication': 'Bearer token required',
            'rate_limit': '100 requests per minute',
            'data_format': 'JSON',
            'version': '2.0'
        }

    def _generate_webhook_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate webhook integration data"""
        
        return {
            'webhook_url': 'https://your-system.com/webhook/cad-analysis',
            'payload_structure': {
                'event': 'analysis_complete',
                'timestamp': 'ISO 8601 format',
                'analysis_id': 'unique identifier',
                'summary': 'compressed summary data',
                'download_links': 'temporary download URLs'
            },
            'authentication': 'HMAC-SHA256 signature',
            'retry_policy': 'Exponential backoff, max 3 retries'
        }

    def _generate_download_helpers(self, export_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate download helper functions"""
        
        download_helpers = {}
        
        for format_name, file_data in export_result.get('files', {}).items():
            if file_data.get('success', True):
                download_helpers[format_name] = {
                    'filename': file_data.get('filename', f'download.{format_name}'),
                    'mimetype': file_data.get('mimetype', 'application/octet-stream'),
                    'size': file_data.get('size', 0),
                    'encoding': file_data.get('encoding', 'utf-8'),
                    'download_ready': True
                }
        
        return download_helpers

    def _generate_sharing_options(self, export_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sharing options for export package"""
        
        return {
            'email_sharing': {
                'enabled': True,
                'max_size': '25MB',
                'supported_formats': ['PDF', 'HTML', 'ZIP']
            },
            'cloud_sharing': {
                'enabled': True,
                'providers': ['Google Drive', 'Dropbox', 'OneDrive'],
                'link_expiry': '30 days'
            },
            'direct_link': {
                'enabled': True,
                'expiry': '7 days',
                'password_protected': False
            },
            'embed_options': {
                'html_embed': True,
                'iframe_embed': True,
                'widget_embed': False
            }
        }

    def _generate_integration_documentation(self, export_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integration documentation"""
        
        return {
            'quick_start': {
                'json_import': 'Import JSON file into your analysis system',
                'csv_analysis': 'Open CSV files in Excel or Google Sheets',
                'html_viewing': 'Open HTML report in any web browser',
                'image_embedding': 'Use PNG/SVG files in presentations'
            },
            'data_structure': {
                'coordinate_system': 'Coordinates in millimeters (mm)',
                'origin': 'Bottom-left corner (0,0)',
                'areas': 'Areas calculated in square meters (m²)',
                'angles': 'Angles in degrees (0-360)'
            },
            'api_integration': {
                'authentication': 'Bearer token in Authorization header',
                'pagination': 'Use offset and limit parameters',
                'filtering': 'Support for element type and date range filters',
                'webhooks': 'Real-time notifications for analysis completion'
            },
            'troubleshooting': {
                'large_files': 'Use ZIP package for files >10MB',
                'browser_compatibility': 'Modern browsers support all formats',
                'mobile_viewing': 'HTML reports are mobile-responsive'
            }
        }

    def create_streamlit_downloads(self, export_package: Dict[str, Any]) -> Dict[str, Any]:
        """Create Streamlit-compatible download buttons and data"""
        
        streamlit_downloads = {}
        
        files = export_package.get('files', {})
        
        for format_name, file_data in files.items():
            if file_data.get('success', True) and 'content' in file_data:
                streamlit_downloads[format_name] = {
                    'data': file_data['content'],
                    'filename': file_data.get('filename', f'download.{format_name}'),
                    'mime': file_data.get('mimetype', 'application/octet-stream'),
                    'encoding': file_data.get('encoding', 'utf-8'),
                    'size_mb': file_data.get('size', 0) / (1024 * 1024)
                }
        
        return streamlit_downloads

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return list(self.format_mapping.keys())

    def get_summary_levels(self) -> List[str]:
        """Get list of available summary levels"""
        return list(self.summary_mapping.keys())

    def get_phase4_capabilities(self) -> Dict[str, Any]:
        """Get information about Phase 4 export and integration capabilities"""
        return {
            'components': {
                'export_integration_system': {
                    'description': 'Comprehensive export with multiple formats and integration',
                    'formats': self.get_supported_formats(),
                    'summary_levels': self.get_summary_levels(),
                    'features': ['Multi-format export', 'API integration', 'Webhook support']
                }
            },
            'export_formats': {
                'json': 'Structured data for API integration',
                'csv': 'Spreadsheet-compatible data files',
                'html': 'Web-viewable comprehensive reports',
                'png': 'High-resolution visualization images',
                'svg': 'Vector graphics for presentations',
                'zip_package': 'Complete export package with all files'
            },
            'summary_levels': {
                'basic': 'Essential metrics and counts',
                'detailed': 'Comprehensive analysis with breakdowns',
                'comprehensive': 'Complete analysis with recommendations',
                'technical': 'Developer-focused technical details'
            },
            'integration_features': [
                'API endpoint generation',
                'Webhook integration support',
                'Download helper functions',
                'Cloud sharing options',
                'Email sharing capabilities',
                'Embed code generation',
                'Direct link sharing'
            ],
            'output_specifications': {
                'max_file_size': '200MB per export',
                'zip_compression': 'Automatic for packages >10MB',
                'encoding': 'UTF-8 for text formats',
                'image_resolution': 'Up to 2400x2400 pixels',
                'api_version': '2.0'
            }
        }

# Create global instance for easy import
phase4_processor = Phase4IntegrationLayer()