
"""
Render Configuration for CAD Analyzer Pro
Single clean configuration file for Render deployment
"""

import os
import logging
from typing import Dict, Any

class RenderConfig:
    """Production configuration for Render deployment"""
    
    def __init__(self):
        self.environment = os.getenv('RENDER_ENVIRONMENT', 'production')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.port = int(os.getenv('PORT', 5000))
        self.host = '0.0.0.0'
        
        # Memory limits for Render
        self.max_file_size = 20 * 1024 * 1024  # 20MB
        self.max_entities = 1000
        self.max_ilots = 100
        
        # Database configuration
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///app.db')
        
        # Logging configuration
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit configuration"""
        return {
            'server': {
                'address': self.host,
                'port': self.port,
                'headless': True,
                'enableCORS': False,
                'enableXsrfProtection': False
            },
            'browser': {
                'gatherUsageStats': False,
                'serverAddress': self.host,
                'serverPort': self.port
            },
            'theme': {
                'primaryColor': '#6366f1',
                'backgroundColor': '#ffffff',
                'secondaryBackgroundColor': '#f0f2f6',
                'textColor': '#262730'
            }
        }
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == 'production'
    
    def get_memory_limits(self) -> Dict[str, int]:
        """Get memory limits"""
        return {
            'max_file_size': self.max_file_size,
            'max_entities': self.max_entities,
            'max_ilots': self.max_ilots
        }

# Global configuration instance
render_config = RenderConfig()
