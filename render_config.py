"""
Render Configuration for CAD Analyzer Pro
Production-ready configuration for Render deployment
"""

import os
import logging
from typing import Dict, Any

class RenderConfig:
    """Configuration class for Render deployment"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_environment()
    
    def setup_logging(self):
        """Setup logging for production"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log')
            ]
        )
    
    def setup_environment(self):
        """Setup environment variables for Render"""
        
        # Render-specific settings
        os.environ.setdefault('PYTHONPATH', '/opt/render/project/src')
        os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
        os.environ.setdefault('STREAMLIT_SERVER_PORT', '10000')
        os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')
        os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
        
        # Performance settings
        os.environ.setdefault('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', '200')
        os.environ.setdefault('STREAMLIT_SERVER_MAX_MESSAGE_SIZE', '200')
        os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
        
        # Security settings
        os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'false')
        os.environ.setdefault('STREAMLIT_CLIENT_TOOLBAR_MODE', 'minimal')
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for Render"""
        return {
            'host': os.environ.get('DATABASE_HOST', 'localhost'),
            'port': int(os.environ.get('DATABASE_PORT', '5432')),
            'database': os.environ.get('DATABASE_NAME', 'cad_analyzer_prod'),
            'username': os.environ.get('DATABASE_USER', 'cad_analyzer_user'),
            'password': os.environ.get('DATABASE_PASSWORD', ''),
            'url': os.environ.get('DATABASE_URL', ''),
            'ssl_mode': 'require' if os.environ.get('RENDER_ENV') == 'production' else 'prefer'
        }
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return {
            'debug': os.environ.get('RENDER_ENV') != 'production',
            'max_file_size': int(os.environ.get('MAX_FILE_SIZE', '200')),
            'cache_enabled': os.environ.get('CACHE_ENABLED', 'true').lower() == 'true',
            'psutil_available': os.environ.get('PSUTIL_AVAILABLE', 'true').lower() == 'true',
            'redis_enabled': os.environ.get('REDIS_ENABLED', 'false').lower() == 'true',
            'redis_url': os.environ.get('REDIS_URL', '')
        }

# Global configuration instance
render_config = RenderConfig()