import os
import logging
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    # Fallback for local development
    DATABASE_URL = "sqlite:///./floor_plan_analyzer.db"

# Create database engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FloorPlanAnalysis(Base):
    """Database model for floor plan analysis results"""
    __tablename__ = "floor_plan_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Analysis results
    zones_data = Column(JSON)
    ilots_data = Column(JSON)
    corridors_data = Column(JSON)
    optimization_data = Column(JSON)
    
    # Metrics
    total_zones = Column(Integer)
    usable_area = Column(Float)
    efficiency_score = Column(Float)
    accessibility_score = Column(Float)
    safety_compliance = Column(Boolean)
    
    # Metadata
    analysis_parameters = Column(JSON)
    processing_time = Column(Float)
    user_session = Column(String)

class ProjectConfiguration(Base):
    """Database model for project configurations"""
    __tablename__ = "project_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Configuration data
    ilot_configuration = Column(JSON)
    corridor_configuration = Column(JSON)
    optimization_settings = Column(JSON)
    analysis_preferences = Column(JSON)
    
    # Project metadata
    description = Column(Text)
    tags = Column(JSON)
    is_active = Column(Boolean, default=True)

class DatabaseManager:
    """Manager for database operations"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def save_analysis(self, analysis_data: Dict[str, Any], session_id: str) -> Optional[int]:
        """Save floor plan analysis to database"""
        try:
            db = self.get_session()
            
            analysis = FloorPlanAnalysis(
                filename=analysis_data.get('filename', 'unknown'),
                file_type=analysis_data.get('file_type', 'unknown'),
                file_size=analysis_data.get('file_size', 0),
                zones_data=analysis_data.get('zones', {}),
                ilots_data=analysis_data.get('ilots', {}),
                corridors_data=analysis_data.get('corridors', {}),
                optimization_data=analysis_data.get('optimization', {}),
                total_zones=analysis_data.get('total_zones', 0),
                usable_area=analysis_data.get('usable_area', 0.0),
                efficiency_score=analysis_data.get('efficiency_score', 0.0),
                accessibility_score=analysis_data.get('accessibility_score', 0.0),
                safety_compliance=analysis_data.get('safety_compliance', False),
                analysis_parameters=analysis_data.get('parameters', {}),
                processing_time=analysis_data.get('processing_time', 0.0),
                user_session=session_id
            )
            
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            
            analysis_id = analysis.id
            db.close()
            
            logger.info(f"Analysis saved with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return None
    
    def get_analysis_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history for a session"""
        try:
            db = self.get_session()
            
            analyses = db.query(FloorPlanAnalysis)\
                        .filter(FloorPlanAnalysis.user_session == session_id)\
                        .order_by(FloorPlanAnalysis.upload_timestamp.desc())\
                        .limit(limit)\
                        .all()
            
            history = []
            for analysis in analyses:
                history.append({
                    'id': analysis.id,
                    'filename': analysis.filename,
                    'file_type': analysis.file_type,
                    'upload_timestamp': analysis.upload_timestamp,
                    'total_zones': analysis.total_zones,
                    'usable_area': analysis.usable_area,
                    'efficiency_score': analysis.efficiency_score,
                    'accessibility_score': analysis.accessibility_score,
                    'safety_compliance': analysis.safety_compliance
                })
            
            db.close()
            return history
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
    
    def save_project_configuration(self, config_data: Dict[str, Any]) -> Optional[int]:
        """Save project configuration"""
        try:
            db = self.get_session()
            
            config = ProjectConfiguration(
                project_name=config_data.get('project_name', 'Untitled Project'),
                ilot_configuration=config_data.get('ilot_config', {}),
                corridor_configuration=config_data.get('corridor_config', {}),
                optimization_settings=config_data.get('optimization_settings', {}),
                analysis_preferences=config_data.get('analysis_preferences', {}),
                description=config_data.get('description', ''),
                tags=config_data.get('tags', [])
            )
            
            db.add(config)
            db.commit()
            db.refresh(config)
            
            config_id = config.id
            db.close()
            
            logger.info(f"Configuration saved with ID: {config_id}")
            return config_id
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return None
    
    def get_project_configurations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all project configurations"""
        try:
            db = self.get_session()
            
            configs = db.query(ProjectConfiguration)\
                       .filter(ProjectConfiguration.is_active == True)\
                       .order_by(ProjectConfiguration.updated_at.desc())\
                       .limit(limit)\
                       .all()
            
            config_list = []
            for config in configs:
                config_list.append({
                    'id': config.id,
                    'project_name': config.project_name,
                    'created_at': config.created_at,
                    'updated_at': config.updated_at,
                    'description': config.description,
                    'tags': config.tags,
                    'ilot_configuration': config.ilot_configuration,
                    'corridor_configuration': config.corridor_configuration,
                    'optimization_settings': config.optimization_settings,
                    'analysis_preferences': config.analysis_preferences
                })
            
            db.close()
            return config_list
            
        except Exception as e:
            logger.error(f"Error getting configurations: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            db = self.get_session()
            db.execute("SELECT 1")
            db.close()
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

def init_database():
    """Initialize database tables"""
    db_manager.create_tables()

def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    return db_manager